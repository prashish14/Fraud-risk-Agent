"""Tests for the A2A JSON-RPC server.

Uses ``httpx.AsyncClient`` with ``ASGITransport`` for in-process testing.
The agent graph is mocked so tests need no database or LLM.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from fraud_risk_agent.a2a.server import app, task_manager
from fraud_risk_agent.a2a.task_manager import TaskStatus


@pytest.fixture(autouse=True)
def _clear_tasks() -> None:
    """Reset the in-memory task store between tests."""
    task_manager._tasks.clear()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ── Health & agent card ──────────────────────────────────────────────────


async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_agent_card(client: AsyncClient) -> None:
    resp = await client.get("/.well-known/agent.json")
    assert resp.status_code == 200
    card = resp.json()
    assert card["name"] == "fraud-risk-agent"
    assert card["version"] == "1.0.0"
    assert len(card["skills"]) == 1
    assert card["skills"][0]["name"] == "assess_customer_risk"
    assert card["url"]  # Must be non-empty at runtime


# ── tasks/send ───────────────────────────────────────────────────────────


async def test_tasks_send_creates_task(client: AsyncClient) -> None:
    """A valid tasks/send returns a task in submitted state."""
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "skill": "assess_customer_risk",
                "customer_id": "C-123",
            },
            "id": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    result = body["result"]
    assert result["status"] == "submitted"
    assert result["skill"] == "assess_customer_risk"
    assert "id" in result


async def test_tasks_send_completes_with_stub(client: AsyncClient) -> None:
    """When the agent graph is unavailable, the task completes with a stub assessment."""
    with patch(
        "fraud_risk_agent.a2a.server._run_agent",
        wraps=None,
    ) as mock_run:
        # Use the real _run_agent but patch the import to fail.
        mock_run.side_effect = None

        resp = await client.post(
            "/a2a",
            json={
                "jsonrpc": "2.0",
                "method": "tasks/send",
                "params": {
                    "skill": "assess_customer_risk",
                    "customer_id": "C-456",
                },
                "id": 2,
            },
        )
        assert resp.status_code == 200
        task_id = resp.json()["result"]["id"]

    # Let the background task run (the real _run_agent with stubbed graph).
    # Re-import and call _run_agent directly to test the stub path.
    from fraud_risk_agent.a2a.server import _run_agent

    with (
        patch(
            "fraud_risk_agent.a2a.server.run_investigation",
            side_effect=ImportError("not available"),
            create=True,
        ),
        patch.dict(
            "sys.modules",
            {"fraud_risk_agent.agent.graph": None},
        ),
    ):
        await _run_agent(task_id, "assess_customer_risk", {"customer_id": "C-456"})

    task = task_manager.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    assert task.result is not None
    assert task.result["customer_id"] == "C-456"


async def test_tasks_send_missing_params(client: AsyncClient) -> None:
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"skill": "assess_customer_risk"},
            "id": 3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32602


# ── tasks/get ────────────────────────────────────────────────────────────


async def test_tasks_get_returns_task(client: AsyncClient) -> None:
    task = task_manager.create_task(
        skill="assess_customer_risk",
        params={"customer_id": "C-789"},
    )
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"task_id": task.id},
            "id": 4,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["id"] == task.id
    assert body["result"]["status"] == "submitted"


async def test_tasks_get_not_found(client: AsyncClient) -> None:
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"task_id": "nonexistent"},
            "id": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["error"]["code"] == -32602


# ── tasks/cancel ─────────────────────────────────────────────────────────


async def test_tasks_cancel_submitted(client: AsyncClient) -> None:
    task = task_manager.create_task(
        skill="assess_customer_risk",
        params={"customer_id": "C-100"},
    )
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"task_id": task.id},
            "id": 6,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["status"] == "canceled"


async def test_tasks_cancel_completed_fails(client: AsyncClient) -> None:
    task = task_manager.create_task(
        skill="assess_customer_risk",
        params={"customer_id": "C-200"},
    )
    await task_manager.update_status(task.id, TaskStatus.COMPLETED, result={"score": 10})

    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {"task_id": task.id},
            "id": 7,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32602
    assert "completed" in body["error"]["message"]


# ── Error cases ──────────────────────────────────────────────────────────


async def test_invalid_method(client: AsyncClient) -> None:
    resp = await client.post(
        "/a2a",
        json={
            "jsonrpc": "2.0",
            "method": "tasks/unknown",
            "params": {},
            "id": 8,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32601


async def test_invalid_json(client: AsyncClient) -> None:
    resp = await client.post(
        "/a2a",
        content=b"not json {{{",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32700


async def test_invalid_jsonrpc_version(client: AsyncClient) -> None:
    resp = await client.post(
        "/a2a",
        json={"jsonrpc": "1.0", "method": "tasks/get", "params": {}, "id": 9},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["code"] == -32600
