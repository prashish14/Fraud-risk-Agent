"""A2A JSON-RPC server — the external interface for the fraud agent.

This module is a protocol adapter. It translates A2A JSON-RPC requests into
agent graph invocations and maps the results back. It contains no business
logic of its own.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from fraud_risk_agent.a2a.agent_card import get_agent_card
from fraud_risk_agent.a2a.auth import verify_a2a_auth
from fraud_risk_agent.a2a.task_manager import TaskManager, TaskStatus
from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialise observability on server start."""
    try:
        from fraud_risk_agent.observability import (
            configure_langsmith,
            configure_logging,
            configure_tracing,
        )

        configure_logging()
        configure_tracing()
        configure_langsmith()
    except Exception:
        logger.warning("Observability setup failed, continuing without it", exc_info=True)
    yield


app = FastAPI(title="Fraud Risk Agent", version="1.0.0", lifespan=_lifespan)
task_manager = TaskManager()
_background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

# ── JSON-RPC error codes ────────────────────────────────────────────────
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32000
RETRYABLE_ERROR = -32001


def _jsonrpc_error(code: int, message: str, req_id: Any = None) -> JSONResponse:
    """Build a JSON-RPC 2.0 error response."""
    return JSONResponse(
        {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id},
    )


def _jsonrpc_result(result: Any, req_id: Any) -> JSONResponse:
    """Build a JSON-RPC 2.0 success response."""
    return JSONResponse({"jsonrpc": "2.0", "result": result, "id": req_id})


# ── Agent graph invocation (lazy import) ─────────────────────────────────


async def _run_agent(task_id: str, skill: str, params: dict) -> None:
    """Run the agent graph for *task_id* and update the task with the result.

    The graph import is deferred so this module loads even when Phase 5
    is incomplete.
    """
    await task_manager.update_status(task_id, TaskStatus.WORKING)
    try:
        try:
            from fraud_risk_agent.agent.graph import run_assessment
        except (ImportError, ModuleNotFoundError):
            logger.warning(
                "agent graph not available — returning stub assessment for task %s",
                task_id,
            )
            from fraud_risk_agent.agent.models import FraudAssessment

            stub = FraudAssessment(
                customer_id=params.get("customer_id", "unknown"),
                order_id=params.get("order_id"),
                score=0,
                risk_level="low",
                recommended_action="allow",
                confidence=0.0,
                rules_version="stub",
                narrative="Agent graph not yet available.",
            )
            await task_manager.update_status(
                task_id,
                TaskStatus.COMPLETED,
                result=stub.model_dump(mode="json"),
            )
            return

        assessment = await run_assessment(
            customer_id=params.get("customer_id", ""),
            order_id=params.get("order_id"),
            context=params.get("context"),
        )
        await task_manager.update_status(
            task_id,
            TaskStatus.COMPLETED,
            result=assessment,
        )
    except Exception:
        logger.exception("agent graph failed for task %s", task_id)
        await task_manager.update_status(
            task_id,
            TaskStatus.FAILED,
            error="Internal agent error",
        )


# ── Route handlers ───────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.get("/.well-known/agent.json")
async def agent_card() -> dict:
    """Serve the A2A agent card."""
    settings = get_settings()
    base_url = f"http://{settings.a2a_host}:{settings.a2a_port}"
    return get_agent_card(base_url)


@app.get("/metrics")
async def metrics() -> dict:
    """Return current fraud assessment metrics."""
    from fraud_risk_agent.observability.metrics import get_metrics

    return get_metrics().snapshot()


@app.post("/a2a")
async def a2a_endpoint(request: Request) -> JSONResponse:
    """JSON-RPC 2.0 dispatcher for A2A protocol methods."""
    # 1. Parse the request body.
    try:
        body = await request.json()
    except Exception:
        return _jsonrpc_error(PARSE_ERROR, "Parse error")

    # Validate basic JSON-RPC structure.
    if not isinstance(body, dict):
        return _jsonrpc_error(INVALID_REQUEST, "Request must be a JSON object")

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params")

    if body.get("jsonrpc") != "2.0" or not method:
        return _jsonrpc_error(INVALID_REQUEST, "Invalid JSON-RPC 2.0 request", req_id)

    # 2. Authenticate.
    try:
        caller_id = await verify_a2a_auth(request)
    except Exception as exc:
        logger.warning("a2a auth error: %s", exc)
        code = INTERNAL_ERROR
        message = "Authentication failed"
        if hasattr(exc, "status_code"):
            message = getattr(exc, "detail", message)
        return _jsonrpc_error(code, message, req_id)

    logger.info("a2a request method=%s caller=%s", method, caller_id)

    # 3. Dispatch.
    if method == "tasks/send":
        return await _handle_tasks_send(params, req_id)
    if method == "tasks/get":
        return await _handle_tasks_get(params, req_id)
    if method == "tasks/cancel":
        return await _handle_tasks_cancel(params, req_id)

    return _jsonrpc_error(METHOD_NOT_FOUND, f"Method not found: {method}", req_id)


# ── Method handlers ──────────────────────────────────────────────────────


async def _handle_tasks_send(params: Any, req_id: Any) -> JSONResponse:
    """Handle ``tasks/send`` — create a task and start the agent."""
    if not isinstance(params, dict):
        return _jsonrpc_error(INVALID_PARAMS, "params must be an object", req_id)

    skill = params.get("skill")
    customer_id = params.get("customer_id")
    if not skill or not customer_id:
        return _jsonrpc_error(
            INVALID_PARAMS,
            "params must include 'skill' and 'customer_id'",
            req_id,
        )

    task = task_manager.create_task(skill=skill, params=params)
    logger.info("task created task_id=%s skill=%s customer_id=%s", task.id, skill, customer_id)

    # Fire-and-forget: run the agent graph in the background.
    bg = asyncio.create_task(_run_agent(task.id, skill, params))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return _jsonrpc_result(task.model_dump(mode="json"), req_id)


async def _handle_tasks_get(params: Any, req_id: Any) -> JSONResponse:
    """Handle ``tasks/get`` — return current task status."""
    if not isinstance(params, dict) or "task_id" not in params:
        return _jsonrpc_error(INVALID_PARAMS, "params must include 'task_id'", req_id)

    task = task_manager.get_task(params["task_id"])
    if task is None:
        return _jsonrpc_error(INVALID_PARAMS, "Task not found", req_id)

    return _jsonrpc_result(task.model_dump(mode="json"), req_id)


async def _handle_tasks_cancel(params: Any, req_id: Any) -> JSONResponse:
    """Handle ``tasks/cancel`` — cancel a task if still pending."""
    if not isinstance(params, dict) or "task_id" not in params:
        return _jsonrpc_error(INVALID_PARAMS, "params must include 'task_id'", req_id)

    cancelled = task_manager.cancel_task(params["task_id"])
    if not cancelled:
        task = task_manager.get_task(params["task_id"])
        if task is None:
            return _jsonrpc_error(INVALID_PARAMS, "Task not found", req_id)
        return _jsonrpc_error(
            INVALID_PARAMS,
            f"Cannot cancel task in '{task.status.value}' state",
            req_id,
        )

    task = task_manager.get_task(params["task_id"])
    return _jsonrpc_result(task.model_dump(mode="json") if task else {}, req_id)
