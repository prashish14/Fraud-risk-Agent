"""A2A client for calling peer agents."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


class A2AClient:
    """Client for sending tasks to peer A2A agents.

    Supports retries with exponential backoff on transient failures.
    """

    def __init__(self, timeout: float = 30.0, max_retries: int = 3) -> None:
        self._timeout = timeout
        self._max_retries = max_retries

    async def send_task(
        self,
        agent_url: str,
        skill: str,
        params: dict,
        *,
        auth_token: str | None = None,
    ) -> dict:
        """Send a task to a peer agent via JSON-RPC ``tasks/send``.

        Returns the parsed JSON-RPC result on success.
        Raises ``httpx.HTTPStatusError`` on non-retryable HTTP errors.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {"skill": skill, **params},
            "id": 1,
        }
        return await self._rpc_call(agent_url, payload, auth_token=auth_token)

    async def get_task(
        self,
        agent_url: str,
        task_id: str,
        *,
        auth_token: str | None = None,
    ) -> dict:
        """Poll a peer agent for task status via JSON-RPC ``tasks/get``."""
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {"task_id": task_id},
            "id": 1,
        }
        return await self._rpc_call(agent_url, payload, auth_token=auth_token)

    async def discover_agent(self, agent_url: str) -> dict:
        """Fetch the agent card from ``/.well-known/agent.json``."""
        url = agent_url.rstrip("/") + "/.well-known/agent.json"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    # ── internal helpers ─────────────────────────────────────────────

    async def _rpc_call(
        self,
        agent_url: str,
        payload: dict,
        *,
        auth_token: str | None = None,
    ) -> dict:
        """Send a JSON-RPC request with retries and exponential backoff."""
        url = agent_url.rstrip("/") + "/a2a"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    body = resp.json()

                    if "error" in body:
                        error = body["error"]
                        # Retryable server-side errors.
                        if error.get("code") == -32001 and attempt < self._max_retries - 1:
                            delay = 2**attempt
                            logger.warning(
                                "retryable A2A error (attempt %d/%d), retrying in %ds: %s",
                                attempt + 1,
                                self._max_retries,
                                delay,
                                error.get("message"),
                            )
                            await asyncio.sleep(delay)
                            continue
                        return body

                    return body
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    delay = 2**attempt
                    logger.warning(
                        "transient A2A error (attempt %d/%d), retrying in %ds: %s",
                        attempt + 1,
                        self._max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

        raise last_exc  # type: ignore[misc]
