"""Authentication middleware for A2A requests."""

import logging

from fastapi import HTTPException, Request

from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)


async def verify_a2a_auth(request: Request) -> str:
    """Verify the A2A request authentication.

    Returns the caller ID extracted from the token if valid.

    Raises:
        HTTPException: 401 if the token is missing or invalid.
        HTTPException: 403 if the caller is not in the allow-list.
    """
    settings = get_settings()
    secret = settings.a2a_auth_secret.get_secret_value()

    # Dev mode: empty secret bypasses authentication entirely.
    if not secret:
        logger.debug("a2a auth skipped — no a2a_auth_secret configured (dev mode)")
        return "anonymous"

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = auth_header.removeprefix("Bearer ").strip()
    if token != secret:
        logger.warning("a2a auth failed — invalid bearer token")
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    # Caller identification from the X-Caller-ID header.
    caller_id = request.headers.get("X-Caller-ID", "unknown")

    allowed_raw = settings.a2a_allowed_callers.strip()
    if allowed_raw:
        allowed = {c.strip() for c in allowed_raw.split(",") if c.strip()}
        if caller_id not in allowed:
            logger.warning("a2a auth rejected caller_id=%s (not in allow-list)", caller_id)
            raise HTTPException(status_code=403, detail="Caller not in allow-list")

    logger.info("a2a auth succeeded for caller_id=%s", caller_id)
    return caller_id
