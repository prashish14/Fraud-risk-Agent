"""Async SQLAlchemy engine and session factories.

Two engines:
- ``replica_engine``: read-only against the e-commerce database,
  enforced at the Postgres level via ``default_transaction_read_only=on``.
- ``fraud_engine``: read-write for the agent's own tables only.

Both are module-level lazy singletons created on first use.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from fraud_risk_agent.config import get_settings

logger = logging.getLogger(__name__)

_replica_engine: AsyncEngine | None = None
_fraud_engine: AsyncEngine | None = None

_replica_session_factory: async_sessionmaker[AsyncSession] | None = None
_fraud_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_replica_engine() -> AsyncEngine:
    """Return the read-only replica engine, creating it on first call."""
    global _replica_engine
    if _replica_engine is None:
        settings = get_settings()
        _replica_engine = create_async_engine(
            settings.replica_database_url,
            pool_size=5,
            pool_pre_ping=True,
            connect_args={
                "server_settings": {
                    "default_transaction_read_only": "on",
                    "statement_timeout": "5000",
                },
            },
        )
        logger.info("Created replica engine (read-only)")
    return _replica_engine


def _get_fraud_engine() -> AsyncEngine:
    """Return the read-write fraud engine, creating it on first call."""
    global _fraud_engine
    if _fraud_engine is None:
        settings = get_settings()
        _fraud_engine = create_async_engine(
            settings.fraud_database_url,
            pool_size=3,
            pool_pre_ping=True,
        )
        logger.info("Created fraud engine (read-write)")
    return _fraud_engine


def _get_replica_session_factory() -> async_sessionmaker[AsyncSession]:
    global _replica_session_factory
    if _replica_session_factory is None:
        _replica_session_factory = async_sessionmaker(
            bind=_get_replica_engine(),
            expire_on_commit=False,
        )
    return _replica_session_factory


def _get_fraud_session_factory() -> async_sessionmaker[AsyncSession]:
    global _fraud_session_factory
    if _fraud_session_factory is None:
        _fraud_session_factory = async_sessionmaker(
            bind=_get_fraud_engine(),
            expire_on_commit=False,
        )
    return _fraud_session_factory


@asynccontextmanager
async def get_replica_session() -> AsyncIterator[AsyncSession]:
    """Yield a read-only session against the e-commerce replica.

    The session is automatically closed when the context manager exits.
    """
    factory = _get_replica_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.close()


@asynccontextmanager
async def get_fraud_session() -> AsyncIterator[AsyncSession]:
    """Yield a read-write session against the fraud-agent database.

    The session is automatically closed when the context manager exits.
    """
    factory = _get_fraud_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.close()


async def dispose_engines() -> None:
    """Dispose both engines. Call during application shutdown."""
    global _replica_engine, _fraud_engine
    global _replica_session_factory, _fraud_session_factory
    if _replica_engine is not None:
        await _replica_engine.dispose()
        _replica_engine = None
        _replica_session_factory = None
        logger.info("Disposed replica engine")
    if _fraud_engine is not None:
        await _fraud_engine.dispose()
        _fraud_engine = None
        _fraud_session_factory = None
        logger.info("Disposed fraud engine")
