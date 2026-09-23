"""A2A protocol layer — server, client, and task management."""

from fraud_risk_agent.a2a.client import A2AClient
from fraud_risk_agent.a2a.server import app
from fraud_risk_agent.a2a.task_manager import TaskManager

__all__ = ["A2AClient", "TaskManager", "app"]
