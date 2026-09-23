"""Async task lifecycle management for A2A requests."""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class TaskStatus(StrEnum):
    """Lifecycle states for an A2A task."""

    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class Task(BaseModel):
    """A single A2A task tracked by the manager."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.SUBMITTED
    skill: str
    params: dict
    result: dict | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskManager:
    """In-memory task store with async task lifecycle.

    Production deployments would back this with Redis or Postgres
    for persistence across restarts.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def create_task(self, skill: str, params: dict) -> Task:
        """Create a new task in *submitted* state and return it."""
        task = Task(skill=skill, params=params)
        self._tasks[task.id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Return a task by ID, or ``None`` if not found."""
        return self._tasks.get(task_id)

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Transition a task to a new status, optionally attaching a result or error."""
        task = self._tasks.get(task_id)
        if task is None:
            return
        task.status = status
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error
        task.updated_at = datetime.now(UTC)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it is still in a cancellable state.

        Returns ``True`` if the task was cancelled, ``False`` otherwise.
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.status in (TaskStatus.SUBMITTED, TaskStatus.WORKING):
            task.status = TaskStatus.CANCELED
            task.updated_at = datetime.now(UTC)
            return True
        return False
