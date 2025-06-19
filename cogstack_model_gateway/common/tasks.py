import logging
import uuid
from datetime import UTC, datetime
from enum import Enum
from functools import wraps

from sqlmodel import Field, Session, SQLModel, select

from cogstack_model_gateway.common.db import DatabaseManager

log = logging.getLogger("cmg.common")


class Status(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    def is_active(self) -> bool:
        """Check if the status indicates that the task is active (i.e. scheduled or running)."""
        return self in {self.RUNNING, self.SCHEDULED}

    def is_final(self) -> bool:
        """Check if the status is final (i.e. succeeded or failed)."""
        return self in {self.SUCCEEDED, self.FAILED}


class UnexpectedStatusError(Exception):
    """Exception raised when an unexpected status is encountered."""

    def __init__(self, task_uuid: str, status: str):
        super().__init__(f"Unexpected status '{status}' for task '{task_uuid}'")
        self.task_uuid = task_uuid
        self.status = status


class Task(SQLModel, table=True):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: Status = Field(default=Status.PENDING)
    model: str = Field(default=None, nullable=True)
    type: str = Field(default=None, nullable=True)
    source: str = Field(default=None, nullable=True)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(), nullable=True)
    started_at: str = Field(default=None, nullable=True)
    finished_at: str = Field(default=None, nullable=True)
    result: str = Field(default=None, nullable=True)
    error_message: str = Field(default=None, nullable=True)
    tracking_id: str = Field(default=None, nullable=True)


class TaskManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @staticmethod
    def with_db_session(func):
        """Decorator to provide a database session to a method."""

        @wraps(func)
        def wrapper(self: "TaskManager", *args, **kwargs):
            with self.db_manager.get_session() as session:
                return func(self, session, *args, **kwargs)

        return wrapper

    @with_db_session
    def create_task(
        self, session: Session, model: str = None, type: str = None, source: str = None
    ) -> Task:
        """Create a new task with the specified status and current timestamp."""
        task = Task(model=model, type=type, source=source)
        session.add(task)
        session.commit()
        log.info("Task '%s' created with status '%s'", task.uuid, task.status.value)
        return task

    @with_db_session
    def get_task(self, session: Session, task_uuid: str) -> Task:
        """Get a task by UUID."""
        statement = select(Task).where(Task.uuid == task_uuid)
        result = session.exec(statement).first()
        if result:
            log.info("Retrieved task '%s'", task_uuid)
        else:
            log.warning("Task '%s' not found", task_uuid)
        return result

    @with_db_session
    def update_task(
        self,
        session: Session,
        task_uuid: str,
        status: Status = None,
        model: str = None,
        type: str = None,
        source: str = None,
        created_at: str = None,
        started_at: str = None,
        finished_at: str = None,
        expected_status: Status = None,
        tracking_id: str = None,
        result: str = None,
        error_message: str = None,
    ) -> Task:
        """Update task details.

        If `status` is provided, the task status is updated only if the current status matches the
        specified `expected_status`.
        """
        if task := session.get(Task, task_uuid):
            original_status = task.status
            now_iso = datetime.now(UTC).isoformat()

            update_fields = {
                "model": model,
                "type": type,
                "source": source,
                "created_at": created_at,
                "started_at": started_at,
                "finished_at": finished_at,
                "tracking_id": tracking_id,
                "result": result,
                "error_message": error_message,
            }

            for field, value in update_fields.items():
                if value is not None:
                    setattr(task, field, value)

            if status:
                if expected_status and task.status != expected_status:
                    raise UnexpectedStatusError(task_uuid, task.status.value)
                task.status = status

            if task.status.is_active() and task.started_at is None:
                task.started_at = now_iso

            if task.status.is_final() and task.finished_at is None:
                task.finished_at = now_iso

            session.commit()
            log.info(
                "Task '%s' updated (original status '%s', current status '%s')",
                task_uuid,
                original_status.value,
                task.status.value,
            )
            return task
        else:
            log.warning("Task '%s' not found", task_uuid)
            return None
