import logging
import uuid
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
    REQUEUED = "requeued"

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
    tracking_id: str = Field(default=None, nullable=True)
    result: str = Field(default=None, nullable=True)
    error_message: str = Field(default=None, nullable=True)


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
    def create_task(self, session: Session, status: str) -> str:
        """Create a new task with the specified status."""
        task = Task(status=status)
        session.add(task)
        session.commit()
        log.info("Task '%s' created with status '%s'", task.uuid, task.status.value)
        return task.uuid

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
        status: str = None,
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
            if status:
                if expected_status and task.status != expected_status:
                    raise UnexpectedStatusError(task_uuid, task.status.value)
                task.status = status
            if tracking_id:
                task.tracking_id = tracking_id
            if result:
                task.result = result
            if error_message:
                task.error_message = error_message
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
