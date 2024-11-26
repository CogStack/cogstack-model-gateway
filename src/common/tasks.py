import uuid
from enum import Enum
from functools import wraps

from sqlmodel import Field, Session, SQLModel, select

from common.db import DatabaseManager


class Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Task(SQLModel, table=True):
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    status: Status = Field(default=Status.PENDING)
    priority: int
    result: str = Field(default=None, nullable=True)
    error_message: str = Field(default=None, nullable=True)


class TaskManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    @staticmethod
    def with_db_session(func):
        @wraps(func)
        def wrapper(self: "TaskManager", *args, **kwargs):
            with self.db_manager.get_session() as session:
                return func(self, session, *args, **kwargs)

        return wrapper

    @with_db_session
    def create_task(self, session: Session, status: str, priority: int) -> str:
        task = Task(status=status, priority=priority)
        session.add(task)
        session.commit()
        return task.uuid

    @with_db_session
    def get_task(self, session: Session, task_uuid: str) -> Task:
        statement = select(Task).where(Task.uuid == task_uuid)
        result = session.exec(statement).first()
        return result

    @with_db_session
    def update_task(
        self,
        session: Session,
        task_uuid: str,
        status: str = None,
        expected_status: str = None,
        result: str = None,
        error_message: str = None,
    ) -> None:
        if task := session.get(Task, task_uuid):
            if status:
                if expected_status and task.status != expected_status:
                    raise ValueError(
                        f"Status of retrieved task '{task_uuid}' is '{task.status}', expected"
                        f" '{expected_status}'"
                    )
                task.status = status
            if result:
                task.result = result
            if error_message:
                task.error_message = error_message
            session.commit()
