from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from cogstack_model_gateway.common.tasks import Status, TaskManager


@pytest.fixture
def db_manager() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for testing."""
    test_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(test_engine)

    class TestDatabaseManager:
        def get_session(self) -> Session:
            return Session(test_engine)

    return TestDatabaseManager()


@pytest.fixture
def task_manager(db_manager: Session) -> TaskManager:
    """Create a TaskManager instance for testing."""
    return TaskManager(db_manager)


def test_status_enum() -> None:
    for status in Status:
        assert isinstance(status, Status)
        assert isinstance(status.value, str)
        assert status.is_final() == (status in {Status.SUCCEEDED, Status.FAILED})


def test_create_task(task_manager: TaskManager) -> None:
    """Test creating a task with different statuses."""
    task_uuid = task_manager.create_task(status=Status.PENDING)
    assert isinstance(task_uuid, str)

    running_task_uuid = task_manager.create_task(status=Status.RUNNING)
    assert isinstance(running_task_uuid, str)
    assert running_task_uuid != task_uuid


def test_get_task(task_manager: TaskManager) -> None:
    """Test retrieving an existing task."""
    task_uuid = task_manager.create_task(status=Status.PENDING)
    retrieved_task = task_manager.get_task(task_uuid)

    assert retrieved_task is not None
    assert retrieved_task.uuid == task_uuid
    assert retrieved_task.status == Status.PENDING


def test_get_nonexistent_task(task_manager: TaskManager) -> None:
    """Test retrieving a non-existent task."""
    non_existent_uuid = "00000000-0000-0000-0000-000000000000"
    retrieved_task = task_manager.get_task(non_existent_uuid)

    assert retrieved_task is None


def test_update_task(task_manager: TaskManager) -> None:
    """Test updating a task's status, result, and error message."""
    task_uuid = task_manager.create_task(status=Status.PENDING)
    updated_task = task_manager.update_task(
        task_uuid, status=Status.RUNNING, expected_status=Status.PENDING
    )
    assert updated_task.status == Status.RUNNING

    final_task = task_manager.update_task(
        task_uuid, status=Status.SUCCEEDED, result="Test result", error_message=None
    )
    assert final_task.status == Status.SUCCEEDED
    assert final_task.result == "Test result"


def test_update_task_with_incorrect_expected_status(task_manager: TaskManager) -> None:
    """Test that updating a task with incorrect expected status raises an error."""
    task_uuid = task_manager.create_task(status=Status.PENDING)
    with pytest.raises(ValueError, match="Status of retrieved task"):
        task_manager.update_task(task_uuid, status=Status.RUNNING, expected_status=Status.SUCCEEDED)


def test_update_nonexistent_task(task_manager: TaskManager) -> None:
    """Test updating a non-existent task."""
    non_existent_uuid = "00000000-0000-0000-0000-000000000000"
    updated_task = task_manager.update_task(non_existent_uuid, status=Status.FAILED)
    assert updated_task is None
    assert task_manager.get_task(non_existent_uuid) is None


@patch("logging.Logger.info")
def test_logging(mock_log_info: MagicMock, task_manager: TaskManager) -> None:
    """Test logged outputs."""
    task_uuid = task_manager.create_task(status=Status.PENDING)
    mock_log_info.assert_called_with(
        "Task '%s' created with status '%s'", task_uuid, Status.PENDING.value
    )

    _ = task_manager.get_task(task_uuid)
    mock_log_info.assert_called_with("Retrieved task '%s'", task_uuid)

    task_manager.update_task(task_uuid, status=Status.RUNNING)
    mock_log_info.assert_called_with(
        "Task '%s' updated (original status '%s', current status '%s')",
        task_uuid,
        Status.PENDING.value,
        Status.RUNNING.value,
    )
