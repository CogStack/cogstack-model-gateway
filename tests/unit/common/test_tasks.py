from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session

from cogstack_model_gateway.common.tasks import Status, Task, TaskManager, UnexpectedStatusError


@pytest.fixture
def task_manager(db_manager: Session) -> TaskManager:
    """Create a TaskManager instance for testing."""
    return TaskManager(db_manager)


def test_status_enum() -> None:
    for status in Status:
        assert isinstance(status, Status)
        assert isinstance(status.value, str)
        assert status.is_active() == (status in {Status.SCHEDULED, Status.RUNNING})
        assert status.is_final() == (status in {Status.SUCCEEDED, Status.FAILED})


def test_create_task(task_manager: TaskManager) -> None:
    """Test creating a task with different statuses."""
    task = task_manager.create_task(model="annotation_model", type="process")
    assert isinstance(task, Task)
    assert task.uuid is not None
    assert isinstance(task.uuid, str)
    assert task.status == Status.PENDING
    assert task.created_at is not None
    assert isinstance(task.created_at, str)
    assert datetime.fromisoformat(task.created_at)
    assert task.model == "annotation_model"
    assert task.type == "process"
    assert task.source is None


def test_get_task(task_manager: TaskManager) -> None:
    """Test retrieving an existing task."""
    task = task_manager.create_task()
    retrieved_task = task_manager.get_task(task.uuid)

    assert retrieved_task is not None
    assert retrieved_task.uuid == task.uuid
    assert retrieved_task.status == Status.PENDING
    assert retrieved_task.created_at is not None
    assert isinstance(retrieved_task.created_at, str)
    assert datetime.fromisoformat(retrieved_task.created_at)


def test_get_nonexistent_task(task_manager: TaskManager) -> None:
    """Test retrieving a non-existent task."""
    non_existent_uuid = "00000000-0000-0000-0000-000000000000"
    retrieved_task = task_manager.get_task(non_existent_uuid)

    assert retrieved_task is None


def test_update_task(task_manager: TaskManager) -> None:
    """Test updating a task's status, result, and error message."""
    task = task_manager.create_task()
    updated_task = task_manager.update_task(
        task.uuid, status=Status.RUNNING, expected_status=Status.PENDING
    )
    assert updated_task.status == Status.RUNNING
    assert updated_task.started_at is not None
    started_at = datetime.fromisoformat(updated_task.started_at)
    assert started_at <= datetime.now(UTC)
    assert updated_task.finished_at is None
    assert updated_task.model is None
    assert updated_task.type is None

    rescheduled_task = task_manager.update_task(
        task.uuid, status=Status.PENDING, model="annotation_model", type="process"
    )
    assert rescheduled_task.status == Status.PENDING
    assert rescheduled_task.model == "annotation_model"
    assert rescheduled_task.type == "process"
    assert rescheduled_task.created_at is not None
    assert rescheduled_task.created_at == updated_task.created_at
    assert rescheduled_task.started_at is not None
    assert rescheduled_task.started_at == updated_task.started_at
    assert rescheduled_task.finished_at is None

    final_task = task_manager.update_task(
        task.uuid, status=Status.SUCCEEDED, result="Test result", error_message=None
    )
    assert final_task.status == Status.SUCCEEDED
    assert final_task.result == "Test result"
    assert final_task.error_message is None
    assert final_task.finished_at is not None
    finished_at = datetime.fromisoformat(final_task.finished_at)
    assert finished_at <= datetime.now(UTC)
    assert finished_at > started_at


def test_update_task_with_incorrect_expected_status(task_manager: TaskManager) -> None:
    """Test that updating a task with incorrect expected status raises an error."""
    task = task_manager.create_task()
    with pytest.raises(UnexpectedStatusError, match="Unexpected status") as exc_info:
        task_manager.update_task(task.uuid, status=Status.RUNNING, expected_status=Status.SUCCEEDED)
    assert exc_info.value.task_uuid == task.uuid
    assert exc_info.value.status == Status.PENDING.value


def test_update_nonexistent_task(task_manager: TaskManager) -> None:
    """Test updating a non-existent task."""
    non_existent_uuid = "00000000-0000-0000-0000-000000000000"
    updated_task = task_manager.update_task(non_existent_uuid, status=Status.FAILED)
    assert updated_task is None
    assert task_manager.get_task(non_existent_uuid) is None


@patch("cogstack_model_gateway.common.tasks.log.info")
def test_logging(mock_log_info: MagicMock, task_manager: TaskManager) -> None:
    """Test logged outputs."""
    task = task_manager.create_task()
    mock_log_info.assert_called_with(
        "Task '%s' created with status '%s'", task.uuid, Status.PENDING.value
    )

    _ = task_manager.get_task(task.uuid)
    mock_log_info.assert_called_with("Retrieved task '%s'", task.uuid)

    task_manager.update_task(task.uuid, status=Status.RUNNING)
    mock_log_info.assert_called_with(
        "Task '%s' updated (original status '%s', current status '%s')",
        task.uuid,
        Status.PENDING.value,
        Status.RUNNING.value,
    )
