import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from dateutil import parser
from sqlmodel import Session

from cogstack_model_gateway.common.models import Model, ModelDeploymentType, ModelManager


@pytest.fixture
def model_manager(db_manager: Session) -> ModelManager:
    """Create a ModelManager instance for testing."""
    return ModelManager(db_manager)


def test_create_model(model_manager: ModelManager) -> None:
    """Test creating a model with different deployment types."""
    model = model_manager.create_model(
        model_name="medcat-snomed", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    assert isinstance(model, Model)
    assert model.model_name == "medcat-snomed"
    assert model.deployment_type == ModelDeploymentType.AUTO
    assert model.idle_ttl == 300
    assert model.ready is False
    assert model.created_at is not None
    assert isinstance(model.created_at, str)
    parser.isoparse(model.created_at)
    assert model.last_used_at is not None
    assert isinstance(model.last_used_at, str)
    parser.isoparse(model.last_used_at)

    # Test creating a manual deployment model without idle_ttl
    model = model_manager.create_model(
        model_name="custom-ner", deployment_type=ModelDeploymentType.MANUAL
    )
    assert model.model_name == "custom-ner"
    assert model.deployment_type == ModelDeploymentType.MANUAL
    assert model.idle_ttl is None
    assert model.ready is False

    # Test creating a static deployment model
    model = model_manager.create_model(
        model_name="static-classifier", deployment_type=ModelDeploymentType.STATIC, ready=True
    )
    assert model.model_name == "static-classifier"
    assert model.deployment_type == ModelDeploymentType.STATIC
    assert model.idle_ttl is None
    assert model.ready is True


def test_create_model_duplicate(model_manager: ModelManager) -> None:
    """Test that creating a duplicate model raises ValueError."""
    model_manager.create_model(
        model_name="duplicate-test", deployment_type=ModelDeploymentType.MANUAL, idle_ttl=300
    )

    # Attempt to create a model with the same name
    with pytest.raises(ValueError, match="Model 'duplicate-test' already exists"):
        model_manager.create_model(
            model_name="duplicate-test", deployment_type=ModelDeploymentType.AUTO, idle_ttl=600
        )


def test_get_model(model_manager: ModelManager) -> None:
    """Test retrieving a model by name."""
    model_manager.create_model(
        model_name="test-get-model",
        deployment_type=ModelDeploymentType.AUTO,
        idle_ttl=300,
        ready=False,
    )

    retrieved_model = model_manager.get_model("test-get-model")
    assert retrieved_model is not None
    assert retrieved_model.model_name == "test-get-model"
    assert retrieved_model.deployment_type == ModelDeploymentType.AUTO
    assert retrieved_model.idle_ttl == 300
    assert retrieved_model.ready is False

    # Test getting a non-existent model
    non_existent = model_manager.get_model("nonexistent-model")
    assert non_existent is None


def test_mark_model_ready(model_manager: ModelManager) -> None:
    """Test marking a model as ready."""
    model = model_manager.create_model(
        model_name="test-mark-ready",
        deployment_type=ModelDeploymentType.MANUAL,
        idle_ttl=300,
        ready=False,
    )
    assert model.ready is False

    updated_model = model_manager.mark_model_ready("test-mark-ready")
    assert updated_model is not None
    assert updated_model.ready is True

    retrieved_model = model_manager.get_model("test-mark-ready")
    assert retrieved_model is not None
    assert retrieved_model.ready is True

    # Test marking a non-existent model as ready
    result = model_manager.mark_model_ready("nonexistent-model")
    assert result is None


def test_record_model_usage(model_manager: ModelManager) -> None:
    """Test recording usage updates last_used_at timestamp."""
    model = model_manager.create_model(
        model_name="api-model", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    original_last_used = parser.isoparse(model.last_used_at)

    time.sleep(0.01)

    updated_model = model_manager.record_model_usage(model.model_name)
    assert updated_model is not None
    updated_last_used = parser.isoparse(updated_model.last_used_at)
    assert updated_last_used > original_last_used

    # Test recording usage for a non-existent model
    updated_model = model_manager.record_model_usage("nonexistent-model")
    assert updated_model is None


def test_delete_model(model_manager: ModelManager) -> None:
    """Test deleting a model."""
    model_manager.create_model(
        model_name="delete-me", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    result = model_manager.delete_model("delete-me")
    assert result is True

    # Test deleting a non-existent model
    result = model_manager.delete_model("nonexistent-model")
    assert result is False


def test_is_model_idle(model_manager: ModelManager) -> None:
    """Test is_model_idle returns tuple of (is_idle, idle_seconds)."""
    old_time = datetime.fromtimestamp(datetime.now(UTC).timestamp() - 1000, UTC)
    # Manually create model with old last_used_at (1000 seconds > 300 second TTL)
    with model_manager.db_manager.get_session() as session:
        model = Model(
            model_name="idle-model",
            deployment_type=ModelDeploymentType.AUTO,
            idle_ttl=300,
            last_used_at=old_time.isoformat(),
        )
        session.add(model)
        session.commit()

    is_idle, idle_seconds = model_manager.is_model_idle("idle-model")
    assert is_idle is True
    assert idle_seconds >= 1000.0
    assert idle_seconds < 1001.0

    # Test is_model_idle returns False for recently used model
    model = model_manager.create_model(
        model_name="active-model",
        deployment_type=ModelDeploymentType.AUTO,
        idle_ttl=300,
    )
    is_idle, idle_seconds = model_manager.is_model_idle("active-model")
    assert is_idle is False
    assert idle_seconds < 300

    # Test is_model_idle returns False for model without TTL
    with model_manager.db_manager.get_session() as session:
        old_time = datetime.fromtimestamp(datetime.now(UTC).timestamp() - 1000, UTC)
        model = Model(
            model_name="no-ttl-model",
            deployment_type=ModelDeploymentType.MANUAL,
            idle_ttl=None,
            last_used_at=old_time.isoformat(),
        )
        session.add(model)
        session.commit()

    is_idle, idle_seconds = model_manager.is_model_idle("no-ttl-model")
    assert is_idle is False
    assert idle_seconds == 0.0

    # Test is_model_idle returns False for non-existent model
    is_idle, idle_seconds = model_manager.is_model_idle("nonexistent-model")
    assert is_idle is False
    assert idle_seconds == 0.0


@patch("cogstack_model_gateway.common.models.log.debug")
def test_logging_create(mock_log_debug: MagicMock, model_manager: ModelManager) -> None:
    """Test logged outputs for create operation."""
    model_manager.create_model(
        model_name="log-test", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    mock_log_debug.assert_called_with(
        "Created model record: %s (type=%s, idle_ttl=%s)",
        "log-test",
        "auto",
        300,
    )


@patch("cogstack_model_gateway.common.models.log.debug")
def test_logging_record_model_usage(mock_log_debug: MagicMock, model_manager: ModelManager) -> None:
    """Test debug logging for record_model_usage operation."""
    model_manager.create_model(
        model_name="debug-test", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    mock_log_debug.reset_mock()

    model_manager.record_model_usage("debug-test")
    mock_log_debug.assert_called_with("Updated last_used_at for model: %s", "debug-test")


@patch("cogstack_model_gateway.common.models.log.debug")
def test_logging_delete(mock_log_debug: MagicMock, model_manager: ModelManager) -> None:
    """Test logged outputs for delete operation."""
    model_manager.create_model(
        model_name="delete-log-test", deployment_type=ModelDeploymentType.AUTO, idle_ttl=300
    )
    mock_log_debug.reset_mock()

    model_manager.delete_model("delete-log-test")
    mock_log_debug.assert_called_with("Deleted model record: %s", "delete-log-test")


@patch("cogstack_model_gateway.common.models.log.debug")
def test_is_model_idle_logging(mock_log_debug: MagicMock, model_manager: ModelManager) -> None:
    """Test debug logging for is_model_idle operation."""
    old_time = datetime.fromtimestamp(datetime.now(UTC).timestamp() - 1000, UTC)
    with model_manager.db_manager.get_session() as session:
        model = Model(
            model_name="idle-model-log",
            deployment_type=ModelDeploymentType.AUTO,
            idle_ttl=300,
            last_used_at=old_time.isoformat(),
        )
        session.add(model)
        session.commit()

    model_manager.is_model_idle("idle-model-log")

    mock_log_debug.assert_called_once()
    call_args = mock_log_debug.call_args[0]
    assert call_args[0] == "Model %s idle check: %s (idle_time=%.0fs, ttl=%ss)"
    assert call_args[1] == "idle-model-log"
    assert call_args[2] is True
    assert call_args[3] >= 1000, f"Expected idle_time >= 1000s, got {call_args[3]}"
    assert call_args[4] == 300


@patch("cogstack_model_gateway.common.models.log.warning")
def test_logging_warning(mock_log_warning: MagicMock, model_manager: ModelManager) -> None:
    """Test logged warnings for non-existent operations."""
    model_manager.record_model_usage("nonexistent")
    mock_log_warning.assert_called_with("Model record not found: %s", "nonexistent")

    mock_log_warning.reset_mock()
    model_manager.delete_model("nonexistent")
    mock_log_warning.assert_called_with("Model record not found for deletion: %s", "nonexistent")
