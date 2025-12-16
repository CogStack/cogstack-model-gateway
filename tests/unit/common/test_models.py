import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from dateutil import parser
from sqlmodel import Session

from cogstack_model_gateway.common.exceptions import ConfigConflictError, ConfigValidationError
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


def test_create_on_demand_config(model_manager: ModelManager) -> None:
    """Test creating an on-demand model configuration."""
    config = model_manager.create_on_demand_config(
        model_name="test-create",
        model_uri="models:/test-create-1/Production",
        idle_ttl=3600,
        description="Test model",
        deploy_spec={"resources": {"limits": {"memory": "4g", "cpus": "2.0"}}},
    )

    assert config.id is not None
    assert config.model_name == "test-create"
    assert config.model_uri == "models:/test-create-1/Production"
    assert config.idle_ttl == 3600
    assert config.description == "Test model"
    assert config.deploy_spec_json == '{"resources": {"limits": {"memory": "4g", "cpus": "2.0"}}}'
    assert config.enabled is True
    assert config.created_at is not None
    assert config.updated_at is not None

    # Test that creating a duplicate config replaces the existing one
    config2 = model_manager.create_on_demand_config(
        model_name="test-create",
        model_uri="models:/test-create-1/v2",
    )

    assert config2.id is not None
    assert config2.id != config.id
    assert config2.model_name == "test-create"
    assert config2.model_uri == "models:/test-create-1/v2"

    # Inherited properties from previous config remain the same
    assert config2.idle_ttl == 3600
    assert config2.description == "Test model"
    assert config2.deploy_spec_json == '{"resources": {"limits": {"memory": "4g", "cpus": "2.0"}}}'
    assert config2.enabled is True
    assert config2.created_at is not None
    assert config2.updated_at is not None

    old_config = model_manager.get_on_demand_config_by_id(config.id)
    assert old_config is not None
    assert old_config.enabled is False

    # Test creating a config without inheritance
    config3 = model_manager.create_on_demand_config(
        model_name="test-create",
        model_uri="models:/test-create-2/v1",
        idle_ttl=7200,
        inherit_config=False,
    )

    assert config3.id is not None
    assert config3.id != config2.id
    assert config3.model_name == "test-create"
    assert config3.model_uri == "models:/test-create-2/v1"
    assert config3.idle_ttl == 7200
    assert config3.description is None  # No inheritance, therefore equals the default value
    assert config3.deploy_spec_json is None  # No inheritance, therefore equals the default value

    # Test that creating a config without inheritance and missing required fields
    # raises ConfigValidationError
    with pytest.raises(ConfigValidationError, match="model_uri is required"):
        model_manager.create_on_demand_config(
            model_name="test-create",
            inherit_config=False,
        )

    # Test that creating a duplicate config with replace_enabled=False raises ConfigConflictError
    with pytest.raises(ConfigConflictError, match="already exists"):
        model_manager.create_on_demand_config(
            model_name="test-create",
            model_uri="models:/test-create-4/v1",
            replace_enabled=False,
        )


def test_get_on_demand_config(model_manager: ModelManager) -> None:
    """Test retrieving an on-demand config by service name."""
    model_manager.create_on_demand_config(
        model_name="test-get",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
        description="Test model",
    )

    config = model_manager.get_on_demand_config("test-get")
    assert config is not None
    assert config.model_name == "test-get"
    assert config.model_uri == "models:/test-1/v1"

    # Test getting a non-existent config
    config = model_manager.get_on_demand_config("nonexistent")
    assert config is None


def test_get_on_demand_config_by_id(model_manager: ModelManager) -> None:
    """Test retrieving an on-demand config by ID."""
    created_config = model_manager.create_on_demand_config(
        model_name="test-id",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )

    config = model_manager.get_on_demand_config_by_id(created_config.id)
    assert config is not None
    assert config.model_name == "test-id"

    # Test getting a non-existent ID
    config = model_manager.get_on_demand_config_by_id(99999)
    assert config is None


def test_list_on_demand_configs(model_manager: ModelManager) -> None:
    """Test listing on-demand configs."""
    model_manager.create_on_demand_config(
        model_name="test-list-1",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )
    model_manager.create_on_demand_config(
        model_name="test-list-1",
        model_uri="models:/test-1/v2",
    )
    model_manager.create_on_demand_config(
        model_name="test-list-2",
        model_uri="models:/test-2/v1",
        idle_ttl=3600,
    )

    configs = model_manager.list_on_demand_configs()
    assert len(configs) == 2
    assert all(c.enabled is True for c in configs)
    assert all(c.model_name in ["test-list-1", "test-list-2"] for c in configs)

    # Test listing on-demand configs including disabled ones
    all_configs = model_manager.list_on_demand_configs(include_disabled=True)
    assert len(all_configs) == 3
    model_names = [c.model_name for c in all_configs]
    assert model_names.count("test-list-1") == 2
    assert model_names.count("test-list-2") == 1


def test_get_on_demand_config_history(model_manager: ModelManager) -> None:
    """Test getting version history for an on-demand config."""
    model_manager.create_on_demand_config(
        model_name="test-history",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )
    model_manager.create_on_demand_config(
        model_name="test-history",
        model_uri="models:/test-1/v2",
        idle_ttl=7200,
    )

    history = model_manager.get_on_demand_config_history("test-history")
    assert len(history) == 2
    # Newest first
    assert history[0].model_uri == "models:/test-1/v2"
    assert history[0].enabled is True
    assert history[1].model_uri == "models:/test-1/v1"
    assert history[1].enabled is False

    # Test for non-existent model
    history = model_manager.get_on_demand_config_history("nonexistent")
    assert history == []


def test_update_on_demand_config(model_manager: ModelManager) -> None:
    """Test updating an on-demand config."""
    model_manager.create_on_demand_config(
        model_name="test-update",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
        description="Original description",
    )

    updated_config = model_manager.update_on_demand_config(
        model_name="test-update",
        model_uri="models:/test-1/v2",
        description="Updated description",
        deploy_spec={"resources": {"limits": {"memory": "8g"}}},
    )

    assert updated_config is not None
    assert updated_config.model_uri == "models:/test-1/v2"
    assert updated_config.description == "Updated description"
    assert updated_config.idle_ttl == 3600  # Unchanged
    assert updated_config.deploy_spec_json == '{"resources": {"limits": {"memory": "8g"}}}'

    # Verify that only a single version exists and is enabled
    history = model_manager.get_on_demand_config_history("test-update")
    assert len(history) == 1
    assert history[0].id == updated_config.id
    assert history[0].enabled is True

    # Test clearing optional fields in an on-demand config
    updated_config2 = model_manager.update_on_demand_config(
        model_name="test-update",
        idle_ttl=7200,
        deploy_spec=None,
        clear_description=True,
    )

    assert updated_config2 is not None
    assert updated_config2.idle_ttl == 7200
    assert updated_config2.description is None
    assert updated_config2.deploy_spec_json is not None  # Can't implicitly clear deploy_spec
    assert updated_config2.deploy_spec_json == '{"resources": {"limits": {"memory": "8g"}}}'

    # Test updating a non-existent config returns None
    assert (
        model_manager.update_on_demand_config(
            model_name="nonexistent",
            model_uri="models:/test-1/v1",
            idle_ttl=3600,
        )
        is None
    )


def test_disable_on_demand_config(model_manager: ModelManager) -> None:
    """Test soft-deleting an on-demand config."""
    model_manager.create_on_demand_config(
        model_name="test-disable",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )

    assert model_manager.disable_on_demand_config("test-disable") is True
    assert model_manager.get_on_demand_config("test-disable") is None

    history = model_manager.get_on_demand_config_history("test-disable")
    assert len(history) == 1
    assert history[0].enabled is False

    # Test disabling a non-existent config returns False
    assert model_manager.disable_on_demand_config("nonexistent") is False


def test_enable_on_demand_config(model_manager: ModelManager) -> None:
    """Test enabling an on-demand config."""
    model_config = model_manager.create_on_demand_config(
        model_name="test-enable",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )
    model_manager.disable_on_demand_config("test-enable")
    assert model_manager.get_on_demand_config("test-enable") is None

    # Re-enable config
    enabled_config = model_manager.enable_on_demand_config(model_config.id)
    assert enabled_config is not None
    assert enabled_config.enabled is True

    enabled_config = model_manager.get_on_demand_config("test-enable")
    assert enabled_config is not None
    assert enabled_config.id == model_config.id

    # Test that enabling a config disables any currently enabled one
    model_config2 = model_manager.create_on_demand_config(
        model_name="test-enable",
        model_uri="models:/test-1/v1",
        idle_ttl=3600,
    )

    model_config1 = model_manager.get_on_demand_config_by_id(model_config.id)
    assert model_config1.enabled is False
    assert model_config2.enabled is True

    model_manager.enable_on_demand_config(model_config1.id)

    history = model_manager.get_on_demand_config_history("test-enable")
    assert next(c for c in history if c.id == model_config1.id).enabled is True
    assert next(c for c in history if c.id == model_config2.id).enabled is False

    # Test enabling a non-existent config returns None
    assert model_manager.enable_on_demand_config(99999) is None

    # Test enabling an already enabled config returns it unchanged
    enabled_config = model_manager.enable_on_demand_config(model_config1.id)
    assert enabled_config is not None
    assert enabled_config.id == model_config1.id
    assert enabled_config.enabled is True


def test_on_demand_config_model_name_validation(model_manager: ModelManager) -> None:
    """Test model_name validation (Docker naming constraints)."""
    valid_names = [
        "medcat-snomed",
        "model_123",
        "my.model",
        "a",  # Single character
        "Model-With-CAPS",
        "123model",  # Starts with number is valid
    ]
    for name in valid_names:
        config = model_manager.create_on_demand_config(
            model_name=name,
            model_uri="models:/test/v1",
            idle_ttl=3600,
        )
        assert config.model_name == name

    # Invalid: starts with special character
    with pytest.raises(ValueError, match="Invalid model name"):
        model_manager.create_on_demand_config(
            model_name="-invalid",
            model_uri="models:/test/v1",
            idle_ttl=3600,
        )

    # Invalid: contains spaces
    with pytest.raises(ValueError, match="Invalid model name"):
        model_manager.create_on_demand_config(
            model_name="invalid name",
            model_uri="models:/test/v1",
            idle_ttl=3600,
        )

    # Invalid: contains special characters
    with pytest.raises(ValueError, match="Invalid model name"):
        model_manager.create_on_demand_config(
            model_name="invalid@model",
            model_uri="models:/test/v1",
            idle_ttl=3600,
        )

    # Invalid: too long (>255 characters)
    with pytest.raises(ValueError, match="Service name too long"):
        model_manager.create_on_demand_config(
            model_name="a" * 256,
            model_uri="models:/test/v1",
            idle_ttl=3600,
        )


def test_on_demand_config_model_uri_validation(model_manager: ModelManager) -> None:
    """Test model_uri validation."""
    valid_uris = [
        "models:/medcat-snomed/Production",
        "s3://bucket/model.zip",
        "runs:/abc123/model",
        "file:///path/to/model",
        "https://example.com/model.zip",
    ]
    for idx, uri in enumerate(valid_uris):
        config = model_manager.create_on_demand_config(
            model_name=f"test-validation-uri-{idx}",
            model_uri=uri,
            idle_ttl=3600,
        )
        assert config.model_uri == uri

    # Invalid: empty string
    with pytest.raises(ValueError, match="Model URI cannot be empty"):
        model_manager.create_on_demand_config(
            model_name="test-empty-uri",
            model_uri="",
            idle_ttl=3600,
        )

    # Invalid: whitespace only
    with pytest.raises(ValueError, match="Model URI cannot be empty"):
        model_manager.create_on_demand_config(
            model_name="test-whitespace-uri",
            model_uri="   ",
            idle_ttl=3600,
        )

    # Test that model_uri validator strips leading/trailing whitespace."""
    model_config = model_manager.create_on_demand_config(
        model_name="test-strip-uri",
        model_uri="  models:/test/v1  ",
        idle_ttl=3600,
    )
    assert model_config.model_uri == "models:/test/v1"


def test_on_demand_config_deploy_property(model_manager: ModelManager) -> None:
    """Test deploy property."""
    model_config = model_manager.create_on_demand_config(
        model_name="no-deploy",
        model_uri="models:/test/v1",
        idle_ttl=3600,
    )

    assert model_config.deploy is None

    # Test deploy property deserializes deploy_spec_json correctly
    model_config = model_manager.create_on_demand_config(
        model_name="with-deploy",
        model_uri="models:/test/v1",
        idle_ttl=3600,
        deploy_spec={
            "resources": {
                "limits": {"memory": "4g", "cpus": "2.0"},
                "reservations": {"memory": "2g"},
            }
        },
    )

    deploy = model_config.deploy
    assert deploy is not None
    assert deploy.resources is not None
    assert deploy.resources.limits is not None
    assert deploy.resources.limits.memory == "4g"
    assert deploy.resources.limits.cpus == "2.0"
    assert deploy.resources.reservations is not None
    assert deploy.resources.reservations.memory == "2g"
    assert deploy.resources.reservations.cpus is None  # Not specified in test data


def test_on_demand_config_model_dump(model_manager: ModelManager) -> None:
    """Test model_dump() for API responses."""
    model_config = model_manager.create_on_demand_config(
        model_name="test-dump",
        model_uri="models:/test/v1",
        idle_ttl=3600,
        description="Test model",
        deploy_spec={"resources": {"limits": {"memory": "4g"}}},
    )

    data = model_config.model_dump()
    assert all(
        k in data
        for k in [
            "id",
            "model_name",
            "model_uri",
            "idle_ttl",
            "description",
            "deploy",
            "enabled",
            "created_at",
            "updated_at",
        ]
    )
    assert data["model_name"] == "test-dump"
    assert data["model_uri"] == "models:/test/v1"
    assert data["idle_ttl"] == 3600
    assert data["description"] == "Test model"
    assert "deploy_spec_json" not in data  # excluded from serialization (internal field)
    assert data["deploy"] is not None  # computed field, included
    assert data["deploy"]["resources"]["limits"]["memory"] == "4g"
    assert data["enabled"] is True
