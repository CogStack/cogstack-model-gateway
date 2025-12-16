from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from cogstack_model_gateway.common.models import ModelDeploymentType
from cogstack_model_gateway.ripper.main import (
    RemovalReason,
    purge_expired_containers,
    should_remove_by_fixed_ttl,
    should_remove_by_idle_ttl,
    stop_and_remove_container,
)


@patch("cogstack_model_gateway.ripper.main.stop_and_remove_model_container")
def test_stop_and_remove_container(mock_stop_and_remove):
    """Test that stop_and_remove_container stops/removes container and deletes from DB."""
    mock_container = MagicMock()
    mock_container.name = "test-container"
    mock_model_manager = MagicMock()
    mock_model_manager.delete_model.return_value = True

    stop_and_remove_container(
        container=mock_container,
        model_name="test-model",
        model_manager=mock_model_manager,
        deployment_type=ModelDeploymentType.MANUAL,
        reason=RemovalReason.FIXED_TTL_EXPIRED,
        idle_time=None,
    )

    mock_stop_and_remove.assert_called_once_with(mock_container)
    mock_model_manager.delete_model.assert_called_once_with("test-model")


@patch("cogstack_model_gateway.ripper.main.get_models")
@patch("cogstack_model_gateway.ripper.main.stop_and_remove_model_container")
@patch("cogstack_model_gateway.ripper.main.time.sleep", side_effect=KeyboardInterrupt)
def test_purge_expired_containers_manual_deployment(
    mock_sleep, mock_stop_and_remove, mock_get_models
):
    """Test that manual deployments are purged based on fixed TTL from labels."""
    mock_config = MagicMock()
    mock_config.labels.managed_by_label = "org.cogstack.model-gateway.managed-by"
    mock_config.labels.managed_by_value = "cmg"
    mock_config.labels.cms_model_label = "org.cogstack.model-serve"
    mock_config.labels.ttl_label = "org.cogstack.model-gateway.ttl"
    mock_config.labels.deployment_type_label = "org.cogstack.model-gateway.deployment-type"
    mock_config.cms.project_name = "cms"
    mock_config.ripper.interval = 60

    mock_model_manager = MagicMock()
    mock_model_manager.delete_model.return_value = True
    mock_config.model_manager = mock_model_manager

    mock_container = MagicMock()
    mock_container.name = "test-container"
    mock_container.labels = {
        "org.cogstack.model-gateway.managed-by": "cmg",
        "org.cogstack.model-serve": "test_model",
        "org.cogstack.model-gateway.ttl": "10",
        "org.cogstack.model-gateway.deployment-type": "manual",
        "com.docker.compose.service": "test-model",
        "com.docker.compose.project": "cms",
    }
    mock_container.attrs = {"Created": f"{(datetime.now(UTC) - timedelta(seconds=20)).isoformat()}"}

    mock_get_models.return_value = [
        {
            "service_name": "test-model",
            "deployment_type": "manual",
            "container": mock_container,
        }
    ]

    with pytest.raises(KeyboardInterrupt):
        purge_expired_containers(mock_config)

    mock_get_models.assert_called_once_with(all=True, managed_only=True)
    mock_stop_and_remove.assert_called_once_with(mock_container)
    mock_model_manager.delete_model.assert_called_once_with("test-model")
    mock_sleep.assert_called_once()


@patch("cogstack_model_gateway.ripper.main.get_models")
@patch("cogstack_model_gateway.ripper.main.stop_and_remove_model_container")
@patch("cogstack_model_gateway.ripper.main.time.sleep", side_effect=KeyboardInterrupt)
def test_purge_expired_containers_static_deployment_not_removed(
    mock_sleep, mock_stop_and_remove, mock_get_models
):
    """Test that static deployments are never auto-removed."""
    mock_config = MagicMock()
    mock_config.labels.managed_by_label = "org.cogstack.model-gateway.managed-by"
    mock_config.labels.managed_by_value = "cmg"
    mock_config.labels.cms_model_label = "org.cogstack.model-serve"
    mock_config.labels.deployment_type_label = "org.cogstack.model-gateway.deployment-type"
    mock_config.cms.project_name = "cms"
    mock_config.ripper.interval = 60

    mock_model_manager = MagicMock()
    mock_config.model_manager = mock_model_manager

    mock_container = MagicMock()
    mock_container.name = "static-model"
    mock_container.labels = {
        "org.cogstack.model-gateway.deployment-type": "static",
        "com.docker.compose.service": "static-model",
    }

    mock_get_models.return_value = [
        {
            "service_name": "static-model",
            "deployment_type": "static",
            "container": mock_container,
        }
    ]

    with pytest.raises(KeyboardInterrupt):
        purge_expired_containers(mock_config)

    mock_stop_and_remove.assert_not_called()
    mock_model_manager.delete_model.assert_not_called()


@patch("cogstack_model_gateway.ripper.main.get_models")
@patch("cogstack_model_gateway.ripper.main.stop_and_remove_model_container")
@patch("cogstack_model_gateway.ripper.main.time.sleep", side_effect=KeyboardInterrupt)
def test_purge_expired_containers_auto_deployment(
    mock_sleep, mock_stop_and_remove, mock_get_models
):
    """Test that auto deployments are purged based on idle TTL from database."""
    mock_config = MagicMock()
    mock_config.labels.managed_by_label = "org.cogstack.model-gateway.managed-by"
    mock_config.labels.managed_by_value = "cmg"
    mock_config.labels.cms_model_label = "org.cogstack.model-serve"
    mock_config.labels.deployment_type_label = "org.cogstack.model-gateway.deployment-type"
    mock_config.cms.project_name = "cms"
    mock_config.ripper.interval = 60

    mock_model_manager = MagicMock()
    mock_model_manager.is_model_idle.return_value = (True, 600.0)
    mock_model_manager.delete_model.return_value = True
    mock_config.model_manager = mock_model_manager

    mock_container = MagicMock()
    mock_container.name = "auto-model"
    mock_container.labels = {
        "org.cogstack.model-gateway.deployment-type": "auto",
        "com.docker.compose.service": "auto-model",
    }
    mock_container.attrs = {"Created": f"{datetime.now(UTC).isoformat()}"}

    mock_get_models.return_value = [
        {
            "service_name": "auto-model",
            "deployment_type": "auto",
            "container": mock_container,
        }
    ]

    with pytest.raises(KeyboardInterrupt):
        purge_expired_containers(mock_config)

    mock_model_manager.is_model_idle.assert_called_once_with("auto-model")
    mock_stop_and_remove.assert_called_once_with(mock_container)
    mock_model_manager.delete_model.assert_called_once_with("auto-model")


def test_should_remove_by_fixed_ttl():
    """Test should_remove_by_fixed_ttl returns True when TTL is exceeded and False otherwise."""
    mock_container = MagicMock()
    mock_container.labels = {"org.cogstack.model-gateway.ttl": "10"}
    mock_container.attrs = {"Created": f"{(datetime.now(UTC) - timedelta(seconds=20)).isoformat()}"}

    assert should_remove_by_fixed_ttl(mock_container, "org.cogstack.model-gateway.ttl") is True

    mock_container.reset_mock()
    mock_container.labels = {"org.cogstack.model-gateway.ttl": "-1"}
    assert should_remove_by_fixed_ttl(mock_container, "org.cogstack.model-gateway.ttl") is False

    mock_container.reset_mock()
    mock_container.labels = {"org.cogstack.model-gateway.ttl": "100"}
    assert should_remove_by_fixed_ttl(mock_container, "org.cogstack.model-gateway.ttl") is False


def test_should_remove_by_idle_ttl():
    """Test should_remove_by_idle_ttl returns True when model is idle and False otherwise."""
    mock_model_manager = MagicMock()
    mock_model_manager.is_model_idle.return_value = (True, 500.0)

    should_remove, idle_time = should_remove_by_idle_ttl("idle-model", mock_model_manager)

    assert should_remove is True
    assert idle_time == 500.0
    mock_model_manager.is_model_idle.assert_called_once_with("idle-model")

    mock_model_manager.reset_mock()
    mock_model_manager.is_model_idle.return_value = (False, 50.0)

    should_remove, idle_time = should_remove_by_idle_ttl("active-model", mock_model_manager)

    assert should_remove is False
    assert idle_time == 50.0
    mock_model_manager.is_model_idle.assert_called_once_with("active-model")
