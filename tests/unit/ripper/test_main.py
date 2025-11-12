from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from cogstack_model_gateway.ripper.main import purge_expired_containers, stop_and_remove_container


def test_stop_and_remove_container():
    mock_container = MagicMock()

    stop_and_remove_container(mock_container)

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


@patch("cogstack_model_gateway.ripper.main.docker.from_env")
@patch("cogstack_model_gateway.ripper.main.time.sleep", side_effect=KeyboardInterrupt)
def test_purge_expired_containers(mock_sleep, mock_docker):
    mock_config = MagicMock()
    mock_config.labels.managed_by_label = "org.cogstack.model-gateway.managed-by"
    mock_config.labels.managed_by_value = "cmg"
    mock_config.labels.cms_model_label = "org.cogstack.model-serve"
    mock_config.labels.ttl_label = "org.cogstack.model-gateway.ttl"
    mock_config.ripper.interval = 60

    mock_container = MagicMock()
    mock_container.labels = {
        "org.cogstack.model-gateway.managed-by": "cmg",
        "org.cogstack.model-serve": "test_model",
        "org.cogstack.model-gateway.ttl": "10",
    }
    mock_container.attrs = {"Created": f"{(datetime.now(UTC) - timedelta(seconds=20)).isoformat()}"}

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    mock_docker.return_value = mock_client

    with pytest.raises(KeyboardInterrupt):
        purge_expired_containers(mock_config)

    mock_client.containers.list.assert_called_once_with(
        filters={
            "label": [
                f"{mock_config.labels.managed_by_label}={mock_config.labels.managed_by_value}",
                mock_config.labels.cms_model_label,
            ]
        },
    )
    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()
    mock_sleep.assert_called_once()
