from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from cogstack_model_gateway.common.containers import (
    IS_MODEL_LABEL,
    MANAGED_BY_LABEL,
    MANAGED_BY_LABEL_VALUE,
    TTL_LABEL,
)
from cogstack_model_gateway.ripper.main import purge_expired_containers, stop_and_remove_container


def test_stop_and_remove_container():
    mock_container = MagicMock()

    stop_and_remove_container(mock_container)

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


@patch("cogstack_model_gateway.ripper.main.docker.from_env")
@patch("cogstack_model_gateway.ripper.main.time.sleep", side_effect=KeyboardInterrupt)
def test_purge_expired_containers(mock_sleep, mock_docker):
    mock_container = MagicMock()
    mock_container.labels = {
        MANAGED_BY_LABEL: MANAGED_BY_LABEL_VALUE,
        IS_MODEL_LABEL: "test_model",
        TTL_LABEL: "10",
    }
    mock_container.attrs = {"Created": f"{(datetime.now() - timedelta(seconds=20)).isoformat()}Z"}

    mock_client = MagicMock()
    mock_client.containers.list.return_value = [mock_container]

    mock_docker.return_value = mock_client

    with pytest.raises(KeyboardInterrupt):
        purge_expired_containers()

    mock_client.containers.list.assert_called_once_with(
        filters={"label": [f"{MANAGED_BY_LABEL}={MANAGED_BY_LABEL_VALUE}", IS_MODEL_LABEL]},
    )
    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()
    mock_sleep.assert_called_once()
