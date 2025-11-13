from unittest.mock import mock_open, patch

import pytest
from pydantic import BaseModel

from cogstack_model_gateway.common.config import (
    Config,
    get_config,
    load_config,
)
from cogstack_model_gateway.common.config.models import OnDemandModel

TEST_ENV_VARS = {
    "CMS_HOST_URL": "http://localhost:8000",
    "CMS_PROJECT_NAME": "test-project",
    "CMG_DB_USER": "test_user",
    "CMG_DB_PASSWORD": "test_password",
    "CMG_DB_HOST": "localhost",
    "CMG_DB_PORT": "5432",
    "CMG_NOT_ACCEPTED_VAR": "test_value",
    "OTHER_NOT_ACCEPTED_VAR": "other_value",
}


def test_config_is_pydantic_model():
    """Test that Config is a Pydantic model."""
    config = Config.model_validate({})
    assert isinstance(config, BaseModel)

    assert hasattr(config, "model_dump")
    assert hasattr(config, "model_validate")


def test_config_has_expected_fields():
    """Test that Config has the expected top-level fields."""
    config = Config.model_validate({})

    assert hasattr(config, "cms")
    assert hasattr(config, "db")
    assert hasattr(config, "object_store")
    assert hasattr(config, "queue")
    assert hasattr(config, "scheduler")
    assert hasattr(config, "ripper")
    assert hasattr(config, "models")


def test_config_validation():
    """Test that Config validates input properly."""
    config = Config.model_validate({})
    assert config is not None

    config = Config.model_validate(
        {
            "cms": {"host_url": "http://localhost:8000"},
            "db": {"user": "test_user"},
        }
    )
    assert config.cms.host_url == "http://localhost:8000"
    assert config.db.user == "test_user"


def test_config_to_dict():
    """Test converting Config to dictionary."""
    config = Config.model_validate(
        {
            "cms": {"host_url": "http://localhost:8000"},
            "db": {"user": "test_user"},
        }
    )

    config_dict = config.model_dump()
    assert isinstance(config_dict, dict)
    assert "cms" in config_dict
    assert "db" in config_dict
    assert config_dict["cms"]["host_url"] == "http://localhost:8000"
    assert config_dict["db"]["user"] == "test_user"


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
def test_get_config_before_load():
    """Test that get_config raises error if config not loaded."""
    try:
        get_config()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "not initialized" in str(e).lower()


def test_get_config_after_load():
    """Test that get_config returns the cached config instance."""
    mock_config = Config.model_validate({})

    with patch("cogstack_model_gateway.common.config._config_instance", new=mock_config):
        config = get_config()
        assert isinstance(config, Config)
        assert config is mock_config


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("os.getenv", side_effect=lambda k, d=None: TEST_ENV_VARS.get(k, d))
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"models": {"deployment": {"auto": {"enabled": true}}}}',
)
def test_load_config(mock_file, mock_getenv, mock_load_dotenv):
    """Test loading config from JSON file and environment variables."""
    config = load_config()

    assert isinstance(config, Config)
    assert config.cms.host_url == "http://localhost:8000"
    assert config.cms.project_name == "test-project"

    assert config.db.user == "test_user"
    assert config.db.password == "test_password"
    assert config.db.host == "localhost"
    assert config.db.port == 5432

    assert config.models is not None


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("os.getenv", side_effect=lambda k, d=None: None)
@patch("builtins.open", new_callable=mock_open, read_data='{"models": {}, "labels": {}}')
def test_load_config_no_env_vars(mock_file, mock_getenv, mock_load_dotenv):
    """Test loading config with no environment variables set."""
    config = load_config()

    assert isinstance(config, Config)

    # Check that config uses defaults when env vars not set
    assert config.cms is not None
    assert config.db is not None
    assert config.models is not None


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("os.getenv", side_effect=lambda k, d=None: TEST_ENV_VARS.get(k, d))
@patch("builtins.open", side_effect=FileNotFoundError)
def test_load_config_missing_file(mock_file, mock_getenv, mock_load_dotenv):
    """Test loading config when JSON file is missing (uses defaults)."""
    config = load_config()

    assert isinstance(config, Config)

    # Should still have env var config
    assert config.cms.host_url == "http://localhost:8000"
    assert config.db.user == "test_user"

    assert config.models is not None


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("os.getenv", return_value=None)
@patch("builtins.open", new_callable=mock_open, read_data='{"models": {}, "labels": {}}')
def test_load_config_caching(mock_file, mock_getenv, mock_load_dotenv):
    """Test that load_config caches the result."""
    config1 = load_config()
    config2 = load_config()

    # Should return the same instance
    assert config1 is config2


def test_config_runtime_managers():
    """Test that runtime managers can be assigned and accessed."""
    config = Config.model_validate({})

    # Initially None
    assert config.database_manager is None
    assert config.task_manager is None
    assert config.model_manager is None
    assert config.task_object_store_manager is None
    assert config.results_object_store_manager is None
    assert config.queue_manager is None

    # Can be assigned
    class MockManager:
        def __init__(self, name):
            self.name = name

    config.database_manager = MockManager("DB")
    config.task_manager = MockManager("Task")

    assert config.database_manager.name == "DB"
    assert config.task_manager.name == "Task"


def test_config_models_deployment():
    """Test models deployment configuration structure."""
    config = Config.model_validate({})

    assert hasattr(config.models.deployment, "auto")
    assert hasattr(config.models.deployment.auto, "config")
    assert hasattr(config.models.deployment.auto, "on_demand")
    auto_config = config.models.deployment.auto.config
    assert auto_config.health_check_timeout == 300
    assert auto_config.default_idle_ttl == 3600
    assert auto_config.deployment_retry_attempts == 2

    assert hasattr(config.models.deployment, "manual")
    assert config.models.deployment.manual.default_ttl == 86400
    assert config.models.deployment.manual.allow_ttl_override is True

    assert hasattr(config.models.deployment, "static")


def test_config_labels():
    """Test Docker labels configuration."""
    config = Config.model_validate({})

    assert config.labels.cms_model_label == "org.cogstack.model-serve"
    assert config.labels.cms_model_uri_label == "org.cogstack.model-serve.uri"
    assert config.labels.deployment_type_label == "org.cogstack.model-gateway.deployment-type"
    assert config.labels.managed_by_label == "org.cogstack.model-gateway.managed-by"
    assert config.labels.managed_by_value == "cmg"
    assert config.labels.ttl_label == "org.cogstack.model-gateway.ttl"


def test_config_helper_get_auto_deployment_config():
    """Test get_auto_deployment_config helper method."""
    config = Config.model_validate(
        {
            "models": {
                "deployment": {
                    "auto": {
                        "config": {
                            "health_check_timeout": 500,
                            "default_idle_ttl": 7200,
                        }
                    }
                }
            }
        }
    )

    auto_config = config.get_auto_deployment_config()
    assert auto_config.health_check_timeout == 500
    assert auto_config.default_idle_ttl == 7200


@pytest.mark.parametrize(
    "config, expected",
    [
        (Config.model_validate({}), []),
        (
            Config.model_validate(
                {
                    "models": {
                        "deployment": {
                            "auto": {
                                "on_demand": [
                                    {
                                        "service_name": "medcat-snomed",
                                        "model_uri": "cogstacksystems/medcat-snomed:latest",
                                        "description": "MedCAT SNOMED model",
                                    },
                                    {
                                        "service_name": "custom-ner",
                                        "model_uri": "custom/ner:v1",
                                        "description": "Custom NER model",
                                    },
                                ]
                            }
                        }
                    }
                }
            ),
            [
                OnDemandModel(
                    service_name="medcat-snomed",
                    model_uri="cogstacksystems/medcat-snomed:latest",
                    description="MedCAT SNOMED model",
                    idle_ttl=3600,  # Applied from AutoDeploymentConfig.default_idle_ttl
                    deploy={},
                ),
                OnDemandModel(
                    service_name="custom-ner",
                    model_uri="custom/ner:v1",
                    description="Custom NER model",
                    idle_ttl=3600,  # Applied from AutoDeploymentConfig.default_idle_ttl
                    deploy={},
                ),
            ],
        ),
    ],
)
def test_config_helper_list_on_demand_models(config, expected):
    """Test list_on_demand_models helper method."""
    result = config.list_on_demand_models()
    assert isinstance(result, list)
    assert result == expected


def test_config_helper_get_on_demand_model():
    """Test get_on_demand_model helper method."""
    config = Config.model_validate(
        {
            "models": {
                "deployment": {
                    "auto": {
                        "on_demand": [
                            {
                                "service_name": "medcat-snomed",
                                "model_uri": "cogstacksystems/medcat-snomed:latest",
                                "description": "MedCAT SNOMED model",
                            }
                        ]
                    }
                }
            }
        }
    )

    model = config.get_on_demand_model("medcat-snomed")
    assert model is not None
    assert model.service_name == "medcat-snomed"
    assert model.model_uri == "cogstacksystems/medcat-snomed:latest"

    model = config.get_on_demand_model("nonexistent")
    assert model is None


@patch("cogstack_model_gateway.common.config._config_instance", new=None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("os.getenv", side_effect=lambda k, d=None: TEST_ENV_VARS.get(k, d))
@patch("builtins.open", new_callable=mock_open, read_data='{"models": {}, "labels": {}}')
def test_config_comprehensive_structure(mock_file, mock_getenv, mock_load_dotenv):
    """Test comprehensive config structure access (similar to manual test scripts)."""
    config = load_config()

    # Test all major sections exist
    assert config.cms is not None
    assert config.db is not None
    assert config.object_store is not None
    assert config.queue is not None
    assert config.scheduler is not None
    assert config.ripper is not None
    assert config.models is not None
    assert config.labels is not None

    # Test CMS config (from env vars)
    assert config.cms.host_url == "http://localhost:8000"
    assert config.cms.project_name == "test-project"

    # Test database config (from env vars)
    assert config.db.host == "localhost"
    assert config.db.port == 5432
    assert config.db.user == "test_user"
    assert config.db.password == "test_password"

    # Test object store config (defaults)
    assert config.object_store.host == "minio"
    assert config.object_store.bucket_tasks == "cmg-tasks"

    # Test queue config (defaults)
    assert config.queue.host == "rabbitmq"
    assert config.queue.name == "cmg_tasks"

    # Test models config structure
    assert config.models.deployment.auto is not None
    assert config.models.deployment.manual is not None
    assert config.models.deployment.static is not None

    # Test labels
    assert "cogstack" in config.labels.cms_model_label.lower()
    assert config.labels.managed_by_value == "cmg"
