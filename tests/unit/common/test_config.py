from unittest.mock import mock_open, patch

from cogstack_model_gateway.common.config import Config, get_config, load_config

TEST_ENV_VARS = {
    "CMG_DB_USER": "test_user",
    "CMG_NOT_ACCEPTED_VAR": "test_value",
    "OTHER_NOT_ACCEPTED_VAR": "other_value",
}


def test_config_bool():
    assert not Config({})
    assert Config({"key": "value"})


def test_config_contains():
    config = Config({"key": "value"})
    assert "key" in config
    assert "missing_key" not in config


def test_config_len():
    assert len(Config({})) == 0
    assert len(Config({"key": "value"})) == 1
    assert len(Config({"key": "value", "nested": {"inner_key": "inner_value"}})) == 2


def test_config_iter():
    config = Config({"key": "value", "nested": {"inner_key": "inner_value"}})
    assert set(iter(config)) == {"key", "nested"}
    assert set(iter(config.nested)) == {"inner_key"}


def test_config_eq():
    assert Config({"key": "value"}) == Config({"key": "value"})
    assert Config({"key": "value"}) != Config({"key": "other_value"})
    assert Config({"key": "value"}) != "not a Config"


def test_config_repr():
    config = Config({"key": "value"})
    assert repr(config) == "Config({'key': 'value'})"

    config.set("nested", {"inner_key": "inner_value"})
    assert (
        repr(config) == "Config({'key': 'value', 'nested': Config({'inner_key': 'inner_value'})})"
    )


def test_config_to_dict():
    d = {"key": "value", "nested": {"inner_key": "inner_value"}}
    assert Config(d).to_dict() == d


def test_config_set():
    config = Config({})
    config.set("key", "value")
    assert config.key == "value"

    config.set("nested", {"inner_key": "inner_value"})
    assert isinstance(config.nested, Config)
    assert config.nested.inner_key == "inner_value"


@patch("cogstack_model_gateway.common.config._config_instance", new=Config({"key": "value"}))
def test_get_config():
    config = get_config()
    assert isinstance(config, Config)
    assert config.key == "value"


@patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
@patch("os.getenv", side_effect=lambda k, d=None: TEST_ENV_VARS.get(k, d))
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("cogstack_model_gateway.common.config._config_instance", new=None)
def test_load_config(mock_load_dotenv, mock_getenv, mock_open):
    config = load_config()
    assert isinstance(config, Config)
    assert config.key == "value"

    assert "env" in config
    assert len(config.env) == 1
    assert "db_user" in config.env
    assert config.env.db_user == "test_user"


@patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
@patch("os.getenv", side_effect=lambda k, d=None: None)
@patch("cogstack_model_gateway.common.config.load_dotenv")
@patch("cogstack_model_gateway.common.config._config_instance", new=None)
def test_load_config_no_env_vars(mock_load_dotenv, mock_getenv, mock_open):
    config = load_config()
    assert isinstance(config, Config)
    assert config.key == "value"
    assert not config.env
    config.set("nested", {"inner_key": "inner_value"})
    assert isinstance(config.nested, Config)
    assert config.nested.inner_key == "inner_value"
