import json
import logging
import os

from dotenv import load_dotenv
from pydantic import ValidationError

from cogstack_model_gateway.common.config.models import Config

log = logging.getLogger("cmg.common")

_config_instance: Config | None = None


def _load_json_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file) as f:
            config = json.load(f)
            log.info(f"Loaded configuration from {config_file}")
            return config
    except FileNotFoundError:
        log.warning(f"Config file {config_file} not found, using defaults.")
        return {}
    except json.JSONDecodeError:
        log.error(f"Config file {config_file} is not a valid JSON file.")
        raise


def _add_from_env_vars(target: dict, mapping: dict) -> None:
    """Recursively construct configuration dict from environment variables.

    Recursively populate `target` from `mapping` where mapping leaf values are environment variable
    names. If the env var is present, the corresponding key is set in `target`.

    Example mapping:
    {
        "db": {"user": "CMG_DB_USER", "host": "CMG_DB_HOST"}
    }
    """
    for key, val in mapping.items():
        if isinstance(val, dict):
            sub = target.setdefault(key, {})
            _add_from_env_vars(sub, val)
        else:
            if (env_val := os.getenv(val)) is not None:
                target[key] = env_val


def _create_config_from_env_vars(env_map: dict) -> dict:
    """Create a configuration dictionary from environment variables."""
    config: dict = {}
    _add_from_env_vars(config, env_map)
    return config


def _load_env_vars() -> dict:
    """Load configuration dict from environment variables, skipping None values."""
    load_dotenv()

    env_map = {
        "db": {
            "user": "CMG_DB_USER",
            "password": "CMG_DB_PASSWORD",
            "name": "CMG_DB_NAME",
            "host": "CMG_DB_HOST",
            "port": "CMG_DB_PORT",
        },
        "object_store": {
            "host": "CMG_OBJECT_STORE_HOST",
            "port": "CMG_OBJECT_STORE_PORT",
            "access_key": "CMG_OBJECT_STORE_ACCESS_KEY",
            "secret_key": "CMG_OBJECT_STORE_SECRET_KEY",
            "bucket_tasks": "CMG_OBJECT_STORE_BUCKET_TASKS",
            "bucket_results": "CMG_OBJECT_STORE_BUCKET_RESULTS",
        },
        "queue": {
            "user": "CMG_QUEUE_USER",
            "password": "CMG_QUEUE_PASSWORD",
            "name": "CMG_QUEUE_NAME",
            "host": "CMG_QUEUE_HOST",
            "port": "CMG_QUEUE_PORT",
        },
        "scheduler": {
            "max_concurrent_tasks": "CMG_SCHEDULER_MAX_CONCURRENT_TASKS",
            "metrics_port": "CMG_SCHEDULER_METRICS_PORT",
        },
        "ripper": {"interval": "CMG_RIPPER_INTERVAL", "metrics_port": "CMG_RIPPER_METRICS_PORT"},
    }

    return _create_config_from_env_vars(env_map)


def load_config(config_file: str | None = "config.json") -> Config:
    """Load and validate configuration from JSON file and environment variables.

    This function:
    1. Loads JSON configuration for tasks and models
    2. Loads environment variables for runtime services
    3. Merges them into a unified structure
    4. Validates everything with Pydantic schema
    5. Caches the result for subsequent calls

    Args:
        config_file (optional): Path to JSON configuration file (default: "config.json")

    Returns:
        Config: Fully validated configuration object

    Raises:
        ValidationError: If configuration doesn't match schema
        JSONDecodeError: If config file is not valid JSON
    """
    global _config_instance

    if _config_instance is None:
        json_config, env_config = _load_json_config(config_file), _load_env_vars()

        merged_config = {
            **env_config,
            "cms": json_config.get("cms", {}),
            "models": json_config.get("models", {}),
            "labels": json_config.get("labels", {}),
            # Only include tracking if explicitly provided in JSON
            # Otherwise, the validator will copy from cms.tracking
            **({"tracking": json_config["tracking"]} if "tracking" in json_config else {}),
        }

        try:
            _config_instance = Config.model_validate(merged_config)
            log.debug(f"Loaded config: {_config_instance.model_dump_json()}")
        except ValidationError as e:
            log.error(f"Configuration validation failed: {e}")
            raise

    return _config_instance


def get_config() -> Config:
    """Get the current configuration instance."""
    if _config_instance is None:
        raise RuntimeError("Config not initialized. Call load_config() first.")
    return _config_instance
