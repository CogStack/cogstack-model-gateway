import json
import logging
import os

from dotenv import load_dotenv

log = logging.getLogger("cmg.common")

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
ACCEPTED_ENVIRONMENT_VARIABLE_PREFIX = "CMG_"
ACCEPTED_ENVIRONMENT_VARIABLES = {
    "CMG_DB_USER",
    "CMG_DB_PASSWORD",
    "CMG_DB_NAME",
    "CMG_DB_HOST",
    "CMG_DB_PORT",
    "CMG_OBJECT_STORE_HOST",
    "CMG_OBJECT_STORE_PORT",
    "CMG_OBJECT_STORE_ACCESS_KEY",
    "CMG_OBJECT_STORE_SECRET_KEY",
    "CMG_OBJECT_STORE_BUCKET_TASKS",
    "CMG_OBJECT_STORE_BUCKET_RESULTS",
    "CMG_QUEUE_USER",
    "CMG_QUEUE_PASSWORD",
    "CMG_QUEUE_NAME",
    "CMG_QUEUE_HOST",
    "CMG_QUEUE_PORT",
    "CMG_SCHEDULER_MAX_CONCURRENT_TASKS",
}


# FIXME: Add validation
class Config:
    def __init__(self, config: dict):
        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __bool__(self):
        return bool(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, Config):
            return self.__dict__ == other.__dict__
        return False

    def set(self, key: str, value):
        if isinstance(value, dict):
            value = Config(value)
        setattr(self, key, value)

    def to_dict(self):
        return {
            key: value.to_dict() if isinstance(value, Config) else value
            for key, value in self.__dict__.items()
        }


_config_instance: Config = None


def load_config() -> Config:
    """Load configuration from the provided JSON file and environment variables."""
    global _config_instance
    if _config_instance is None:
        try:
            with open(CONFIG_FILE) as f:
                config = json.load(f)
        except FileNotFoundError:
            log.error(f"Config file {CONFIG_FILE} not found.")
            raise
        except json.JSONDecodeError:
            log.error(f"Config file {CONFIG_FILE} is not a valid JSON file.")
            raise

        load_dotenv()
        config["env"] = {
            var.replace(ACCEPTED_ENVIRONMENT_VARIABLE_PREFIX, "", 1).lower(): value
            for var in ACCEPTED_ENVIRONMENT_VARIABLES
            if (value := os.getenv(var))
        }
        log.info(f"Loaded config: {config}")
        _config_instance = Config(config)
    return _config_instance


def get_config() -> Config:
    """Get the current configuration instance."""
    if _config_instance is None:
        raise RuntimeError("Config not initialized. Call load_config() first.")
    return _config_instance
