import json
import logging
import os

from dotenv import load_dotenv

log = logging.getLogger("cmg.common")

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
ACCEPTED_ENVIRONMENT_VARIABLES = {
    "cms": {
        "CMS_HOST_URL": "",
        "CMS_PROJECT_NAME": "cms",
        "CMS_SERVER_PORT": "8000",
    },
    "cmg": {
        "CMG_DB_USER": "admin",
        "CMG_DB_PASSWORD": "admin",
        "CMG_DB_NAME": "cmg_tasks",
        "CMG_DB_HOST": "postgres",
        "CMG_DB_PORT": "5432",
        "CMG_OBJECT_STORE_HOST": "minio",
        "CMG_OBJECT_STORE_PORT": "9000",
        "CMG_OBJECT_STORE_ACCESS_KEY": "admin",
        "CMG_OBJECT_STORE_SECRET_KEY": "admin123",
        "CMG_OBJECT_STORE_BUCKET_TASKS": "cmg-tasks",
        "CMG_OBJECT_STORE_BUCKET_RESULTS": "cmg-results",
        "CMG_QUEUE_USER": "admin",
        "CMG_QUEUE_PASSWORD": "admin",
        "CMG_QUEUE_NAME": "cmg_tasks",
        "CMG_QUEUE_HOST": "rabbitmq",
        "CMG_QUEUE_PORT": "5672",
        "CMG_SCHEDULER_MAX_CONCURRENT_TASKS": "1",
    },
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
            log.warning(f"Config file {CONFIG_FILE} not found.")
            config = {}
        except json.JSONDecodeError:
            log.error(f"Config file {CONFIG_FILE} is not a valid JSON file.")
            raise

        load_dotenv()
        for key, env_vars in ACCEPTED_ENVIRONMENT_VARIABLES.items():
            config[key] = {
                var.replace(f"{key.upper()}_", "", 1).lower(): os.getenv(var, default)
                for var, default in env_vars.items()
            }
        log.info(f"Loaded config: {config}")
        _config_instance = Config(config)
    return _config_instance


def get_config() -> Config:
    """Get the current configuration instance."""
    if _config_instance is None:
        raise RuntimeError("Config not initialized. Call load_config() first.")
    return _config_instance
