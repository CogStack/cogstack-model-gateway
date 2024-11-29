import json
import logging
import os

from dotenv import load_dotenv

log = logging.getLogger("cmg")

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
ACCEPTED_ENVIRONMENT_VARIABLE_PREFIX = "CMG_"
ACCEPTED_ENVIRONMENT_VARIABLES = {
    "CMG_DB_USER",
    "CMG_DB_PASSWORD",
    "CMG_DB_NAME",
    "CMG_DB_HOST",
    "CMG_DB_PORT",
    "CMG_QUEUE_USER",
    "CMG_QUEUE_PASSWORD",
    "CMG_QUEUE_NAME",
    "CMG_QUEUE_HOST",
    "CMG_QUEUE_PORT",
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

    def set(self, key: str, value):
        if isinstance(value, dict):
            value = Config(value)
        setattr(self, key, value)


def load_config() -> Config:
    """Load configuration from the provided JSON file and environment variables."""
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file {CONFIG_FILE} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Config file {CONFIG_FILE} is not a valid JSON file.")
        raise

    load_dotenv()
    config["env"] = {
        var.replace(ACCEPTED_ENVIRONMENT_VARIABLE_PREFIX, "", 1).lower(): value
        for var in ACCEPTED_ENVIRONMENT_VARIABLES
        if (value := os.getenv(var))
    }
    log.debug(f"Loaded config: {config}")
    return Config(config)


config = load_config()


def get_config() -> Config:
    return config
