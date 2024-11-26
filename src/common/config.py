import json
import os

from dotenv import load_dotenv

CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")


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
    for key, value in os.environ.items():
        if key in config:
            config[key] = value

    return Config(config)


config = load_config()


def get_config() -> Config:
    return config
