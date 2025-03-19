import logging
import os

COMMON_LOG_LEVEL_VAR = "CMG_COMMON_LOG_LEVEL"
GATEWAY_LOG_LEVEL_VAR = "CMG_GATEWAY_LOG_LEVEL"
RIPPER_LOG_LEVEL_VAR = "CMG_RIPPER_LOG_LEVEL"
SCHEDULER_LOG_LEVEL_VAR = "CMG_SCHEDULER_LOG_LEVEL"


def configure_logging():
    """Configure logging for the CogStack Model Gateway packages.

    The logging level for each package can be set using the corresponding environment variable.
    """
    parent_logger = logging.getLogger("cmg")
    parent_logger.setLevel(logging.DEBUG)

    if not any(isinstance(handler, logging.StreamHandler) for handler in parent_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        parent_logger.addHandler(handler)

    logging.getLogger("cmg.common").setLevel(os.getenv(COMMON_LOG_LEVEL_VAR) or logging.INFO)
    logging.getLogger("cmg.gateway").setLevel(os.getenv(GATEWAY_LOG_LEVEL_VAR) or logging.INFO)
    logging.getLogger("cmg.ripper").setLevel(os.getenv(RIPPER_LOG_LEVEL_VAR) or logging.INFO)
    logging.getLogger("cmg.scheduler").setLevel(os.getenv(SCHEDULER_LOG_LEVEL_VAR) or logging.INFO)
