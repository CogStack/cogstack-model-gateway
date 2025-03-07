import logging
import os


def configure_logging():
    parent_logger = logging.getLogger("cmg")
    parent_logger.setLevel(logging.DEBUG)

    if not any(isinstance(handler, logging.StreamHandler) for handler in parent_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        parent_logger.addHandler(handler)

    logging.getLogger("cmg.common").setLevel(os.getenv("CMG_COMMON_LOG_LEVEL", logging.INFO))
    logging.getLogger("cmg.gateway").setLevel(os.getenv("CMG_GATEWAY_LOG_LEVEL", logging.INFO))
    logging.getLogger("cmg.ripper").setLevel(os.getenv("CMG_RIPPER_LOG_LEVEL", logging.INFO))
    logging.getLogger("cmg.scheduler").setLevel(os.getenv("CMG_SCHEDULER_LOG_LEVEL", logging.INFO))
