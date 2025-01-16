import logging

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-cleanup-cms",
        action="store_true",
        default=False,
        help="Skip cleanup for the CMS resources after completing the tests.",
    )


@pytest.fixture(scope="module")
def cleanup_cms(request: pytest.FixtureRequest) -> bool:
    return not request.config.getoption("--skip-cleanup-cms")


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    # Suppress logging from testcontainers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("testcontainers"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    parent_logger = logging.getLogger("cmg.tests")
    parent_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    parent_logger.addHandler(handler)

    # Configure child loggers
    logging.getLogger("cmg.tests.integration").setLevel(logging.INFO)
    logging.getLogger("cmg.tests.unit").setLevel(logging.INFO)
