import logging
from collections.abc import Generator

import pytest
from sqlmodel import Session, SQLModel, create_engine


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to the pytest command line."""
    parser.addoption(
        "--skip-cleanup-cms",
        action="store_true",
        default=False,
        help="Skip cleanup for the CMS resources after completing the tests.",
    )

    parser.addoption(
        "--enable-cmg-logging",
        action="store_true",
        default=False,
        help="Enable logging for the CogStack Model Gateway resources (i.e. Gateway, Scheduler).",
    )


@pytest.fixture(scope="module")
def cleanup_cms(request: pytest.FixtureRequest) -> bool:
    """Determine whether to cleanup the CMS resources after completing the tests."""
    return not request.config.getoption("--skip-cleanup-cms")


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    """Set up logging for the CogStack Model Gateway test resources.

    Suppress logging from the `testcontainers` package, set the logging level for the CogStack Model
    Gateway to DEBUG, and for the tests to INFO.
    """
    # Suppress logging from testcontainers
    for logger_name in logging.root.manager.loggerDict:
        if logger_name.startswith("testcontainers"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    parent_logger = logging.getLogger("cmg")
    parent_logger.setLevel(logging.DEBUG)

    if not any(isinstance(handler, logging.StreamHandler) for handler in parent_logger.handlers):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        parent_logger.addHandler(handler)

    logging.getLogger("cmg.tests").setLevel(logging.INFO)


@pytest.fixture(scope="function")
def db_manager() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database for testing."""
    test_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(test_engine)

    class TestDatabaseManager:
        def get_session(self) -> Session:
            return Session(test_engine)

    return TestDatabaseManager()
