import logging
from contextlib import contextmanager

from sqlalchemy.exc import ProgrammingError
from sqlmodel import Session, SQLModel, create_engine

log = logging.getLogger("cmg.common")

DEFAULT_POSTGRES_URL = "postgresql+psycopg2://user:password@postgres/cmg_tasks"


class DatabaseManager:
    def __init__(
        self,
        user: str = None,
        password: str = None,
        host: str = None,
        port: int = None,
        db_name: str = None,
        connection_url: str = None,
    ):
        if user and password and host and port and db_name:
            self.connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        elif connection_url:
            self.connection_url = connection_url
        else:
            self.connection_url = DEFAULT_POSTGRES_URL

        self.engine = create_engine(self.connection_url)

    def init_db(self):
        """Initialize the database by creating all tables."""
        log.info("Initializing database")
        try:
            SQLModel.metadata.create_all(self.engine)
            log.info("Database setup completed")
        except ProgrammingError:
            log.info("Database already exists")

    @contextmanager
    def get_session(self):
        """Get a database session."""
        with Session(self.engine) as session:
            yield session
