from contextlib import contextmanager

from sqlalchemy.exc import ProgrammingError
from sqlmodel import Session, SQLModel, create_engine

POSTGRES_URL = "postgresql+psycopg2://user:password@postgres/taskdb"


class DatabaseManager:
    def __init__(self, database_url: str = POSTGRES_URL):
        self.database_url = database_url
        self.engine = create_engine(self.database_url, echo=True)

    def init_db(self):
        # FIXME: Error handling
        try:
            SQLModel.metadata.create_all(self.engine)
            print("Database setup complete.")
        except ProgrammingError:
            print("Database already initialized.")

    @contextmanager
    def get_session(self):
        with Session(self.engine) as session:
            yield session
