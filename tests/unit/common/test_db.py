from unittest.mock import MagicMock, patch

from sqlmodel import Session

from cogstack_model_gateway.common.db import DEFAULT_POSTGRES_URL, DatabaseManager


def test_database_manager_init_with_params():
    db_manager = DatabaseManager(
        user="test_user", password="test_password", host="test_host", port=5432, db_name="test_db"
    )
    expected_url = "postgresql+psycopg2://test_user:test_password@test_host:5432/test_db"
    assert db_manager.connection_url == expected_url


def test_database_manager_init_with_connection_url():
    connection_url = "postgresql+psycopg2://test_user:test_password@test_host:5432/test_db"
    db_manager = DatabaseManager(connection_url=connection_url)
    assert db_manager.connection_url == connection_url


def test_database_manager_init_with_default_url():
    db_manager = DatabaseManager()
    assert db_manager.connection_url == DEFAULT_POSTGRES_URL


@patch("cogstack_model_gateway.common.db.create_engine")
def test_database_manager_engine_creation(mock_create_engine: MagicMock):
    _ = DatabaseManager()
    mock_create_engine.assert_called_once_with(DEFAULT_POSTGRES_URL)


@patch("cogstack_model_gateway.common.db.SQLModel.metadata.create_all")
def test_database_manager_init_db(mock_create_all: MagicMock):
    db_manager = DatabaseManager()
    db_manager.init_db()
    mock_create_all.assert_called_once_with(db_manager.engine)


def test_database_manager_get_session():
    db_manager = DatabaseManager()
    with patch.object(db_manager, "engine", create=True):
        with db_manager.get_session() as session:
            assert isinstance(session, Session)
