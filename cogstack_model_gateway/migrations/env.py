import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

sys.path.append(str(Path(__file__).parent.parent.parent))

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
from cogstack_model_gateway.common.tasks import SQLModel, Task  # noqa: E402, F401

target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

load_dotenv()

user = os.getenv("CMG_DB_USER")
password = os.getenv("CMG_DB_PASSWORD")
host = os.getenv("CMG_DB_HOST")
port = os.getenv("CMG_DB_PORT")
db_name = os.getenv("CMG_DB_NAME")

missing = [
    var
    for var, val in {
        "CMG_DB_USER": user,
        "CMG_DB_PASSWORD": password,
        "CMG_DB_HOST": host,
        "CMG_DB_PORT": port,
        "CMG_DB_NAME": db_name,
    }.items()
    if not val
]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

config.set_main_option(
    "sqlalchemy.url", f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
