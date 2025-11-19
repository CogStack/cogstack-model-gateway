"""Extend Task with request info and timestamps

Revision ID: 1859866dbf86
Revises: 000000000000
Create Date: 2025-06-19 17:29:59.695124

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1859866dbf86"
down_revision: str | Sequence[str] | None = "000000000000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("task", sa.Column("model", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column("task", sa.Column("type", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column("task", sa.Column("source", sqlmodel.sql.sqltypes.AutoString(), nullable=True))
    op.add_column(
        "task", sa.Column("created_at", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.add_column(
        "task", sa.Column("started_at", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )
    op.add_column(
        "task", sa.Column("finished_at", sqlmodel.sql.sqltypes.AutoString(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("task", "finished_at")
    op.drop_column("task", "started_at")
    op.drop_column("task", "created_at")
    op.drop_column("task", "source")
    op.drop_column("task", "type")
    op.drop_column("task", "model")
