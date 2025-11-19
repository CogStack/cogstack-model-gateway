"""Initial schema

Revision ID: 000000000000
Revises:
Create Date: 2025-11-19 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "000000000000"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create initial task table."""
    op.create_table(
        "task",
        sa.Column("uuid", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "SCHEDULED", "RUNNING", "SUCCEEDED", "FAILED", name="status"),
            nullable=False,
        ),
        sa.Column("result", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("tracking_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("uuid"),
    )


def downgrade() -> None:
    """Drop initial task table."""
    op.drop_table("task")
