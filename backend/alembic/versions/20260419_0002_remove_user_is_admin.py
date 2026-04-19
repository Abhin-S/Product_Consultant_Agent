"""Remove user role column

Revision ID: 20260419_0002
Revises: 20260418_0001
Create Date: 2026-04-19 22:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260419_0002"
down_revision = "20260418_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("users") as batch_op:
            batch_op.drop_column("is_admin")
    else:
        op.drop_column("users", "is_admin")


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("users") as batch_op:
            batch_op.add_column(sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()))
    else:
        op.add_column("users", sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()))
