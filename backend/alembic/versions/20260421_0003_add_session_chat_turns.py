"""Add persistent session chat turns table

Revision ID: 20260421_0003
Revises: 20260419_0002
Create Date: 2026-04-21 09:30:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260421_0003"
down_revision = "20260419_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_chat_turns",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_message", sa.Text(), nullable=False),
        sa.Column("insight_output", sa.JSON(), nullable=False),
        sa.Column("grounding_status", sa.String(length=30), nullable=False),
        sa.Column("faithfulness_corrected", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("used_fallback", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("retrieval_diagnostics", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_session_chat_turns_session_id"), "session_chat_turns", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_session_chat_turns_session_id"), table_name="session_chat_turns")
    op.drop_table("session_chat_turns")
