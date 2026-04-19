"""Initial schema

Revision ID: 20260418_0001
Revises:
Create Date: 2026-04-18 11:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260418_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "user_integrations",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=20), nullable=False),
        sa.Column("access_token_encrypted", sa.Text(), nullable=False),
        sa.Column("workspace_id", sa.String(length=255), nullable=True),
        sa.Column("database_id", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "provider", name="uq_user_integrations_user_provider"),
    )
    op.create_index(op.f("ix_user_integrations_user_id"), "user_integrations", ["user_id"], unique=False)

    op.create_table(
        "analysis_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("idea_text", sa.Text(), nullable=False),
        sa.Column("raw_output", sa.JSON(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("used_fallback", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_analysis_sessions_user_id"), "analysis_sessions", ["user_id"], unique=False)

    op.create_table(
        "evaluation_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("avg_similarity_score", sa.Float(), nullable=False),
        sa.Column("min_similarity_score", sa.Float(), nullable=False),
        sa.Column("max_similarity_score", sa.Float(), nullable=False),
        sa.Column("docs_above_threshold", sa.Integer(), nullable=False),
        sa.Column("total_docs_retrieved", sa.Integer(), nullable=False),
        sa.Column("context_total_tokens", sa.Integer(), nullable=False),
        sa.Column("context_local_ratio", sa.Float(), nullable=False),
        sa.Column("context_dynamic_ratio", sa.Float(), nullable=False),
        sa.Column("used_fallback", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("articles_fetched", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("articles_surviving", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("avg_fallback_relevance", sa.Float(), nullable=True),
        sa.Column("llm_latency_ms", sa.Float(), nullable=False),
        sa.Column("llm_retry_count", sa.Integer(), nullable=False),
        sa.Column("llm_validation_passed", sa.Boolean(), nullable=False),
        sa.Column("context_precision", sa.Float(), nullable=True),
        sa.Column("context_recall", sa.Float(), nullable=True),
        sa.Column("faithfulness", sa.Float(), nullable=True),
        sa.Column("answer_relevance", sa.Float(), nullable=True),
        sa.Column("ragas_eval_status", sa.String(length=20), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("retrieved_docs", sa.JSON(), nullable=False),
        sa.Column("generated_output", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id"),
    )

    op.create_table(
        "action_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("action_type", sa.String(length=30), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("priority", sa.String(length=20), nullable=False),
        sa.Column("target_provider", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["analysis_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_action_logs_session_id"), "action_logs", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_action_logs_session_id"), table_name="action_logs")
    op.drop_table("action_logs")
    op.drop_table("evaluation_logs")
    op.drop_index(op.f("ix_analysis_sessions_user_id"), table_name="analysis_sessions")
    op.drop_table("analysis_sessions")
    op.drop_index(op.f("ix_user_integrations_user_id"), table_name="user_integrations")
    op.drop_table("user_integrations")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")