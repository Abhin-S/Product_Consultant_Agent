import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    idea_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_output: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    used_fallback: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    user = relationship("User", back_populates="analysis_sessions")
    evaluation_log = relationship(
        "EvaluationLog", back_populates="session", cascade="all, delete-orphan", uselist=False
    )
    action_logs = relationship("ActionLog", back_populates="session", cascade="all, delete-orphan")


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    # Tier 1
    avg_similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    min_similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_similarity_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    docs_above_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_docs_retrieved: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_local_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    context_dynamic_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    used_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    articles_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    articles_surviving: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_fallback_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Tier 2
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevance: Mapped[float | None] = mapped_column(Float, nullable=True)
    ragas_eval_status: Mapped[str] = mapped_column(String(20), nullable=False, default="skipped")

    # Audit
    query: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_docs: Mapped[list] = mapped_column(JSON, nullable=False)
    generated_output: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    session = relationship("AnalysisSession", back_populates="evaluation_log")


class ActionLog(Base):
    __tablename__ = "action_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("analysis_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action_type: Mapped[str] = mapped_column(String(30), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[str] = mapped_column(String(20), nullable=False)
    target_provider: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    session = relationship("AnalysisSession", back_populates="action_logs")