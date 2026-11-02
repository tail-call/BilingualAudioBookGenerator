import enum
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobType(str, enum.Enum):
    COLLECTOR = "collector"
    PREPROCESSOR = "preprocessor"
    VOICER = "voicer"
    GLUE = "glue"


class Book(Base):
    __tablename__ = "books"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    pdf_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    pdf_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_language: Mapped[str] = mapped_column(String(50), default="English")
    target_language: Mapped[str] = mapped_column(String(50), default="Russian")

    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.PENDING, index=True
    )
    total_sentences: Mapped[int] = mapped_column(Integer, default=0)
    completed_sentences: Mapped[int] = mapped_column(Integer, default=0)
    output_file: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    sentences: Mapped[list[Sentence]] = relationship(
        "Sentence",
        back_populates="book",
        cascade="all, delete-orphan",
        order_by="Sentence.sequence_num",
    )
    jobs: Mapped[list[Job]] = relationship(
        "Job", back_populates="book", cascade="all, delete-orphan"
    )


class Sentence(Base):
    __tablename__ = "sentences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    book_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    sequence_num: Mapped[int] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    book: Mapped[Book] = relationship("Book", back_populates="sentences")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    book_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("books.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(
        String(20), default=JobStatus.PENDING, index=True
    )
    sequence_num: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    sentence_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )

    book: Mapped[Book] = relationship("Book", back_populates="jobs")
