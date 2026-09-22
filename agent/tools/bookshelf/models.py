from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BookshelfModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class BookAsset(BookshelfModel):
    id: str
    book_id: str
    ordinal: int = Field(ge=0)
    relative_path: str
    media_type: str
    locator: str | None = None
    source_name: str | None = None


class BookChunk(BookshelfModel):
    id: str
    book_id: str
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)
    locator: str
    heading: str | None = None
    media_paths: list[str] = Field(default_factory=list)


class BookRecord(BookshelfModel):
    id: str
    source_hash: str
    source_filename: str
    source_path: str
    source_format: Literal["pdf", "epub"]
    markdown_path: str
    title: str
    author: str | None = None
    status: Literal["pending", "processing", "ready", "failed"] = "pending"
    error: str | None = None
    chunk_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ReadingProgress(BookshelfModel):
    reader_id: str
    book_id: str
    next_ordinal: int = Field(default=0, ge=0)
    status: Literal["unread", "reading", "completed"] = "unread"
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None


class ReadingEvent(BookshelfModel):
    reader_id: str
    book_id: str
    chunk_id: str
    raw_timestamp: str
    reflection: str
    memory_text: str
    memory_synced: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class BookshelfStatus(BookshelfModel):
    reader_id: str
    current_book: BookRecord | None = None
    progress: ReadingProgress | None = None
    current_locator: str | None = None
    percent_complete: float = Field(default=0.0, ge=0.0, le=100.0)
    available_books: int = Field(default=0, ge=0)
    pending_books: int = Field(default=0, ge=0)
    failed_books: int = Field(default=0, ge=0)
