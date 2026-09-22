from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import BookAsset, BookChunk, BookRecord, ReadingEvent, ReadingProgress


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BookshelfRepository:
    """Transactional, per-agent bookshelf metadata and reading state."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS books (
            id TEXT PRIMARY KEY,
            source_hash TEXT NOT NULL UNIQUE,
            source_filename TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_format TEXT NOT NULL CHECK(source_format IN ('pdf', 'epub')),
            markdown_path TEXT NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending', 'processing', 'ready', 'failed')),
            error TEXT,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS book_assets (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            relative_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            locator TEXT,
            source_name TEXT,
            UNIQUE(book_id, relative_path)
        );
        CREATE TABLE IF NOT EXISTS book_chunks (
            id TEXT PRIMARY KEY,
            book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            locator TEXT NOT NULL,
            heading TEXT,
            media_json TEXT NOT NULL DEFAULT '[]',
            UNIQUE(book_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_book_chunks_book_ordinal
            ON book_chunks(book_id, ordinal);
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT NOT NULL REFERENCES book_chunks(id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            version TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(chunk_id, provider, model, version)
        );
        CREATE TABLE IF NOT EXISTS reader_progress (
            reader_id TEXT NOT NULL,
            book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            next_ordinal INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL CHECK(status IN ('unread', 'reading', 'completed')),
            started_at TEXT,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            PRIMARY KEY(reader_id, book_id)
        );
        CREATE TABLE IF NOT EXISTS reader_current (
            reader_id TEXT PRIMARY KEY,
            book_id TEXT REFERENCES books(id) ON DELETE SET NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reading_events (
            reader_id TEXT NOT NULL,
            book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL REFERENCES book_chunks(id) ON DELETE CASCADE,
            raw_timestamp TEXT NOT NULL,
            reflection TEXT NOT NULL,
            memory_text TEXT NOT NULL,
            memory_synced INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY(reader_id, book_id, chunk_id)
        );
        """
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(schema)
            columns = {
                row["name"] for row in connection.execute(
                    "PRAGMA table_info(reading_events)"
                ).fetchall()
            }
            if "memory_synced" not in columns:
                connection.execute(
                    "ALTER TABLE reading_events ADD COLUMN memory_synced INTEGER NOT NULL DEFAULT 0"
                )

    @staticmethod
    def _book(row: sqlite3.Row | None) -> BookRecord | None:
        if row is None:
            return None
        return BookRecord(
            id=row["id"], source_hash=row["source_hash"],
            source_filename=row["source_filename"], source_path=row["source_path"],
            source_format=row["source_format"], markdown_path=row["markdown_path"],
            title=row["title"], author=row["author"], status=row["status"],
            error=row["error"], chunk_count=row["chunk_count"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _chunk(row: sqlite3.Row) -> BookChunk:
        return BookChunk(
            id=row["id"], book_id=row["book_id"], ordinal=row["ordinal"],
            text=row["text"], locator=row["locator"], heading=row["heading"],
            media_paths=json.loads(row["media_json"] or "[]"),
        )

    @staticmethod
    def _progress(row: sqlite3.Row | None) -> ReadingProgress | None:
        if row is None:
            return None
        return ReadingProgress(
            reader_id=row["reader_id"], book_id=row["book_id"],
            next_ordinal=row["next_ordinal"], status=row["status"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    def register_book(self, book: BookRecord) -> BookRecord:
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM books WHERE source_hash=?", (book.source_hash,)
            ).fetchone()
            if existing:
                return self._book(existing)
            connection.execute(
                """INSERT INTO books
                   (id, source_hash, source_filename, source_path, source_format,
                    markdown_path, title, author, status, error, chunk_count,
                    metadata_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (book.id, book.source_hash, book.source_filename, book.source_path,
                 book.source_format, book.markdown_path, book.title, book.author,
                 book.status, book.error, book.chunk_count,
                 json.dumps(book.metadata, ensure_ascii=False),
                 book.created_at.isoformat(), book.updated_at.isoformat()),
            )
        return book

    def get_book(self, book_id: str) -> BookRecord | None:
        with self._lock, self._connect() as connection:
            return self._book(connection.execute(
                "SELECT * FROM books WHERE id=?", (book_id,)
            ).fetchone())

    def books(self, statuses: Iterable[str] | None = None) -> list[BookRecord]:
        query = "SELECT * FROM books"
        args: tuple = ()
        values = list(statuses or [])
        if values:
            query += f" WHERE status IN ({','.join('?' for _ in values)})"
            args = tuple(values)
        query += " ORDER BY title COLLATE NOCASE, created_at"
        with self._lock, self._connect() as connection:
            return [self._book(row) for row in connection.execute(query, args).fetchall()]

    def set_book_status(self, book_id: str, status: str, error: str | None = None) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE books SET status=?, error=?, updated_at=? WHERE id=?",
                (status, error, _now(), book_id),
            )

    def recover_interrupted_ingestion(self, stale_seconds: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)).isoformat()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE books SET status='pending', error='recovered interrupted ingestion',
                   updated_at=? WHERE status='processing' AND updated_at < ?""",
                (_now(), cutoff),
            )
            return cursor.rowcount

    def finish_ingestion(
        self,
        book: BookRecord,
        chunks: list[BookChunk],
        assets: list[BookAsset],
        embedding_profile: tuple[str, str, str] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError("one embedding is required for every chunk")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM book_assets WHERE book_id=?", (book.id,))
            connection.execute("DELETE FROM book_chunks WHERE book_id=?", (book.id,))
            connection.executemany(
                """INSERT INTO book_assets
                   (id, book_id, ordinal, relative_path, media_type, locator, source_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(a.id, a.book_id, a.ordinal, a.relative_path, a.media_type,
                  a.locator, a.source_name) for a in assets],
            )
            connection.executemany(
                """INSERT INTO book_chunks
                   (id, book_id, ordinal, text, locator, heading, media_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(c.id, c.book_id, c.ordinal, c.text, c.locator, c.heading,
                  json.dumps(c.media_paths, ensure_ascii=False)) for c in chunks],
            )
            if embedding_profile and embeddings is not None:
                provider, model, version = embedding_profile
                connection.executemany(
                    """INSERT INTO chunk_embeddings
                       (chunk_id, provider, model, version, dimensions, vector_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [(chunk.id, provider, model, version, len(vector),
                      json.dumps(vector), _now())
                     for chunk, vector in zip(chunks, embeddings)],
                )
            connection.execute(
                """UPDATE books SET title=?, author=?, markdown_path=?, status='ready',
                   error=NULL, chunk_count=?, metadata_json=?, updated_at=? WHERE id=?""",
                (book.title, book.author, book.markdown_path, len(chunks),
                 json.dumps(book.metadata, ensure_ascii=False), _now(), book.id),
            )

    def chunks_for_book(self, book_id: str) -> list[BookChunk]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM book_chunks WHERE book_id=? ORDER BY ordinal", (book_id,)
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def chunks_for_books(self, book_ids: list[str]) -> list[BookChunk]:
        if not book_ids:
            return []
        result: list[BookChunk] = []
        with self._lock, self._connect() as connection:
            for offset in range(0, len(book_ids), 800):
                batch = book_ids[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT * FROM book_chunks WHERE book_id IN ({placeholders}) "
                    "ORDER BY book_id, ordinal", tuple(batch),
                ).fetchall()
                result.extend(self._chunk(row) for row in rows)
        return result

    def get_chunk(self, book_id: str, ordinal: int) -> BookChunk | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM book_chunks WHERE book_id=? AND ordinal=?",
                (book_id, ordinal),
            ).fetchone()
        return self._chunk(row) if row else None

    def put_embeddings(
        self, provider: str, model: str, version: str,
        values: list[tuple[str, list[float]]],
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.executemany(
                """INSERT OR REPLACE INTO chunk_embeddings
                   (chunk_id, provider, model, version, dimensions, vector_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(chunk_id, provider, model, version, len(vector),
                  json.dumps(vector), _now()) for chunk_id, vector in values],
            )

    def upgrade_chunks_in_place(
        self,
        book: BookRecord,
        chunks: list[BookChunk],
        embedding_profile: tuple[str, str, str],
        embeddings: list[list[float]],
    ) -> None:
        """Replace derived chunk text/vectors without disturbing reader state."""

        if len(chunks) != len(embeddings):
            raise ValueError("one embedding is required for every upgraded chunk")
        provider, model, version = embedding_profile
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing_ids = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM book_chunks WHERE book_id=?", (book.id,)
                ).fetchall()
            }
            if existing_ids != {chunk.id for chunk in chunks}:
                raise ValueError("in-place EPUB upgrade cannot change stable chunk IDs")
            connection.executemany(
                """UPDATE book_chunks SET text=?, locator=?, heading=?, media_json=?
                   WHERE id=? AND book_id=?""",
                [(chunk.text, chunk.locator, chunk.heading,
                  json.dumps(chunk.media_paths, ensure_ascii=False), chunk.id, book.id)
                 for chunk in chunks],
            )
            connection.executemany(
                """INSERT OR REPLACE INTO chunk_embeddings
                   (chunk_id, provider, model, version, dimensions, vector_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(chunk.id, provider, model, version, len(vector),
                  json.dumps(vector), _now())
                 for chunk, vector in zip(chunks, embeddings)],
            )
            connection.execute(
                """UPDATE books SET metadata_json=?, updated_at=? WHERE id=?""",
                (json.dumps(book.metadata, ensure_ascii=False), _now(), book.id),
            )

    def get_embeddings(
        self, chunk_ids: list[str], provider: str, model: str, version: str,
    ) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        result: dict[str, list[float]] = {}
        with self._lock, self._connect() as connection:
            for offset in range(0, len(chunk_ids), 800):
                batch = chunk_ids[offset:offset + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""SELECT chunk_id, vector_json FROM chunk_embeddings
                        WHERE chunk_id IN ({placeholders}) AND provider=? AND model=? AND version=?""",
                    (*batch, provider, model, version),
                ).fetchall()
                result.update({row["chunk_id"]: json.loads(row["vector_json"]) for row in rows})
        return result

    def start_book(self, reader_id: str, book_id: str) -> ReadingProgress:
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            book = connection.execute(
                "SELECT status FROM books WHERE id=?", (book_id,)
            ).fetchone()
            if book is None or book["status"] != "ready":
                raise ValueError("book is not ready")
            existing_progress = connection.execute(
                "SELECT status FROM reader_progress WHERE reader_id=? AND book_id=?",
                (reader_id, book_id),
            ).fetchone()
            if existing_progress and existing_progress["status"] == "completed":
                raise ValueError("book has already been completed by this reader")
            connection.execute(
                """INSERT INTO reader_progress
                   (reader_id, book_id, next_ordinal, status, started_at, updated_at)
                   VALUES (?, ?, 0, 'reading', ?, ?)
                   ON CONFLICT(reader_id, book_id) DO UPDATE SET
                     status='reading',
                     started_at=COALESCE(reader_progress.started_at, excluded.started_at),
                     updated_at=excluded.updated_at""",
                (reader_id, book_id, now, now),
            )
            connection.execute(
                """INSERT INTO reader_current(reader_id, book_id, updated_at)
                   VALUES (?, ?, ?) ON CONFLICT(reader_id) DO UPDATE SET
                   book_id=excluded.book_id, updated_at=excluded.updated_at""",
                (reader_id, book_id, now),
            )
            row = connection.execute(
                "SELECT * FROM reader_progress WHERE reader_id=? AND book_id=?",
                (reader_id, book_id),
            ).fetchone()
        return self._progress(row)

    def current(self, reader_id: str) -> tuple[BookRecord | None, ReadingProgress | None]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT b.* FROM reader_current c JOIN books b ON b.id=c.book_id
                   WHERE c.reader_id=?""", (reader_id,)
            ).fetchone()
            book = self._book(row)
            if book is None:
                return None, None
            progress_row = connection.execute(
                "SELECT * FROM reader_progress WHERE reader_id=? AND book_id=?",
                (reader_id, book.id),
            ).fetchone()
        return book, self._progress(progress_row)

    def eligible_prior_chunks(self, reader_id: str) -> list[BookChunk]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT c.* FROM book_chunks c
                   JOIN reader_progress p ON p.book_id=c.book_id
                   WHERE p.reader_id=? AND c.ordinal < p.next_ordinal
                   ORDER BY c.book_id, c.ordinal""", (reader_id,)
            ).fetchall()
        return [self._chunk(row) for row in rows]

    def has_event(self, reader_id: str, book_id: str, chunk_id: str) -> bool:
        with self._lock, self._connect() as connection:
            return connection.execute(
                """SELECT 1 FROM reading_events
                   WHERE reader_id=? AND book_id=? AND chunk_id=?""",
                (reader_id, book_id, chunk_id),
            ).fetchone() is not None

    def commit_reading(self, event: ReadingEvent, next_ordinal: int, chunk_count: int) -> None:
        completed = next_ordinal >= chunk_count
        now = _now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO reading_events
                   (reader_id, book_id, chunk_id, raw_timestamp, reflection, memory_text,
                    memory_synced, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event.reader_id, event.book_id, event.chunk_id, event.raw_timestamp,
                 event.reflection, event.memory_text, int(event.memory_synced),
                 event.created_at.isoformat()),
            )
            connection.execute(
                """UPDATE reader_progress SET next_ordinal=?, status=?, updated_at=?, completed_at=?
                   WHERE reader_id=? AND book_id=?""",
                (next_ordinal, "completed" if completed else "reading", now,
                 now if completed else None, event.reader_id, event.book_id),
            )
            if completed:
                connection.execute(
                    "UPDATE reader_current SET book_id=NULL, updated_at=? WHERE reader_id=?",
                    (now, event.reader_id),
                )

    def progress_for(self, reader_id: str, book_id: str) -> ReadingProgress | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reader_progress WHERE reader_id=? AND book_id=?",
                (reader_id, book_id),
            ).fetchone()
        return self._progress(row)

    def reading_events(self, reader_id: str, book_id: str | None = None) -> list[ReadingEvent]:
        query = "SELECT * FROM reading_events WHERE reader_id=?"
        args: tuple = (reader_id,)
        if book_id is not None:
            query += " AND book_id=?"
            args += (book_id,)
        query += " ORDER BY created_at"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return [ReadingEvent(
            reader_id=row["reader_id"], book_id=row["book_id"],
            chunk_id=row["chunk_id"], raw_timestamp=row["raw_timestamp"],
            reflection=row["reflection"], memory_text=row["memory_text"],
            memory_synced=bool(row["memory_synced"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        ) for row in rows]

    def unsynced_events(self, reader_id: str) -> list[ReadingEvent]:
        return [event for event in self.reading_events(reader_id) if not event.memory_synced]

    def mark_event_synced(self, reader_id: str, book_id: str, chunk_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE reading_events SET memory_synced=1
                   WHERE reader_id=? AND book_id=? AND chunk_id=?""",
                (reader_id, book_id, chunk_id),
            )
