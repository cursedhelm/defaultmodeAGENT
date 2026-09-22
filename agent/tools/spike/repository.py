from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import SpikeActionEvent, SpikeExecution


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SpikeRepository:
    """SQLite outbox for spike actions and their memory reflections."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS spike_events (
            id TEXT PRIMARY KEY,
            source_user_id TEXT NOT NULL,
            source_memory_id INTEGER,
            source_memory_hash TEXT NOT NULL,
            source_memory TEXT NOT NULL,
            action TEXT,
            status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'failed')),
            target_id TEXT,
            target_label TEXT,
            query TEXT,
            executions_json TEXT NOT NULL,
            grounded INTEGER NOT NULL DEFAULT 0,
            release_recommended INTEGER NOT NULL DEFAULT 0,
            raw_timestamp TEXT NOT NULL,
            reflection TEXT,
            memory_text TEXT,
            memory_synced INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_spike_reflection_outbox
            ON spike_events(memory_synced, status, updated_at);
        CREATE INDEX IF NOT EXISTS idx_spike_source_attempts
            ON spike_events(source_user_id, source_memory_hash, status);
        """
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(schema)

    @staticmethod
    def _event(row: sqlite3.Row) -> SpikeActionEvent:
        return SpikeActionEvent(
            id=row["id"], source_user_id=row["source_user_id"],
            source_memory_id=row["source_memory_id"],
            source_memory_hash=row["source_memory_hash"],
            source_memory=row["source_memory"], action=row["action"],
            status=row["status"], target_id=row["target_id"],
            target_label=row["target_label"], query=row["query"],
            executions=[SpikeExecution.model_validate(value) for value in json.loads(row["executions_json"] or "[]")],
            grounded=bool(row["grounded"]),
            release_recommended=bool(row["release_recommended"]),
            raw_timestamp=row["raw_timestamp"], reflection=row["reflection"],
            memory_text=row["memory_text"], memory_synced=bool(row["memory_synced"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save(self, event: SpikeActionEvent) -> None:
        event.updated_at = datetime.now(timezone.utc)
        values = (
            event.id, event.source_user_id, event.source_memory_id,
            event.source_memory_hash, event.source_memory, event.action,
            event.status, event.target_id, event.target_label, event.query,
            json.dumps([item.model_dump(mode="json") for item in event.executions], ensure_ascii=False),
            int(event.grounded), int(event.release_recommended),
            event.raw_timestamp, event.reflection, event.memory_text,
            int(event.memory_synced), event.created_at.isoformat(),
            event.updated_at.isoformat(),
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO spike_events
                   (id, source_user_id, source_memory_id, source_memory_hash,
                    source_memory, action, status, target_id, target_label, query,
                    executions_json, grounded, release_recommended, raw_timestamp,
                    reflection, memory_text, memory_synced, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     action=excluded.action, status=excluded.status,
                     target_id=excluded.target_id, target_label=excluded.target_label,
                     query=excluded.query, executions_json=excluded.executions_json,
                     grounded=excluded.grounded,
                     release_recommended=excluded.release_recommended,
                     reflection=excluded.reflection, memory_text=excluded.memory_text,
                     memory_synced=excluded.memory_synced, updated_at=excluded.updated_at""",
                values,
            )

    def get(self, event_id: str) -> SpikeActionEvent | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM spike_events WHERE id=?", (event_id,)
            ).fetchone()
        return self._event(row) if row else None

    def latest(self) -> SpikeActionEvent | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM spike_events ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self._event(row) if row else None

    def completed_count(self, source_user_id: str, source_memory_hash: str) -> int:
        """Return completed SEEKING episodes that already spent this trace's energy."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT COUNT(*) AS count FROM spike_events
                   WHERE source_user_id=? AND source_memory_hash=?
                     AND status='completed'""",
                (source_user_id, source_memory_hash),
            ).fetchone()
        return int(row["count"]) if row else 0

    def awaiting_reflection(self) -> list[SpikeActionEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM spike_events
                   WHERE status != 'pending' AND reflection IS NULL
                   ORDER BY created_at"""
            ).fetchall()
        return [self._event(row) for row in rows]

    def pending(self) -> list[SpikeActionEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM spike_events WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return [self._event(row) for row in rows]

    def unsynced(self) -> list[SpikeActionEvent]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM spike_events
                   WHERE memory_text IS NOT NULL AND memory_synced=0
                   ORDER BY created_at"""
            ).fetchall()
        return [self._event(row) for row in rows]

    def mark_synced(self, event_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE spike_events SET memory_synced=1, updated_at=? WHERE id=?",
                (_now(), event_id),
            )
