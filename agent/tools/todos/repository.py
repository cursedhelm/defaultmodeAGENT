from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Principal, TodoGrant, TodoItem, TodoList


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


class TodoRepository:
    """Thread-safe transactional todo storage shared across bot runtimes."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS todo_lists (
            owner_key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            is_bot INTEGER NOT NULL DEFAULT 0,
            goal TEXT,
            visibility TEXT NOT NULL DEFAULT 'private',
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS todo_items (
            id TEXT PRIMARY KEY,
            owner_key TEXT NOT NULL REFERENCES todo_lists(owner_key) ON DELETE CASCADE,
            text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            rank REAL,
            position INTEGER NOT NULL DEFAULT 0,
            UNIQUE(owner_key, normalized_text)
        );
        CREATE INDEX IF NOT EXISTS idx_todo_items_owner_position
            ON todo_items(owner_key, position, created_at);
        CREATE TABLE IF NOT EXISTS todo_grants (
            owner_key TEXT NOT NULL REFERENCES todo_lists(owner_key) ON DELETE CASCADE,
            principal_key TEXT NOT NULL,
            permission TEXT NOT NULL CHECK(permission IN ('view', 'edit')),
            PRIMARY KEY(owner_key, principal_key)
        );
        CREATE TABLE IF NOT EXISTS todo_embeddings (
            cache_key TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS todo_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_key TEXT NOT NULL,
            agent_key TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            action TEXT NOT NULL,
            source TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS todo_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(schema)
            audit_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(todo_audit)").fetchall()
            }
            if "agent_key" not in audit_columns:
                try:
                    connection.execute(
                        "ALTER TABLE todo_audit ADD COLUMN agent_key TEXT NOT NULL DEFAULT 'legacy:unknown'"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).casefold():
                        raise

    def ensure_list(self, principal: Principal) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO todo_lists(owner_key, display_name, is_bot, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(owner_key) DO UPDATE SET
                     display_name=excluded.display_name, is_bot=excluded.is_bot""",
                (principal.key, principal.display_name, int(principal.is_bot), _now()),
            )

    def get_list(self, principal: Principal) -> TodoList:
        self.ensure_list(principal)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT * FROM todo_lists WHERE owner_key=?", (principal.key,)
            ).fetchone()
            items = connection.execute(
                "SELECT * FROM todo_items WHERE owner_key=? ORDER BY position, created_at, id",
                (principal.key,),
            ).fetchall()
        owner = Principal(
            key=row["owner_key"], display_name=row["display_name"], is_bot=bool(row["is_bot"])
        )
        return TodoList(
            owner=owner,
            goal=row["goal"],
            visibility=row["visibility"],
            revision=row["revision"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
            items=[
                TodoItem(
                    id=item["id"], text=item["text"], created_by=item["created_by"],
                    created_at=datetime.fromisoformat(item["created_at"]),
                    rank=item["rank"], position=item["position"],
                )
                for item in items
            ],
        )

    def set_goal(self, principal: Principal, goal: str | None) -> None:
        self.ensure_list(principal)
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE todo_lists SET goal=?, revision=revision+1, updated_at=? WHERE owner_key=?",
                (goal, _now(), principal.key),
            )

    def add_item(self, principal: Principal, text: str, created_by: str) -> TodoItem:
        self.ensure_list(principal)
        item = TodoItem(
            id=str(uuid.uuid4()), text=" ".join(text.split()), created_by=created_by,
            created_at=datetime.now(timezone.utc), position=2_147_483_647,
        )
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """INSERT INTO todo_items
                       (id, owner_key, text, normalized_text, created_by, created_at, rank, position)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
                    (item.id, principal.key, item.text, normalize_text(item.text), created_by,
                     item.created_at.isoformat(), item.position),
                )
                connection.execute(
                    "UPDATE todo_lists SET revision=revision+1, updated_at=? WHERE owner_key=?",
                    (_now(), principal.key),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("that item is already on the todo list") from exc
        return item

    def delete_items(self, owner_key: str, item_ids: Iterable[str]) -> int:
        ids = list(item_ids)
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM todo_items WHERE owner_key=? AND id IN ({placeholders})",
                (owner_key, *ids),
            )
            if cursor.rowcount:
                connection.execute(
                    "UPDATE todo_lists SET revision=revision+1, updated_at=? WHERE owner_key=?",
                    (_now(), owner_key),
                )
            return cursor.rowcount

    def clear(self, owner_key: str) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM todo_items WHERE owner_key=?", (owner_key,))
            connection.execute(
                "UPDATE todo_lists SET goal=NULL, revision=revision+1, updated_at=? WHERE owner_key=?",
                (_now(), owner_key),
            )
            return cursor.rowcount

    def apply_ranking(
        self,
        owner_key: str,
        ordered: list[tuple[str, float | None]],
        max_items: int,
        *,
        expected_revision: int,
    ) -> bool:
        keep = ordered[:max_items]
        keep_ids = [item_id for item_id, _ in keep]
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT revision FROM todo_lists WHERE owner_key=?", (owner_key,)
            ).fetchone()
            if row is None or row["revision"] != expected_revision:
                connection.rollback()
                return False
            for position, (item_id, rank) in enumerate(keep):
                connection.execute(
                    "UPDATE todo_items SET position=?, rank=? WHERE owner_key=? AND id=?",
                    (position, rank, owner_key, item_id),
                )
            if keep_ids:
                placeholders = ",".join("?" for _ in keep_ids)
                connection.execute(
                    f"DELETE FROM todo_items WHERE owner_key=? AND id NOT IN ({placeholders})",
                    (owner_key, *keep_ids),
                )
            else:
                connection.execute("DELETE FROM todo_items WHERE owner_key=?", (owner_key,))
            connection.execute(
                "UPDATE todo_lists SET revision=revision+1, updated_at=? WHERE owner_key=?",
                (_now(), owner_key),
            )
            return True

    def set_grant(self, owner_key: str, principal_key: str, permission: str | None) -> None:
        with self._lock, self._connect() as connection:
            if permission is None:
                connection.execute(
                    "DELETE FROM todo_grants WHERE owner_key=? AND principal_key=?",
                    (owner_key, principal_key),
                )
            else:
                TodoGrant(owner_key=owner_key, principal_key=principal_key, permission=permission)
                connection.execute(
                    """INSERT INTO todo_grants(owner_key, principal_key, permission) VALUES (?, ?, ?)
                       ON CONFLICT(owner_key, principal_key) DO UPDATE SET permission=excluded.permission""",
                    (owner_key, principal_key, permission),
                )
            shared = connection.execute(
                "SELECT 1 FROM todo_grants WHERE owner_key=? LIMIT 1", (owner_key,)
            ).fetchone()
            connection.execute(
                "UPDATE todo_lists SET visibility=?, revision=revision+1, updated_at=? WHERE owner_key=?",
                ("shared" if shared else "private", _now(), owner_key),
            )

    def has_grant(self, owner_key: str, principal_key: str, permissions: set[str]) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT permission FROM todo_grants WHERE owner_key=? AND principal_key=?",
                (owner_key, principal_key),
            ).fetchone()
        return bool(row and row["permission"] in permissions)

    @staticmethod
    def embedding_key(provider: str, model: str, text: str, version: str = "v1") -> tuple[str, str]:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        raw = f"{version}\0{provider}\0{model}\0{text_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest(), text_hash

    def get_embedding(
        self, provider: str, model: str, text: str, *, version: str = "v1"
    ) -> list[float] | None:
        cache_key, _ = self.embedding_key(provider, model, text, version)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT dimensions, vector_json FROM todo_embeddings WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        vector = json.loads(row["vector_json"])
        if len(vector) != row["dimensions"]:
            return None
        return [float(value) for value in vector]

    def put_embedding(
        self,
        provider: str,
        model: str,
        text: str,
        vector: list[float],
        *,
        version: str = "v1",
    ) -> None:
        if not vector:
            raise ValueError("embedding vector is empty")
        cache_key, text_hash = self.embedding_key(provider, model, text, version)
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO todo_embeddings
                   (cache_key, provider, model, dimensions, text_hash, vector_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (cache_key, provider, model, len(vector), text_hash,
                 json.dumps(vector, separators=(",", ":")), _now()),
            )

    def audit(
        self,
        actor_key: str,
        agent_key: str,
        owner_key: str,
        action: str,
        source: str,
        detail: dict,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT INTO todo_audit(actor_key, agent_key, owner_key, action, source, detail_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (actor_key, agent_key, owner_key, action, source,
                 json.dumps(detail, ensure_ascii=False, default=str), _now()),
            )

    def get_metadata(self, key: str) -> str | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT value FROM todo_metadata WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO todo_metadata(key, value) VALUES (?, ?)", (key, value)
            )

    def claim_metadata(self, key: str, value: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO todo_metadata(key, value) VALUES (?, ?)", (key, value)
            )
            return cursor.rowcount == 1

    def delete_metadata(self, key: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM todo_metadata WHERE key=?", (key,))

    def get_or_create_runtime_profile(self, requested: dict) -> dict:
        """The first bot establishes shared ranking semantics for every bot process."""
        key = "todo:runtime_profile:v1"
        encoded = json.dumps(requested, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT value FROM todo_metadata WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO todo_metadata(key, value) VALUES (?, ?)", (key, encoded)
                )
                return requested
            try:
                profile = json.loads(row["value"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("stored todo runtime profile is invalid") from exc
            if not isinstance(profile, dict):
                raise RuntimeError("stored todo runtime profile is invalid")
            return profile
