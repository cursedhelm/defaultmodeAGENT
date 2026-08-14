"""Read-only, UI-agnostic memory graph used by TUI and future clients.

The agent's memory pickle remains authoritative.  This module only snapshots
that schema, builds a sparse graph from surviving postings, and persists
disposable derived data under ``viz_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional
import hashlib
import json
import math
import os
import pickle
import re
import threading

import numpy as np

from tui.shared import (
    PATHS,
    STATE,
    VIZ_PIPELINE_VERSION,
    build_tfidf_vectors,
    load_memory_cache,
    tokenize,
)
from viz_live import (
    discover_live_bot_names,
    fetch_live_snapshot,
    validate_live_payload,
)


GRAPH_CACHE_VERSION = 1


@dataclass(frozen=True)
class ThemeSnapshot:
    """Theme values read from their existing caches without refreshing them."""

    global_themes: tuple[str, ...] = ()
    user_themes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    updated_at: Optional[float] = None

    def for_user(self, user_id: Optional[str]) -> tuple[str, ...]:
        if not user_id:
            return self.global_themes
        return tuple(dict.fromkeys(
            self.global_themes + self.user_themes.get(user_id, ())
        ))


@dataclass(frozen=True)
class GraphSearchHit:
    mid: int
    score: float
    text: str
    user_id: Optional[str]


def _read_pickle(path: Path, default: Any) -> Any:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return default


def load_theme_snapshot(bot_name: str) -> ThemeSnapshot:
    """Read existing theme files only; never invoke attention refresh logic."""
    root = PATHS.cache_dir / bot_name / "themes"
    global_themes = tuple(_read_pickle(root / "themes.pkl", []) or [])
    updated_at = None
    try:
        meta = json.loads((root / "themes.meta.json").read_text(encoding="utf-8"))
        updated_at = float(meta.get("updated_at_epoch", 0)) or None
    except Exception:
        pass

    users: dict[str, tuple[str, ...]] = {}
    users_root = root / "users"
    if users_root.exists():
        try:
            children = list(users_root.iterdir())
        except OSError:
            children = []
        for child in children:
            if child.is_dir():
                values = tuple(_read_pickle(child / "themes.pkl", []) or [])
                if values:
                    users[child.name] = values
    return ThemeSnapshot(global_themes, users, updated_at)


def _themes_from_live(bot_name: str, values: Any) -> ThemeSnapshot:
    """Overlay warm in-memory themes on the last read-only disk snapshot."""
    disk = load_theme_snapshot(bot_name)
    if not isinstance(values, dict):
        return disk
    live_global = tuple(values.get("global_themes", ()) or ())
    merged_users = dict(disk.user_themes)
    raw_users = values.get("user_themes", {})
    if isinstance(raw_users, dict):
        for user_id, themes in raw_users.items():
            if themes:
                merged_users[str(user_id)] = tuple(themes)
    return ThemeSnapshot(
        live_global or disk.global_themes,
        merged_users,
        disk.updated_at,
    )


def live_bot_names() -> list[str]:
    """Bot names advertised by Discord live hooks."""
    return discover_live_bot_names(PATHS.cache_dir)


def _disk_source_key(bot_name: str) -> Optional[str]:
    try:
        stat = PATHS.bot_memory(bot_name).stat()
        return f"disk:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return None


def _graph_fingerprint(bot_name: str, source_key: str) -> str:
    raw = (
        f"graph-v{GRAPH_CACHE_VERSION}:projection-v{VIZ_PIPELINE_VERSION}:"
        f"{bot_name}:{source_key}"
    )
    return hashlib.blake2b(raw.encode(), digest_size=10).hexdigest()


def _graph_paths(bot_name: str, source_key: str) -> tuple[Path, Path]:
    root = PATHS.cache_dir / bot_name / "viz_cache"
    fp = _graph_fingerprint(bot_name, source_key)
    return root / f"graph-{fp}.pkl", root / f"graph-{fp}.sparse.npz"


def _live_source_key(memory_ids: np.ndarray, sparse: Any, terms: Iterable[str]) -> str:
    """Content identity for a consistent live snapshot, including pruned edges."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(memory_ids.astype(np.int32, copy=False).tobytes())
    digest.update(sparse.indptr.astype(np.int64, copy=False).tobytes())
    digest.update(sparse.indices.astype(np.int32, copy=False).tobytes())
    digest.update(sparse.data.astype(np.float32, copy=False).tobytes())
    for term in terms:
        digest.update(term.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return f"live:{digest.hexdigest()}"


class VizGraph:
    """Immutable-enough graph snapshot with navigation and API-friendly hooks."""

    def __init__(
        self,
        *,
        bot_name: str,
        memory_ids: np.ndarray,
        texts: list[str],
        owners: list[Optional[str]],
        sparse: Any,
        terms: list[str],
        raw_magnitudes: np.ndarray,
        source_key: str,
        source_label: str,
        themes: Optional[ThemeSnapshot] = None,
        runtime_state: Optional[dict] = None,
        derived_cached: bool = False,
        stale: bool = False,
    ):
        self.bot_name = bot_name
        self.memory_ids = np.asarray(memory_ids, dtype=np.int32)
        self.texts = list(texts)
        self.owners = list(owners)
        self.sparse = sparse.tocsr().astype(np.float32)
        self.terms = list(terms)
        self.raw_magnitudes = np.asarray(raw_magnitudes, dtype=np.float32)
        self.source_key = source_key
        self.source_label = source_label
        self.themes = themes or ThemeSnapshot()
        self.runtime_state = dict(runtime_state or {})
        self.derived_cached = derived_cached
        self.stale = stale

        self._row_by_mid = {
            int(mid): row for row, mid in enumerate(self.memory_ids.tolist())
        }
        self._term_to_col = {term: col for col, term in enumerate(self.terms)}
        self._rows_by_user: dict[str, np.ndarray] = {}
        buckets: dict[str, list[int]] = {}
        for row, owner in enumerate(self.owners):
            if owner is not None:
                buckets.setdefault(str(owner), []).append(row)
        for owner, rows in buckets.items():
            self._rows_by_user[owner] = np.asarray(rows, dtype=np.int32)
        self._connection_cache: dict[tuple[int, int, Optional[str]], list] = {}
        self._theme_mid_cache: dict[Optional[str], set[int]] = {}

    @property
    def users(self) -> list[str]:
        return sorted(self._rows_by_user)

    def mids_for_user(self, user_id: Optional[str]) -> set[int]:
        if not user_id:
            return set(map(int, self.memory_ids.tolist()))
        rows = self._rows_by_user.get(str(user_id), np.empty(0, dtype=np.int32))
        return set(map(int, self.memory_ids[rows].tolist()))

    def row_for_mid(self, mid: int) -> Optional[int]:
        return self._row_by_mid.get(int(mid))

    def text_for_mid(self, mid: int) -> str:
        row = self.row_for_mid(mid)
        return self.texts[row] if row is not None else ""

    def owner_for_mid(self, mid: int) -> Optional[str]:
        row = self.row_for_mid(mid)
        return self.owners[row] if row is not None else None

    def _scope_rows(self, user_id: Optional[str]) -> np.ndarray:
        if not user_id:
            return np.arange(len(self.memory_ids), dtype=np.int32)
        return self._rows_by_user.get(str(user_id), np.empty(0, dtype=np.int32))

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        limit: int = 200,
    ) -> list[GraphSearchHit]:
        """Search projected terms first, then raw snapshot text as a fallback."""
        query = query.strip()
        if not query:
            return []
        match = re.fullmatch(r"#?(\d+)", query)
        if match:
            mid = int(match.group(1))
            row = self.row_for_mid(mid)
            if row is not None and (not user_id or self.owners[row] == user_id):
                return [GraphSearchHit(mid, 1.0, self.texts[row], self.owners[row])]

        scope_rows = self._scope_rows(user_id)
        if not len(scope_rows):
            return []
        query_lower = query.casefold()
        cols = [
            self._term_to_col[token]
            for token in dict.fromkeys(tokenize(query))
            if token in self._term_to_col
        ]
        scores = np.zeros(len(self.memory_ids), dtype=np.float32)
        if cols:
            scores = np.asarray(self.sparse[:, cols].sum(axis=1)).ravel()

        # Exact phrase/substring remains useful for terms intentionally absent
        # from the associative graph, but it never changes the graph itself.
        substring_rows = []
        if not cols:
            for row in scope_rows.tolist():
                if query_lower in self.texts[row].casefold():
                    scores[row] += 1.0
                    substring_rows.append(row)

        candidates = scope_rows[scores[scope_rows] > 0]
        if not len(candidates) and substring_rows:
            candidates = np.asarray(substring_rows, dtype=np.int32)
        ordered = candidates[np.argsort(scores[candidates])[::-1]][:limit]
        return [
            GraphSearchHit(
                int(self.memory_ids[row]),
                float(scores[row]),
                self.texts[row],
                self.owners[row],
            )
            for row in ordered.tolist()
        ]

    def connections(
        self,
        mid: int,
        *,
        top_k: int = 6,
        user_id: Optional[str] = None,
    ) -> list[tuple[int, float, list[str]]]:
        """Return cosine neighbors directly from the authoritative sparse graph."""
        cache_key = (int(mid), int(top_k), user_id)
        if cache_key in self._connection_cache:
            return list(self._connection_cache[cache_key])
        row = self.row_for_mid(mid)
        if row is None or self.sparse[row].nnz == 0:
            return []

        scores = np.asarray((self.sparse @ self.sparse[row].T).toarray()).ravel()
        scores[row] = 0.0
        allowed = self._scope_rows(user_id)
        if len(allowed) != len(self.memory_ids):
            mask = np.zeros(len(self.memory_ids), dtype=bool)
            mask[allowed] = True
            scores[~mask] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if not len(candidates):
            return []
        if len(candidates) > top_k:
            part = np.argpartition(scores[candidates], -top_k)[-top_k:]
            candidates = candidates[part]
        candidates = candidates[np.argsort(scores[candidates])[::-1]]

        source_row = self.sparse[row]
        source_weights = dict(zip(source_row.indices, source_row.data))
        result = []
        for other_row in candidates.tolist():
            other = self.sparse[other_row]
            shared = set(source_weights).intersection(other.indices.tolist())
            ranked = sorted(
                shared,
                key=lambda col: source_weights[col] * float(other[0, col]),
                reverse=True,
            )[:8]
            result.append((
                int(self.memory_ids[other_row]),
                float(scores[other_row]),
                [self.terms[col] for col in ranked],
            ))
        self._connection_cache[cache_key] = result
        return list(result)

    @staticmethod
    def _theme_words(theme: str) -> list[str]:
        return re.findall(r"[\w'-]+", theme.casefold())

    def themes_for_mid(
        self,
        mid: int,
        user_id: Optional[str] = None,
    ) -> list[str]:
        row = self.row_for_mid(mid)
        if row is None:
            return []
        text = self.texts[row].casefold()
        live_cols = set(self.sparse[row].indices.tolist())
        matches = []
        for theme in self.themes.for_user(user_id or self.owners[row]):
            if theme.casefold() in text:
                matches.append(theme)
                continue
            words = self._theme_words(theme)
            cols = [self._term_to_col[w] for w in words if w in self._term_to_col]
            threshold = max(1, math.ceil(len(cols) / 2))
            if cols and sum(col in live_cols for col in cols) >= threshold:
                matches.append(theme)
        return matches

    def themed_mids(self, user_id: Optional[str] = None) -> set[int]:
        if user_id in self._theme_mid_cache:
            return set(self._theme_mid_cache[user_id])
        words = {
            word
            for theme in self.themes.for_user(user_id)
            for word in self._theme_words(theme)
        }
        cols = [self._term_to_col[word] for word in words if word in self._term_to_col]
        if not cols:
            themed = set()
        else:
            rows = np.flatnonzero(np.asarray(
                self.sparse[:, cols].getnnz(axis=1)
            ).ravel())
            allowed = set(self._scope_rows(user_id).tolist())
            themed = {
                int(self.memory_ids[row]) for row in rows.tolist()
                if row in allowed
            }
        self._theme_mid_cache[user_id] = themed
        return set(themed)

    def node_payload(self, mid: int) -> dict:
        """Stable serializable hook suitable for a future web transport."""
        row = self.row_for_mid(mid)
        if row is None:
            return {}
        return {
            "id": int(mid),
            "user_id": self.owners[row],
            "text": self.texts[row],
            "strength": float(self.raw_magnitudes[row]),
            "themes": self.themes_for_mid(mid),
        }


def _owners_for(cache: dict, memory_ids: list[int]) -> list[Optional[str]]:
    owner_by_mid = {
        int(mid): str(user_id)
        for user_id, mids in cache.get("user_memories", {}).items()
        for mid in mids
    }
    return [owner_by_mid.get(mid) for mid in memory_ids]


def build_graph(
    bot_name: str,
    cache: dict,
    *,
    source_label: str,
    source_key: Optional[str] = None,
    runtime_state: Optional[dict] = None,
    themes: Optional[ThemeSnapshot] = None,
) -> VizGraph:
    memories = cache.get("memories", [])
    memory_ids = [mid for mid, text in enumerate(memories) if text is not None]
    sparse, terms, raw_magnitudes = build_tfidf_vectors(cache, memory_ids)
    ids_array = np.asarray(memory_ids, dtype=np.int32)
    if source_key is None:
        source_key = _live_source_key(ids_array, sparse, terms)
    return VizGraph(
        bot_name=bot_name,
        memory_ids=ids_array,
        texts=[memories[mid] for mid in memory_ids],
        owners=_owners_for(cache, memory_ids),
        sparse=sparse,
        terms=terms,
        raw_magnitudes=raw_magnitudes,
        source_key=source_key,
        source_label=source_label,
        themes=themes or load_theme_snapshot(bot_name),
        runtime_state=runtime_state,
    )


def save_graph_cache(graph: VizGraph) -> bool:
    """Persist disposable graph data; never writes the memory source pickle."""
    if not graph.source_key.startswith("disk:"):
        return False
    meta_path, sparse_path = _graph_paths(graph.bot_name, graph.source_key)
    try:
        from scipy.sparse import save_npz

        meta_path.parent.mkdir(parents=True, exist_ok=True)
        token = f"{os.getpid()}-{threading.get_ident()}"
        sparse_tmp = sparse_path.with_name(
            sparse_path.stem + f".{token}.tmp.npz"
        )
        meta_tmp = meta_path.with_name(meta_path.name + f".{token}.tmp")
        # This cache optimizes latency, not storage. Uncompressed sparse arrays
        # are materially faster to save/load and remain much smaller than the
        # canonical pickle because raw postings are not duplicated.
        save_npz(str(sparse_tmp), graph.sparse, compressed=False)
        with meta_tmp.open("wb") as fh:
            pickle.dump({
                "version": GRAPH_CACHE_VERSION,
                "bot_name": graph.bot_name,
                "source_key": graph.source_key,
                "memory_ids": graph.memory_ids,
                "texts": graph.texts,
                "owners": graph.owners,
                "terms": graph.terms,
                "raw_magnitudes": graph.raw_magnitudes,
            }, fh, protocol=5)
        os.replace(sparse_tmp, sparse_path)
        os.replace(meta_tmp, meta_path)
        # Only the latest disk-derived base graph is useful. Coordinates are
        # independently keyed, while stale base graphs can be very large.
        for pattern in ("graph-*.pkl", "graph-*.sparse.npz"):
            for old_path in meta_path.parent.glob(pattern):
                if old_path not in (meta_path, sparse_path):
                    try:
                        old_path.unlink()
                    except OSError:
                        pass
        return True
    except Exception:
        return False


def load_cached_graph(
    bot_name: str,
    *,
    allow_stale: bool = False,
) -> Optional[VizGraph]:
    """Load a derived graph using only a cheap stat, optionally as stale LKG."""
    source_key = _disk_source_key(bot_name)
    meta_path = sparse_path = None
    stale = False
    if source_key is not None:
        current_meta, current_sparse = _graph_paths(bot_name, source_key)
        if current_meta.exists() and current_sparse.exists():
            meta_path, sparse_path = current_meta, current_sparse
    if meta_path is None and allow_stale:
        root = PATHS.cache_dir / bot_name / "viz_cache"
        try:
            candidates = sorted(
                root.glob("graph-*.pkl"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
        except OSError:
            candidates = []
        for candidate in candidates:
            fp = candidate.stem.removeprefix("graph-")
            candidate_sparse = root / f"graph-{fp}.sparse.npz"
            if candidate_sparse.exists():
                meta_path, sparse_path = candidate, candidate_sparse
                stale = True
                break
    if meta_path is None or sparse_path is None:
        return None
    if not (meta_path.exists() and sparse_path.exists()):
        return None
    try:
        from scipy.sparse import load_npz

        with meta_path.open("rb") as fh:
            meta = pickle.load(fh)
        if (
            meta.get("version") != GRAPH_CACHE_VERSION
            or meta.get("bot_name") != bot_name
        ):
            return None
        cached_source_key = str(meta.get("source_key", ""))
        if not stale and cached_source_key != source_key:
            return None
        return VizGraph(
            bot_name=bot_name,
            memory_ids=meta["memory_ids"],
            texts=meta["texts"],
            owners=meta["owners"],
            sparse=load_npz(str(sparse_path)),
            terms=meta["terms"],
            raw_magnitudes=meta["raw_magnitudes"],
            source_key=cached_source_key,
            source_label="derived cache (stale)" if stale else "derived cache",
            themes=load_theme_snapshot(bot_name),
            derived_cached=True,
            stale=stale,
        )
    except Exception:
        return None


def resolve_graph(
    bot_name: str,
    *,
    force_source_reload: bool = False,
    persist: bool = True,
) -> VizGraph:
    """Resolve live RAM first, then derived graph cache, then disk checkpoint."""
    context = STATE.get_live_context(bot_name)
    if context is not None:
        try:
            payload = validate_live_payload(context.snapshot_payload())
            return build_graph(
                bot_name,
                payload["memory"],
                source_label="live Chat RAM",
                runtime_state=payload.get("runtime"),
                themes=_themes_from_live(bot_name, payload.get("themes")),
            )
        except Exception:
            # A context can disappear while a page is refreshing. Continue to
            # the Discord/derived/offline sources instead of blanking the Viz.
            pass

    payload = fetch_live_snapshot(PATHS.cache_dir, bot_name)
    if payload is not None:
        return build_graph(
            bot_name,
            payload["memory"],
            source_label="live Discord RAM",
            runtime_state=payload.get("runtime"),
            themes=_themes_from_live(bot_name, payload.get("themes")),
        )

    if not force_source_reload:
        cached = load_cached_graph(bot_name, allow_stale=True)
        if cached is not None:
            return cached

    cache = load_memory_cache(bot_name)
    if not cache:
        raise FileNotFoundError(f"no memory state for {bot_name}")
    source_key = _disk_source_key(bot_name) or "disk:missing"
    graph = build_graph(
        bot_name,
        cache,
        source_label="offline checkpoint",
        source_key=source_key,
        themes=load_theme_snapshot(bot_name),
    )
    if persist:
        save_graph_cache(graph)
    return graph
