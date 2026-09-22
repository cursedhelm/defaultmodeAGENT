from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecentLogExport:
    """A bounded, complete-line view over the tail of a JSONL log."""

    payload: bytes
    entry_count: int
    source_bytes: int
    truncated: bool


def read_recent_jsonl(path: str | Path, max_bytes: int) -> RecentLogExport:
    """Read at most ``max_bytes`` from a JSONL tail without scanning the file.

    When the window starts inside an older record, that partial record is
    discarded so every returned line remains valid JSONL. The source may be
    concurrently appended; a final non-newline-terminated record is retained
    because both project loggers write each record in one append operation.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    source = Path(path)
    source_bytes = source.stat().st_size
    if source_bytes == 0:
        return RecentLogExport(b"", 0, 0, False)

    start = max(0, source_bytes - max_bytes)
    with source.open("rb") as handle:
        handle.seek(start)
        payload = handle.read(max_bytes)

    if start:
        boundary = payload.find(b"\n")
        complete_tail = payload[boundary + 1:] if boundary >= 0 else b""
        if not complete_tail.strip():
            notice = json.dumps({
                "event": "oversized_log_entry",
                "source_bytes": source_bytes,
            }, ensure_ascii=False).encode("utf-8") + b"\n"
            if len(notice) > max_bytes:
                notice = b""
            return RecentLogExport(notice, int(bool(notice)), source_bytes, True)
        payload = complete_tail

    entry_count = sum(1 for line in payload.splitlines() if line.strip())
    return RecentLogExport(
        payload=payload,
        entry_count=entry_count,
        source_bytes=source_bytes,
        truncated=start > 0,
    )
