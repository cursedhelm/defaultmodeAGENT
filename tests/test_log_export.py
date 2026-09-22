import json
import sys
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from log_export import read_recent_jsonl


def test_short_jsonl_export_is_complete(tmp_path):
    path = tmp_path / "bot.jsonl"
    path.write_bytes(b'{"index": 1}\n{"index": 2}\n')

    export = read_recent_jsonl(path, 1024)

    assert export.payload == path.read_bytes()
    assert export.entry_count == 2
    assert export.truncated is False


def test_large_jsonl_export_reads_complete_tail_records(tmp_path):
    path = tmp_path / "bot.jsonl"
    records = [json.dumps({"index": index, "value": "x" * 40}) for index in range(100)]
    path.write_text("\n".join(records) + "\n", encoding="utf-8")

    export = read_recent_jsonl(path, 300)
    decoded = [json.loads(line) for line in export.payload.splitlines()]

    assert export.truncated is True
    assert export.entry_count == len(decoded)
    assert decoded[-1]["index"] == 99
    assert decoded[0]["index"] > 0
    assert len(export.payload) <= 300


def test_oversized_latest_record_returns_valid_warning(tmp_path):
    path = tmp_path / "bot.jsonl"
    path.write_text(json.dumps({"value": "x" * 1000}) + "\n", encoding="utf-8")

    export = read_recent_jsonl(path, 100)
    warning = json.loads(export.payload)

    assert export.truncated is True
    assert warning["event"] == "oversized_log_entry"
