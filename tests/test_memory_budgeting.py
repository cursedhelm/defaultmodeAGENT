import sys
import threading
from collections import defaultdict
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from context import fit_ranked_entries
from memory import UserMemoryIndex
from tokenizer import count_tokens


class StubLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass


def _memory_index_without_disk(tmp_path):
    index = UserMemoryIndex.__new__(UserMemoryIndex)
    index.cache_dir = str(tmp_path)
    index.max_tokens = 1  # Must no longer constrain lexical candidate retrieval.
    index.memories = ["needle " * 200, "needle compact result"]
    index.user_memories = defaultdict(list, {"user": [0, 1]})
    index.inverted_index = defaultdict(list, {"needle": [0] * 200 + [1]})
    index.stopwords = set()
    index._global_stops = set()
    index._user_stops = {}
    index._mut = threading.RLock()
    index._cache_mtime = 0.0
    index.logger = StubLogger()
    return index


def test_lexical_search_is_bounded_by_candidate_count_not_text_tokens(tmp_path):
    index = _memory_index_without_disk(tmp_path)

    results = index.search("needle", k=2, user_id="user", dedup_threshold=1.1)

    assert len(results) == 2
    assert {memory for memory, _ in results} == set(index.memories)


def test_prompt_budget_skips_oversized_entry_and_keeps_shorter_candidate():
    header = "{count} memories:\n"
    oversized = "oversized " * 200
    shorter = "short meaningful candidate\n"
    budget = count_tokens(header.format(count=1) + shorter)

    block = fit_ranked_entries(
        [oversized, shorter],
        header,
        max_tokens=budget,
    )

    assert oversized not in block
    assert shorter in block
    assert block.startswith("1 memories:")
    assert count_tokens(block) <= budget
