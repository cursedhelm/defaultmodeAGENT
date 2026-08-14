import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import yaml


AGENT_DIR = Path(__file__).resolve().parents[1] / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from adapters.base import NormalizedMessage, PlatformAdapter
from agent_core import process_message
import api_client
from api_client import build_chat_messages


class StubAdapter(PlatformAdapter):
    async def send(self, channel_id: str, text: str) -> None:
        raise AssertionError("An empty model response should not be sent")

    @asynccontextmanager
    async def thinking(self, channel_id: str):
        yield

    async def fetch_history(self, channel_id: str, limit: int, skip_id=None):
        return [], {}

    def format_context_header(self, msg: NormalizedMessage) -> str:
        return "Current channel: #test\n"

    def format_response(self, response: str, msg: NormalizedMessage) -> str:
        return response

    def sanitize_content(self, content: str, msg: NormalizedMessage) -> str:
        return content


class StubMemoryIndex:
    user_memories = {}

    async def search_async(self, query, k, user_id=None):
        return []


class StubLogger:
    def debug(self, message):
        pass

    def info(self, message):
        pass

    def error(self, message):
        pass

    def log(self, data):
        pass


class StubRuntime:
    processing_enabled = True
    amygdala_response = 50
    agent_id = "agent"
    agent_name = "loop"
    spike_processor = None
    logger = StubLogger()

    def __init__(self):
        self.call = None

    async def call_api(self, **kwargs):
        self.call = kwargs
        return ""


def test_build_chat_messages_keeps_transport_fields_distinct():
    messages = build_chat_messages("persona", "explicit system note", "current turn")

    assert messages == [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "explicit system note"},
        {"role": "user", "content": "current turn"},
    ]


def test_foreground_turn_embeds_assembled_context_once_in_user_content():
    runtime = StubRuntime()
    msg = NormalizedMessage(
        id="message-1",
        content="hello",
        author_id="user-1",
        author_name="user",
        channel_id="channel-1",
        channel_name="test",
        guild_name="guild",
        is_dm=False,
        mentions_bot=True,
    )
    prompt_formats = {
        "introduction": "{assembled_context}<user_message>{user_message}</user_message>",
        "chat_with_memory": "{assembled_context}<user_message>{user_message}</user_message>",
    }
    system_prompts = {"default_chat": "loop {amygdala_response} {themes}"}

    with patch("agent_core._themes_memoized", return_value=""):
        asyncio.run(
            process_message(
                msg=msg,
                adapter=StubAdapter(),
                runtime=runtime,
                memory_index=StubMemoryIndex(),
                prompt_formats=prompt_formats,
                system_prompts=system_prompts,
            )
        )

    assembled_context = "Current channel: #test\n**Ongoing Channel Conversation:**\n\n<conversation>\n</conversation>\n"
    user_content = runtime.call["user_content"]
    messages = build_chat_messages(
        runtime.call["system_prompt"],
        runtime.call.get("supplemental_system_context", ""),
        user_content,
    )

    assert runtime.call.get("supplemental_system_context", "") == ""
    assert [message["role"] for message in messages] == ["system", "user"]
    assert user_content.count(assembled_context) == 1
    assert sum(message["content"].count(assembled_context) for message in messages) == 1


def test_loop_templates_use_explicit_assembled_context_name():
    prompt_path = AGENT_DIR / "prompts" / "loop" / "prompt_formats.yaml"
    prompt_formats = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))

    separately_contextualized = {
        "chat_with_memory",
        "introduction",
        "analyze_image",
        "analyze_audio",
        "analyze_video",
        "analyze_file",
        "analyze_combined",
        "repo_file_chat",
        "ask_repo",
    }

    for prompt_name in separately_contextualized:
        assert "{context}" not in prompt_formats[prompt_name]
        assert "{assembled_context}" in prompt_formats[prompt_name]


def test_no_persona_uses_the_ambiguous_context_placeholder():
    for prompt_path in (AGENT_DIR / "prompts").glob("*/prompt_formats.yaml"):
        assert "{context}" not in prompt_path.read_text(encoding="utf-8"), prompt_path


def test_api_jsonl_logging_is_batched_off_thread(tmp_path, monkeypatch):
    log_path = tmp_path / "api.jsonl"
    writer = api_client._BatchedJsonlWriter(
        queue_size=16,
        batch_size=8,
        flush_interval=0.02,
    )
    caller_thread = threading.get_ident()
    writer_threads = []
    original_write_batch = writer._write_batch

    def tracked_write_batch(batch):
        writer_threads.append(threading.get_ident())
        original_write_batch(batch)

    writer._write_batch = tracked_write_batch
    monkeypatch.setattr(api_client, "_api_log_writer", writer)

    for index in range(5):
        api_client.log_to_jsonl({"index": index}, str(log_path))
    writer.shutdown(timeout=2.0)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records == [{"index": index} for index in range(5)]
    assert writer_threads
    assert all(thread_id != caller_thread for thread_id in writer_threads)


def test_api_jsonl_logger_rejects_records_after_shutdown(tmp_path):
    writer = api_client._BatchedJsonlWriter(flush_interval=0.01)
    writer.shutdown(timeout=2.0)

    assert writer.submit({"late": True}, str(tmp_path / "late.jsonl")) is False
