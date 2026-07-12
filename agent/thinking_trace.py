import re
from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel, Field


class ThinkingTracePrompts(BaseModel):
    """Memory-string template for stored model thinking traces.

    Load-bearing prefix: it is how extracted reasoning identifies itself when
    it resurfaces in prompts, and the (HH:MM [DD/MM/YY]) timestamp is
    regex-parsed across the framework.
    """
    trace_memory: str = Field(default="Thinking trace from interaction with @{user_name} ({timestamp}):\n {trace}")


PROMPTS = ThinkingTracePrompts()


_THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def separate_thinking_traces(text: str | None) -> Tuple[str, List[str]]:
    """Return response text with <think> blocks removed plus extracted traces."""
    if not text:
        return text or "", []

    traces = [m.group(1).strip() for m in _THINK_RE.finditer(text) if m.group(1).strip()]
    visible = _THINK_RE.sub("", text)
    visible = re.sub(r"\n{3,}", "\n\n", visible).strip()
    return visible, traces


async def store_thinking_trace(memory_index, user_id: str, user_name: str, trace: str) -> None:
    """Store a model thinking trace as a private user memory."""
    trace = (trace or "").strip()
    if not trace:
        return

    storage_timestamp = datetime.now().strftime("%H:%M [%d/%m/%y]")
    memory_string = PROMPTS.trace_memory.format(
        user_name=user_name, timestamp=storage_timestamp, trace=trace,
    )
    await memory_index.add_memory_async(user_id, memory_string)


async def store_thinking_traces(memory_index, user_id: str, user_name: str, traces: List[str]) -> None:
    for trace in traces:
        await store_thinking_trace(memory_index, user_id, user_name, trace)
