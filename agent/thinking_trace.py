import re
from datetime import datetime
from typing import List, Tuple


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
    memory_string = (
        f"Thinking trace from interaction with @{user_name} ({storage_timestamp}):\n {trace}"
    )
    await memory_index.add_memory_async(user_id, memory_string)


async def store_thinking_traces(memory_index, user_id: str, user_name: str, traces: List[str]) -> None:
    for trace in traces:
        await store_thinking_trace(memory_index, user_id, user_name, trace)
