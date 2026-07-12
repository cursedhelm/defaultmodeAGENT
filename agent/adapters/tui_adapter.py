"""TUI implementation of PlatformAdapter.

Delivers agent responses via a caller-supplied callback (no Discord, no network).
All platform-specific no-ops (typing indicator, history, mention formatting) are
stubs — the real behaviour lives in the caller's RichLog widget.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable, Tuple

from pydantic import BaseModel, Field

from .base import NormalizedMessage, PlatformAdapter


class TUIAdapterPrompts(BaseModel):
    """Hardcoded prompt strings this adapter injects into LLM context."""
    context_header: str = Field(default="Current channel: TUI\n")


PROMPTS = TUIAdapterPrompts()


class TUIAdapter(PlatformAdapter):
    """
    Routes agent output to whatever callback ``on_send`` the caller supplies.

    Typical usage inside ChatPage::

        def _on_send(channel_id: str, text: str) -> None:
            log.write(text)

        adapter = TUIAdapter(_on_send)
    """

    def __init__(self, on_send: Callable[[str, str], None]):
        self._on_send = on_send

    async def send(self, channel_id: str, text: str) -> None:
        self._on_send(channel_id, text)

    @asynccontextmanager
    async def thinking(self, channel_id: str):
        yield  # no typing indicator in the TUI

    async def fetch_history(
        self, channel_id: str, limit: int, skip_id: str | None = None
    ) -> Tuple[list, dict]:
        return [], {}

    def format_context_header(self, msg: NormalizedMessage) -> str:
        return PROMPTS.context_header

    def format_response(self, response: str, msg: NormalizedMessage) -> str:
        return response

    def sanitize_content(self, content: str, msg: NormalizedMessage) -> str:
        return content
