"""
Platform abstraction layer for defaultMODE.

New platform adapters implement PlatformAdapter and pass themselves into
agent_core.process_message / process_files alongside an AgentRuntime.
See HOOKS.md for the full contract and a minimal adapter skeleton.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


@dataclass
class NormalizedAttachment:
    """Platform-agnostic representation of a file attached to a message."""
    filename: str
    content_type: str   # e.g. "image/png", "text/plain", ""
    size: int           # bytes
    _fetch: Callable    # async () -> bytes

    async def read(self) -> bytes:
        return await self._fetch()


@dataclass
class NormalizedMessage:
    """Platform-agnostic representation of an inbound message."""
    id: str
    content: str
    author_id: str
    author_name: str
    channel_id: str
    channel_name: str
    guild_name: Optional[str]       # None for DMs / platforms without guilds
    is_dm: bool
    mentions_bot: bool
    attachments: List[NormalizedAttachment] = field(default_factory=list)
    reply_to: Optional[NormalizedMessage] = None
    raw: Any = None                 # original platform object — escape hatch


class PlatformAdapter(ABC):
    """
    Contract between the platform-agnostic agent core and a specific chat platform.

    Implement all abstract methods for each new platform.  The Discord
    implementation lives in adapters/discord_adapter.py.
    """

    # ------------------------------------------------------------------ #
    # Outbound                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def send(self, channel_id: str, text: str) -> None:
        """Send text to the given channel, splitting if the platform requires it."""
        ...

    @abstractmethod
    async def thinking(self, channel_id: str):
        """
        Async context manager that signals the platform the agent is working.

        Usage::

            async with adapter.thinking(channel_id):
                response = await runtime.call_api(...)

        On Discord this shows the typing indicator.  On other platforms it
        may be a no-op or a status update.
        """
        ...

    # ------------------------------------------------------------------ #
    # Inbound / history                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    async def fetch_history(
        self, channel_id: str, limit: int, skip_id: str | None = None
    ) -> Tuple[list, dict]:
        """
        Return ``(messages, reactions_map)`` in the format expected by
        ``context.process_history_dual``.

        ``skip_id`` is the ID of the triggering message and should be omitted
        from the returned list.
        """
        ...

    # ------------------------------------------------------------------ #
    # Formatting                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def format_context_header(self, msg: NormalizedMessage) -> str:
        """
        Return the platform-context line injected at the top of every LLM
        prompt, e.g.::

            "Current Discord server: MyServer, channel: #general\\n"
            "Current channel: CLI\\n"
        """
        ...

    @abstractmethod
    def format_response(self, response: str, msg: NormalizedMessage) -> str:
        """
        Post-process the LLM response for platform delivery.

        On Discord this converts ``@username`` strings to snowflake mentions.
        On other platforms it may be a no-op.
        """
        ...

    # ------------------------------------------------------------------ #
    # Content parsing                                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def sanitize_content(self, content: str, msg: NormalizedMessage) -> str:
        """
        Strip or convert platform-specific markup from user content before
        it reaches the LLM.  Discord strips ``<@id>`` mention tokens;
        other platforms may need different handling.
        """
        ...

    # ------------------------------------------------------------------ #
    # Optional — platform-specific command dispatch                       #
    # ------------------------------------------------------------------ #

    async def invoke_embedded_commands(
        self, response: str, msg: NormalizedMessage
    ) -> None:
        """
        Scan the LLM response for whitelisted commands and invoke them.

        Default is a no-op.  The Discord adapter overrides this to call
        ``discord_bot.invoke_embedded_commands``.
        """
        return
