"""
Discord implementation of PlatformAdapter.

Owns all Discord-specific I/O:  message normalisation, chunked sending,
typing indicators, history fetching, mention formatting.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Optional, Tuple

import discord

from .base import NormalizedAttachment, NormalizedMessage, PlatformAdapter
from context import fetch_history_with_reactions
from discord_utils import format_discord_mentions, sanitize_mentions
from chunker import balance_wraps

if TYPE_CHECKING:
    from discord.ext import commands


class DiscordAdapter(PlatformAdapter):
    """
    Bridges Discord.py objects into the platform-agnostic agent core.

    Pass one instance of this into ``agent_core.process_message`` /
    ``process_files`` alongside an ``AgentRuntime``.
    """

    def __init__(self, bot: "commands.Bot"):
        self._bot = bot

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _get_channel(self, channel_id: str):
        cid = int(channel_id)
        ch = self._bot.get_channel(cid)
        if ch is None:
            ch = await self._bot.fetch_channel(cid)
        return ch

    # ------------------------------------------------------------------ #
    # Normalisation — discord.Message → NormalizedMessage                 #
    # ------------------------------------------------------------------ #

    async def normalize(
        self,
        message: discord.Message,
        is_command: bool = False,
    ) -> NormalizedMessage:
        """
        Convert a raw Discord message into a NormalizedMessage.

        Handles bot-mention stripping, reply-chain resolution, and
        attachment wrapping.  Replaces ``extract_content_and_reply``.
        """
        is_dm = isinstance(message.channel, discord.DMChannel)
        channel_name = message.channel.name if hasattr(message.channel, "name") else "DM"
        guild_name = message.guild.name if message.guild else None

        # Strip bot mention from content
        if is_command:
            parts = message.content.split(maxsplit=1)
            content = parts[1] if len(parts) > 1 else ""
        elif message.guild and message.guild.me:
            content = (
                message.content
                .replace(f"<@!{message.guild.me.id}>", "")
                .replace(f"<@{message.guild.me.id}>", "")
                .strip()
            )
        else:
            content = message.content.strip()

        # Resolve reply chain
        reply_to: Optional[NormalizedMessage] = None
        if message.reference and not is_command:
            try:
                original = await message.channel.fetch_message(
                    message.reference.message_id
                )
                original_content = original.content.strip()
                for m in original.mentions:
                    original_content = (
                        original_content
                        .replace(f"<@{m.id}>", f"@{m.name}")
                        .replace(f"<@!{m.id}>", f"@{m.name}")
                    )
                for ch in original.channel_mentions:
                    original_content = original_content.replace(
                        f"<#{ch.id}>", f"#{ch.name}"
                    )

                reply_attachments = []
                for att in original.attachments:
                    ext = os.path.splitext(att.filename.lower())[1]
                    if (att.content_type and att.content_type.startswith("image/")) or ext in {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv"}:
                        reply_attachments.append(self._wrap_attachment(att))

                reply_to = NormalizedMessage(
                    id=str(original.id),
                    content=original_content,
                    author_id=str(original.author.id),
                    author_name=original.author.name,
                    channel_id=str(original.channel.id),
                    channel_name=channel_name,
                    guild_name=guild_name,
                    is_dm=is_dm,
                    mentions_bot=False,
                    attachments=reply_attachments,
                    raw=original,
                )

                # Augment content to give the LLM reply context (preserves
                # the same string the old extract_content_and_reply produced)
                if original_content:
                    content = (
                        f"[@{message.author.name} replying to "
                        f"@{original.author.name}'s message: {original_content}]"
                        f"\n\n@{message.author.name}: {content}"
                    )
            except (discord.NotFound, discord.Forbidden):
                pass

        # Wrap attachments
        attachments = [self._wrap_attachment(a) for a in message.attachments]

        mentions_bot = (
            is_dm
            or (message.guild and self._bot.user in message.mentions)
            or (
                message.guild
                and any(r in message.guild.me.roles for r in message.role_mentions)
            )
        )

        return NormalizedMessage(
            id=str(message.id),
            content=content,
            author_id=str(message.author.id),
            author_name=message.author.name,
            channel_id=str(message.channel.id),
            channel_name=channel_name,
            guild_name=guild_name,
            is_dm=is_dm,
            mentions_bot=mentions_bot,
            attachments=attachments,
            reply_to=reply_to,
            raw=message,
        )

    def _wrap_attachment(self, att: discord.Attachment) -> NormalizedAttachment:
        return NormalizedAttachment(
            filename=att.filename,
            content_type=att.content_type or "",
            size=att.size,
            _fetch=att.read,
        )

    # ------------------------------------------------------------------ #
    # PlatformAdapter — outbound                                           #
    # ------------------------------------------------------------------ #

    async def send(self, channel_id: str, text: str) -> None:
        """Send text to a Discord channel, chunking at 1800 chars."""
        channel = await self._get_channel(channel_id)
        await _send_chunked(channel, text, logger=self._bot.logger)

    @asynccontextmanager
    async def thinking(self, channel_id: str):
        """Show Discord typing indicator while the agent is working."""
        try:
            channel = await self._get_channel(channel_id)
        except Exception:
            yield
            return
        task = asyncio.create_task(_maintain_typing(channel))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------ #
    # PlatformAdapter — inbound / history                                  #
    # ------------------------------------------------------------------ #

    async def fetch_history(
        self, channel_id: str, limit: int, skip_id: str | None = None
    ) -> Tuple[list, dict]:
        channel = await self._get_channel(channel_id)
        return await fetch_history_with_reactions(
            channel, limit, skip_id=int(skip_id) if skip_id else None
        )

    # ------------------------------------------------------------------ #
    # PlatformAdapter — formatting                                         #
    # ------------------------------------------------------------------ #

    def format_context_header(self, msg: NormalizedMessage) -> str:
        if msg.is_dm:
            return "Current channel: Direct Message\n"
        return f"Current Discord server: {msg.guild_name}, channel: #{msg.channel_name}\n"

    def format_response(self, response: str, msg: NormalizedMessage) -> str:
        guild = msg.raw.guild if msg.raw else None
        return format_discord_mentions(
            response,
            guild,
            getattr(self._bot, "mentions_enabled", False),
            self._bot,
        )

    def sanitize_content(self, content: str, msg: NormalizedMessage) -> str:
        if not msg.raw:
            return content
        raw = msg.raw
        combined = (
            list(raw.mentions)
            + list(raw.channel_mentions)
            + list(raw.role_mentions)
        )
        return sanitize_mentions(content, combined)

    async def invoke_embedded_commands(
        self, response: str, msg: NormalizedMessage
    ) -> None:
        """Delegate to discord_bot.invoke_embedded_commands via the stored bot."""
        from discord_bot import invoke_embedded_commands as _invoke
        if msg.raw is not None:
            await _invoke(response, msg.raw, self._bot)


# ------------------------------------------------------------------ #
# Private helpers (used only by DiscordAdapter)                       #
# ------------------------------------------------------------------ #

async def _maintain_typing(channel) -> None:
    """Keep Discord typing indicator alive for up to 5 minutes."""
    try:
        async with channel.typing():
            await asyncio.sleep(300)
    except Exception:
        pass


async def _send_chunked(channel, text: str, max_length: int = 1800, logger=None) -> None:
    """
    Send a (possibly long) string to a Discord channel, splitting it into
    chunks that respect code blocks and XML-style tags.

    Extracted verbatim from discord_bot.send_long_message so that
    discord_bot.py can delegate to DiscordAdapter.send().
    """
    if not text:
        return

    segments = []
    lines = text.split("\n")
    current_segment: list[str] = []
    in_code_block = False
    tag_stack: list[str] = []

    for line in lines:
        if "```" in line:
            if not in_code_block:
                if current_segment:
                    segments.append(("\n".join(current_segment), False))
                    current_segment = []
                in_code_block = True
            else:
                in_code_block = False
                current_segment.append(line)
                segments.append(("\n".join(current_segment), True))
                current_segment = []
                continue
        if not in_code_block:
            opens = line.count("<")
            closes = line.count(">")
            if opens > closes:
                tag_stack.extend(["<"] * (opens - closes))
            elif closes > opens and tag_stack:
                tag_stack = tag_stack[: (opens - closes)]
            if tag_stack and not current_segment:
                if current_segment:
                    segments.append(("\n".join(current_segment), False))
                    current_segment = []
            elif (
                not tag_stack
                and current_segment
                and any("<" in s or ">" in s for s in current_segment)
            ):
                current_segment.append(line)
                segments.append(("\n".join(current_segment), True))
                current_segment = []
                continue
        current_segment.append(line)
    if current_segment:
        segments.append(("\n".join(current_segment), in_code_block or bool(tag_stack)))

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for content, is_wrapped in segments:
        if is_wrapped:
            if len(content) > max_length:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                if "\n" not in content:
                    remaining = content
                    while remaining:
                        chunk_size = max_length - 6
                        if remaining.startswith("```"):
                            chunk = remaining[:chunk_size] + "\n```"
                            remaining = "```\n" + remaining[chunk_size:] if remaining[chunk_size:] else ""
                        else:
                            chunk = remaining[:chunk_size]
                            remaining = remaining[chunk_size:]
                        chunks.append(chunk)
                else:
                    balanced = balance_wraps(content)
                    while balanced:
                        original_length = len(balanced)
                        split_point = balanced.rfind("\n", 0, max_length)
                        if split_point == -1:
                            split_point = max_length - 6
                        chunk = balanced[:split_point]
                        if "```" in chunk and chunk.count("```") % 2 != 0:
                            chunk += "\n```"
                        chunks.append(chunk)
                        balanced = balanced[split_point:].lstrip()
                        if len(balanced) >= original_length:
                            chunks.append(balanced[: max_length - 6])
                            balanced = balanced[max_length - 6:].lstrip()
                        if len(balanced) < 6:
                            if balanced:
                                chunks.append(balanced)
                            break
                        if "```" in chunk and chunk.endswith("```") and balanced:
                            balanced = "```\n" + balanced
            else:
                if current_length + len(content) + 1 > max_length:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [content]
                    current_length = len(content)
                else:
                    current_chunk.append(content)
                    current_length += len(content) + 1
        else:
            for line in content.split("\n"):
                while len(line) > max_length:
                    if current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                    chunks.append(line[:max_length])
                    line = line[max_length:]
                    current_length = 0
                if current_length + len(line) + 1 > max_length:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line) + 1
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    for chunk in chunks:
        if not chunk.strip():
            continue
        max_retries = 3
        retry_count = 0
        base_delay = 0.5
        while retry_count < max_retries:
            try:
                await channel.send(chunk.strip())
                await asyncio.sleep(0.1)
                break
            except discord.HTTPException as e:
                if e.status == 429:
                    retry_count += 1
                    if retry_count == max_retries:
                        if logger:
                            logger.error("Max retries reached for message chunk. Skipping.")
                        break
                    retry_after = getattr(e, "retry_after", base_delay * (2 ** retry_count))
                    if logger:
                        logger.warning(
                            f"Rate limited. Waiting {retry_after:.2f}s before retry "
                            f"{retry_count}/{max_retries}"
                        )
                    await asyncio.sleep(retry_after)
                else:
                    if logger:
                        logger.error(f"Error sending message chunk: {str(e)}")
                    break
