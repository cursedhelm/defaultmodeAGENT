"""Energy-bounded SEEKING for associations the DMN can no longer resolve.

Each episode exposes only eligible Discord surfaces, related users, scoped
memory search, and the agent's authorized background tools. Its action and
private reflection are persisted through a SQLite outbox before graph cleanup.
"""


import asyncio
import hashlib
import json
import os
import pickle
import threading
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, List, Dict, Set, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict, Field
import yaml
import re
from tools.chronpression import chronomic_filter
from chunker import truncate_middle, clean_response
from discord_utils import sanitize_mentions, format_discord_mentions
from attention import format_themes_for_prompt, get_current_themes
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
from bot_config import config as bot_config
from memory import AtomicSaver
from api_schema import ToolSpec
from tools.bundle import AgentToolBundle, merge_tool_bundles
from tools.bookshelf.toolset import build_bookshelf_tool_bundle
from tools.todos.models import Principal, TodoRequestContext
from tools.todos.toolset import build_todo_tool_bundle
from tools.spike.models import SpikeActionEvent, SpikeActionOutcome, SpikeExecution
from tools.spike.repository import SpikeRepository
from context import (
    fetch_history_with_reactions,
    process_history_dual,
    build_memory_context,
    build_conversation_context,
    rerank_if_enabled
)
import discord


class SpikePrompts(BaseModel):
    """Hardcoded prompt scaffolding for spike outreach.

    The tension ladder translates the match score into felt language injected
    as {tension_desc} into the YAML spike_engagement prompt. The silence
    vocabulary is the code half of a contract whose instruction half lives in
    the YAML — keep them in sync. Memory templates are load-bearing (prefix
    terms enter the inverted index; timestamps are regex-parsed).
    """
    # (upper score bound, description) — first bound the score is below wins
    tension_ladder: List[Tuple[float, str]] = Field(default=[
        (0.4, "distant, tenuous"),
        (0.5, "loosely connected"),
        (0.6, "resonant but uncertain"),
    ])
    tension_ceiling: str = Field(default="strongly drawn")
    # Responses recognised as the agent opting out of sending
    silence_prefix: str = Field(default="[SILENCE]")
    silence_tokens: Set[str] = Field(default={"", "none", "pass"})
    # Location strings injected as {location}
    location_channel: str = Field(default="#{channel_name} in {guild_name}")
    location_dm: str = Field(default="DM")
    # Memory-string templates
    outreach_memory: str = Field(default="spike reached {location} ({timestamp}):\norphan: {orphan}\nresponse: {response}")
    silence_memory: str = Field(default="spike considered {location} ({timestamp}):\norphan: {orphan}\n[chose silence]")
    reflection_memory: str = Field(default="Reflections on spike to {location} ({timestamp}):\n{thought}")


PROMPTS = SpikePrompts()


class SpikeToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SpikeSearchInput(SpikeToolInput):
    user_id: str
    query: str = Field(min_length=3, max_length=1000)


class SpikeSilenceInput(SpikeToolInput):
    reason: str = Field(min_length=1, max_length=1000)


@dataclass
class Surface:
    channel: discord.abc.Messageable
    last_engaged: datetime
    compressed: str = ""
    raw_conversation: str = ""
    score: float = 0.0

@dataclass
class ChannelMessageBuffer:
    """Pre-fetched message buffer for a channel."""
    channel_id: int
    messages: List[str]
    fetched_at: datetime

@dataclass
class SpikeEvent:
    orphaned_memory: str
    target: Surface
    surface_seed: str
    memories: List[Tuple[str, float]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SpikePromptState:
    location: str
    timestamp: str
    tension_desc: str
    orphan_memory: str
    memory_context: str
    conversation_context: str

class SpikeProcessor:
    def __init__(self, bot, memory_index, cache_path: str = None):
        self.bot = bot
        self.memory_index = memory_index
        self.config = bot_config.spike
        self.last_spike: datetime = datetime.min
        self.enabled: bool = bool(self.config.enabled)
        self.logger = bot.logger
        self.temporal_parser = TemporalParser()
        # Persistence for engagement log
        self.cache_path = cache_path or os.path.join('cache', getattr(bot, 'bot_id', 'default'), 'spike')
        self.engagement_log_path = os.path.join(self.cache_path, 'engagement_log.pkl')
        self.repository = SpikeRepository(
            Path(self.cache_path) / self.config.database_filename
        )
        self.action_prompt_formats = self._load_action_yaml("spike_action_prompt_formats.yaml")
        self.action_system_prompts = self._load_action_yaml("spike_action_system_prompts.yaml")
        for key in ("spike_action_selection", "spike_action_reflection"):
            if key in getattr(bot, "prompt_formats", {}):
                self.action_prompt_formats[key] = bot.prompt_formats[key]
            if key in getattr(bot, "system_prompts", {}):
                self.action_system_prompts[key] = bot.system_prompts[key]
        self._mut = threading.RLock()
        self.engagement_log: Dict[int, datetime] = self._load_engagement_log()
        self._saver = AtomicSaver(self.engagement_log_path, self._snapshot_engagement, debounce=1.0, logger=self.logger)

    @staticmethod
    def _load_action_yaml(filename: str) -> dict[str, str]:
        path = Path(__file__).resolve().parent / "prompts" / filename
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    async def initialize(self) -> None:
        """Replay durable reflection work left by an interrupted process."""
        pending = await asyncio.to_thread(self.repository.pending)
        for event in pending:
            event.status = "failed"
            event.executions.append(SpikeExecution(
                sequence=len(event.executions), kind=event.action or "silence",
                name="spike_interrupted", arguments={}, ok=False,
                error="SEEKING episode was interrupted before its result was committed",
            ))
            event.action = event.action or "silence"
            await asyncio.to_thread(self.repository.save, event)
        await self._recover_reflections()
        await self._sync_pending_reflections()

    def _load_engagement_log(self) -> Dict[int, datetime]:
        """Load engagement log from disk or return empty defaultdict."""
        if os.path.exists(self.engagement_log_path):
            try:
                with open(self.engagement_log_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"spike.engagement.load path={self.engagement_log_path} entries={len(data)}")
                print(f"\n[spike] loaded engagement log: {len(data)} entries from {self.engagement_log_path}")
                for cid, ts in sorted(data.items(), key=lambda x: x[1], reverse=True):
                    age = datetime.now() - ts
                    print(f"  channel_id={cid}  last_engaged={ts.strftime('%Y-%m-%d %H:%M:%S')}  ({int(age.total_seconds() // 3600)}h ago)")
                # Convert to defaultdict
                log = defaultdict(lambda: datetime.min)
                log.update(data)
                return log
            except Exception as e:
                self.logger.warning(f"spike.engagement.load.err path={self.engagement_log_path} msg={e}")
                print(f"[spike] ERROR loading engagement log: {e}")
        else:
            print(f"[spike] no engagement log found at {self.engagement_log_path} — starting fresh")
        return defaultdict(lambda: datetime.min)

    def _snapshot_engagement(self) -> dict:
        """Return a copy of engagement log for atomic save."""
        with self._mut:
            return dict(self.engagement_log)

    def log_engagement(self, channel_id: int):
        with self._mut:
            self.engagement_log[channel_id] = datetime.now()
        self._saver.request()

    def get_recent_surfaces(self) -> List[Surface]:
        now = datetime.now()
        cutoff = now - timedelta(hours=self.config.recency_window_hours)
        surfaces = []
        with self._mut:
            items = list(self.engagement_log.items())
        print(f"\n[spike] get_recent_surfaces: {len(items)} log entries, recency_window={self.config.recency_window_hours}h, cutoff={cutoff.strftime('%Y-%m-%d %H:%M:%S')}")
        for cid, ts in items:
            if ts < cutoff:
                print(f"  SKIP  channel_id={cid}  ts={ts.strftime('%Y-%m-%d %H:%M:%S')}  (older than cutoff)")
                continue
            ch = self.bot.get_channel(cid)
            if ch and isinstance(ch, (discord.TextChannel, discord.DMChannel)):
                surfaces.append(Surface(channel=ch, last_engaged=ts))
                print(f"  OK    channel_id={cid}  name=#{ch.name}  ts={ts.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"  MISS  channel_id={cid}  ts={ts.strftime('%Y-%m-%d %H:%M:%S')}  (bot.get_channel returned {ch!r})")
        surfaces.sort(key=lambda s: s.last_engaged, reverse=True)
        print(f"[spike] viable surfaces: {len(surfaces)}/{len(items)}")
        return surfaces[:self.config.max_surfaces]

    async def fetch_channel_messages(self, channel: discord.abc.Messageable, limit: int) -> list[str]:
        """Fetch messages from channel, returns list of formatted message strings."""
        msgs = []
        try:
            async for msg in channel.history(limit=limit):
                if msg.author == self.bot.user:
                    continue
                content = msg.content.strip()
                if not content:
                    continue
                mentions = list(msg.mentions) + list(msg.channel_mentions) + list(msg.role_mentions)
                sanitized = sanitize_mentions(content, mentions)
                msgs.append(f"@{msg.author.name}: {sanitized}")
        except (discord.Forbidden, discord.HTTPException) as e:
            self.logger.warning(f"spike.fetch.err channel={channel.id} msg={e}")
            return []
        msgs.reverse()
        return msgs

    async def prefetch_surfaces(self, surfaces: List[Surface]) -> Dict[int, ChannelMessageBuffer]:
        """Batch fetch messages for all surfaces at max_expansion count (one API call per channel)."""
        buffers: Dict[int, ChannelMessageBuffer] = {}

        async def fetch_one(surface: Surface) -> Tuple[int, ChannelMessageBuffer]:
            channel_id = surface.channel.id
            msgs = await self.fetch_channel_messages(surface.channel, self.config.max_expansion)
            return channel_id, ChannelMessageBuffer(
                channel_id=channel_id,
                messages=msgs,
                fetched_at=datetime.now()
            )

        # Fetch all channels in parallel
        tasks = [fetch_one(s) for s in surfaces]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"spike.prefetch.err msg={result}")
                continue
            channel_id, buffer = result
            buffers[channel_id] = buffer

        self.logger.info(f"spike.prefetch.ok channels={len(buffers)} max_n={self.config.max_expansion}")
        return buffers

    async def compress_surface_from_buffer(self, surface: Surface, buffer: ChannelMessageBuffer, n: int) -> str:
        """Compress surface using pre-fetched messages sliced to n (no API calls)."""
        # Slice messages to n (buffer contains max_expansion messages in chronological order)
        # We want the most recent n messages, which are at the end
        msgs = buffer.messages[-n:] if len(buffer.messages) >= n else buffer.messages
        if not msgs:
            return ""
        raw = "\n".join(msgs)
        try:
            compressed = await asyncio.to_thread(
                chronomic_filter,
                raw,
                compression=self.config.compression_ratio,
                fuzzy_strength=1.0
            )
            return compressed
        except Exception as e:
            self.logger.warning(f"spike.chronpress.err msg={e}")
            return truncate_middle(raw, max_tokens=500)

    async def compress_surface(self, surface: Surface, n: int) -> str:
        msgs = await self.fetch_channel_messages(surface.channel, n)
        if not msgs:
            return ""
        raw = "\n".join(msgs)
        try:
            compressed = await asyncio.to_thread(
                chronomic_filter,
                raw,
                compression=self.config.compression_ratio,
                fuzzy_strength=1.0
            )
            return compressed
        except Exception as e:
            self.logger.warning(f"spike.chronpress.err msg={e}")
            return truncate_middle(raw, max_tokens=500)

    def extract_memory_content(self, memory: str) -> str:
        """Strip metadata prefix from DMN-generated memories, return semantic content."""
        # pattern: "Reflections on ... (timestamp):\n<content>"
        if ':\n' in memory:
            return memory.split(':\n', 1)[1].strip()
        return memory

    async def score_match(self, orphaned: str, compressed: str) -> float:
        if not compressed:
            return 0.0

        # extract actual content from orphan metadata wrapper
        content = self.extract_memory_content(orphaned)

        clean_content = self.memory_index.clean_text(content)
        clean_ctx = self.memory_index.clean_text(compressed)

        if not clean_content or not clean_ctx:
            return 0.0

        # bm25-style scoring inline (avoids index mutation)
        content_terms = clean_content.split()
        ctx_terms = clean_ctx.split()
        ctx_counter = Counter(ctx_terms)
        doc_len = len(ctx_terms)
        avg_len = doc_len  # single doc
        k1, b = 1.2, 0.75

        score = 0.0
        for term in set(content_terms):
            if term not in ctx_counter:
                continue
            tf = ctx_counter[term]
            # idf approximation: term present = 1 doc, treat as meaningful
            idf = 1.0
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_len, 1)))
            score += idf * (numerator / denominator)

        # normalize by query length
        if content_terms:
            score /= len(set(content_terms))

        # theme resonance scoring
        theme_score = 0.0
        themes = get_current_themes(self.memory_index)
        if themes:
            combined = f"{content} {compressed}".lower()
            hits = sum(1 for t in themes if t.lower() in combined)
            theme_score = min(1.0, hits / max(1, len(themes) * 0.3))

        tw = self.config.theme_weight
        final = (1 - tw) * min(1.0, score) + tw * theme_score

        self.logger.debug(f"spike.score bm25={score:.3f} theme={theme_score:.3f} final={final:.3f}")
        return final

    async def find_targets(self, orphaned_memory: str) -> List[Surface]:
        surfaces = self.get_recent_surfaces()
        if not surfaces:
            self.logger.info("spike.no_surfaces")
            return []

        # Batch prefetch all channels at max_expansion (one API call per channel)
        buffers = await self.prefetch_surfaces(surfaces)

        n = self.config.context_n
        viable = []
        while n <= self.config.max_expansion:
            for surface in surfaces:
                buffer = buffers.get(surface.channel.id)
                if buffer:
                    surface.compressed = await self.compress_surface_from_buffer(surface, buffer, n)
                else:
                    surface.compressed = ""
                surface.score = await self.score_match(orphaned_memory, surface.compressed)
            viable = [s for s in surfaces if s.score >= self.config.match_threshold]
            if not viable:
                max_score = max((s.score for s in surfaces), default=0.0)
                self.logger.info(f"spike.no_viable n={n} max_score={max_score:.3f}")
                n += self.config.expansion_step
                continue
            if len(viable) == 1 or n >= self.config.max_expansion:
                break
            top_score = max(s.score for s in viable)
            ties = [s for s in viable if abs(s.score - top_score) < 0.05]
            if len(ties) == 1:
                break
            n += self.config.expansion_step
            self.logger.info(f"spike.expand n={n} ties={len(ties)}")
        if not viable:
            self.logger.log({
                'event': 'spike_no_target',
                'orphaned_memory': orphaned_memory[:300],
                'surfaces_evaluated': len(surfaces),
                'scores': {str(s.channel.id): round(s.score, 3) for s in surfaces},
                'threshold': self.config.match_threshold,
                'final_n': n,
            })
            return []
        viable.sort(key=lambda surface: surface.score, reverse=True)
        return viable

    async def find_target(self, orphaned_memory: str) -> Optional[SpikeEvent]:
        viable = await self.find_targets(orphaned_memory)
        if not viable:
            return None
        target = viable[0]
        self.logger.info(f"spike.target channel={target.channel.id} score={target.score:.3f}")
        self.logger.log({
            'event': 'spike_target_found',
            'orphaned_memory': orphaned_memory[:300],
            'target_channel': target.channel.id,
            'target_score': round(target.score, 3),
            'surfaces_evaluated': len(viable),
            'scores': {str(s.channel.id): round(s.score, 3) for s in viable},
            'viable_count': len(viable),
        })
        return SpikeEvent(
            orphaned_memory=orphaned_memory,
            target=target,
            surface_seed=target.compressed
        )

    def _natural_now(self) -> str:
        expression = self.temporal_parser.get_temporal_expression(datetime.now())
        return " ".join(
            value for value in (expression.base_expression, expression.time_context)
            if value
        )

    def _temporalize(self, text: str) -> str:
        timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
        return re.sub(
            timestamp_pattern,
            lambda match: "(" + self.temporal_parser.get_temporal_expression(
                datetime.strptime(
                    f"{match.group(1)}:{match.group(2)} {match.group(3)}",
                    "%H:%M %d/%m/%y",
                )
            ).base_expression + ")",
            text,
        )

    @staticmethod
    def _tool_schema(model: type[BaseModel]) -> dict[str, Any]:
        value = model.model_json_schema()
        value.pop("title", None)
        return value

    @staticmethod
    def _principal(user) -> Principal:
        return Principal(
            key=f"discord:{user.id}",
            display_name=getattr(user, "display_name", None) or user.name,
            is_bot=bool(getattr(user, "bot", False)),
        )

    async def _related_users(self, source_user_id: str, memory: str) -> Dict[str, Any]:
        users: Dict[str, Any] = {}
        bot_user = getattr(self.bot, "user", None)
        if source_user_id and (bot_user is None or str(bot_user.id) != str(source_user_id)):
            try:
                user = await self.bot.fetch_user(int(source_user_id))
                if user:
                    users[str(user.id)] = user
            except Exception:
                pass

        mentioned_names = {name.casefold() for name in re.findall(r"@([\w.\-]+)", memory)}
        get_all_members = getattr(self.bot, "get_all_members", None)
        if mentioned_names and get_all_members:
            for member in get_all_members():
                names = {
                    str(getattr(member, "name", "")).casefold(),
                    str(getattr(member, "display_name", "")).casefold(),
                }
                if mentioned_names & names and (bot_user is None or member.id != bot_user.id):
                    users[str(member.id)] = member
        return users

    def _background_tool_bundle(self, related_users: Dict[str, Any]):
        bot_user = getattr(self.bot, "user", None)
        if not bot_user or not self.config.allow_agent_tools:
            return None
        agent = self._principal(bot_user)
        known_targets: dict[str, Principal] = {}
        for user in related_users.values():
            principal = self._principal(user)
            known_targets[str(user.id)] = principal
            known_targets[principal.key] = principal
            known_targets[principal.display_name.casefold()] = principal
            known_targets[str(getattr(user, "name", "")).casefold()] = principal

        bundles = []
        todo_service = getattr(self.bot, "todo_service", None)
        if todo_service is not None:
            context = TodoRequestContext(
                actor=agent, agent=agent, source="spike",
                known_targets=known_targets,
            )
            bundles.append(build_todo_tool_bundle(todo_service, context))
        bookshelf = getattr(self.bot, "bookshelf_service", None)
        if bookshelf is not None:
            reader_id = str(
                getattr(self.bot, "reader_id", None)
                or getattr(self.bot, "agent_name", None)
                or "default"
            )
            bundles.append(build_bookshelf_tool_bundle(bookshelf, reader_id))
        return merge_tool_bundles(*bundles)

    async def process_orphan(
        self,
        orphaned_memory: str,
        *,
        source_user_id: str | None = None,
        source_memory_id: int | None = None,
    ) -> SpikeActionOutcome:
        """Run one bounded SEEKING episode and durably reflect on its action."""
        await self._sync_pending_reflections()
        source_user_id = str(source_user_id or getattr(self.bot, "agent_id", "unknown"))
        raw_timestamp = datetime.now().strftime("(%H:%M [%d/%m/%y])")
        event = SpikeActionEvent(
            source_user_id=source_user_id,
            source_memory_id=source_memory_id,
            source_memory_hash=hashlib.sha256(orphaned_memory.encode("utf-8")).hexdigest(),
            source_memory=orphaned_memory,
            raw_timestamp=raw_timestamp,
        )
        await asyncio.to_thread(self.repository.save, event)

        try:
            source_name = await self.bot.resolve_user(source_user_id)
        except Exception:
            source_name = f"User({source_user_id})"

        prior_attempts = await asyncio.to_thread(
            self.repository.completed_count,
            event.source_user_id,
            event.source_memory_hash,
        )
        if prior_attempts >= self.config.max_attempts_per_memory:
            event.action = "silence"
            event.status = "completed"
            event.release_recommended = True
            event.executions.append(SpikeExecution(
                sequence=0,
                kind="silence",
                name="spike_energy_exhausted",
                arguments={
                    "reason": "completed SEEKING energy budget already spent",
                    "prior_attempts": prior_attempts,
                },
                result={"silent": True, "release_source": True},
                ok=True,
            ))
            await asyncio.to_thread(self.repository.save, event)
            try:
                await self._reflect_action_event(event, source_name)
            except Exception as exc:
                self.logger.error(f"spike.action.reflect.err event={event.id} msg={exc}")
            refreshed = await asyncio.to_thread(self.repository.get, event.id) or event
            self.logger.log({
                "event": "spike_energy_exhausted",
                "event_id": refreshed.id,
                "source_memory_hash": refreshed.source_memory_hash,
                "prior_attempts": prior_attempts,
            })
            return SpikeActionOutcome(
                event_id=refreshed.id,
                action="silence",
                status="completed",
                grounded=False,
                release_recommended=True,
                reflection_memory=refreshed.memory_text,
            )

        surfaces = (
            await self.find_targets(orphaned_memory)
            if self.config.allow_channel_outreach else []
        )
        surface_by_id = {str(surface.channel.id): surface for surface in surfaces}
        related_users = await self._related_users(source_user_id, orphaned_memory)
        user_by_id = {
            user_id: user for user_id, user in related_users.items()
            if getattr(user, "id", None) != getattr(getattr(self.bot, "user", None), "id", None)
            and not bool(getattr(user, "bot", False))
        }

        surface_lines = []
        for surface in surfaces:
            channel = surface.channel
            if isinstance(channel, discord.TextChannel):
                label = f"#{channel.name} in {channel.guild.name}"
            else:
                label = "DM"
            surface_lines.append(
                f"- channel_id={channel.id}; location={label}; resonance={surface.score:.3f}\n"
                f"  {truncate_middle(surface.compressed, max_tokens=240)}"
            )
        related_lines = [
            f"- user_id={user_id}; name=@{getattr(user, 'name', user_id)}"
            for user_id, user in user_by_id.items()
        ]

        state_lock = threading.RLock()
        origin_loop = asyncio.get_running_loop()

        def reserve(kind: str, name: str, arguments: dict) -> SpikeExecution:
            with state_lock:
                if len(event.executions) >= self.config.max_tool_actions:
                    raise RuntimeError("SEEKING action budget exhausted")
                if kind in {"reach_channel", "message_user"} and any(
                    item.kind in {"reach_channel", "message_user"}
                    for item in event.executions
                ):
                    raise RuntimeError("only one outward message is allowed per SEEKING episode")
                execution = SpikeExecution(
                    sequence=len(event.executions), kind=kind, name=name,
                    arguments=dict(arguments), ok=False, error="pending",
                )
                event.executions.append(execution)
                event.action = kind
                self.repository.save(event)
                return execution

        def finish(execution: SpikeExecution, *, result: Any = None, error: Exception | None = None):
            with state_lock:
                execution.result = result
                execution.ok = error is None
                execution.error = str(error) if error else None
                self.repository.save(event)

        async def on_gateway(coroutine):
            if asyncio.get_running_loop() is origin_loop:
                return await coroutine
            future = asyncio.run_coroutine_threadsafe(coroutine, origin_loop)
            return await asyncio.wrap_future(future)

        action_specs: list[ToolSpec] = []
        action_runtime: dict[str, Any] = {}

        if surface_by_id:
            async def reach_channel(arguments: dict):
                channel_id = str(arguments.get("channel_id", ""))
                content = str(arguments.get("content", "")).strip()
                execution = reserve("reach_channel", "spike_reach_channel", arguments)
                try:
                    if channel_id not in surface_by_id:
                        raise ValueError("channel_id must be an eligible recent surface")
                    if not content:
                        raise ValueError("content cannot be empty")
                    now = datetime.now()
                    if (now - self.last_spike).total_seconds() < self.config.cooldown_seconds:
                        raise RuntimeError("outward-message cooldown is active")
                    surface = surface_by_id[channel_id]
                    channel = surface.channel
                    formatted = format_discord_mentions(
                        content, getattr(channel, "guild", None),
                        self.bot.mentions_enabled, self.bot,
                    )
                    await on_gateway(self._send_chunked(channel, formatted))
                    self.last_spike = now
                    self.log_engagement(channel.id)
                    result = {
                        "sent": True, "channel_id": channel_id,
                        "location": getattr(channel, "name", "DM"),
                        "resonance": surface.score, "content": content,
                    }
                    finish(execution, result=result)
                    return result
                except Exception as exc:
                    finish(execution, error=exc)
                    raise

            action_specs.append(ToolSpec(
                name="spike_reach_channel",
                description="Send one relevant message to an eligible recent channel.",
                parameters={
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "enum": sorted(surface_by_id)},
                        "content": {"type": "string", "minLength": 1, "maxLength": 1800},
                    },
                    "required": ["channel_id", "content"],
                    "additionalProperties": False,
                },
            ))
            action_runtime["spike_reach_channel"] = reach_channel

        if user_by_id and self.config.allow_direct_messages:
            async def message_user(arguments: dict):
                user_id = str(arguments.get("user_id", ""))
                content = str(arguments.get("content", "")).strip()
                execution = reserve("message_user", "spike_message_user", arguments)
                try:
                    if user_id not in user_by_id:
                        raise ValueError("user_id must be an eligible related user")
                    if not content:
                        raise ValueError("content cannot be empty")
                    now = datetime.now()
                    if (now - self.last_spike).total_seconds() < self.config.cooldown_seconds:
                        raise RuntimeError("outward-message cooldown is active")
                    user = user_by_id[user_id]
                    sent_message = await on_gateway(user.send(content))
                    self.last_spike = now
                    sent_channel = getattr(sent_message, "channel", None)
                    if sent_channel is not None:
                        self.log_engagement(sent_channel.id)
                    result = {
                        "sent": True, "user_id": user_id,
                        "user_name": getattr(user, "name", user_id), "content": content,
                    }
                    finish(execution, result=result)
                    return result
                except Exception as exc:
                    finish(execution, error=exc)
                    raise

            action_specs.append(ToolSpec(
                name="spike_message_user",
                description="Send one direct message to an eligible user related to the unresolved association.",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "enum": sorted(user_by_id)},
                        "content": {"type": "string", "minLength": 1, "maxLength": 1800},
                    },
                    "required": ["user_id", "content"],
                    "additionalProperties": False,
                },
            ))
            action_runtime["spike_message_user"] = message_user

        if self.config.allow_memory_search:
            searchable_ids = sorted({source_user_id, *user_by_id.keys()})

            async def search_memories(arguments: dict):
                data = SpikeSearchInput.model_validate(arguments)
                execution = reserve(
                    "search_user_memories", "spike_search_user_memories", arguments
                )
                try:
                    if data.user_id not in searchable_ids:
                        raise ValueError("user_id must be related to the unresolved association")
                    results = await self.memory_index.search_async(
                        data.query, k=self.config.memory_k, user_id=data.user_id
                    )
                    filtered = [
                        (memory, score) for memory, score in results
                        if memory != orphaned_memory
                    ]
                    result = {
                        "user_id": data.user_id, "query": data.query,
                        "count": len(filtered),
                        "memories": [
                            {
                                "relevance": round(float(score), 3),
                                "memory": truncate_middle(
                                    self._temporalize(memory),
                                    max_tokens=self.config.memory_truncation,
                                ),
                            }
                            for memory, score in filtered
                        ],
                    }
                    event.query = data.query
                    finish(execution, result=result)
                    return result
                except Exception as exc:
                    finish(execution, error=exc)
                    raise

            search_schema = self._tool_schema(SpikeSearchInput)
            search_schema["properties"]["user_id"]["enum"] = searchable_ids
            action_specs.append(ToolSpec(
                name="spike_search_user_memories",
                description="Search a related user's memories with a new query you generate, excluding the unresolved source itself.",
                parameters=search_schema,
            ))
            action_runtime["spike_search_user_memories"] = search_memories

        async def choose_silence(arguments: dict):
            data = SpikeSilenceInput.model_validate(arguments)
            execution = reserve("silence", "spike_choose_silence", arguments)
            result = {"silent": True, "reason": data.reason}
            finish(execution, result=result)
            return result

        action_specs.append(ToolSpec(
            name="spike_choose_silence",
            description="Conclude that this unresolved association does not warrant action now.",
            parameters=self._tool_schema(SpikeSilenceInput),
        ))
        action_runtime["spike_choose_silence"] = choose_silence

        domain_bundle = self._background_tool_bundle(related_users)
        if domain_bundle:
            wrapped_runtime = {}
            for name, implementation in domain_bundle.runtime.items():
                async def invoke(arguments: dict, *, _name=name, _implementation=implementation):
                    execution = reserve("invoke_tool", _name, arguments)
                    try:
                        result = await _implementation(arguments)
                        finish(execution, result=result)
                        return result
                    except Exception as exc:
                        finish(execution, error=exc)
                        raise
                wrapped_runtime[name] = invoke
            domain_bundle = AgentToolBundle(
                specs=domain_bundle.specs, runtime=wrapped_runtime
            )

        bundle = merge_tool_bundles(
            AgentToolBundle(specs=action_specs, runtime=action_runtime),
            domain_bundle,
        )
        surface_context = "\n\n".join(surface_lines) or "No recent channel met the outreach threshold."
        related_context = "\n".join(related_lines) or "No directly messageable related user was resolved."
        themes = format_themes_for_prompt(self.memory_index, source_user_id, mode="sections")
        selection_prompt = self.action_prompt_formats["spike_action_selection"].format(
            source_user=source_name,
            memory=self._temporalize(orphaned_memory),
            surface_context=surface_context,
            related_users=related_context,
            timestamp=self._natural_now(),
        )
        selection_system = self.action_system_prompts["spike_action_selection"].format(
            agent_name=getattr(self.bot, "agent_name", getattr(self.bot.user, "name", "agent")),
            amygdala_response=self.bot.amygdala_response,
            themes=themes,
        )

        try:
            response = await self.bot.call_api(
                user_content=selection_prompt,
                system_prompt=selection_system,
                temperature=self.config.decision_temperature,
                tools=bundle.specs,
                tool_runtime=bundle.runtime,
                auto_execute_tools=True,
            )
            _, traces = separate_thinking_traces(response)
            await store_thinking_traces(
                self.memory_index, str(self.bot.user.id), self.bot.user.name, traces
            )
        except Exception as exc:
            successful = [execution for execution in event.executions if execution.ok]
            event.status = "completed" if successful else "failed"
            if event.executions:
                event.action = event.executions[-1].kind
            else:
                event.action = "silence"
                event.executions.append(SpikeExecution(
                    sequence=0, kind="silence", name="spike_api_failure",
                    arguments={}, ok=False, error=str(exc),
                ))
            event.grounded = any(
                execution.kind in {"reach_channel", "message_user", "invoke_tool"}
                or (
                    execution.kind == "search_user_memories"
                    and isinstance(execution.result, dict)
                    and int(execution.result.get("count", 0)) > 0
                )
                for execution in successful
            )
            event.release_recommended = bool(
                event.status == "completed" and not event.grounded
                and any(
                    (execution.kind == "silence" and self.config.release_on_silence)
                    or (
                        execution.kind == "search_user_memories"
                        and isinstance(execution.result, dict)
                        and int(execution.result.get("count", 0)) == 0
                    )
                    for execution in successful
                )
            )
            await asyncio.to_thread(self.repository.save, event)
        else:
            successful = [execution for execution in event.executions if execution.ok]
            if not event.executions:
                event.executions.append(SpikeExecution(
                    sequence=0, kind="silence", name="spike_implicit_silence",
                    arguments={"reason": clean_response(response) or "no action selected"},
                    result={"silent": True}, ok=True,
                ))
                event.action = "silence"
                successful = list(event.executions)
            priority = {
                "reach_channel": 5, "message_user": 4,
                "invoke_tool": 3, "search_user_memories": 2, "silence": 1,
            }
            event.action = max(
                (execution.kind for execution in (successful or event.executions)),
                key=lambda kind: priority[kind],
            )
            event.status = "completed" if successful else "failed"
            event.grounded = any(
                execution.ok and (
                    execution.kind in {"reach_channel", "message_user", "invoke_tool"}
                    or (
                        execution.kind == "search_user_memories"
                        and isinstance(execution.result, dict)
                        and int(execution.result.get("count", 0)) > 0
                    )
                )
                for execution in event.executions
            )
            weak_search = any(
                execution.ok
                and execution.kind == "search_user_memories"
                and isinstance(execution.result, dict)
                and int(execution.result.get("count", 0)) == 0
                for execution in successful
            )
            event.release_recommended = bool(
                event.status == "completed" and not event.grounded
                and (
                    weak_search
                    or (
                        self.config.release_on_silence
                        and any(execution.kind == "silence" for execution in successful)
                    )
                )
            )
            for execution in reversed(event.executions):
                if execution.ok and execution.kind in {"reach_channel", "message_user"}:
                    result = execution.result if isinstance(execution.result, dict) else {}
                    event.target_id = str(result.get("channel_id") or result.get("user_id") or "") or None
                    event.target_label = str(result.get("location") or result.get("user_name") or "") or None
                    break
            await asyncio.to_thread(self.repository.save, event)

        try:
            await self._reflect_action_event(event, source_name)
        except Exception as exc:
            self.logger.error(f"spike.action.reflect.err event={event.id} msg={exc}")

        refreshed = await asyncio.to_thread(self.repository.get, event.id) or event
        self.logger.log({
            "event": "spike_action_completed",
            "event_id": refreshed.id,
            "action": refreshed.action,
            "status": refreshed.status,
            "grounded": refreshed.grounded,
            "release_recommended": refreshed.release_recommended,
            "executions": [item.model_dump(mode="json") for item in refreshed.executions],
        })
        return SpikeActionOutcome(
            event_id=refreshed.id,
            action=refreshed.action or "silence",
            status=refreshed.status,
            grounded=refreshed.grounded,
            release_recommended=refreshed.release_recommended,
            reflection_memory=refreshed.memory_text,
        )

    async def _reflect_action_event(self, event: SpikeActionEvent, source_name: str | None = None) -> None:
        source_name = source_name or await self.bot.resolve_user(event.source_user_id)
        action_context = json.dumps(
            [execution.model_dump(mode="json") for execution in event.executions],
            ensure_ascii=False, indent=2,
        )
        result_context = "completed" if event.status == "completed" else "failed"
        grounding_context = (
            "independent evidence was found; a successor reflection may re-enter the graph"
            if event.grounded else
            "no independent grounding was established; this reflection must not rescue the source by itself"
        )
        themes = format_themes_for_prompt(
            self.memory_index, event.source_user_id, mode="sections"
        )
        prompt = self.action_prompt_formats["spike_action_reflection"].format(
            memory=self._temporalize(event.source_memory),
            action_context=truncate_middle(action_context, max_tokens=1600),
            result_context=result_context,
            grounding_context=grounding_context,
            timestamp=self._natural_now(),
        )
        system = self.action_system_prompts["spike_action_reflection"].format(
            agent_name=getattr(self.bot, "agent_name", getattr(self.bot.user, "name", "agent")),
            amygdala_response=self.bot.amygdala_response,
            themes=themes,
        )
        response = await self.bot.call_api(
            user_content=prompt, system_prompt=system,
            temperature=self.bot.amygdala_response / 100,
        )
        reflection, traces = separate_thinking_traces(response)
        await store_thinking_traces(
            self.memory_index, str(self.bot.user.id), self.bot.user.name, traces
        )
        reflection = clean_response(reflection).strip()
        if not reflection:
            raise ValueError("spike action reflection was empty")
        event.reflection = reflection
        event.memory_text = (
            f"Reflections on SEEKING {event.action} for @{source_name} "
            f"{event.raw_timestamp}:\n{reflection}"
        )
        await asyncio.to_thread(self.repository.save, event)
        await self._sync_action_reflection(event)

    async def _sync_action_reflection(self, event: SpikeActionEvent) -> None:
        if not event.memory_text:
            return
        with self.memory_index._mut:
            already_present = event.memory_text in self.memory_index.memories
        if not already_present:
            await self.memory_index.add_memory_async(
                str(self.bot.user.id), event.memory_text
            )
        await asyncio.to_thread(self.repository.mark_synced, event.id)

    async def _sync_pending_reflections(self) -> None:
        events = await asyncio.to_thread(self.repository.unsynced)
        for event in events:
            await self._sync_action_reflection(event)

    async def _recover_reflections(self) -> None:
        events = await asyncio.to_thread(self.repository.awaiting_reflection)
        for event in events:
            try:
                await self._reflect_action_event(event)
            except Exception as exc:
                self.logger.error(f"spike.action.recover.err event={event.id} msg={exc}")

    async def process_spike(self, event: SpikeEvent) -> Optional[str]:
        now = datetime.now()
        if (now - self.last_spike).total_seconds() < self.config.cooldown_seconds:
            self.logger.info("spike.cooldown")
            return None
        self.last_spike = now
        channel = event.target.channel

        # --- shared context pipeline (identical to process_message) ---

        # Fetch conversation history with reactions + memory search in parallel
        # Search seeded by conversation context only — the orphan is already in the
        # prompt verbatim, so including it here just biases retrieval toward its own
        # semantic neighbours instead of memories relevant to the conversation.
        search_key = event.surface_seed
        history_task = asyncio.create_task(
            fetch_history_with_reactions(channel, bot_config.conversation.max_history)
        )
        memory_task = asyncio.create_task(
            self.memory_index.search_async(search_key, k=self.config.memory_k, user_id=None)
        )
        history_result, candidate_memories = await asyncio.gather(history_task, memory_task)
        history_msgs, reactions_map = history_result

        # Process history into formatted context (temporal timestamps, reactions, bot msgs visible)
        simple_ctx, formatted_msgs = process_history_dual(
            history_msgs, reactions_map, self.temporal_parser,
            bot_config.conversation.truncation_length
        )
        conversation_context = build_conversation_context(formatted_msgs)

        # Rerank memories using shared hippocampus logic
        relevant_memories = await rerank_if_enabled(
            self.bot, candidate_memories, search_key, logger=self.logger
        )
        event.memories = relevant_memories

        memory_context = build_memory_context(
            relevant_memories, self.temporal_parser,
            bot_config.conversation.truncation_length,
            max_tokens=getattr(self.memory_index, "max_tokens", bot_config.search.max_tokens),
        )

        # --- spike-specific prompt assembly ---

        # Parse timestamps in orphaned memory to natural language
        timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
        orphan_memory = re.sub(
            timestamp_pattern,
            lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
            event.orphaned_memory
        )

        # Compute tension description based on match score
        score = event.target.score
        tension_desc = PROMPTS.tension_ceiling
        for bound, desc in PROMPTS.tension_ladder:
            if score < bound:
                tension_desc = desc
                break

        prompt_state = self._build_prompt_state(
            channel=channel,
            tension_desc=tension_desc,
            orphan_memory=orphan_memory,
            memory_context=memory_context,
            conversation_context=conversation_context,
            now=now
        )
        location = prompt_state.location

        themes = format_themes_for_prompt(self.memory_index, None, mode="user")
        rendered_user_content = self.bot.prompt_formats['spike_engagement'].format(
            location=prompt_state.location,
            timestamp=prompt_state.timestamp,
            tension_desc=prompt_state.tension_desc,
            memory=prompt_state.orphan_memory,
            memory_context=prompt_state.memory_context,
            conversation_context=prompt_state.conversation_context,
            themes=themes,
        )
        system_prompt = self.bot.system_prompts['spike_engagement'].replace(
            '{amygdala_response}', str(self.bot.amygdala_response)
        ).replace('{themes}', themes)

        # Log full model context before API call
        temperature = self.bot.amygdala_response / 100
        self.logger.info(f"spike.api_call location={location} score={score:.3f} tension={tension_desc} temp={temperature:.2f}")
        self.logger.log({
            'event': 'spike_api_call',
            'timestamp': now.isoformat(),
            'channel_id': channel.id,
            'location': location,
            'score': event.target.score,
            'tension': tension_desc,
            'temperature': temperature,
            'orphaned_memory': event.orphaned_memory,
            'formatted_orphan': orphan_memory,
            'system_prompt': system_prompt,
            'user_content': rendered_user_content,
            'memory_context': memory_context,
            'conversation_context': conversation_context,
            'themes': themes,
            'memory_count': len(relevant_memories),
            'conversation_msgs': len(formatted_msgs),
        })
        try:
            # Show typing indicator during API call
            async with channel.typing():
                response = await self.bot.call_api(
                    user_content=rendered_user_content,
                    system_prompt=system_prompt,
                    temperature=temperature
                )
            response, thinking_traces = separate_thinking_traces(response)
            await store_thinking_traces(
                self.memory_index,
                str(self.bot.user.id),
                self.bot.user.name,
                thinking_traces,
            )
            response = clean_response(response)
            timestamp_label = prompt_state.timestamp

            # Detect [SILENCE] choice — agent opts out of sending but still reflects
            chose_silence = (
                not response
                or response.strip().lower() in PROMPTS.silence_tokens
                or response.strip().upper().startswith(PROMPTS.silence_prefix)
            )
            if chose_silence:
                self.logger.info("spike.silence chosen")
                self.logger.log({
                    'event': 'spike_silence',
                    'timestamp': now.isoformat(),
                    'channel_id': channel.id,
                    'location': location,
                    'score': event.target.score,
                    'raw_response': response,
                })
                # Still reflect so the silence itself becomes a memory
                memory_text = PROMPTS.silence_memory.format(
                    location=location, timestamp=timestamp_label,
                    orphan=event.orphaned_memory[:200],
                )
                await self.memory_index.add_memory_async(str(self.bot.user.id), memory_text)
                asyncio.create_task(self._reflect_on_spike(
                    memory_text=memory_text,
                    location=location,
                    conversation_context=simple_ctx
                ))
                return None
            formatted = format_discord_mentions(response, getattr(channel, 'guild', None), self.bot.mentions_enabled, self.bot)
            await self._send_chunked(channel, formatted)
            self.log_engagement(channel.id)
            memory_text = PROMPTS.outreach_memory.format(
                location=location, timestamp=timestamp_label,
                orphan=event.orphaned_memory[:200], response=response,
            )
            await self.memory_index.add_memory_async(str(self.bot.user.id), memory_text)
            # Fire reflection as background task (mirrors generate_and_save_thought in process_message)
            asyncio.create_task(self._reflect_on_spike(
                memory_text=memory_text,
                location=location,
                conversation_context=simple_ctx
            ))
            self.logger.log({
                'event': 'spike_fired',
                'timestamp': now.isoformat(),
                'channel_id': channel.id,
                'location': location,
                'orphaned_memory': event.orphaned_memory[:200],
                'memory_context_size': len(memory_context),
                'response': response,
                'score': event.target.score
            })
            return response
        except Exception as e:
            self.logger.error(f"spike.process.err msg={e}")
            return None

    def _build_prompt_state(
        self,
        *,
        channel: discord.abc.Messageable,
        tension_desc: str,
        orphan_memory: str,
        memory_context: str,
        conversation_context: str,
        now: datetime
    ) -> SpikePromptState:
        if isinstance(channel, discord.TextChannel):
            location = PROMPTS.location_channel.format(channel_name=channel.name, guild_name=channel.guild.name)
        else:
            location = PROMPTS.location_dm
        timestamp = now.strftime("%H:%M [%d/%m/%y]")
        return SpikePromptState(
            location=location,
            timestamp=timestamp,
            tension_desc=tension_desc,
            orphan_memory=orphan_memory,
            memory_context=memory_context,
            conversation_context=conversation_context,
        )

    async def _reflect_on_spike(self, memory_text: str, location: str, conversation_context: str):
        """Generate and save a reflection on spike outreach, mirroring generate_and_save_thought."""
        try:
            current_time = datetime.now()
            storage_timestamp = current_time.strftime("%H:%M [%d/%m/%y]")
            temporal_expr = self.temporal_parser.get_temporal_expression(current_time)
            temporal_timestamp = temporal_expr.base_expression
            if temporal_expr.time_context:
                temporal_timestamp = f"{temporal_timestamp} in the {temporal_expr.time_context}"

            # Parse timestamps in the memory_text to natural language
            timestamp_pattern = r'\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)'
            temporal_memory_text = re.sub(
                timestamp_pattern,
                lambda m: f"({self.temporal_parser.get_temporal_expression(datetime.strptime(f'{m.group(1)}:{m.group(2)} {m.group(3)}', '%H:%M %d/%m/%y')).base_expression})",
                memory_text
            )

            rendered_user_content = self.bot.prompt_formats['generate_thought'].format(
                user_name=self.bot.user.name,
                memory_text=temporal_memory_text,
                timestamp=temporal_timestamp,
                conversation_context=conversation_context if conversation_context else ""
            )
            themes = format_themes_for_prompt(self.memory_index, None, mode="global")
            thought_system = self.bot.system_prompts['thought_generation'].replace(
                '{amygdala_response}', str(self.bot.amygdala_response)
            ).replace('{themes}', themes)

            self.logger.info(f"spike.reflect location={location}")
            self.logger.log({
                'event': 'spike_reflect_call',
                'timestamp': current_time.isoformat(),
                'location': location,
                'system_prompt': thought_system,
                'user_content': rendered_user_content,
                'memory_text': memory_text,
            })

            thought_response = await self.bot.call_api(
                user_content=rendered_user_content,
                system_prompt=thought_system,
                temperature=self.bot.amygdala_response / 100
            )
            thought_response, thinking_traces = separate_thinking_traces(thought_response)
            await store_thinking_traces(
                self.memory_index,
                str(self.bot.user.id),
                self.bot.user.name,
                thinking_traces,
            )
            thought_response = clean_response(thought_response)
            reflection = PROMPTS.reflection_memory.format(
                location=location, timestamp=storage_timestamp, thought=thought_response,
            )
            await self.memory_index.add_memory_async(str(self.bot.user.id), reflection)

            self.logger.info(f"spike.reflect.ok location={location} len={len(thought_response)}")
            self.logger.log({
                'event': 'spike_reflection_saved',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'reflection': thought_response,
            })
        except Exception as e:
            self.logger.error(f"spike.reflect.err msg={e}")

    async def _send_chunked(self, channel, text: str, max_len: int = 1800):
        while text:
            chunk = text[:max_len]
            if len(text) > max_len:
                split = chunk.rfind('\n')
                if split > max_len // 2:
                    chunk = text[:split]
            await channel.send(chunk.strip())
            text = text[len(chunk):].strip()
            await asyncio.sleep(0.1)


async def handle_orphaned_memory(
    spike_processor: SpikeProcessor,
    orphaned_memory: str,
    *,
    source_user_id: str | None = None,
    source_memory_id: int | None = None,
) -> SpikeActionOutcome | None:
    if not spike_processor.enabled:
        spike_processor.logger.info("spike.disabled skipping orphan handling")
        return None
    return await spike_processor.process_orphan(
        orphaned_memory,
        source_user_id=source_user_id,
        source_memory_id=source_memory_id,
    )
