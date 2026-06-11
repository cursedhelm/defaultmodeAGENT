'''
## memory flow
spike triggered
    │
    └─► find_target()
            │
            ├─► prefetch_surfaces() (batch fetch all channels)
            ├─► compress_surface_from_buffer() + score_match() per surface
            │
            └─► winner surface found
                    │
                    └─► process_spike()
                            │
                            ├─► fetch_history_with_reactions + rerank_if_enabled (shared pipeline)
                            │       │
                            │       ├─► search_key = compressed_context + orphaned_memory
                            │       ├─► memory_index.search_async(combined_key, k=12)
                            │       ├─► hippocampus reranking (shared with discord_bot)
                            │       ├─► temporal parse timestamps
                            │       └─► return <memories> block (identical format to process_message)
                            │
                            ├─► prompt = orphan + memory_context + conversation_context
                            ├─► call api (main bot api, no override)
                            ├─► send response
                            ├─► store interaction memory under bot.user.id
                            └─► _reflect_on_spike() (background task)
                                    │
                                    ├─► call api (generate_thought / thought_generation)
                                    └─► store reflection memory under bot.user.id

'''


import asyncio
import os
import pickle
import threading
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import re
from tools.chronpression import chronomic_filter
from chunker import truncate_middle, clean_response
from discord_utils import sanitize_mentions, format_discord_mentions
from attention import format_themes_for_prompt, get_current_themes
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
from bot_config import config as bot_config
from memory import AtomicSaver
from context import (
    fetch_history_with_reactions,
    process_history_dual,
    build_memory_context,
    build_conversation_context,
    rerank_if_enabled
)
import discord

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
        self.enabled: bool = True
        self.logger = bot.logger
        self.temporal_parser = TemporalParser()
        # Persistence for engagement log
        self.cache_path = cache_path or os.path.join('cache', getattr(bot, 'bot_id', 'default'), 'spike')
        self.engagement_log_path = os.path.join(self.cache_path, 'engagement_log.pkl')
        self._mut = threading.RLock()
        self.engagement_log: Dict[int, datetime] = self._load_engagement_log()
        self._saver = AtomicSaver(self.engagement_log_path, self._snapshot_engagement, debounce=1.0, logger=self.logger)

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

    async def find_target(self, orphaned_memory: str) -> Optional[SpikeEvent]:
        surfaces = self.get_recent_surfaces()
        if not surfaces:
            self.logger.info("spike.no_surfaces")
            return None

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
            return None
        target = max(viable, key=lambda s: s.score)
        self.logger.info(f"spike.target channel={target.channel.id} score={target.score:.3f}")
        self.logger.log({
            'event': 'spike_target_found',
            'orphaned_memory': orphaned_memory[:300],
            'target_channel': target.channel.id,
            'target_score': round(target.score, 3),
            'surfaces_evaluated': len(surfaces),
            'scores': {str(s.channel.id): round(s.score, 3) for s in surfaces},
            'viable_count': len(viable),
            'final_n': n,
        })
        return SpikeEvent(
            orphaned_memory=orphaned_memory,
            target=target,
            surface_seed=target.compressed
        )

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
            bot_config.conversation.truncation_length
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
        if score < 0.4:
            tension_desc = "distant, tenuous"
        elif score < 0.5:
            tension_desc = "loosely connected"
        elif score < 0.6:
            tension_desc = "resonant but uncertain"
        else:
            tension_desc = "strongly drawn"

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
        prompt = self.bot.prompt_formats['spike_engagement'].format(
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
            'prompt': prompt,
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
                    prompt=prompt,
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
                or response.strip().lower() in ('', 'none', 'pass')
                or response.strip().upper().startswith('[SILENCE]')
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
                memory_text = f"spike considered {location} ({timestamp_label}):\norphan: {event.orphaned_memory[:200]}\n[chose silence]"
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
            memory_text = f"spike reached {location} ({timestamp_label}):\norphan: {event.orphaned_memory[:200]}\nresponse: {response}"
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
            location = f"#{channel.name} in {channel.guild.name}"
        else:
            location = "DM"
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

            thought_prompt = self.bot.prompt_formats['generate_thought'].format(
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
                'prompt': thought_prompt,
                'memory_text': memory_text,
            })

            thought_response = await self.bot.call_api(
                prompt=thought_prompt,
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
            reflection = f"Reflections on spike to {location} ({storage_timestamp}):\n{thought_response}"
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


async def handle_orphaned_memory(spike_processor: SpikeProcessor, orphaned_memory: str) -> bool:
    if not spike_processor.enabled:
        spike_processor.logger.info("spike.disabled skipping orphan handling")
        return False
    event = await spike_processor.find_target(orphaned_memory)
    if not event:
        return False
    response = await spike_processor.process_spike(event)
    return response is not None
