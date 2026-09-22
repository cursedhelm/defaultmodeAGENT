from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from api_schema import ToolSpec
from attention import format_themes_for_prompt
from chunker import clean_response, truncate_middle
from context import build_memory_context, fit_ranked_entries, rerank_if_enabled
from temporality import TemporalParser
from thinking_trace import separate_thinking_traces, store_thinking_traces
from tools.bookshelf.models import ReadingEvent


class ReadingProcessor:
    """Restart-safe background book ingestion, recall, and reflection loop."""

    def __init__(self, bookshelf, memory_index, prompt_formats, system_prompts,
                 runtime, reader_id: str, reading_config=None):
        if reading_config is None:
            from bot_config import config
            reading_config = config.reading
        self.bookshelf = bookshelf
        self.memory_index = memory_index
        self.runtime = runtime
        self.reader_id = str(reader_id)
        self.config = reading_config
        if reading_config.tick_rate is None:
            from bot_config import config
            self.tick_rate = config.system.tick_rate
        else:
            self.tick_rate = reading_config.tick_rate
        self.logger = getattr(runtime, "logger", logging.getLogger("bot.reader"))
        self.temporal_parser = TemporalParser()
        self.enabled = False
        self.task: asyncio.Task | None = None
        self.prompt_formats = dict(self._load_yaml("reading_prompt_formats.yaml"))
        self.prompt_formats.update({
            key: value for key, value in (prompt_formats or {}).items()
            if key in {"bookshelf_choose_book", "reading_reflection"}
        })
        self.system_prompts = dict(self._load_yaml("reading_system_prompts.yaml"))
        self.system_prompts.update({
            key: value for key, value in (system_prompts or {}).items()
            if key in {"bookshelf_selection", "reading_reflection"}
        })

    @staticmethod
    def _load_yaml(filename: str) -> dict[str, str]:
        path = Path(__file__).resolve().parent / "prompts" / filename
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    async def start(self) -> None:
        if self.enabled or not self.config.enabled:
            return
        self.enabled = True
        try:
            await self.bookshelf.initialize()
            await self._sync_pending_memories()
        except Exception:
            self.enabled = False
            raise
        self.task = asyncio.create_task(self._process_loop())
        self.logger.info(
            f"READER loop started: cache={self.bookshelf.root} api="
            f"{self.config.reader_api_type or 'default'} model={self.config.reader_model or 'default'}"
        )

    async def stop(self) -> None:
        self.enabled = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
        self.logger.info("READER loop stopped")

    async def _process_loop(self) -> None:
        while self.enabled:
            try:
                if self.runtime.processing_enabled:
                    await self.process_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.logger.error(f"READER cycle failed: {exc}")
            await asyncio.sleep(self.tick_rate)

    def _densest_memory(self) -> str | None:
        with self.memory_index._mut:
            density: Counter[int] = Counter()
            for postings in self.memory_index.inverted_index.values():
                density.update(postings)
            allowed = None
            if self.config.prior_scope == "agent":
                allowed = set(self.memory_index.user_memories.get(str(self.runtime.agent_id), []))
            candidates = [
                (score, memory_id, self.memory_index.memories[memory_id])
                for memory_id, score in density.items()
                if memory_id < len(self.memory_index.memories)
                and self.memory_index.memories[memory_id] is not None
                and (allowed is None or memory_id in allowed)
            ]
        return max(candidates, default=(0, -1, None))[2]

    def _natural_now(self) -> str:
        expression = self.temporal_parser.get_temporal_expression(datetime.now())
        return " ".join(value for value in (expression.base_expression, expression.time_context) if value)

    def _temporalize(self, text: str) -> str:
        pattern = r"\((\d{2}):(\d{2})\s*\[(\d{2}/\d{2}/\d{2})\]\)"
        return re.sub(
            pattern,
            lambda match: "(" + self.temporal_parser.get_temporal_expression(
                datetime.strptime(
                    f"{match.group(1)}:{match.group(2)} {match.group(3)}",
                    "%H:%M %d/%m/%y",
                )
            ).base_expression + ")",
            text,
        )

    def _themes(self) -> str:
        try:
            return format_themes_for_prompt(
                self.memory_index, str(self.runtime.agent_id), mode="sections"
            )
        except Exception:
            return "No stable themes yet."

    def _api_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.config.reader_api_type:
            kwargs["api_type_override"] = self.config.reader_api_type
        if self.config.reader_model:
            kwargs["model_override"] = self.config.reader_model
        return kwargs

    async def process_once(self) -> bool:
        await self._sync_pending_memories()
        await self.bookshelf.ingest_pending(limit=1)
        book, progress = await self.bookshelf.current(self.reader_id)
        if not book:
            selected = await self._choose_book()
            if not selected:
                return False
        return await self._read_next()

    async def _choose_book(self) -> bool:
        seed = self._densest_memory() or "curiosity without an established memory prior"
        candidates = await self.bookshelf.selection_candidates(
            self.reader_id, seed, self.config.selection_limit
        )
        if not candidates:
            return False
        rendered = []
        valid_ids = set()
        for book, chunk, score in candidates:
            valid_ids.add(book.id)
            rendered.append(
                f"- id={book.id}; title={book.title!r}; author={book.author or 'unknown'}; "
                f"relevance={score:.3f}; evidence={truncate_middle(self._temporalize(chunk.text), 120)!r}"
            )
        chosen: dict[str, str] = {}

        async def begin(arguments: dict):
            book_id = str(arguments.get("book_id", ""))
            if book_id not in valid_ids:
                raise ValueError("book_id must be one of the supplied candidates")
            await self.bookshelf.start_book(self.reader_id, book_id)
            chosen["id"] = book_id
            return {"started": book_id}

        spec = ToolSpec(
            name="bookshelf_begin",
            description="Begin reading exactly one of the candidate books.",
            parameters={
                "type": "object",
                "properties": {"book_id": {"type": "string", "enum": sorted(valid_ids)}},
                "required": ["book_id"], "additionalProperties": False,
            },
        )
        prompt = self.prompt_formats["bookshelf_choose_book"].format(
            curiosity_seed=self._temporalize(seed),
            candidate_books="\n".join(rendered), timestamp=self._natural_now(),
        )
        system = self.system_prompts["bookshelf_selection"].format(
            amygdala_response=self.runtime.amygdala_response, themes=self._themes(),
        )
        try:
            await self.runtime.call_api(
                user_content=prompt, system_prompt=system,
                temperature=self.config.temperature, tools=[spec],
                tool_runtime={"bookshelf_begin": begin}, auto_execute_tools=True,
                **self._api_kwargs(),
            )
        except Exception as exc:
            self.logger.warning(f"READER book choice degraded to top hybrid result: {exc}")
        if not chosen:
            await self.bookshelf.start_book(self.reader_id, candidates[0][0].id)
        return True

    async def _read_next(self) -> bool:
        book, progress, chunk = await self.bookshelf.next_chunk(self.reader_id)
        if not book or not progress or not chunk:
            return False
        if await asyncio.to_thread(
            self.bookshelf.repository.has_event, self.reader_id, book.id, chunk.id
        ):
            return False
        owner = str(self.runtime.agent_id) if self.config.prior_scope == "agent" else None
        memories = await self.memory_index.search_async(
            chunk.text, k=self.config.memory_candidates, user_id=owner
        )
        memories = await rerank_if_enabled(
            self.runtime, memories, chunk.text, logger=self.logger
        )
        memories = memories[:self.config.memory_limit]
        memory_context = build_memory_context(
            memories, self.temporal_parser, self.config.memory_truncation,
            max_tokens=self.config.prompt_context_tokens,
        ) or "<memories></memories>"
        priors = await self.bookshelf.prior_chunks(
            self.reader_id, chunk.text, self.config.prior_limit
        )
        prior_entries = [
            f"[Relevance: {score:.2f}] {prior.locator}: "
            f"{truncate_middle(self._temporalize(prior.text), self.config.memory_truncation)}\n"
            for prior, score in priors if prior.id != chunk.id
        ]
        prior_context = fit_ranked_entries(
            prior_entries, "<reading_priors>\n", "</reading_priors>",
            max_tokens=self.config.prompt_context_tokens,
        ) or "<reading_priors></reading_priors>"
        prompt = self.prompt_formats["reading_reflection"].format(
            book_metadata=f"{book.title} by {book.author or 'unknown author'} ({book.source_format.upper()})",
            position=f"{chunk.locator}; chunk {chunk.ordinal + 1} of {book.chunk_count}",
            section_text=self._temporalize(chunk.text), memory_context=memory_context,
            prior_reading_context=prior_context, timestamp=self._natural_now(),
        )
        system = self.system_prompts["reading_reflection"].format(
            amygdala_response=self.runtime.amygdala_response, themes=self._themes(),
        )
        response = await self.runtime.call_api(
            user_content=prompt, system_prompt=system,
            temperature=self.config.temperature,
            image_paths=self.bookshelf.media_paths(book, chunk),
            **self._api_kwargs(),
        )
        reflection, traces = separate_thinking_traces(response)
        await store_thinking_traces(
            self.memory_index, str(self.runtime.agent_id), self.runtime.agent_name, traces
        )
        reflection = clean_response(reflection).strip()
        if not reflection:
            raise ValueError("READER returned an empty reflection")
        # Raw timestamps are authoritative in storage. TemporalParser is applied
        # only when memories are rendered into later prompts.
        raw_timestamp = datetime.now().strftime("(%H:%M [%d/%m/%y])")
        memory_text = (
            f"Reading reflection on {book.title!r}, {chunk.locator} {raw_timestamp}:\n"
            f"{reflection}"
        )
        event = ReadingEvent(
            reader_id=self.reader_id, book_id=book.id, chunk_id=chunk.id,
            raw_timestamp=raw_timestamp, reflection=reflection, memory_text=memory_text,
        )
        await self.bookshelf.commit(event, chunk.ordinal + 1, book.chunk_count)
        await self._sync_event(event)
        self.logger.info(
            f"READER reflected on {book.title} at {chunk.locator} "
            f"({chunk.ordinal + 1}/{book.chunk_count})"
        )
        return True

    async def _sync_event(self, event: ReadingEvent) -> None:
        with self.memory_index._mut:
            already_present = event.memory_text in self.memory_index.memories
        if not already_present:
            await self.memory_index.add_memory_async(
                str(self.runtime.agent_id), event.memory_text
            )
        await asyncio.to_thread(
            self.bookshelf.repository.mark_event_synced,
            event.reader_id, event.book_id, event.chunk_id,
        )

    async def _sync_pending_memories(self) -> None:
        events = await asyncio.to_thread(
            self.bookshelf.repository.unsynced_events, self.reader_id
        )
        for event in events:
            await self._sync_event(event)
