from __future__ import annotations

import asyncio
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from .converter import (
    EPUB_CONVERSION_VERSION,
    chunk_markdown,
    clean_epub_markdown,
    convert_book,
    refresh_epub_chunks,
)
from .index import bm25_rank, hybrid_rank
from .models import BookRecord, BookshelfStatus, ReadingEvent
from .repository import BookshelfRepository


class BookshelfService:
    """Owns one agent's managed library, index, and restart-safe reader cursor."""

    EMBEDDING_VERSION = "bookshelf-chunks-v1"

    def __init__(self, root: str | Path, repository: BookshelfRepository, embedder, config, logger=None):
        self.root = Path(root).resolve()
        self.books_dir = self.root / "books"
        self.inbox_dir = self.root / "inbox"
        self.repository = repository
        self.embedder = embedder
        self.config = config
        self.logger = logger
        self._ingest_lock = asyncio.Lock()

    @property
    def embedding_profile(self) -> tuple[str, str, str]:
        return (
            self.config.embedding_provider,
            self.config.embedding_model,
            self.EMBEDDING_VERSION,
        )

    async def initialize(self) -> None:
        self.books_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        recovered = await asyncio.to_thread(
            self.repository.recover_interrupted_ingestion,
            self.config.ingestion_stale_seconds,
        )
        if recovered:
            self._log("warning", f"Bookshelf recovered {recovered} interrupted ingestion job(s)")
        candidates = [
            path for path in self.root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".pdf", ".epub"}
        ]
        for source in candidates:
            resolved = source.resolve()
            if self.books_dir == resolved.parent or self.books_dir in resolved.parents:
                continue
            try:
                await self.add_path(source)
            except Exception as exc:
                self._log("warning", f"Bookshelf could not register {source.name}: {exc}")
        await self.upgrade_legacy_epubs()
        await self.ingest_pending()

    async def upgrade_legacy_epubs(self) -> list[BookRecord]:
        """Rebuild legacy EPUB derivatives without changing cursors or event FKs."""

        upgraded: list[BookRecord] = []
        for book in self.repository.books({"ready"}):
            if book.source_format != "epub":
                continue
            if int(book.metadata.get("conversion_version", 0) or 0) >= EPUB_CONVERSION_VERSION:
                continue
            try:
                markdown_path = Path(book.markdown_path)
                old_chunks = await asyncio.to_thread(self.repository.chunks_for_book, book.id)
                converted = await asyncio.to_thread(
                    convert_book, book.source_path, markdown_path.parent, book.id,
                )
                cleaned_markdown = clean_epub_markdown(converted.markdown)
                # Keep the original chunk boundaries and IDs because reading
                # events and cursors refer to them. Source markers let us enrich
                # those chunks with accurate page/section locators in place.
                rebuilt_chunks = refresh_epub_chunks(cleaned_markdown, old_chunks)
                embeddings = await self._embed_many([chunk.text for chunk in rebuilt_chunks])
                temporary = markdown_path.with_suffix(".md.tmp")
                await asyncio.to_thread(temporary.write_text, cleaned_markdown + "\n", "utf-8")
                await asyncio.to_thread(os.replace, temporary, markdown_path)
                updated = book.model_copy(update={
                    "metadata": {
                        **book.metadata,
                        **converted.metadata,
                        "conversion_version": EPUB_CONVERSION_VERSION,
                    }
                })
                await asyncio.to_thread(
                    self.repository.upgrade_chunks_in_place,
                    updated, rebuilt_chunks, self.embedding_profile, embeddings,
                )
                upgraded.append(self.repository.get_book(book.id) or updated)
                self._log(
                    "info",
                    f"Bookshelf rebuilt legacy EPUB locators for {book.title} "
                    f"without changing its reading cursor",
                )
            except Exception as exc:
                self._log("error", f"Bookshelf EPUB cleanup failed for {book.title}: {exc}")
        return upgraded

    def _log(self, level: str, message: str) -> None:
        if self.logger and hasattr(self.logger, level):
            getattr(self.logger, level)(message)

    async def add_path(self, source: str | Path, *, title: str | None = None,
                       author: str | None = None) -> BookRecord:
        path = Path(source)
        return await self.add_bytes(path.name, await asyncio.to_thread(path.read_bytes), title=title, author=author)

    async def add_bytes(self, filename: str, data: bytes, *, title: str | None = None,
                        author: str | None = None) -> BookRecord:
        extension = Path(filename).suffix.casefold()
        if extension not in {".pdf", ".epub"}:
            raise ValueError("bookshelf accepts PDF and EPUB files only")
        if not data:
            raise ValueError("book file is empty")
        if len(data) > self.config.max_file_bytes:
            raise ValueError(f"book exceeds the {self.config.max_file_bytes // (1024 * 1024)} MB limit")
        digest = hashlib.sha256(data).hexdigest()
        existing = self.repository.books()
        duplicate = next((book for book in existing if book.source_hash == digest), None)
        if duplicate:
            if duplicate.status == "failed":
                self.repository.set_book_status(duplicate.id, "pending")
                return self.repository.get_book(duplicate.id) or duplicate
            return duplicate
        book_id = digest[:24]
        safe_stem = "".join(c if c.isalnum() or c in "-_." else "-" for c in Path(filename).stem).strip("-.")[:80] or "book"
        book_dir = self.books_dir / f"{safe_stem}--{digest[:8]}"
        book_dir.mkdir(parents=True, exist_ok=True)
        source_path = book_dir / f"source{extension}"
        temporary = book_dir / f".source{extension}.tmp"
        await asyncio.to_thread(temporary.write_bytes, data)
        await asyncio.to_thread(os.replace, temporary, source_path)
        record = BookRecord(
            id=book_id, source_hash=digest, source_filename=Path(filename).name,
            source_path=str(source_path), source_format=extension[1:],
            markdown_path=str(book_dir / "book.md"), title=title or Path(filename).stem,
            author=author, metadata={"requested_title": title, "requested_author": author},
        )
        record = await asyncio.to_thread(self.repository.register_book, record)
        self._log("info", f"Bookshelf registered {record.title} ({record.id})")
        return record

    async def ingest_pending(self, limit: int | None = None) -> list[BookRecord]:
        async with self._ingest_lock:
            pending = self.repository.books({"pending"})
            if limit is not None:
                pending = pending[:limit]
            results = []
            for book in pending:
                results.append(await self._ingest(book))
            return results

    async def _ingest(self, book: BookRecord) -> BookRecord:
        self.repository.set_book_status(book.id, "processing")
        try:
            output_dir = Path(book.markdown_path).parent
            converted = await asyncio.to_thread(convert_book, book.source_path, output_dir, book.id)
            markdown = converted.markdown.strip()
            if not markdown:
                raise ValueError("document conversion produced no readable text")
            markdown_path = Path(book.markdown_path)
            temporary = markdown_path.with_suffix(".md.tmp")
            await asyncio.to_thread(temporary.write_text, markdown + "\n", "utf-8")
            await asyncio.to_thread(os.replace, temporary, markdown_path)
            chunks = chunk_markdown(markdown, book.id, self.config.chunk_target_tokens)
            if not chunks:
                raise ValueError("document conversion produced no reading chunks")
            embeddings = await self._embed_many([chunk.text for chunk in chunks])
            requested_title = book.metadata.get("requested_title")
            requested_author = book.metadata.get("requested_author")
            updated = book.model_copy(update={
                "title": requested_title or converted.title,
                "author": requested_author or converted.author,
                "metadata": {**converted.metadata, **book.metadata},
                "chunk_count": len(chunks),
                "status": "ready",
                "error": None,
            })
            await asyncio.to_thread(
                self.repository.finish_ingestion, updated, chunks, converted.assets,
                self.embedding_profile, embeddings,
            )
            self._log("info", f"Bookshelf indexed {updated.title}: {len(chunks)} chunks")
            return self.repository.get_book(book.id) or updated
        except Exception as exc:
            self.repository.set_book_status(book.id, "failed", str(exc))
            self._log("error", f"Bookshelf ingestion failed for {book.title}: {exc}")
            return self.repository.get_book(book.id) or book

    async def _embed_many(self, texts: list[str]) -> list[list[float]]:
        values: list[list[float]] = []
        for offset in range(0, len(texts), self.config.embedding_batch_size):
            batch = texts[offset:offset + self.config.embedding_batch_size]
            result = await self.embedder(
                batch, provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                max_tokens=self.config.max_embed_tokens,
            )
            if len(result) != len(batch):
                raise ValueError("embedding provider returned an unexpected vector count")
            for vector in result:
                clean = [float(value) for value in vector]
                if not clean or any(not math.isfinite(value) for value in clean):
                    raise ValueError("embedding provider returned an invalid vector")
                values.append(clean)
        return values

    async def search_chunks(self, query: str, chunks, limit: int) -> list[tuple[Any, float]]:
        if not chunks or limit <= 0:
            return []
        lexical = bm25_rank(query, chunks, limit=max(self.config.candidate_pool, limit))
        lexical_by_id = {chunk.id: score for chunk, score in lexical}
        # A bounded full-vector scan lets semantically related passages compete even
        # when they share no literal BM25 terms.
        if len(chunks) <= self.config.semantic_scan_limit:
            candidates = list(chunks)
        else:
            candidates = [chunk for chunk, _ in lexical]
        if not candidates:
            return []
        ranked_input = [(chunk, lexical_by_id.get(chunk.id, 0.0)) for chunk in candidates]
        try:
            query_vector = await self.embedder(
                query, provider=self.config.embedding_provider,
                model=self.config.embedding_model,
                max_tokens=self.config.max_embed_tokens,
            )
            vectors = await asyncio.to_thread(
                self.repository.get_embeddings, [chunk.id for chunk in candidates],
                *self.embedding_profile,
            )
        except Exception as exc:
            self._log("warning", f"Bookshelf semantic search degraded to BM25: {exc}")
            query_vector, vectors = None, {}
        return hybrid_rank(
            ranked_input, query_vector, vectors,
            blend=self.config.hybrid_blend, limit=limit,
        )

    async def selection_candidates(self, reader_id: str, curiosity: str, limit: int = 5):
        books = []
        for book in self.repository.books({"ready"}):
            progress = self.repository.progress_for(reader_id, book.id)
            if progress is None or progress.status != "completed":
                books.append(book)
        chunks = self.repository.chunks_for_books([book.id for book in books])
        ranked = await self.search_chunks(curiosity, chunks, max(limit * 6, limit))
        by_book: dict[str, tuple[Any, float]] = {}
        for chunk, score in ranked:
            if chunk.book_id not in by_book or score > by_book[chunk.book_id][1]:
                by_book[chunk.book_id] = (chunk, score)
        records = {book.id: book for book in books}
        return [
            (records[book_id], chunk, score)
            for book_id, (chunk, score) in sorted(by_book.items(), key=lambda item: item[1][1], reverse=True)[:limit]
        ]

    async def prior_chunks(self, reader_id: str, query: str, limit: int):
        chunks = await asyncio.to_thread(self.repository.eligible_prior_chunks, reader_id)
        return await self.search_chunks(query, chunks, limit)

    async def start_book(self, reader_id: str, book_id: str):
        return await asyncio.to_thread(self.repository.start_book, reader_id, book_id)

    async def current(self, reader_id: str):
        return await asyncio.to_thread(self.repository.current, reader_id)

    async def next_chunk(self, reader_id: str):
        book, progress = await self.current(reader_id)
        if not book or not progress:
            return book, progress, None
        chunk = await asyncio.to_thread(self.repository.get_chunk, book.id, progress.next_ordinal)
        return book, progress, chunk

    async def commit(self, event: ReadingEvent, next_ordinal: int, chunk_count: int) -> None:
        await asyncio.to_thread(self.repository.commit_reading, event, next_ordinal, chunk_count)

    async def status(self, reader_id: str) -> BookshelfStatus:
        books = await asyncio.to_thread(self.repository.books)
        book, progress = await self.current(reader_id)
        chunk = None
        percent = 0.0
        if book and progress:
            chunk = await asyncio.to_thread(self.repository.get_chunk, book.id, progress.next_ordinal)
            percent = min(100.0, 100.0 * progress.next_ordinal / max(1, book.chunk_count))
        return BookshelfStatus(
            reader_id=reader_id, current_book=book, progress=progress,
            current_locator=chunk.locator if chunk else None, percent_complete=percent,
            available_books=sum(1 for value in books if value.status == "ready"),
            pending_books=sum(1 for value in books if value.status in {"pending", "processing"}),
            failed_books=sum(1 for value in books if value.status == "failed"),
        )

    def media_paths(self, book: BookRecord, chunk) -> list[str]:
        book_dir = Path(book.markdown_path).parent.resolve()
        result = []
        for relative in chunk.media_paths[:self.config.max_media_per_chunk]:
            candidate = (book_dir / relative).resolve()
            if candidate.is_file() and book_dir in candidate.parents:
                result.append(str(candidate))
        return result
