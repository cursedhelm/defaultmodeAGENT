from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from api_schema import ToolSpec
from tools.bundle import AgentToolBundle

from .service import BookshelfService


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddBookInput(ToolInput):
    filename: str | None = Field(
        default=None,
        description="Attached PDF/EPUB filename. Omit when exactly one supported file is attached.",
    )
    title: str | None = None
    author: str | None = None


class ListBooksInput(ToolInput):
    status: str | None = Field(default=None, description="Optional: ready, pending, processing, or failed.")


class BeginBookInput(ToolInput):
    book_id: str = Field(description="Stable book ID from bookshelf_list.")


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    value = model.model_json_schema()
    value.pop("title", None)
    return value


def build_bookshelf_tool_bundle(
    service: BookshelfService,
    reader_id: str,
    attachments: dict[str, bytes] | None = None,
) -> AgentToolBundle:
    files = attachments or {}

    def public_book(book) -> dict[str, Any]:
        return {
            "id": book.id, "title": book.title, "author": book.author,
            "format": book.source_format, "status": book.status,
            "chunk_count": book.chunk_count, "error": book.error,
        }

    async def bookshelf_status(arguments: dict):
        ToolInput.model_validate(arguments)
        status = await service.status(reader_id)
        value = status.model_dump(mode="json", exclude={"current_book"})
        value["current_book"] = public_book(status.current_book) if status.current_book else None
        return value

    async def bookshelf_list(arguments: dict):
        data = ListBooksInput.model_validate(arguments)
        allowed = {"ready", "pending", "processing", "failed"}
        if data.status and data.status not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        books = await asyncio.to_thread(
            service.repository.books, {data.status} if data.status else None
        )
        return {"books": [public_book(book) for book in books]}

    async def bookshelf_begin(arguments: dict):
        data = BeginBookInput.model_validate(arguments)
        progress = await service.start_book(reader_id, data.book_id)
        return progress.model_dump(mode="json")

    async def bookshelf_add(arguments: dict):
        data = AddBookInput.model_validate(arguments)
        if not files:
            raise ValueError("attach a PDF or EPUB to add it to the bookshelf")
        if data.filename is None:
            if len(files) != 1:
                raise ValueError("filename is required when multiple books are attached")
            filename = next(iter(files))
        else:
            filename = next((name for name in files if name.casefold() == data.filename.casefold()), None)
            if filename is None:
                raise ValueError(f"attached book not found: {data.filename}")
        book = await service.add_bytes(filename, files[filename], title=data.title, author=data.author)
        return public_book(book)

    specs = [
        ToolSpec(
            name="bookshelf_status",
            description="See what this agent is reading, its exact progress, and library counts.",
            parameters=_schema(ToolInput),
        ),
        ToolSpec(
            name="bookshelf_list",
            description="List this agent's local books and stable IDs.",
            parameters=_schema(ListBooksInput),
        ),
        ToolSpec(
            name="bookshelf_begin",
            description="Ask this agent to begin or resume a ready book from its own bookshelf.",
            parameters=_schema(BeginBookInput),
        ),
    ]
    runtime = {
        "bookshelf_status": bookshelf_status,
        "bookshelf_list": bookshelf_list,
        "bookshelf_begin": bookshelf_begin,
    }
    if files:
        specs.append(ToolSpec(
            name="bookshelf_add",
            description="Add an attached PDF or EPUB to this agent's private bookshelf for background reading.",
            parameters=_schema(AddBookInput),
        ))
        runtime["bookshelf_add"] = bookshelf_add
    return AgentToolBundle(specs=specs, runtime=runtime)
