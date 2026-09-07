from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
from pathlib import Path

from .models import Principal, TodoRequestContext
from .service import TodoService


def parse_todont_markdown(text: str) -> tuple[str | None, list[str]]:
    section = None
    goal: str | None = None
    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.casefold() == "# goal":
            section = "goal"
            continue
        if line.casefold() == "# items":
            section = "items"
            continue
        if not line:
            continue
        if section == "goal" and goal is None:
            goal = line
        elif section == "items" and line.startswith("- "):
            items.append(line[2:].strip())
    return goal, items


async def migrate_todont_directory(service: TodoService, source: str | Path) -> dict[str, int]:
    """Import legacy Markdown only; pickle embeddings are never deserialized."""
    directory = Path(source)
    if not directory.is_dir():
        raise FileNotFoundError(f"todo import directory does not exist: {directory}")
    marker = "migration:todont:" + hashlib.sha256(
        str(directory.resolve()).encode("utf-8")
    ).hexdigest()
    if not service.repository.claim_metadata(marker, "in_progress"):
        return {"lists": 0, "items": 0, "skipped": 0, "already_migrated": 1}
    imported_lists = 0
    imported_items = 0
    skipped = 0
    try:
        for path in sorted(directory.glob("*.md")):
            if not re.fullmatch(r"\d+", path.stem):
                skipped += 1
                continue
            owner = Principal(key=f"discord:{path.stem}", display_name=f"User({path.stem})")
            migration_agent = Principal(
                key="system:todont-migration", display_name="todont migration", is_bot=True
            )
            context = TodoRequestContext(
                actor=owner,
                agent=migration_agent,
                source="discord_command",
            )
            goal, items = parse_todont_markdown(path.read_text(encoding="utf-8"))
            await service.set_goal(context, owner, goal)
            for item in items:
                try:
                    await service.add(context, owner, item)
                    imported_items += 1
                except ValueError as exc:
                    if "already" not in str(exc):
                        raise
            imported_lists += 1
    except Exception:
        service.repository.delete_metadata(marker)
        raise
    service.repository.set_metadata(marker, "complete")
    return {"lists": imported_lists, "items": imported_items, "skipped": skipped, "already_migrated": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Import todont Markdown lists")
    parser.add_argument("source", help="Directory containing legacy <discord_id>.md files")
    parser.add_argument("--database", default="cache/shared/todos.sqlite3")
    args = parser.parse_args()

    from .embeddings import TodoEmbeddings
    from .repository import TodoRepository

    repository = TodoRepository(args.database)
    service = TodoService(repository, TodoEmbeddings(repository, None, provider="none", model="none"))
    print(asyncio.run(migrate_todont_directory(service, args.source)))


if __name__ == "__main__":
    main()
