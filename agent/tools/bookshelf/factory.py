from __future__ import annotations

from pathlib import Path

from .repository import BookshelfRepository
from .service import BookshelfService


def create_bookshelf_service(cache_dir, bookshelf_config, embedder, logger=None) -> BookshelfService:
    root = Path(cache_dir).resolve()
    repository = BookshelfRepository(root / bookshelf_config.database_filename)
    service = BookshelfService(root, repository, embedder, bookshelf_config, logger)
    if logger:
        logger.info(
            f"Bookshelf ready: root={root} embeddings="
            f"{bookshelf_config.embedding_provider}/{bookshelf_config.embedding_model}"
        )
    return service
