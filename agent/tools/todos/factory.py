from __future__ import annotations

from typing import Any

from .embeddings import TodoEmbeddings
from .repository import TodoRepository
from .service import TodoService


def create_todo_service(todo_config, embedder, logger: Any = None) -> TodoService:
    repository = TodoRepository(todo_config.database_path)
    requested_profile = {
        "algorithm": "goal-cosine-v1",
        "embedding_provider": todo_config.embedding_provider,
        "embedding_model": todo_config.embedding_model,
        "max_embed_tokens": todo_config.max_embed_tokens,
        "max_items": todo_config.max_items,
        "remove_threshold": todo_config.semantic_remove_threshold,
        "ambiguity_margin": todo_config.semantic_ambiguity_margin,
    }
    profile = repository.get_or_create_runtime_profile(requested_profile)
    if logger and profile != requested_profile:
        logger.warning(
            "Todo service is using the shared store's canonical runtime profile "
            f"instead of this bot's requested profile: {profile}"
        )
    embeddings = TodoEmbeddings(
        repository,
        embedder,
        provider=profile["embedding_provider"],
        model=profile["embedding_model"],
        max_tokens=int(profile["max_embed_tokens"]),
    )
    service = TodoService(
        repository,
        embeddings,
        max_items=int(profile["max_items"]),
        remove_threshold=float(profile["remove_threshold"]),
        ambiguity_margin=float(profile["ambiguity_margin"]),
    )
    if logger:
        logger.info(
            f"Todo service ready: database={todo_config.database_path} "
            f"embeddings={profile['embedding_provider']}/{profile['embedding_model']}"
        )
    return service
