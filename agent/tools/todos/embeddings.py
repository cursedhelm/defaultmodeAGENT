from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any

from .repository import TodoRepository


Embedder = Callable[..., Awaitable[Any]]


class TodoEmbeddings:
    def __init__(
        self,
        repository: TodoRepository,
        embedder: Embedder | None,
        *,
        provider: str,
        model: str,
        max_tokens: int = 256,
    ):
        self.repository = repository
        self.embedder = embedder
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.cache_version = f"v1:max_tokens={max_tokens}"

    async def get_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_indexes: list[int] = []
        for index, text in enumerate(texts):
            cached = self.repository.get_embedding(
                self.provider, self.model, text, version=self.cache_version
            )
            if cached is None:
                missing_texts.append(text)
                missing_indexes.append(index)
            else:
                vectors[index] = cached

        if missing_texts:
            if self.embedder is None:
                raise RuntimeError("todo embedding provider is not configured")
            response = await self.embedder(
                missing_texts,
                provider=self.provider,
                model=self.model,
                max_tokens=self.max_tokens,
            )
            if len(missing_texts) == 1 and response and isinstance(response[0], (int, float)):
                response = [response]
            if not isinstance(response, list) or len(response) != len(missing_texts):
                raise RuntimeError("embedding provider returned an unexpected batch shape")
            for index, text, vector in zip(missing_indexes, missing_texts, response):
                clean = [float(value) for value in vector]
                if not clean or not all(math.isfinite(value) for value in clean):
                    raise RuntimeError("embedding provider returned an invalid vector")
                self.repository.put_embedding(
                    self.provider, self.model, text, clean, version=self.cache_version
                )
                vectors[index] = clean

        result = [vector for vector in vectors if vector is not None]
        if len(result) != len(texts) or len({len(vector) for vector in result}) != 1:
            raise RuntimeError("embedding dimensions do not match")
        return result

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            raise ValueError("embedding dimensions do not match")
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
