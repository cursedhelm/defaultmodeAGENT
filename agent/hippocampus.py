from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
import numpy as np
import aiohttp
from logger import logging
from pydantic import BaseModel, Field
from api_client import get_embeddings
import asyncio
from chunker import truncate_middle
from bot_config import HippocampusConfig, EmbeddingConfig
from tools.chronpression import chronomic_filter
from tokenizer import count_tokens


class Hippocampus:
    def __init__(self, config: HippocampusConfig, logger=None):
        self.config = config
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._embedding_cache_max = int(getattr(config, 'embedding_cache_max', 5000))
        self.embedding_config = EmbeddingConfig()
        self.logger = logger or logging.getLogger("bot.default")

    def _cache_put(self, k: str, v: np.ndarray) -> None:
        self._embedding_cache[k] = v
        self._embedding_cache.move_to_end(k)
        while len(self._embedding_cache) > self._embedding_cache_max:
            self._embedding_cache.popitem(last=False)

    def _cache_get(self, k: str) -> Optional[np.ndarray]:
        v = self._embedding_cache.get(k)
        if v is not None:
            self._embedding_cache.move_to_end(k)
        return v

    def _smart_compress_memory(self, text: str, max_tokens: int) -> str:
        """
        Intelligently compress memory text using chronpression before truncation.

        Strategy:
        1. Check if text exceeds token limit
        2. If yes, apply chronomic compression with adaptive ratio
        3. If still too long, fall back to truncate_middle

        Args:
            text: Memory text to compress
            max_tokens: Maximum allowed tokens

        Returns:
            Compressed text that fits within max_tokens
        """
        current_tokens = count_tokens(text)

        # If already within limit, return as-is
        if current_tokens <= max_tokens:
            return text

        # Calculate how much we need to compress
        # Add 10% safety margin to account for compression variance
        target_ratio = (max_tokens * 0.9) / current_tokens

        # Use chronpression for intelligent compression
        # compression parameter is how much to REMOVE (0.5 = remove 50%)
        compression_ratio = max(0.3, min(0.85, 1.0 - target_ratio))

        try:
            compressed = chronomic_filter(
                text,
                compression=compression_ratio,
                fuzzy_strength=1.0
            )

            # Verify it's actually shorter
            compressed_tokens = count_tokens(compressed)

            if compressed_tokens <= max_tokens:
                self.logger.debug(
                    f"Chronpression: {current_tokens} → {compressed_tokens} tokens "
                    f"(ratio: {compression_ratio:.2f})"
                )
                return compressed
            else:
                # Still too long, use truncate_middle as fallback
                self.logger.debug(
                    f"Chronpression insufficient ({compressed_tokens} > {max_tokens}), "
                    f"falling back to truncate_middle"
                )
                return truncate_middle(compressed, max_tokens=max_tokens)

        except Exception as e:
            self.logger.warning(f"Chronpression failed: {e}, using truncate_middle")
            return truncate_middle(text, max_tokens=max_tokens)

    async def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embeddings specifically from Ollama API."""
        # Compress text to fit model's context window with safety margin
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        compressed_text = self._smart_compress_memory(text, max_tokens=max_tokens)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.embedding_config.api_base}/api/embeddings",
                    json={
                        "model": self.embedding_config.model,
                        "prompt": compressed_text
                    },
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                        raise Exception(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                    result = await response.json()
                    embedding = np.array(result["embedding"])
                    if embedding.shape[0] != self.embedding_config.dimensions:
                        raise ValueError(
                            f"Unexpected embedding dimensions: got {embedding.shape[0]}, "
                            f"expected {self.embedding_config.dimensions}"
                        )
                    return embedding
            except Exception as e:
                self.logger.error(f"Ollama embedding error: {str(e)}")
                return None

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embeddings with provider-specific handling."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        try:
            if self.config.embedding_provider == "ollama":
                embedding = await self._get_ollama_embedding(text)
            else:
                max_tokens = getattr(self.embedding_config, "max_embed_tokens", 256)
                embedding = await get_embeddings(
                    text,
                    provider=self.config.embedding_provider,
                    model=self.config.embedding_model,
                    max_tokens=max_tokens,
                )
                embedding = np.array(embedding)

            if embedding is not None:
                embedding = embedding / np.linalg.norm(embedding)
                self._cache_put(text, embedding)
            return embedding
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return None

    async def _get_ollama_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for multiple texts in a single batch from Ollama API."""
        # Compress texts to fit model's context window with safety margin
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        compressed_texts = [self._smart_compress_memory(text, max_tokens=max_tokens) for text in texts]

        async with aiohttp.ClientSession() as session:
            try:
                tasks = []
                for text in compressed_texts:
                    tasks.append(
                        session.post(
                            f"{self.embedding_config.api_base}/api/embeddings",
                            json={
                                "model": self.embedding_config.model,
                                "prompt": text
                            },
                        )
                    )
                responses = await asyncio.gather(*tasks)
                embeddings = []
                for response in responses:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(
                            f"Ollama API returned status {response.status}: {error_text}"
                        )
                        continue
                    result = await response.json()
                    embedding = np.array(result["embedding"])
                    if embedding.shape[0] != self.embedding_config.dimensions:
                        self.logger.error(
                            f"Unexpected embedding dimensions: {embedding.shape[0]}"
                        )
                        continue
                    embeddings.append(embedding)
                return np.array(embeddings)
            except Exception as e:
                self.logger.error(f"Batch embedding error: {str(e)}")
                return None

    async def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get cached embeddings for multiple texts with batch processing."""
        uncached_texts = []
        uncached_indices = []
        embeddings = np.zeros((len(texts), self.embedding_config.dimensions))

        for i, text in enumerate(texts):
            cached = self._cache_get(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            try:
                if self.config.embedding_provider == "ollama":
                    new_embeddings = await self._get_ollama_embeddings_batch(uncached_texts)
                else:
                    max_tokens = getattr(self.embedding_config, "max_embed_tokens", 256)
                    new_embeddings = await get_embeddings(
                        uncached_texts,
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        max_tokens=max_tokens,
                    )
                    new_embeddings = np.array(new_embeddings)

                if new_embeddings is not None and len(new_embeddings) > 0:
                    for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                        normalized_embedding = embedding / np.linalg.norm(embedding)
                        self._cache_put(text, normalized_embedding)
                        embeddings[uncached_indices[i]] = normalized_embedding
            except Exception as e:
                self.logger.error(f"Batch embedding generation failed: {str(e)}")

        return embeddings

    async def rerank_memories(
        self,
        query: str,
        memories: List[Tuple[str, float]],
        threshold: float = 0.6,
        blend_factor: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank memories using batched vector similarity and blend with original scores."""
        self.logger.info(
            f"Starting batch reranking for query: {query[:100]}... with {len(memories)} candidates"
        )
        blend = self.config.blend_factor if blend_factor is None else blend_factor
        self.logger.info(
            f"Using blend factor: {blend:.2f} (initial:{blend:.2f}/embedding:{1 - blend:.2f})"
        )

        # Keep original memories for returning to agent
        original_memories = [str(m[0]) for m in memories]
        initial_scores = np.array([m[1] for m in memories])

        # Compress memories ONLY for embedding generation
        max_tokens = getattr(self.embedding_config, "max_embed_tokens", 160)
        compressed_for_embedding = [self._smart_compress_memory(text, max_tokens=max_tokens) for text in original_memories]

        # Compress query ONLY for embedding generation
        compressed_query = self._smart_compress_memory(query, max_tokens=max_tokens)
        query_embedding = await self._get_embedding(compressed_query)
        if query_embedding is None:
            self.logger.error("Failed to generate query embedding")
            return []

        # Generate embeddings from compressed versions
        memory_embeddings = await self._get_embeddings_batch(compressed_for_embedding)
        cosine = np.dot(memory_embeddings, query_embedding)
        embedding_similarities = 0.5 * (cosine + 1.0)
        initial_scores = np.clip(initial_scores, 0.0, 1.0)
        combined_scores = np.clip(
            (blend * initial_scores) + ((1 - blend) * embedding_similarities), 0.0, 1.0
        )

        # Return ORIGINAL uncompressed memories with their reranked scores
        reranked = [(original_memories[i], float(combined_scores[i]))
                    for i in range(len(original_memories))
                    if combined_scores[i] >= threshold]
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            f"Batch reranking complete – {len(reranked)}/{len(memories)} memories above threshold"
        )
        return reranked
