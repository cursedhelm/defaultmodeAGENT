from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import BookChunk


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these",
    "those", "have", "has", "had", "can", "could", "will", "would", "should",
}


def tokenize(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[\w']+", text.casefold())
        if token not in _STOPWORDS and (not token.isdigit() or len(token) == 4)
    ]


def bm25_rank(query: str, chunks: list[BookChunk], limit: int = 32) -> list[tuple[BookChunk, float]]:
    """Use the same k1/b/length-normalized BM25 shape as UserMemoryIndex."""

    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return []
    documents = {chunk.id: tokenize(chunk.text) for chunk in chunks}
    avg_length = sum(len(tokens) for tokens in documents.values()) / max(1, len(documents))
    postings: dict[str, dict[str, int]] = defaultdict(dict)
    for chunk_id, tokens in documents.items():
        for term, count in Counter(tokens).items():
            postings[term][chunk_id] = count
    by_id = {chunk.id: chunk for chunk in chunks}
    scores: Counter = Counter()
    total = len(chunks)
    k1, b = 1.2, 0.75
    for term in query_terms:
        matching = postings.get(term, {})
        if not matching:
            continue
        document_frequency = len(matching)
        inverse_frequency = math.log(
            (total - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0
        )
        for chunk_id, frequency in matching.items():
            length = len(documents[chunk_id]) or 1
            normalizer = k1 * ((1 - b) + b * (length / max(avg_length, 1.0)))
            scores[chunk_id] += inverse_frequency * ((k1 + 1) * frequency) / (normalizer + frequency)
    for chunk_id in list(scores):
        scores[chunk_id] /= max(1e-9, math.log(1 + len(documents[chunk_id])))
    maximum = max(scores.values(), default=1.0)
    return [
        (by_id[chunk_id], float(score / maximum))
        for chunk_id, score in scores.most_common(limit)
    ]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def hybrid_rank(
    lexical: list[tuple[BookChunk, float]],
    query_vector: list[float] | None,
    vectors: dict[str, list[float]],
    *,
    blend: float,
    limit: int,
) -> list[tuple[BookChunk, float]]:
    result: list[tuple[BookChunk, float]] = []
    for chunk, lexical_score in lexical:
        vector = vectors.get(chunk.id)
        if query_vector is None or vector is None:
            score = lexical_score
        else:
            semantic = 0.5 * (cosine(query_vector, vector) + 1.0)
            score = (blend * lexical_score) + ((1.0 - blend) * semantic)
        result.append((chunk, max(0.0, min(1.0, float(score)))))
    result.sort(key=lambda item: item[1], reverse=True)
    return result[:limit]
