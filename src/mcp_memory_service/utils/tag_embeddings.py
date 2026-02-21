"""Semantic tag matching via embedding k-NN.

Builds an in-memory index of tag embeddings and finds semantically
similar tags for a given query embedding using cosine similarity.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class TagEmbeddingIndex(TypedDict):
    tags: list[str]
    matrix: NDArray[np.float32]


def build_tag_embedding_index(
    tags: list[str],
    embeddings: list[list[float]],
) -> TagEmbeddingIndex:
    """Build a normalised tag embedding index for k-NN search."""
    if not tags:
        return {"tags": [], "matrix": np.empty((0, 0))}

    if len(tags) != len(embeddings):
        raise ValueError(f"tags ({len(tags)}) and embeddings ({len(embeddings)}) must have same length")

    matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix = matrix / norms

    return {"tags": list(tags), "matrix": matrix}


def find_semantic_tags(
    query_embedding: list[float] | NDArray,
    index: TagEmbeddingIndex,
    threshold: float = 0.5,
    max_tags: int = 10,
) -> list[str]:
    """Find tags semantically similar to a query embedding."""
    if not index["tags"]:
        return []

    query = np.array(query_embedding, dtype=np.float32)
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    query = query / norm

    similarities = index["matrix"] @ query

    mask = similarities >= threshold
    if not mask.any():
        return []

    indices = np.where(mask)[0]
    scores = similarities[indices]
    top_indices = indices[np.argsort(-scores)[:max_tags]]

    return [index["tags"][i] for i in top_indices]
