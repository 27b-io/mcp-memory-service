"""Tests for semantic tag matching via embedding k-NN."""

from mcp_memory_service.utils.tag_embeddings import (
    build_tag_embedding_index,
    find_semantic_tags,
)


class TestBuildTagEmbeddingIndex:
    def test_builds_index_from_tags_and_embeddings(self):
        tags = ["python", "redis", "docker"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        index = build_tag_embedding_index(tags, embeddings)
        assert index["tags"] == ["python", "redis", "docker"]
        assert index["matrix"].shape == (3, 3)

    def test_empty_tags_returns_empty_index(self):
        index = build_tag_embedding_index([], [])
        assert index["tags"] == []
        assert index["matrix"].shape[0] == 0


class TestFindSemanticTags:
    def test_finds_similar_tags(self):
        tags = ["python", "redis", "docker"]
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [0.9, 0.1, 0.0]
        matches = find_semantic_tags(query_embedding, index, threshold=0.5, max_tags=5)
        assert "python" in matches
        assert "docker" not in matches

    def test_respects_threshold(self):
        tags = ["python", "redis"]
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [0.7, 0.7]
        matches = find_semantic_tags(query_embedding, index, threshold=0.99, max_tags=5)
        assert len(matches) == 0

    def test_respects_max_tags(self):
        tags = [f"tag{i}" for i in range(20)]
        embeddings = [[1.0, 0.0] for _ in range(20)]
        index = build_tag_embedding_index(tags, embeddings)
        query_embedding = [1.0, 0.0]
        matches = find_semantic_tags(query_embedding, index, threshold=0.0, max_tags=5)
        assert len(matches) == 5

    def test_empty_index_returns_empty(self):
        index = build_tag_embedding_index([], [])
        matches = find_semantic_tags([1.0, 0.0], index, threshold=0.0, max_tags=5)
        assert matches == []
