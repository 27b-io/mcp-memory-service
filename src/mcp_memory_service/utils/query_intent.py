"""
Query intent extraction for multi-vector search fan-out.

Decomposes natural language queries into concept sub-queries using spaCy NLP.
Falls back to simple keyword extraction if spaCy is unavailable.

Lazy-loaded singleton pattern (matches emotional_analysis.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_FALLBACK_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "my",
        "your",
        "our",
        "their",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "about",
        "up",
        "out",
        "if",
        "then",
        "so",
        "than",
        "too",
        "very",
        "just",
        "also",
        "any",
        "each",
        "every",
        "all",
        "some",
        "me",
        "i",
    }
)
_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


@dataclass
class QueryIntentResult:
    """Result of query intent analysis."""

    original_query: str
    sub_queries: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)


@runtime_checkable
class QueryIntentAnalyzer(Protocol):
    """Protocol for pluggable query intent analyzers."""

    def analyze(self, query: str) -> QueryIntentResult:
        """Analyze a query and return intent decomposition."""


class SpaCyAnalyzer:
    """spaCy-based concept extractor. Lazy-loads model as class-level singleton."""

    _nlp = None

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        max_sub_queries: int = 4,
        min_query_tokens: int = 3,
    ):
        self._model_name = model_name
        self._max_sub_queries = max_sub_queries
        self._min_query_tokens = min_query_tokens

    def _ensure_model(self) -> bool:
        """Lazy-load spaCy model. Returns False if model unavailable."""
        if SpaCyAnalyzer._nlp is None:
            try:
                import spacy

                SpaCyAnalyzer._nlp = spacy.load(self._model_name)
                logger.info("Loaded spaCy model: %s", self._model_name)
            except (ImportError, OSError) as e:
                logger.warning("spaCy model '%s' unavailable: %s", self._model_name, e)
                return False
        return True

    def analyze(self, query: str) -> QueryIntentResult:
        query = query.strip()
        if not query:
            return QueryIntentResult(original_query=query, sub_queries=[query])

        tokens = [t for t in _TOKEN_PATTERN.split(query.lower()) if t and t not in _FALLBACK_STOP_WORDS and len(t) > 1]
        if len(tokens) < self._min_query_tokens:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        if not self._ensure_model():
            # Model unavailable — fall back to no fan-out
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])
        doc = SpaCyAnalyzer._nlp(query)

        concepts: list[str] = []
        seen: set[str] = set()

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            lower = text.lower()
            if lower not in seen and len(text) > 1:
                seen.add(lower)
                concepts.append(text)

        for ent in doc.ents:
            text = ent.text.strip()
            lower = text.lower()
            if lower not in seen and len(text) > 1:
                seen.add(lower)
                concepts.append(text)

        covered_tokens = set()
        for c in concepts:
            for t in _TOKEN_PATTERN.split(c.lower()):
                if t:
                    covered_tokens.add(t)

        for token in doc:
            if (
                token.text.lower() not in covered_tokens
                and token.text.lower() not in _FALLBACK_STOP_WORDS
                and not token.is_punct
                and not token.is_space
                and len(token.text) > 1
                and token.text.lower() not in seen
            ):
                seen.add(token.text.lower())
                concepts.append(token.text)

        concepts = concepts[: self._max_sub_queries]

        sub_queries: list[str] = []
        seen_queries: set[str] = set()
        for concept in concepts:
            lower = concept.lower()
            if lower not in seen_queries:
                seen_queries.add(lower)
                sub_queries.append(concept)

        if query.lower() not in seen_queries:
            sub_queries.append(query)

        if len(sub_queries) <= 1:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        return QueryIntentResult(original_query=query, sub_queries=sub_queries, concepts=concepts)


class FallbackAnalyzer:
    """Simple keyword-based fallback when spaCy is unavailable."""

    def __init__(self, min_query_tokens: int = 3, max_sub_queries: int = 4):
        self._min_query_tokens = min_query_tokens
        self._max_sub_queries = max_sub_queries

    def analyze(self, query: str) -> QueryIntentResult:
        query = query.strip()
        if not query:
            return QueryIntentResult(original_query=query, sub_queries=[query])

        tokens = [t for t in _TOKEN_PATTERN.split(query.lower()) if t and t not in _FALLBACK_STOP_WORDS and len(t) > 1]
        if len(tokens) < self._min_query_tokens:
            return QueryIntentResult(original_query=query, sub_queries=[query], concepts=[])

        concepts = tokens[: self._max_sub_queries]
        sub_queries = list(dict.fromkeys(concepts))
        if query.lower() not in {q.lower() for q in sub_queries}:
            sub_queries.append(query)

        return QueryIntentResult(original_query=query, sub_queries=sub_queries, concepts=concepts)


_analyzer: QueryIntentAnalyzer | None = None


def get_analyzer(
    model_name: str = "en_core_web_sm",
    max_sub_queries: int = 4,
    min_query_tokens: int = 3,
) -> QueryIntentAnalyzer:
    """Get or create the query intent analyzer singleton."""
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    try:
        import spacy  # noqa: F401

        _analyzer = SpaCyAnalyzer(
            model_name=model_name,
            max_sub_queries=max_sub_queries,
            min_query_tokens=min_query_tokens,
        )
        logger.info("Query intent: using SpaCyAnalyzer")
    except ImportError:
        _analyzer = FallbackAnalyzer(min_query_tokens=min_query_tokens, max_sub_queries=max_sub_queries)
        logger.info("Query intent: spaCy not available, using FallbackAnalyzer")

    return _analyzer
