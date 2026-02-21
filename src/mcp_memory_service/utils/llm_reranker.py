"""
Optional LLM-based re-ranking for search results.

Uses Anthropic Haiku via anthropic-lb to re-score search candidates
by inferred relevance to the user's query intent.

Non-fatal: all errors return empty results, letting the existing ranking pass through.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMReranker(Protocol):
    """Protocol for pluggable LLM re-rankers."""

    async def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[str, float]]:
        """Re-rank candidates by inferred relevance to the query."""


class AnthropicReranker:
    def __init__(self, client: Any, model: str, timeout_ms: int = 2000):
        self._client = client
        self._model = model
        self._timeout_ms = timeout_ms

    async def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[tuple[str, float]]:
        if not candidates:
            return []

        try:
            candidate_text = "\n".join(
                f"- [{c['content_hash'][:12]}]: {c.get('summary', c.get('content', '')[:100])}" for c in candidates
            )

            prompt = (
                f'Given this search query: "{query}"\n\n'
                f"Rate the relevance of each memory (0.0 = irrelevant, 1.0 = highly relevant):\n\n"
                f"{candidate_text}\n\n"
                f"Return ONLY a JSON array of objects with 'hash' and 'score' fields. "
                f"Use the hash prefixes shown in brackets. Example: "
                f'[{{"hash": "abc123", "score": 0.8}}]'
            )

            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._timeout_ms / 1000.0,
            )

            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            scores_data = json.loads(text)
            hash_map = {c["content_hash"][:12]: c["content_hash"] for c in candidates}

            results = []
            for item in scores_data:
                short_hash = item.get("hash", "")
                score = float(item.get("score", 0.0))
                full_hash = hash_map.get(short_hash)
                if full_hash:
                    results.append((full_hash, min(1.0, max(0.0, score))))

            return results

        except Exception as e:
            logger.warning(f"LLM re-ranking failed (non-fatal): {e}")
            return []
