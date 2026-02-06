"""Extractive summariser for memory content.

Zero-cost, pure-function approach: extracts a one-line summary (~50 tokens)
from memory content using simple heuristics. No LLM, no ML inference.

Strategy (in priority order):
1. Client-provided summary (passthrough, skips extraction)
2. Lines starting with Decision:/Conclusion:/Key insight:/Summary:
3. First sentence of content
4. Truncate to MAX_SUMMARY_TOKENS word boundary
"""

import re

# ~50 tokens ≈ ~200 chars (rough 1:4 token-to-char ratio for English)
MAX_SUMMARY_CHARS = 200

# Signal prefixes that indicate a summary-worthy line
_SIGNAL_PREFIXES = re.compile(
    r"^\s*(?:Decision|Conclusion|Key insight|Summary|TL;?DR|Result|Finding|Outcome)\s*:\s*",
    re.IGNORECASE,
)

# Sentence boundary: period/question/exclamation followed by space or end
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def extract_summary(content: str, client_summary: str | None = None) -> str | None:
    """Extract a one-line summary from memory content.

    Args:
        content: The full memory content to summarise.
        client_summary: Optional client-provided summary (used as-is if provided).

    Returns:
        A summary string (~50 tokens max), or None if content is empty.
    """
    # Passthrough: client-provided summary takes precedence
    if client_summary is not None:
        return _truncate(client_summary.strip()) or None

    if not content or not content.strip():
        return None

    # Strategy 1: Look for signal lines (Decision:, Conclusion:, etc.)
    signal = _find_signal_line(content)
    if signal:
        return _truncate(signal)

    # Strategy 2: First sentence
    first = _first_sentence(content)
    return _truncate(first)


def _find_signal_line(content: str) -> str | None:
    """Find the first line matching a signal prefix pattern."""
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _SIGNAL_PREFIXES.match(line)
        if match:
            # Return the content after the prefix
            return line[match.end() :].strip() or line
    return None


def _first_sentence(content: str) -> str:
    """Extract the first sentence from content."""
    # Collapse whitespace for clean extraction
    text = " ".join(content.split())

    match = _SENTENCE_END.search(text)
    if match:
        # Include the punctuation
        return text[: match.end()].strip()

    # No sentence boundary found — return the whole thing (will be truncated)
    return text


def _truncate(text: str) -> str:
    """Truncate text to MAX_SUMMARY_CHARS on a word boundary."""
    if len(text) <= MAX_SUMMARY_CHARS:
        return text

    # Find last space before limit
    truncated = text[:MAX_SUMMARY_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > MAX_SUMMARY_CHARS // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip() + "..."
