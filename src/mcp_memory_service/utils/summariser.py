"""Memory content summarisation: extractive (zero-cost) and LLM-powered.

Provides two summarisation strategies:

1. **Extractive** (zero-cost, pure-function):
   - Lines starting with Decision:/Conclusion:/Key insight:/Summary:
   - First sentence of content
   - Truncated to MAX_SUMMARY_CHARS word boundary

2. **LLM-powered**:
   - **Anthropic** (Claude Haiku/Sonnet, size-based routing):
     - Short memories (<500 chars): Haiku (fast, cheap)
     - Long memories (≥500 chars): Sonnet (better quality)
   - **Gemini** (Flash, legacy):
     - Async HTTP call to Gemini API
   - Falls back to extractive on any error

Configuration via MCP_SUMMARY_* env vars:
- MCP_SUMMARY_MODE: 'extractive' or 'llm' (auto-detect if None)
- MCP_SUMMARY_PROVIDER: 'anthropic' or 'gemini' (default: anthropic)
- MCP_SUMMARY_ANTHROPIC_BASE_URL: Anthropic API URL (default: https://api.anthropic.com)
"""

import logging
import re

import httpx

logger = logging.getLogger(__name__)

# ~50 tokens ≈ ~200 chars (rough 1:4 token-to-char ratio for English)
MAX_SUMMARY_CHARS = 200

# Signal prefixes that indicate a summary-worthy line
_SIGNAL_PREFIXES = re.compile(
    r"^\s*(?:Decision|Conclusion|Key insight|Summary|TL;?DR|Result|Finding|Outcome)\s*:\s*",
    re.IGNORECASE,
)

# Sentence boundary: period/question/exclamation followed by space or end
# Negative lookbehind (?<!\d) prevents matching numbered lists like "1. Item"
_SENTENCE_END = re.compile(r"(?<!\d)[.!?](?:\s|$)")


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


# =============================================================================
# LLM-powered summarisation
# =============================================================================


async def anthropic_summarise(
    content: str,
    api_key: str | None,
    base_url: str = "https://api.anthropic.com",
    model_small: str = "claude-3-5-haiku-20241022",
    model_large: str = "claude-3-5-sonnet-20241022",
    size_threshold: int = 500,
    max_tokens: int = 50,
    timeout: float = 5.0,
) -> str | None:
    """Generate summary using Anthropic API with size-based model routing.

    Args:
        content: The full memory content to summarise.
        api_key: Anthropic API key (can be None if using proxy).
        base_url: Anthropic API base URL (use proxy URL for load balancing).
        model_small: Model for short memories (<size_threshold chars).
        model_large: Model for long memories (≥size_threshold chars).
        size_threshold: Character count threshold for model selection.
        max_tokens: Maximum output tokens for summary.
        timeout: HTTP request timeout in seconds.

    Returns:
        LLM-generated summary string, or None on error (caller should fall back to extractive).

    Raises:
        None - all errors are caught and logged, returning None for fallback.
    """
    if not content or not content.strip():
        return None

    # Size-based model selection
    content_len = len(content)
    model = model_small if content_len < size_threshold else model_large

    # Anthropic API endpoint
    url = f"{base_url.rstrip('/')}/v1/messages"

    # Prompt for one-sentence summary
    prompt = (
        f"Summarise this memory in one sentence (max {max_tokens} tokens). "
        f"Capture the key decision, fact, or conclusion:\n\n{content}"
    )

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    # Add API key header only if provided (proxy might not need it)
    if api_key:
        headers["x-api-key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            data = response.json()

            # Extract text from Anthropic response structure
            content_blocks = data.get("content", [])
            if not content_blocks:
                logger.warning("Anthropic summarise: no content blocks in response")
                return None

            text_block = next((block for block in content_blocks if block.get("type") == "text"), None)
            if not text_block:
                logger.warning("Anthropic summarise: no text block in content")
                return None

            summary = text_block.get("text", "").strip()
            if not summary:
                logger.warning("Anthropic summarise: empty text in response")
                return None

            logger.debug(f"Anthropic summary generated ({model}, {content_len} chars input, {len(summary)} chars output)")
            return summary

    except httpx.TimeoutException:
        logger.warning(f"Anthropic summarise timeout after {timeout}s - falling back to extractive")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Anthropic summarise HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Anthropic summarise error: {type(e).__name__}")
        return None


async def llm_summarise(
    content: str, api_key: str, model: str = "gemini-2.5-flash", max_tokens: int = 50, timeout: float = 5.0
) -> str | None:
    """Generate summary using LLM (Gemini API).

    Args:
        content: The full memory content to summarise.
        api_key: Gemini API key.
        model: Gemini model identifier (default: gemini-2.5-flash).
        max_tokens: Maximum output tokens for summary.
        timeout: HTTP request timeout in seconds.

    Returns:
        LLM-generated summary string, or None on error (caller should fall back to extractive).

    Raises:
        None - all errors are caught and logged, returning None for fallback.
    """
    if not content or not content.strip():
        return None

    if not api_key:
        logger.warning("LLM summarise called without API key - falling back to extractive")
        return None

    # Gemini API endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    # Prompt for one-sentence summary
    prompt = (
        f"Summarise this memory in one sentence (max {max_tokens} tokens). "
        f"Capture the key decision, fact, or conclusion:\n\n{content}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.3},
    }

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                params={"key": api_key},
            )
            response.raise_for_status()

            data = response.json()

            # Extract text from Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                logger.warning("LLM summarise: no candidates in response")
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                logger.warning("LLM summarise: no parts in candidate")
                return None

            summary = parts[0].get("text", "").strip()
            if not summary:
                logger.warning("LLM summarise: empty text in response")
                return None

            return summary

    except httpx.TimeoutException:
        logger.warning(f"LLM summarise timeout after {timeout}s - falling back to extractive")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"LLM summarise HTTP error {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"LLM summarise error: {type(e).__name__}")
        return None


async def summarise(content: str, client_summary: str | None = None, config=None) -> str | None:
    """Generate summary using configured mode (extractive or LLM).

    This is the main entrypoint for summary generation. It checks the config
    to determine whether to use LLM or extractive summarisation.

    Args:
        content: The full memory content to summarise.
        client_summary: Optional client-provided summary (used as-is if provided).
        config: Settings object (from config.py). If None, uses extractive mode.

    Returns:
        A summary string (~50 tokens max), or None if content is empty.
    """
    # Passthrough: client-provided summary takes precedence
    if client_summary is not None:
        return _truncate(client_summary.strip()) or None

    if not content or not content.strip():
        return None

    # Determine mode from config
    mode = "extractive"
    if config and hasattr(config, "summary"):
        mode = config.summary.get_effective_mode()

    # LLM mode: try LLM first, fall back to extractive on error
    if mode == "llm" and config:
        try:
            provider = config.summary.provider if hasattr(config.summary, "provider") else "gemini"

            if provider == "anthropic":
                # Use Anthropic with size-based routing
                api_key = None
                if config.summary.anthropic_api_key:
                    api_key = config.summary.anthropic_api_key.get_secret_value()

                summary = await anthropic_summarise(
                    content,
                    api_key=api_key,
                    base_url=config.summary.anthropic_base_url,
                    model_small=config.summary.anthropic_model_small,
                    model_large=config.summary.anthropic_model_large,
                    size_threshold=config.summary.anthropic_size_threshold,
                    max_tokens=config.summary.max_tokens,
                    timeout=config.summary.timeout_seconds,
                )
                if summary:
                    return summary

            elif provider == "gemini" and config.summary.api_key:
                # Use Gemini (legacy)
                summary = await llm_summarise(
                    content,
                    api_key=config.summary.api_key.get_secret_value(),
                    model=config.summary.model,
                    max_tokens=config.summary.max_tokens,
                    timeout=config.summary.timeout_seconds,
                )
                if summary:
                    return summary

            else:
                logger.warning(f"Unknown summary provider '{provider}', falling back to extractive")

            # Fall through to extractive on None
        except Exception as e:
            logger.warning(f"LLM summarise failed, falling back to extractive: {e}")

    # Extractive mode (or LLM fallback)
    return extract_summary(content)
