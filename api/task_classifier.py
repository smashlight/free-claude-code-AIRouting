"""Fast pre-flight task complexity classifier via DeepSeek API.

AUTO_ROUTE uses this module to classify incoming coding tasks before
routing them to the appropriate model tier (flash for simple, pro for complex).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

import httpx
from loguru import logger

from config.settings import Settings

CLASSIFICATION_PROMPT = """\
Classify this coding task complexity. Respond with ONLY one word: SIMPLE, COMPLEX, or VERY_COMPLEX.

Criteria:
- SIMPLE: Chat, questions, file reads, simple search, one-line fixes, documentation lookups
- COMPLEX: Multi-file refactoring, architecture changes, debugging, code generation with tools, security-sensitive changes
- VERY_COMPLEX: System design, complex algorithms, performance optimization, large-scale refactoring

Task:
{messages_text}

Response (one word):"""


class TaskComplexity(StrEnum):
    """Complexity levels for coding tasks."""

    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"
    VERY_COMPLEX = "VERY_COMPLEX"


_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a task complexity classifier. "
    "Respond with ONLY one word: SIMPLE, COMPLEX, or VERY_COMPLEX. "
    "Do NOT include any reasoning, thinking, or explanation."
)


@dataclass(frozen=True)
class ComplexityResult:
    """Result of a task complexity classification call."""

    complexity: TaskComplexity
    classifier_model: str
    latency_ms: float


class TaskClassifier:
    """Fast pre-flight task complexity classifier via DeepSeek API.

    Makes a synchronous HTTP call to the classifier model (default:
    deepseek-v4-flash) with a tiny prompt to determine task complexity.

    Falls back to COMPLEX on any error to avoid routing a simple task
    to the wrong model when the classifier is unavailable.

    Usage:
        classifier = TaskClassifier(settings)
        result = classifier.classify("What is 2+2?")
        print(result.complexity)  # TaskComplexity.SIMPLE
        classifier.close()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        model_ref = settings.auto_route_classifier_model
        self._classifier_model = Settings.parse_model_name(model_ref)
        self._classifier_provider = Settings.parse_provider_type(model_ref)
        self._classifier_api_key = self._resolve_api_key(
            settings, self._classifier_provider
        )
        self._base_url = self._resolve_base_url(self._classifier_provider)
        self._http_client = httpx.Client(timeout=10.0)

    @staticmethod
    def _resolve_api_key(settings: Settings, provider: str) -> str:
        """Get the API key for the classifier provider."""
        if provider == "deepseek":
            return settings.deepseek_api_key
        if provider == "open_router":
            return settings.open_router_api_key
        return ""

    @staticmethod
    def _resolve_base_url(provider: str) -> str:
        """Get the base URL for the classifier provider's Anthropic API."""
        if provider == "deepseek":
            return "https://api.deepseek.com/anthropic"
        if provider == "open_router":
            return "https://openrouter.ai/api/v1"
        return ""

    def classify(self, messages_text: str) -> ComplexityResult:
        """Classify task complexity from extracted message text.

        Args:
            messages_text: Concatenated recent messages for classification.

        Returns:
            A ComplexityResult with the predicted complexity level,
            classifier model name, and latency in milliseconds.
        """
        start = time.monotonic()

        prompt = CLASSIFICATION_PROMPT.format(messages_text=messages_text[:2000])

        try:
            if self._classifier_provider == "deepseek":
                complexity = self._classify_via_deepseek(prompt)
            else:
                logger.warning(
                    "TaskClassifier: unsupported provider '{}', falling back to COMPLEX",
                    self._classifier_provider,
                )
                complexity = TaskComplexity.COMPLEX
        except Exception as exc:
            logger.warning(
                "TaskClassifier failed: {} type={} — falling back to COMPLEX",
                exc,
                type(exc).__name__,
            )
            complexity = TaskComplexity.COMPLEX

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "TaskClassifier: complexity={} latency={:.1f}ms model={}",
            complexity,
            elapsed_ms,
            self._classifier_model,
        )
        return ComplexityResult(
            complexity=complexity,
            classifier_model=self._classifier_model,
            latency_ms=round(elapsed_ms, 1),
        )

    def _classify_via_deepseek(self, prompt: str) -> TaskComplexity:
        """Send classification prompt to DeepSeek Anthropic-compatible API."""
        response = self._http_client.post(
            f"{self._base_url}/v1/messages",
            headers={
                "x-api-key": self._classifier_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self._classifier_model,
                "max_tokens": 300,
                "temperature": 0,
                "system": _CLASSIFIER_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return self._parse_complexity(response.json())

    @staticmethod
    def _parse_complexity(data: dict) -> TaskComplexity:
        """Parse DeepSeek response content for complexity classification.

        Searches ``text`` blocks first, then falls back to ``thinking``
        blocks in case the model didn't finish its text output before
        hitting ``max_tokens``.
        """
        for block in data.get("content", []):
            if block.get("type") == "text":
                text = block.get("text", "").strip().upper()
                result = TaskClassifier._match_complexity(text)
                if result is not None:
                    return result

        for block in data.get("content", []):
            if block.get("type") == "thinking":
                text = block.get("thinking", "").strip().upper()
                result = TaskClassifier._match_complexity(text)
                if result is not None:
                    return result

        return TaskComplexity.COMPLEX

    @staticmethod
    def _match_complexity(text: str) -> TaskComplexity | None:
        """Check if the given text contains a complexity keyword."""
        if "SIMPLE" in text:
            return TaskComplexity.SIMPLE
        if "VERY_COMPLEX" in text:
            return TaskComplexity.VERY_COMPLEX
        if "COMPLEX" in text:
            return TaskComplexity.COMPLEX
        return None

    def close(self) -> None:
        """Close the underlying HTTP client to release connections."""
        self._http_client.close()
