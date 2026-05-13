"""Tests for the TaskClassifier AUTO_ROUTE service."""

from unittest.mock import MagicMock, patch

import pytest

from api.task_classifier import (
    CLASSIFICATION_PROMPT,
    ComplexityResult,
    TaskClassifier,
    TaskComplexity,
)
from config.settings import Settings


@pytest.fixture
def settings():
    s = Settings()
    s.auto_route_enabled = True
    s.auto_route_classifier_model = "deepseek/deepseek-v4-flash"
    s.deepseek_api_key = "test-key"
    return s


# =============================================================================
# Structure tests (no API call)
# =============================================================================


def test_classifier_returns_complexity_result(settings):
    """Classifier returns a valid ComplexityResult structure."""
    classifier = TaskClassifier(settings)
    with patch.object(classifier, "_http_client") as mock_http:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "SIMPLE"}]
        }
        mock_http.post.return_value = mock_response

        result = classifier.classify("What is the capital of France?")

    assert result.complexity == TaskComplexity.SIMPLE
    assert isinstance(result.latency_ms, (int, float))
    assert result.latency_ms > 0
    assert result.classifier_model == "deepseek-v4-flash"
    classifier.close()


def test_classifier_routes_complex_to_flash_model(settings):
    """Classifier should call DeepSeek API with flash model."""
    classifier = TaskClassifier(settings)
    with patch.object(classifier, "_http_client") as mock_http:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "COMPLEX"}]
        }
        mock_http.post.return_value = mock_response

        result = classifier.classify("Refactor the auth module")

    # Verify the API call used the flash model
    call_kwargs = mock_http.post.call_args[1]
    assert call_kwargs["json"]["model"] == "deepseek-v4-flash"
    assert result.complexity == TaskComplexity.COMPLEX
    classifier.close()


# =============================================================================
# Parsing tests
# =============================================================================


@pytest.mark.parametrize(
    "api_response_text,expected",
    [
        ("SIMPLE", TaskComplexity.SIMPLE),
        ("COMPLEX", TaskComplexity.COMPLEX),
        ("VERY_COMPLEX", TaskComplexity.VERY_COMPLEX),
        (" simple ", TaskComplexity.SIMPLE),
        ("complex", TaskComplexity.COMPLEX),
        ("very_complex", TaskComplexity.VERY_COMPLEX),
        ("SIMPLE.", TaskComplexity.SIMPLE),
        ("I think this is COMPLEX", TaskComplexity.COMPLEX),
        ("", TaskComplexity.COMPLEX),  # empty => fallback
        ("UNKNOWN", TaskComplexity.COMPLEX),  # unknown => fallback
    ],
)
def test_parse_complexity(api_response_text, expected, settings):
    """_parse_complexity extracts the correct complexity from API responses."""
    data = {"content": [{"type": "text", "text": api_response_text}]}
    result = TaskClassifier._parse_complexity(data)
    assert result == expected


def test_parse_complexity_falls_back_on_missing_content(settings):
    """Empty content list falls back to COMPLEX."""
    data = {"content": []}
    result = TaskClassifier._parse_complexity(data)
    assert result == TaskComplexity.COMPLEX


def test_parse_complexity_skips_non_text_blocks(settings):
    """Non-text content blocks are skipped when looking for classification."""
    data = {
        "content": [
            {"type": "thinking", "text": "SIMPLE"},
            {"type": "text", "text": "COMPLEX"},
        ]
    }
    result = TaskClassifier._parse_complexity(data)
    assert result == TaskComplexity.COMPLEX


# =============================================================================
# Error handling tests
# =============================================================================


def test_classifier_fallback_on_timeout(settings):
    """HTTP timeout should fall back to COMPLEX."""
    classifier = TaskClassifier(settings)
    with patch.object(classifier, "_http_client") as mock_http:
        import httpx

        mock_http.post.side_effect = httpx.TimeoutException(
            "timeout", request=MagicMock()
        )

        result = classifier.classify("Simple question")

    assert result.complexity == TaskComplexity.COMPLEX  # fallback
    classifier.close()


def test_classifier_fallback_on_http_error(settings):
    """HTTP error should fall back to COMPLEX."""
    classifier = TaskClassifier(settings)
    with patch.object(classifier, "_http_client") as mock_http:
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403 Forbidden", request=MagicMock(), response=MagicMock()
        )
        mock_http.post.return_value = mock_response

        result = classifier.classify("Simple question")

    assert result.complexity == TaskComplexity.COMPLEX  # fallback
    classifier.close()


def test_classifier_fallback_on_invalid_api_key(settings):
    """Invalid API key should fall back to COMPLEX, not crash."""
    settings.deepseek_api_key = "invalid-key-that-will-fail"
    classifier = TaskClassifier(settings)
    # We can't mock the actual HTTP call here since the key is invalid
    # but the classifier should handle the error gracefully
    with patch.object(classifier, "_http_client") as mock_http:
        mock_http.post.side_effect = Exception("API error")
        result = classifier.classify("Simple question")
    assert result.complexity == TaskComplexity.COMPLEX
    classifier.close()


# =============================================================================
# Prompt tests
# =============================================================================


def test_classification_prompt_contains_messages_text():
    """The prompt format string includes the messages_text placeholder."""
    prompt = CLASSIFICATION_PROMPT.format(messages_text="test")
    assert "test" in prompt
    assert "SIMPLE" in prompt
    assert "COMPLEX" in prompt
    assert "VERY_COMPLEX" in prompt


def test_classifier_truncates_long_input(settings):
    """Input longer than 2000 chars should be truncated before API call."""
    classifier = TaskClassifier(settings)
    long_text = "hello " * 500  # 3000 chars

    with patch.object(classifier, "_http_client") as mock_http:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": [{"type": "text", "text": "SIMPLE"}]
        }
        mock_http.post.return_value = mock_response

        result = classifier.classify(long_text)

    assert result.complexity == TaskComplexity.SIMPLE
    # Verify the prompt was truncated to 2000 chars
    call_kwargs = mock_http.post.call_args[1]
    prompt_sent = call_kwargs["json"]["messages"][0]["content"]
    assert len(prompt_sent) <= 2000 + len("...") + len(CLASSIFICATION_PROMPT.split("{messages_text}")[0]) + len(CLASSIFICATION_PROMPT.split("{messages_text}")[1])
    classifier.close()


# =============================================================================
# API key resolution tests
# =============================================================================


def test_resolve_api_key_deepseek(settings):
    """DeepSeek provider uses deepseek_api_key."""
    key = TaskClassifier._resolve_api_key(settings, "deepseek")
    assert key == "test-key"


def test_resolve_api_key_openrouter(settings):
    """OpenRouter provider uses open_router_api_key."""
    settings.open_router_api_key = "or-key"
    key = TaskClassifier._resolve_api_key(settings, "open_router")
    assert key == "or-key"


def test_resolve_api_key_unknown(settings):
    """Unknown provider returns empty key."""
    key = TaskClassifier._resolve_api_key(settings, "unknown")
    assert key == ""
