from unittest.mock import MagicMock, patch

import pytest

from api.model_router import ModelRouter
from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from api.task_classifier import ComplexityResult, TaskClassifier, TaskComplexity
from config.settings import Settings


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.enable_model_thinking = True
    settings.enable_opus_thinking = None
    settings.enable_sonnet_thinking = None
    settings.enable_haiku_thinking = None
    return settings


def test_model_router_resolves_default_model(settings):
    resolved = ModelRouter(settings).resolve("claude-3-opus")

    assert resolved.original_model == "claude-3-opus"
    assert resolved.provider_id == "nvidia_nim"
    assert resolved.provider_model == "fallback-model"
    assert resolved.provider_model_ref == "nvidia_nim/fallback-model"
    assert resolved.thinking_enabled is True


def test_model_router_applies_opus_override(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"

    request = MessagesRequest(
        model="claude-opus-4-20250514",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    assert routed.request.model == "deepseek/deepseek-r1"
    assert routed.resolved.provider_model_ref == "open_router/deepseek/deepseek-r1"
    assert routed.resolved.original_model == "claude-opus-4-20250514"
    assert routed.resolved.thinking_enabled is True
    assert request.model == "claude-opus-4-20250514"


def test_model_router_resolves_per_model_thinking(settings):
    settings.enable_model_thinking = False
    settings.enable_opus_thinking = True
    settings.enable_haiku_thinking = False

    router = ModelRouter(settings)

    assert router.resolve("claude-opus-4-20250514").thinking_enabled is True
    assert router.resolve("claude-sonnet-4-20250514").thinking_enabled is False
    assert router.resolve("claude-3-haiku-20240307").thinking_enabled is False
    assert router.resolve("claude-2.1").thinking_enabled is False


def test_model_router_applies_haiku_override(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-7b"
    assert routed.resolved.provider_model_ref == "lmstudio/qwen2.5-7b"


def test_model_router_applies_sonnet_override(settings):
    settings.model_sonnet = "nvidia_nim/meta/llama-3.3-70b-instruct"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "meta/llama-3.3-70b-instruct"
    assert (
        routed.resolved.provider_model_ref == "nvidia_nim/meta/llama-3.3-70b-instruct"
    )


def test_model_router_routes_prefixed_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="deepseek/deepseek-chat",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-chat"
    assert routed.resolved.original_model == "deepseek/deepseek-chat"
    assert routed.resolved.provider_id == "deepseek"
    assert routed.resolved.provider_model == "deepseek-chat"
    assert routed.resolved.provider_model_ref == "deepseek/deepseek-chat"


def test_model_router_routes_wafer_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="wafer/DeepSeek-V4-Pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "DeepSeek-V4-Pro"
    assert routed.resolved.provider_id == "wafer"
    assert routed.resolved.provider_model == "DeepSeek-V4-Pro"
    assert routed.resolved.provider_model_ref == "wafer/DeepSeek-V4-Pro"


def test_model_router_routes_gateway_encoded_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.provider_id == "nvidia_nim"
    assert routed.resolved.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.provider_model_ref
        == "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )


def test_model_router_routes_no_thinking_gateway_model_directly(settings):
    settings.enable_model_thinking = True

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.provider_id == "nvidia_nim"
    assert routed.resolved.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert routed.resolved.thinking_enabled is False


def test_model_router_direct_prefixed_model_uses_provider_model_for_thinking(settings):
    settings.enable_model_thinking = False
    settings.enable_opus_thinking = True

    resolved = ModelRouter(settings).resolve("open_router/anthropic/claude-opus-4")

    assert resolved.provider_id == "open_router"
    assert resolved.provider_model == "anthropic/claude-opus-4"
    assert resolved.thinking_enabled is True


def test_model_router_routes_token_count_request(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    request = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_token_count_request(request)

    assert routed.request.model == "qwen2.5-7b"
    assert request.model == "claude-3-haiku-20240307"


def test_model_router_logs_mapping(settings):
    with patch("api.model_router.logger.debug") as mock_log:
        ModelRouter(settings).resolve("claude-2.1")

    mock_log.assert_called()
    args = mock_log.call_args[0]
    assert "MODEL MAPPING" in args[0]
    assert args[1] == "claude-2.1"
    assert args[2] == "fallback-model"


# =============================================================================
# AUTO_ROUTE tests
# =============================================================================


def test_auto_route_routes_simple_task_to_sonnet_tier(settings):
    """SIMPLE task routes to the sonnet-tier (flash) model."""
    settings.auto_route_enabled = True
    settings.model_sonnet = "deepseek/deepseek-v4-flash"
    settings.model_opus = "deepseek/deepseek-v4-pro"
    router = ModelRouter(settings)

    with patch.object(router, "_get_classifier") as mock_get:
        mock_cls = MagicMock()
        mock_cls.classify.return_value = ComplexityResult(
            complexity=TaskComplexity.SIMPLE,
            classifier_model="deepseek-v4-flash",
            latency_ms=100.0,
        )
        mock_get.return_value = mock_cls

        resolved = router.resolve_with_classification(
            "claude-sonnet-4-20250514", "What is 2+2?"
        )

    assert resolved.provider_model == "deepseek-v4-flash"
    assert resolved.provider_id == "deepseek"


def test_auto_route_routes_complex_task_to_opus_tier(settings):
    """COMPLEX task routes to the opus-tier (pro) model."""
    settings.auto_route_enabled = True
    settings.model_sonnet = "deepseek/deepseek-v4-flash"
    settings.model_opus = "deepseek/deepseek-v4-pro"
    router = ModelRouter(settings)

    with patch.object(router, "_get_classifier") as mock_get:
        mock_cls = MagicMock()
        mock_cls.classify.return_value = ComplexityResult(
            complexity=TaskComplexity.COMPLEX,
            classifier_model="deepseek-v4-flash",
            latency_ms=150.0,
        )
        mock_get.return_value = mock_cls

        resolved = router.resolve_with_classification(
            "claude-sonnet-4-20250514", "Refactor auth module to use JWT"
        )

    assert resolved.provider_model == "deepseek-v4-pro"
    assert resolved.provider_id == "deepseek"


def test_auto_route_routes_very_complex_task_to_opus_tier(settings):
    """VERY_COMPLEX task also routes to the opus-tier (pro) model."""
    settings.auto_route_enabled = True
    settings.model_sonnet = "deepseek/deepseek-v4-flash"
    settings.model_opus = "deepseek/deepseek-v4-pro"
    router = ModelRouter(settings)

    with patch.object(router, "_get_classifier") as mock_get:
        mock_cls = MagicMock()
        mock_cls.classify.return_value = ComplexityResult(
            complexity=TaskComplexity.VERY_COMPLEX,
            classifier_model="deepseek-v4-flash",
            latency_ms=200.0,
        )
        mock_get.return_value = mock_cls

        resolved = router.resolve_with_classification(
            "claude-sonnet-4-20250514", "Design a distributed cache system"
        )

    assert resolved.provider_model == "deepseek-v4-pro"
    assert resolved.provider_id == "deepseek"
    assert resolved.original_model == "claude-opus-4-20250514"


def test_auto_route_falls_back_to_default_when_disabled(settings):
    """When auto_route_enabled=False, resolve_with_classification == resolve()."""
    settings.auto_route_enabled = False
    settings.model = "nvidia_nim/fallback-model"
    router = ModelRouter(settings)

    resolved = router.resolve_with_classification(
        "claude-sonnet-4-20250514", "Some task"
    )

    assert resolved.provider_model == "fallback-model"
    assert resolved.provider_id == "nvidia_nim"
    assert resolved.original_model == "claude-sonnet-4-20250514"


def test_auto_route_lazy_inits_classifier(settings):
    """TaskClassifier is only created when resolve_with_classification is called."""
    settings.auto_route_enabled = True
    settings.model_sonnet = "deepseek/deepseek-v4-flash"
    router = ModelRouter(settings)

    # Before any classification call, _task_classifier should be None
    assert router._task_classifier is None

    # Calling resolve (without classification) should not init the classifier
    router.resolve("claude-sonnet-4-20250514")
    assert router._task_classifier is None

    # Calling resolve_with_classification should lazy-init
    with (
        patch.object(router, "_get_classifier", wraps=router._get_classifier) as spy,
        patch.object(TaskClassifier, "classify") as mock_classify,
    ):
        mock_classify.return_value = ComplexityResult(
            complexity=TaskComplexity.SIMPLE,
            classifier_model="deepseek-v4-flash",
            latency_ms=50.0,
        )
        router.resolve_with_classification("claude-sonnet-4-20250514", "test")
        spy.assert_called_once()
