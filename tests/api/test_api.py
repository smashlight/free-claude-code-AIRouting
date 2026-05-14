from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from providers.nvidia_nim import NvidiaNimProvider

app = create_app()

# Mock provider
mock_provider = MagicMock(spec=NvidiaNimProvider)

# Track stream_response calls for test_model_mapping
_stream_response_calls: list = []


async def _mock_stream_response(*args, **kwargs):
    """Minimal async generator for streaming tests."""
    _stream_response_calls.append((args, kwargs))
    yield "event: message_start\ndata: {}\n\n"
    yield "[DONE]\n\n"


mock_provider.stream_response = _mock_stream_response


@pytest.fixture(scope="module")
def client():
    """HTTP client with provider resolution stubbed; patch only for this file."""
    with (
        patch("api.dependencies.resolve_provider", return_value=mock_provider),
        patch(
            "providers.registry.ProviderRegistry.validate_configured_models",
            new_callable=AsyncMock,
        ),
        patch("providers.registry.ProviderRegistry.start_model_list_refresh"),
        TestClient(app) as test_client,
    ):
        yield test_client


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models_list(client: TestClient):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["has_more"] is False
    ids = [item["id"] for item in data["data"]]
    assert "claude-sonnet-4-20250514" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]


def test_probe_endpoints_return_204_with_allow_headers(client: TestClient):
    responses = [
        client.head("/"),
        client.options("/"),
        client.head("/health"),
        client.options("/health"),
        client.head("/v1/messages"),
        client.options("/v1/messages"),
        client.head("/v1/messages/count_tokens"),
        client.options("/v1/messages/count_tokens"),
    ]

    for response in responses:
        assert response.status_code == 204
        assert "Allow" in response.headers


def test_create_message_stream(client: TestClient):
    """Create message returns streaming response."""
    payload = {
        "model": "claude-3-sonnet",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    response = client.post("/v1/messages", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = b"".join(response.iter_bytes())
    assert b"message_start" in content or b"event:" in content


def test_model_mapping(client: TestClient):
    # Test Haiku mapping
    _stream_response_calls.clear()
    payload_haiku = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "stream": True,
    }
    client.post("/v1/messages", json=payload_haiku)
    assert len(_stream_response_calls) == 1
    args = _stream_response_calls[0][0]
    kwargs = _stream_response_calls[0][1]
    assert args[0].model != "claude-3-haiku-20240307"
    assert kwargs["thinking_enabled"] is False


def test_error_fallbacks(client: TestClient):
    from providers.exceptions import (
        AuthenticationError,
        OverloadedError,
        RateLimitError,
    )

    base_payload = {
        "model": "test",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "stream": True,
    }

    def _raise_auth(*args, **kwargs):
        raise AuthenticationError("Invalid Key")

    def _raise_rate_limit(*args, **kwargs):
        raise RateLimitError("Too Many Requests")

    def _raise_overloaded(*args, **kwargs):
        raise OverloadedError("Server Overloaded")

    # 1. Authentication Error (401)
    mock_provider.stream_response = _raise_auth
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"

    # 2. Rate Limit (429)
    mock_provider.stream_response = _raise_rate_limit
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"

    # 3. Overloaded (529)
    mock_provider.stream_response = _raise_overloaded
    response = client.post("/v1/messages", json=base_payload)
    assert response.status_code == 529
    assert response.json()["error"]["type"] == "overloaded_error"

    # Reset for subsequent tests
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_returns_500(client: TestClient):
    """Non-ProviderError exceptions are caught and returned as HTTPException(500)."""

    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("unexpected crash")

    mock_provider.stream_response = _raise_runtime
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_with_status_code(client: TestClient):
    """Unexpected errors always map to HTTP 500 (ignore ad-hoc status_code attrs)."""

    class ExceptionWithStatus(RuntimeError):
        def __init__(self, msg: str, status_code: int = 500):
            super().__init__(msg)
            self.status_code = status_code

    def _raise_with_status(*args, **kwargs):
        raise ExceptionWithStatus("bad gateway", 502)

    mock_provider.stream_response = _raise_with_status
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    mock_provider.stream_response = _mock_stream_response


def test_generic_exception_empty_message_returns_non_empty_detail(client: TestClient):
    """Exceptions with empty __str__ still return a readable HTTP detail."""

    class SilentError(RuntimeError):
        def __str__(self):
            return ""

    def _raise_silent(*args, **kwargs):
        raise SilentError()

    mock_provider.stream_response = _raise_silent
    response = client.post(
        "/v1/messages",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    assert response.status_code == 500
    assert response.json()["detail"] != ""
    mock_provider.stream_response = _mock_stream_response


def test_count_tokens_endpoint(client: TestClient):
    """count_tokens endpoint returns token count."""
    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    assert "input_tokens" in response.json()


def test_stop_endpoint_no_handler_no_cli_503(client: TestClient):
    """POST /stop without handler or cli_manager returns 503."""
    # Ensure no handler or cli_manager on app state
    if hasattr(app.state, "message_handler"):
        delattr(app.state, "message_handler")
    if hasattr(app.state, "cli_manager"):
        delattr(app.state, "cli_manager")
    response = client.post("/stop")
    assert response.status_code == 503


# =============================================================================
# _extract_messages_text tests
# =============================================================================


def test_extract_messages_text_empty():
    """Empty messages produce empty string."""
    from api.services import _extract_messages_text

    assert _extract_messages_text([]) == ""


def test_extract_messages_text_simple_content():
    """String content messages are extracted correctly."""
    from api.models.anthropic import Message
    from api.services import _extract_messages_text

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="What is the capital of France?"),
    ]
    text = _extract_messages_text(messages)
    assert "Hello" in text
    assert "Hi there!" in text
    assert "What is the capital" in text
    assert text.count("user:") == 2
    assert text.count("assistant:") == 1


def test_extract_messages_text_respects_max_messages():
    """Only the last N messages are included."""
    from api.models.anthropic import Message
    from api.services import _extract_messages_text

    messages = [Message(role="user", content=str(i)) for i in range(10)]
    text = _extract_messages_text(messages, max_messages=3)
    assert "7" in text
    assert "9" in text
    # "0" should be beyond the window
    assert "user: 0" not in text


def test_extract_messages_text_truncates_long_content():
    """Content longer than 500 chars is truncated."""
    from api.models.anthropic import Message
    from api.services import _extract_messages_text

    long_content = "a" * 1000
    messages = [Message(role="user", content=long_content)]
    text = _extract_messages_text(messages)
    assert text.endswith("...")
    assert len(text) < 550  # role prefix + 500 trunc + ...


def test_extract_messages_text_content_blocks():
    """ContentBlock list messages are handled correctly."""
    from api.models.anthropic import (
        ContentBlockText,
        ContentBlockToolResult,
        ContentBlockToolUse,
        Message,
    )
    from api.services import _extract_messages_text

    messages = [
        Message(
            role="user",
            content=[ContentBlockText(type="text", text="Read file foo.py")],
        ),
        Message(
            role="assistant",
            content=[
                ContentBlockText(type="text", text="Here's the content:"),
                ContentBlockToolUse(
                    type="tool_use", id="tu1", name="Read", input={"path": "foo.py"}
                ),
            ],
        ),
        Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    type="tool_result", tool_use_id="tu1", content="file content here"
                ),
                ContentBlockText(type="text", text="Now refactor it to use async"),
            ],
        ),
    ]
    text = _extract_messages_text(messages)
    assert "Read file foo.py" in text
    assert "Here's the content:" in text
    assert "Now refactor it" in text
    # tool_result content should be replaced with marker
    assert "file content here" not in text
    assert "[tool_result]" in text
    # tool_use should remain as descriptive marker
    assert "Read" in text or "[tool_use]" in text


def test_extract_messages_text_skips_thinking_blocks():
    """Thinking and redacted_thinking blocks are excluded."""
    from api.models.anthropic import (
        ContentBlockText,
        ContentBlockThinking,
        Message,
    )
    from api.services import _extract_messages_text

    messages = [
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(
                    type="thinking", thinking="internal reasoning", signature="sig1"
                ),
                ContentBlockText(type="text", text="Final answer"),
            ],
        )
    ]
    text = _extract_messages_text(messages)
    assert "Final answer" in text
    assert "internal reasoning" not in text


def test_extract_messages_text_keeps_prior_context_for_short_confirmation():
    """AUTO_ROUTE sees task intent when the last user message is only confirmation."""
    from api.models.anthropic import Message
    from api.services import _extract_messages_text

    messages = [
        Message(
            role="user",
            content="Redesign auth: JWT, refresh tokens, RBAC, migrations, tests.",
        ),
        Message(
            role="assistant",
            content="Plan: update auth middleware, database schema, and tests.",
        ),
        Message(role="user", content="Yes, proceed"),
    ]

    text = _extract_messages_text(messages)

    assert "Redesign auth" in text
    assert "refresh tokens" in text
    assert "Yes, proceed" in text


def test_extract_messages_text_tool_followup_keeps_previous_task_not_tool_payload():
    """Tool follow-ups keep surrounding task intent without leaking file payloads."""
    from api.models.anthropic import ContentBlockText, ContentBlockToolResult, Message
    from api.services import _extract_messages_text

    messages = [
        Message(
            role="user", content="Refactor the routing layer across multiple files"
        ),
        Message(role="assistant", content="I'll inspect the relevant files first."),
        Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    type="tool_result",
                    tool_use_id="read1",
                    content="very large source code payload that should not classify",
                )
            ],
        ),
        Message(role="user", content=[ContentBlockText(type="text", text="Continue")]),
    ]

    text = _extract_messages_text(messages)

    assert "Refactor the routing layer" in text
    assert "[tool_result]" in text
    assert "very large source code payload" not in text
    assert "Continue" in text


def test_extract_messages_text_strips_system_reminder_blocks():
    """Claude Code system reminders are excluded but adjacent user text survives."""
    from api.models.anthropic import ContentBlockText, Message
    from api.services import _extract_messages_text

    messages = [
        Message(
            role="user",
            content=[
                ContentBlockText(
                    type="text",
                    text="<system-reminder>Do not mention this</system-reminder>",
                ),
                ContentBlockText(
                    type="text", text="Implement AUTO_ROUTE context window"
                ),
            ],
        )
    ]

    text = _extract_messages_text(messages)

    assert "system-reminder" not in text
    assert "Do not mention this" not in text
    assert "Implement AUTO_ROUTE context window" in text
