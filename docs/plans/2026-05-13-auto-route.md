# AUTO_ROUTE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add automatic task complexity classification to `free-claude-code` proxy — simple tasks route to `deepseek-v4-flash` (cheap), complex tasks route to `deepseek-v4-pro` (powerful).

**Architecture:** Synchronous `TaskClassifier` makes a fast HTTP call to DeepSeek flash API with a classification prompt (~2s latency). `ModelRouter.resolve_with_classification()` calls the classifier, selects the appropriate model tier (opus/sonnet), and delegates to existing `resolve()`.

**Tech Stack:** Python 3.12+, Pydantic Settings, httpx, DeepSeek Anthropic-compatible API

---

## Task 1: Settings — add AUTO_ROUTE fields

**Files:**
- Modify: `config/settings.py` (fields + validation)
- Test: `tests/config/test_config.py`

### Step 1: Add fields to config/settings.py

Insert after the `enable_haiku_thinking` field block (~line 170):

```python
# ==================== AUTO_ROUTE (Task Complexity Routing) ====================
auto_route_enabled: bool = Field(default=False, validation_alias="AUTO_ROUTE_ENABLED")
auto_route_classifier_model: str = Field(
    default="deepseek/deepseek-v4-flash",
    validation_alias="AUTO_ROUTE_CLASSIFIER_MODEL",
)
auto_route_complexity_threshold: float = Field(
    default=0.5,
    validation_alias="AUTO_ROUTE_COMPLEXITY_THRESHOLD",
)
```

### Step 2: Expand model validator

In `@field_validator("model", "model_opus", "model_sonnet", "model_haiku")` add `"auto_route_classifier_model"` to the list so it validates provider format.

### Step 3: Add tests in tests/config/test_config.py

```python
def test_auto_route_settings_defaults():
    settings = Settings()
    assert settings.auto_route_enabled is False
    assert settings.auto_route_classifier_model == "deepseek/deepseek-v4-flash"
    assert settings.auto_route_complexity_threshold == 0.5
```

### Step 4: Verify

```bash
uv run pytest tests/config/test_config.py -v
```

### Step 5: Commit

```bash
git add config/settings.py tests/config/test_config.py
git commit -m "feat(config): add AUTO_ROUTE settings fields"
```

---

## Task 2: TaskClassifier — new service

**Files:**
- Create: `api/task_classifier.py`
- Test: `tests/api/test_task_classifier.py`

### Step 1: Create api/task_classifier.py

```python
"""Fast pre-flight task complexity classifier via DeepSeek API."""

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
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"
    VERY_COMPLEX = "VERY_COMPLEX"


@dataclass(frozen=True)
class ComplexityResult:
    complexity: TaskComplexity
    classifier_model: str
    latency_ms: float


class TaskClassifier:
    """Fast pre-flight task complexity classifier via DeepSeek API.

    Makes a synchronous HTTP call to the classifier model (default: deepseek-v4-flash)
    with a tiny prompt to determine task complexity. Falls back to COMPLEX on any error.

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
        self._classifier_api_key = self._resolve_api_key(settings, self._classifier_provider)
        self._base_url = self._resolve_base_url(self._classifier_provider)
        self._http_client = httpx.Client(timeout=10.0)

    @staticmethod
    def _resolve_api_key(settings: Settings, provider: str) -> str:
        if provider == "deepseek":
            return settings.deepseek_api_key
        if provider == "open_router":
            return settings.open_router_api_key
        return ""

    @staticmethod
    def _resolve_base_url(provider: str) -> str:
        if provider == "deepseek":
            return "https://api.deepseek.com/anthropic"
        if provider == "open_router":
            return "https://openrouter.ai/api/v1"
        return ""

    def classify(self, messages_text: str) -> ComplexityResult:
        """Classify task complexity from extracted message text."""
        start = time.monotonic()

        prompt = CLASSIFICATION_PROMPT.format(
            messages_text=messages_text[:2000]
        )

        try:
            if self._classifier_provider == "deepseek":
                response = self._http_client.post(
                    f"{self._base_url}/v1/messages",
                    headers={
                        "x-api-key": self._classifier_api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": self._classifier_model,
                        "max_tokens": 10,
                        "temperature": 0,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                response.raise_for_status()
                data = response.json()
                complexity = self._parse_complexity(data)
            else:
                logger.warning(
                    "TaskClassifier: unsupported provider '{}', falling back to COMPLEX",
                    self._classifier_provider,
                )
                complexity = TaskComplexity.COMPLEX

        except Exception as exc:
            logger.warning(
                "TaskClassifier failed: {} type={} — falling back to COMPLEX",
                exc, type(exc).__name__,
            )
            complexity = TaskComplexity.COMPLEX

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "TaskClassifier: complexity={} latency={:.1f}ms model={}",
            complexity, elapsed_ms, self._classifier_model,
        )
        return ComplexityResult(
            complexity=complexity,
            classifier_model=self._classifier_model,
            latency_ms=round(elapsed_ms, 1),
        )

    @staticmethod
    def _parse_complexity(data: dict) -> TaskComplexity:
        """Parse DeepSeek response for complexity classification."""
        for block in data.get("content", []):
            if block.get("type") != "text":
                continue
            text = block.get("text", "").strip().upper()
            if "SIMPLE" in text:
                return TaskComplexity.SIMPLE
            if "VERY_COMPLEX" in text:
                return TaskComplexity.VERY_COMPLEX
            if "COMPLEX" in text:
                return TaskComplexity.COMPLEX
        return TaskComplexity.COMPLEX

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http_client.close()
```

### Step 2: Create tests/api/test_task_classifier.py

```python
"""Tests for the TaskClassifier AUTO_ROUTE service."""

import pytest
from api.task_classifier import TaskClassifier, TaskComplexity
from config.settings import Settings


@pytest.fixture
def settings():
    s = Settings()
    s.auto_route_enabled = True
    s.auto_route_classifier_model = "deepseek/deepseek-v4-flash"
    s.deepseek_api_key = "test-key"
    return s


def test_classifier_returns_complexity(settings):
    """Classifier returns a valid ComplexityResult structure."""
    classifier = TaskClassifier(settings)
    result = classifier.classify("What is the capital of France?")
    assert result.complexity in TaskComplexity
    assert isinstance(result.latency_ms, (int, float))
    assert result.latency_ms > 0
    assert result.classifier_model == "deepseek-v4-flash"
    classifier.close()


def test_simple_question_classified_as_simple(settings):
    """Real DeepSeek API: simple factual question returns SIMPLE."""
    classifier = TaskClassifier(settings)
    result = classifier.classify("What is the capital of France?")
    assert result.complexity == TaskComplexity.SIMPLE
    classifier.close()


def test_complex_refactor_classified_as_complex(settings):
    """Real DeepSeek API: multi-file refactor returns COMPLEX or VERY_COMPLEX."""
    classifier = TaskClassifier(settings)
    result = classifier.classify(
        "Refactor the authentication module to use JWT tokens instead of session-based auth. "
        "Update all middleware, controllers, and add unit tests."
    )
    assert result.complexity in (TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX)
    classifier.close()


def test_classifier_fallback_on_api_error(settings):
    """Invalid API key should fall back to COMPLEX, not crash."""
    settings.deepseek_api_key = "invalid-key"
    classifier = TaskClassifier(settings)
    result = classifier.classify("Simple question")
    assert result.complexity == TaskComplexity.COMPLEX  # fallback
    classifier.close()
```

### Step 3: Verify

```bash
uv run pytest tests/api/test_task_classifier.py -v
```

### Step 4: Commit

```bash
git add api/task_classifier.py tests/api/test_task_classifier.py
git commit -m "feat(api): add TaskClassifier for AUTO_ROUTE complexity detection"
```

---

## Task 3: ModelRouter — resolve_with_classification

**Files:**
- Modify: `api/model_router.py`
- Test: `tests/api/test_model_router.py`

### Step 1: Add to api/model_router.py

Import at top:

```python
from api.task_classifier import TaskClassifier, TaskComplexity
```

In `__init__`:

```python
self._task_classifier: TaskClassifier | None = None
```

New methods:

```python
def resolve_with_classification(
    self, claude_model_name: str, messages_text: str
) -> ResolvedModel:
    """Resolve model with pre-flight AUTO_ROUTE complexity classification.

    When auto_route_enabled is False, behaves exactly like resolve().
    When enabled, classifies the task and routes to the appropriate tier:
    - SIMPLE -> sonnet-tier model (cheaper)
    - COMPLEX / VERY_COMPLEX -> opus-tier model (more powerful)
    """
    if not self._settings.auto_route_enabled:
        return self.resolve(claude_model_name)

    classifier = self._get_classifier()
    result = classifier.classify(messages_text)

    # Route based on complexity
    if result.complexity == TaskComplexity.SIMPLE:
        tier_model = "claude-sonnet-4-20250514"
    else:
        tier_model = "claude-opus-4-20250514"

    resolved = self.resolve(tier_model)

    logger.info(
        "AUTO_ROUTE: {} -> {}/{} ({}ms classifier={})",
        result.complexity,
        resolved.provider_id,
        resolved.provider_model,
        result.latency_ms,
        result.classifier_model,
    )
    return resolved

def _get_classifier(self) -> TaskClassifier:
    """Lazy-init TaskClassifier (only created if AUTO_ROUTE is used)."""
    if self._task_classifier is None:
        self._task_classifier = TaskClassifier(self._settings)
    return self._task_classifier
```

### Step 2: Add tests to tests/api/test_model_router.py

```python
from unittest.mock import patch, MagicMock
from api.task_classifier import TaskComplexity, ComplexityResult


def test_router_routes_simple_task_to_sonnet(settings):
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


def test_router_routes_complex_task_to_opus(settings):
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


def test_router_routes_very_complex_task_to_opus(settings):
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


def test_router_ignores_auto_route_when_disabled(settings):
    """When auto_route_enabled=False, resolve_with_classification == resolve()."""
    settings.auto_route_enabled = False
    settings.model = "nvidia_nim/fallback-model"
    router = ModelRouter(settings)

    resolved = router.resolve_with_classification(
        "claude-sonnet-4-20250514", "Some task"
    )

    assert resolved.provider_model == "fallback-model"
    assert resolved.provider_id == "nvidia_nim"
```

### Step 3: Verify

```bash
uv run pytest tests/api/test_model_router.py -v -k "auto_route or simple or complex"
# Then full test suite
uv run pytest tests/api/test_model_router.py -v
```

### Step 4: Commit

```bash
git add api/model_router.py tests/api/test_model_router.py
git commit -m "feat(api): add AUTO_ROUTE routing to ModelRouter"
```

---

## Task 4: Services — integrate with MessagesRequest

**Files:**
- Modify: `api/services.py`
- Test: `tests/api/test_api.py`

### Step 1: Add _extract_messages_text utility

Add to `api/services.py`:

```python
def _extract_messages_text(messages: list, max_messages: int = 5) -> str:
    """Extract recent message text for classification, excluding large tool results.

    Takes the last N messages and extracts text content blocks only.
    Tool results and non-text blocks are replaced with block-type markers.
    Each message is truncated to 500 chars to keep the classification prompt small.
    """
    parts: list[str] = []
    for msg in messages[-max_messages:]:
        role = msg.role if hasattr(msg, 'role') else '?'
        content = msg.content if hasattr(msg, 'content') else ''

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_blocks: list[str] = []
            for block in content:
                btype = getattr(block, 'type', None) or ''
                if btype == 'tool_result':
                    text_blocks.append('[tool_result]')
                elif btype in ('thinking', 'redacted_thinking'):
                    continue  # skip internal thinking
                elif hasattr(block, 'text'):
                    text_blocks.append(block.text)
                else:
                    text_blocks.append(f'[{btype}]')
            text = ' | '.join(text_blocks)
        else:
            continue

        if len(text) > 500:
            text = text[:500] + '...'
        parts.append(f'{role}: {text}')

    return '\n\n'.join(parts)
```

### Step 2: Update resolve_messages_request

In `ClaudeProxyService`, add the auto_route check before calling `resolve()`:

```python
# In create_message method, before routed = self._model_router.resolve_messages_request(request_data)
# Or better, modify the service call:
```

Actually, the cleanest approach is to modify `ClaudeProxyService.create_message()` directly right after `_require_non_empty_messages`:

```python
# In ClaudeProxyService.create_message():
routed = self._model_router.resolve_messages_request(request_data)
```

Change to:

```python
if self._settings.auto_route_enabled:
    messages_text = _extract_messages_text(request_data.messages)
    resolved = self._model_router.resolve_with_classification(
        request_data.model, messages_text
    )
    routed = RoutedMessagesRequest(
        request=request_data.model_copy(update={"model": resolved.provider_model}, deep=True),
        resolved=resolved,
    )
else:
    routed = self._model_router.resolve_messages_request(request_data)
```

### Step 3: Add test in tests/api/test_api.py

```python
def test_messages_text_extraction():
    from api.services import _extract_messages_text
    from api.models.anthropic import Message

    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="What is the capital of France?"),
    ]
    text = _extract_messages_text(messages)
    assert "Hello" in text
    assert "What is the capital" in text
    assert text.count("user:") == 2
    assert text.count("assistant:") == 1


def test_extract_messages_text_handles_empty():
    from api.services import _extract_messages_text
    assert _extract_messages_text([]) == ""


def test_extract_messages_text_handles_content_blocks():
    from api.services import _extract_messages_text
    from api.models.anthropic import Message, ContentBlockText, ContentBlockToolUse, ContentBlockToolResult

    messages = [
        Message(role="user", content=[
            ContentBlockText(type="text", text="Read file foo.py"),
        ]),
        Message(role="assistant", content=[
            ContentBlockText(type="text", text="Here's the content:"),
            ContentBlockToolUse(type="tool_use", id="tu1", name="Read", input={"path": "foo.py"}),
        ]),
        Message(role="user", content=[
            ContentBlockToolResult(type="tool_result", tool_use_id="tu1", content="file content here"),
            ContentBlockText(type="text", text="Now refactor it to use async"),
        ]),
    ]
    text = _extract_messages_text(messages)
    assert "Read file foo.py" in text
    assert "Now refactor it" in text
    # tool_result content stripped
    assert "file content here" not in text
    # tool_use name present
    assert "Read" in text
```

### Step 4: Verify

```bash
uv run pytest tests/api/test_api.py::test_messages_text_extraction -v
uv run pytest tests/api/test_api.py::test_extract_messages_text_handles_empty -v
uv run pytest tests/api/test_api.py::test_extract_messages_text_handles_content_blocks -v
# Full regression
uv run pytest tests/api/test_api.py tests/api/test_model_router.py -v
```

### Step 5: Commit

```bash
git add api/services.py tests/api/test_api.py
git commit -m "feat(api): integrate AUTO_ROUTE classification into proxy service"
```

---

## Task 5: .env.example documentation

**Files:**
- Modify: `.env.example`

### Step 1: Add AUTO_ROUTE section

```bash
# =============================================================================
# AUTO_ROUTE: Automatic task complexity routing
# When enabled, classifies each request before routing:
# - SIMPLE tasks -> sonnet-tier model (cheaper, faster)
# - COMPLEX/VERY_COMPLEX tasks -> opus-tier model (more powerful)
# =============================================================================
AUTO_ROUTE_ENABLED=false
AUTO_ROUTE_CLASSIFIER_MODEL=deepseek/deepseek-v4-flash
AUTO_ROUTE_COMPLEXITY_THRESHOLD=0.5
```

### Step 2: Full CI check

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

### Step 3: Commit

```bash
git add .env.example
git commit -m "docs: add AUTO_ROUTE configuration to env example"
```

---

## Success Criteria

1. `AUTO_ROUTE_ENABLED=true` in .env enables automatic model selection
2. Simple tasks (questions, file reads) routed to flash model
3. Complex tasks (refactoring, debugging) routed to pro model
4. `AUTO_ROUTE_ENABLED=false` — no change to existing behavior
5. API errors in classifier → safe fallback to COMPLEX (pro model)
6. All existing tests pass, no regressions
7. `uv run ruff check` and `uv run ty check` pass
