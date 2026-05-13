"""Request builder and DeepSeek native Anthropic compatibility sanitizer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError

# Block types not supported on DeepSeek partial Anthropic-compatible API.
_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)

# Block types silently stripped for DeepSeek since the content is typically
# also provided via tool_result (e.g. Claude Code attaches PDFs as document
# blocks alongside a Read tool_result containing the text).
_STRIPPABLE_MESSAGE_BLOCK_TYPES = frozenset({"image", "document"})
_OMITTED_ATTACHMENT_TEXT = (
    "[attachment omitted: DeepSeek does not support image or document inputs]"
)
_OMITTED_ATTACHMENT_BLOCK = {"type": "text", "text": _OMITTED_ATTACHMENT_TEXT}


def _strip_unsupported_attachment_blocks(messages: Any) -> Any:
    """Remove image/document blocks that DeepSeek cannot process.

    Claude Code sends PDFs as ``document`` blocks alongside a Read ``tool_result``
    that already contains the extracted text. Stripping preserves the request
    instead of failing with an unsupported block error.
    """
    if not isinstance(messages, list):
        return messages

    stripped: list[Any] = []
    top_level_dropped: dict[str, int] = {}
    nested_dropped: dict[str, int] = {}
    placeholder_replacements = 0

    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            stripped.append(message)
            continue

        new_content: list[Any] = []
        message_dropped_attachment = False
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype in _STRIPPABLE_MESSAGE_BLOCK_TYPES:
                    top_level_dropped[btype] = top_level_dropped.get(btype, 0) + 1
                    message_dropped_attachment = True
                    continue
                if btype == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        filtered_inner: list[Any] = []
                        for sub in inner:
                            if (
                                isinstance(sub, dict)
                                and sub.get("type") in _STRIPPABLE_MESSAGE_BLOCK_TYPES
                            ):
                                sub_type = sub["type"]
                                nested_dropped[sub_type] = (
                                    nested_dropped.get(sub_type, 0) + 1
                                )
                                continue
                            filtered_inner.append(sub)
                        if not filtered_inner:
                            filtered_inner = [_OMITTED_ATTACHMENT_BLOCK]
                            placeholder_replacements += 1
                        new_block = dict(block)
                        new_block["content"] = filtered_inner
                        new_content.append(new_block)
                        continue
            new_content.append(block)
        if not new_content and message_dropped_attachment:
            new_content = [_OMITTED_ATTACHMENT_BLOCK]
            placeholder_replacements += 1
        new_msg = dict(message)
        new_msg["content"] = new_content
        stripped.append(new_msg)

    if top_level_dropped or nested_dropped:
        logger.warning(
            "DEEPSEEK_REQUEST: stripped unsupported attachment blocks "
            "(top_level={} nested_in_tool_result={} placeholder_tool_results={}). "
            "DeepSeek has no vision/document support; the model will not see this content.",
            dict(top_level_dropped),
            dict(nested_dropped),
            placeholder_replacements,
        )
    return stripped


def _is_server_listed_tool(tool: Mapping[str, Any]) -> bool:
    """True for Anthropic web_search / web_fetch-style tool definitions (listed tools)."""
    name = (tool.get("name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.get("type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(blocks: Any, *, where: str) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result" and "content" in block:
            _walk_block_list_for_unsupported(
                block["content"], where=f"{where} (tool_result content)"
            )


def _validate_deepseek_native_request_dict(data: dict[str, Any]) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError(
            "DeepSeek native does not support mcp_servers on requests."
        )

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek native does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        c = message.get("content")
        if isinstance(c, list):
            _walk_block_list_for_unsupported(c, where=f"messages[{i}].content")
        if isinstance(c, str) and "<think>" in c:
            # Unusual, but block encoded redacted content — treat as unsafe for DeepSeek.
            pass

    system = data.get("system")
    if isinstance(system, list):
        _walk_block_list_for_unsupported(system, where="system")


def _has_tool_history_blocks(message: Mapping[str, Any]) -> bool:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if role == "assistant" and btype == "tool_use":
            return True
        if role == "user" and btype == "tool_result":
            return True
    return False


def _has_replayable_thinking_before_tool_use(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False

    has_reasoning_content = isinstance(message.get("reasoning_content"), str) and bool(
        message["reasoning_content"].strip()
    )
    has_thinking = has_reasoning_content
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking" and isinstance(block.get("thinking"), str):
            has_thinking = bool(block["thinking"])
            continue
        if btype == "tool_use":
            return has_thinking
    return False


def _materialize_reasoning_content_for_native(messages: Any) -> Any:
    """Replay assistant ``reasoning_content`` as native thinking blocks.

    Some upstream conversion paths keep provider reasoning in the OpenAI-style
    top-level ``reasoning_content`` field. DeepSeek's native Anthropic endpoint
    does not accept that field, but thinking-mode tool follow-ups must replay
    the prior assistant reasoning before the matching ``tool_use``. Convert that
    metadata into a DeepSeek/Anthropic ``thinking`` content block before later
    stripping the top-level helper field.
    """
    if not isinstance(messages, list):
        return messages

    out: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        if message.get("role") != "assistant":
            out.append(message)
            continue
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue

        has_tool_use = any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in content
        )
        has_native_thinking_before_tool_use = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "thinking" and isinstance(
                block.get("thinking"), str
            ):
                has_native_thinking_before_tool_use = bool(block["thinking"])
                continue
            if block.get("type") == "tool_use":
                break
        if not has_tool_use or has_native_thinking_before_tool_use:
            out.append(message)
            continue

        new_content: list[Any] = []
        inserted = False
        for block in content:
            if (
                not inserted
                and isinstance(block, dict)
                and block.get("type") == "tool_use"
            ):
                new_content.append({"type": "thinking", "thinking": reasoning})
                inserted = True
            new_content.append(block)
        new_msg = dict(message)
        new_msg["content"] = new_content
        out.append(new_msg)
    return out


def _has_tool_history(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_tool_history_blocks(message):
            return True
    return False


def _has_replayable_tool_thinking(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_replayable_thinking_before_tool_use(
            message
        ):
            return True
    return False


def _remove_deepseek_thinking_hints(data: dict[str, Any]) -> None:
    """Remove request hints that can keep DeepSeek in thinking mode after fallback."""
    output_config = data.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        cleaned_output_config = dict(output_config)
        cleaned_output_config.pop("effort", None)
        if cleaned_output_config:
            data["output_config"] = cleaned_output_config
        else:
            data.pop("output_config", None)

    context_management = data.get("context_management")
    if not isinstance(context_management, dict):
        return
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return
    filtered_edits = [
        edit
        for edit in edits
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("type"), str)
            and edit["type"].startswith("clear_thinking_")
        )
    ]
    if len(filtered_edits) == len(edits):
        return
    cleaned_context_management = dict(context_management)
    if filtered_edits:
        cleaned_context_management["edits"] = filtered_edits
        data["context_management"] = cleaned_context_management
    else:
        cleaned_context_management.pop("edits", None)
        if cleaned_context_management:
            data["context_management"] = cleaned_context_management
        else:
            data.pop("context_management", None)


def sanitize_deepseek_messages_for_native(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Strip thinking content blocks from all messages.

    When ``thinking_enabled`` is False all ``thinking`` and
    ``redacted_thinking`` blocks are removed from every message
    (not just assistant — DeepSeek V4 Pro rejects requests that
    carry unmanaged thinking blocks).

    When ``thinking_enabled`` is True only ``redacted_thinking`` is
    removed (DeepSeek does not support it).
    """
    if not isinstance(messages, list):
        return messages

    def _has_signature(block: dict) -> bool:
        """Check if a thinking block has a signature (signed thinking)."""
        sig = block.get("signature")
        return bool(sig and isinstance(sig, str))

    sanitized: list[Any] = []
    strip_types = {"redacted_thinking"}
    if not thinking_enabled:
        strip_types.add("thinking")

    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        filtered = [
            block
            for block in content
            if not (isinstance(block, dict) and block.get("type") in strip_types)
        ]
        new_msg = dict(message)
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _serialize_tool_result_content(content: Any) -> str:
    """Serialize tool_result content to string for DeepSeek API.

    DeepSeek's Anthropic-compatible API expects tool_result.content to be a string,
    not an array of content blocks.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _normalize_tool_result_content(messages: Any) -> Any:
    """Normalize tool_result content to strings for DeepSeek API compatibility."""
    if not isinstance(messages, list):
        return messages

    normalized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(message)
            continue

        # Process content blocks
        new_content: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            if block.get("type") == "tool_result":
                # Normalize tool_result content to string
                normalized_block = dict(block)
                normalized_block["content"] = _serialize_tool_result_content(
                    block.get("content")
                )
                new_content.append(normalized_block)
            else:
                new_content.append(block)

        new_msg = dict(message)
        new_msg["content"] = new_content
        normalized.append(new_msg)

    return normalized


def _ensure_text_after_tool_results(messages: list) -> list:
    """Ensure user messages with tool_result blocks also have a text block.

    DeepSeek V4 Pro requires that every ``tool_result`` block is followed by
    at least one ``text`` block in the same user message. If a user message
    contains only ``tool_result`` blocks, an empty continuation text block
    is appended.

    Without this, DeepSeek returns HTTP 400 with a misleading error about
    ``content[].thinking`` mode.
    """
    result: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        # Check if every block is a tool_result
        if content and all(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            new_msg = dict(msg)
            new_msg["content"] = [*content, {"type": "text", "text": "[continuation]"}]
            result.append(new_msg)
        else:
            result.append(msg)
    return result


def _strip_reasoning_content_when_native(messages: Any) -> Any:
    """``reasoning_content`` is OpenAI-helper metadata; not part of native Anthropic body."""
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        msg = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(msg)
    return out


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a DeepSeek ``/v1/messages`` JSON body (Anthropic format)."""
    logger.debug(
        "DEEPSEEK_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    if "messages" in data:
        data["messages"] = _strip_unsupported_attachment_blocks(data["messages"])
    _validate_deepseek_native_request_dict(data)
    data.pop("extra_body", None)

    has_tool_history = _has_tool_history(data)
    has_replayable_tool_thinking = _has_replayable_tool_thinking(data)
    unsafe_tool_followup = has_tool_history and not has_replayable_tool_thinking
    effective_thinking_enabled = thinking_enabled and not unsafe_tool_followup
    if thinking_enabled:
        if unsafe_tool_followup:
            logger.debug(
                "DEEPSEEK_REQUEST: disabling thinking for tool follow-up without "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
            _remove_deepseek_thinking_hints(data)
        elif has_tool_history:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for tool follow-up with "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
        elif data.get("tools") or data.get("tool_choice"):
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for initial tool request "
                "model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )

    thinking_cfg = data.pop("thinking", None)
    if effective_thinking_enabled and isinstance(thinking_cfg, dict):
        thinking_payload: dict[str, Any] = {"type": "enabled"}
        budget_tokens = thinking_cfg.get("budget_tokens")
        if isinstance(budget_tokens, int):
            thinking_payload["budget_tokens"] = budget_tokens
        data["thinking"] = thinking_payload

    if "messages" in data:
        if effective_thinking_enabled:
            data["messages"] = _materialize_reasoning_content_for_native(
                data["messages"]
            )
        data["messages"] = _strip_reasoning_content_when_native(
            _ensure_text_after_tool_results(
                _normalize_tool_result_content(
                    sanitize_deepseek_messages_for_native(
                        data["messages"],
                        thinking_enabled=effective_thinking_enabled,
                    )
                )
            )
        )
    if "max_tokens" not in data or data.get("max_tokens") is None:
        data["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    data["stream"] = True

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        data.get("model"),
        len(data.get("messages", [])),
        len(data.get("tools", [])),
    )

    return data
