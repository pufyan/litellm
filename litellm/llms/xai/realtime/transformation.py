"""
xAI Grok Voice realtime event normalizer.

xAI's Grok Voice realtime API is structurally OpenAI-compatible but ships
several wire-format quirks that cause strict GA clients (e.g. pipecat's
``OpenAIRealtimeLLMService``) to crash before they can process tool calls:

  - ``ping`` keepalive events (unknown to GA clients)
  - ``usage: {}`` on ``response.created`` / ``response.done``
  - ``role: "tool"`` on ``conversation.item.added`` function_call items
  - Missing ``output_index`` / ``content_index`` on streaming response events
  - Missing ``part`` on ``response.content_part.done``

``XAIRealtimeNormalizer`` is plugged into ``RealTimeStreaming`` at handler
construction time (see ``handler.py``) so all normalization is isolated here
and ``RealTimeStreaming`` stays provider-agnostic.
"""

from typing import Any, FrozenSet, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.realtime_correlation import (
    RealtimeCorrelationState,
    track_content_index,
    track_output_index,
)


def _derive_ga_server_event_types() -> Optional[FrozenSet[str]]:
    """Derive the canonical GA server-event vocabulary from the openai SDK.

    Returns None when the SDK shape is unavailable; filtering then fails open
    (unknown event types pass through) instead of dropping legitimate events.
    """
    try:
        import typing

        from openai.types.realtime.realtime_server_event import RealtimeServerEvent

        args = typing.get_args(RealtimeServerEvent)
        members = typing.get_args(args[0]) if len(args) == 2 else args
        types = frozenset(
            literal
            for member in members
            for literal in typing.get_args(getattr(member, "model_fields", {}).get("type").annotation)  # type: ignore[union-attr]
            if isinstance(literal, str)
        )
        return types or None
    except Exception:
        return None


GA_SERVER_EVENT_TYPES: Optional[FrozenSet[str]] = _derive_ga_server_event_types()


class XAIRealtimeNormalizer:
    """Per-session normalizer that fixes xAI Grok Voice wire-format quirks."""

    # ---------------------------------------------------------------------------
    # Event-type sets used by the index-injection logic
    # ---------------------------------------------------------------------------
    _EVENTS_NEEDING_OUTPUT_INDEX = frozenset(
        [
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ]
    )
    _EVENTS_NEEDING_CONTENT_INDEX = frozenset(
        [
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
        ]
    )

    def __init__(self) -> None:
        # Cache content-part objects keyed by (response_id, item_id, content_index)
        # so that ``response.content_part.done`` events missing ``part`` can be
        # back-filled from earlier ``content_part.added`` / delta-done events.
        self._content_part_by_key: dict[tuple, dict[str, Any]] = {}

    # ---------------------------------------------------------------------------
    # Public interface consumed by RealTimeStreaming
    # ---------------------------------------------------------------------------

    def should_drop(self, event: object) -> bool:
        """Drop provider-specific events unknown to the canonical GA vocabulary.

        Structural guarantee for the outbound contract: any xAI event whose
        type is outside the GA server-event set derived from the openai SDK is
        dropped instead of leaking a provider-native shape to the client. When
        the vocabulary cannot be derived, filtering fails open and only the
        known ``ping`` keepalive is dropped.
        """
        if not isinstance(event, dict):
            return False
        event_type = event.get("type")
        if event_type == "ping":
            return True
        if (
            GA_SERVER_EVENT_TYPES is not None
            and isinstance(event_type, str)
            and event_type not in GA_SERVER_EVENT_TYPES
        ):
            verbose_logger.debug("XAIRealtimeNormalizer: dropping non-GA event type %s", event_type)
            return True
        return False

    def normalize(
        self, event: "dict[str, Any]", state: RealtimeCorrelationState
    ) -> "Tuple[dict[str, Any], RealtimeCorrelationState]":
        """Apply all xAI normalization passes in order."""
        event = self._normalize_content_part_events(event)
        event_type = event.get("type") or ""
        event = self._normalize_conversation_item_added(event, event_type)
        event, state = self._inject_missing_indices(event, event_type, state)
        event = self._normalize_response_usage_event(event, event_type)
        return event, state

    def patch_outgoing_session(self, session: dict) -> dict:
        """Patch a client ``session.update`` payload before forwarding to xAI.

        Unlike OpenAI, xAI does not default ``turn_detection.create_response``
        to ``True`` for ``server_vad``. Clients such as Pipecat omit the field,
        which leaves VAD detecting speech but never auto-creating a response.
        Only fill the default when the client did not set ``create_response``.
        """
        session = dict(session)
        self._default_server_vad_create_response(session)
        return session

    @staticmethod
    def _default_server_vad_create_response(session: dict) -> None:
        turn_detection = session.get("turn_detection")
        if isinstance(turn_detection, dict):
            XAIRealtimeNormalizer._ensure_server_vad_create_response(turn_detection)

        audio = session.get("audio")
        if isinstance(audio, dict):
            audio_input = audio.get("input")
            if isinstance(audio_input, dict):
                nested_td = audio_input.get("turn_detection")
                if isinstance(nested_td, dict):
                    XAIRealtimeNormalizer._ensure_server_vad_create_response(nested_td)

    @staticmethod
    def _ensure_server_vad_create_response(turn_detection: dict) -> None:
        if turn_detection.get("type") == "server_vad" and "create_response" not in turn_detection:
            turn_detection["create_response"] = True

    # ---------------------------------------------------------------------------
    # Pass 1: content-part caching and back-fill
    # ---------------------------------------------------------------------------

    @staticmethod
    def _content_part_key(event: dict) -> tuple:
        return (
            event.get("response_id"),
            event.get("item_id"),
            event.get("content_index", 0),
        )

    def _remember_content_part(self, event: dict) -> None:
        part = event.get("part")
        if isinstance(part, dict):
            self._content_part_by_key[self._content_part_key(event)] = part

    def _update_content_part_field(self, event: dict, *, part_type: str, field: str, value: object) -> None:
        if value is None:
            return
        key = self._content_part_key(event)
        existing = self._content_part_by_key.get(key)
        if not isinstance(existing, dict):
            updated = {"type": part_type, field: value}
        else:
            updated = {
                **existing,
                "type": existing.get("type", part_type),
                field: value,
            }
        self._content_part_by_key[key] = updated

    def _resolve_content_part(self, event: dict) -> dict[str, Any]:
        part = event.get("part")
        if isinstance(part, dict):
            return part
        cached = self._content_part_by_key.get(self._content_part_key(event))
        if isinstance(cached, dict):
            return cached
        return {"type": "audio", "transcript": ""}

    def _normalize_content_part_events(self, event: dict) -> dict:
        event_type = event.get("type")

        if event_type == "response.content_part.added":
            self._remember_content_part(event)
            if not isinstance(event.get("part"), dict):
                return {**event, "part": self._resolve_content_part(event)}
            return event

        if event_type == "response.output_text.done":
            self._update_content_part_field(event, part_type="text", field="text", value=event.get("text"))
            return event

        if event_type == "response.output_audio_transcript.done":
            self._update_content_part_field(
                event,
                part_type="audio",
                field="transcript",
                value=event.get("transcript"),
            )
            return event

        if event_type == "response.content_part.done":
            self._remember_content_part(event)
            if not isinstance(event.get("part"), dict):
                return {**event, "part": self._resolve_content_part(event)}
            return event

        return event

    # ---------------------------------------------------------------------------
    # Pass 2: conversation.item.added role normalisation
    # ---------------------------------------------------------------------------

    @staticmethod
    def _normalize_conversation_item_added(event: dict, event_type: str) -> dict:
        """Map ``role: "tool"`` → ``role: "assistant"`` on function_call items.

        xAI uses ``role: "tool"`` which is not in the GA-allowed set
        ("user" | "assistant" | "system").
        """
        if event_type != "conversation.item.added":
            return event
        item = event.get("item")
        if not isinstance(item, dict):
            return event
        if item.get("role") == "tool":
            return {**event, "item": {**item, "role": "assistant"}}
        return event

    # ---------------------------------------------------------------------------
    # Pass 3: inject missing output_index / content_index
    # ---------------------------------------------------------------------------

    # Which logical content-part modality an event belongs to. output_audio and
    # output_audio_transcript are two facets of the SAME content part (audio
    # bytes + its transcript), so both map to "audio" and must resolve to the
    # same content_index — mirrors Gemini's reference behavior (always
    # content_index=0 for output_audio_transcript.delta alongside output_audio.delta).
    _CONTENT_PART_MODALITY = {
        "response.content_part.added": "content",
        "response.content_part.done": "content",
        "response.output_text.delta": "text",
        "response.output_text.done": "text",
        "response.output_audio.delta": "audio",
        "response.output_audio.done": "audio",
        "response.output_audio_transcript.delta": "audio",
        "response.output_audio_transcript.done": "audio",
    }

    # Events whose id lives at a top-level "item_id" field, per the OpenAI GA
    # realtime event shapes (litellm/types/llms/openai.py). The two
    # response.output_item.* events are the exception (see _event_item_id).
    _EVENTS_WITH_TOP_LEVEL_ITEM_ID = frozenset(
        [
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ]
    )

    @staticmethod
    def _event_item_id(event: "dict[str, Any]", event_type: str) -> Optional[str]:
        """Extract the item id an event refers to.

        response.output_item.added/.done carry it nested at item.id (they have
        no top-level item_id field); every other index-bearing event type
        carries a top-level item_id.
        """
        if event_type in ("response.output_item.added", "response.output_item.done"):
            item = event.get("item")
            return item.get("id") if isinstance(item, dict) else None
        if event_type in XAIRealtimeNormalizer._EVENTS_WITH_TOP_LEVEL_ITEM_ID:
            item_id = event.get("item_id")
            return item_id if isinstance(item_id, str) else None
        return None

    def _inject_missing_indices(
        self, event: "dict[str, Any]", event_type: str, state: RealtimeCorrelationState
    ) -> "Tuple[dict[str, Any], RealtimeCorrelationState]":
        """Inject real ``output_index`` / ``content_index`` when xAI omits them.

        xAI omits both fields on every streaming response event; pydantic GA
        clients require them as non-optional ints. Previously defaulted to a
        hardcoded 0 (correct only for single-item responses); now resolved via
        the shared realtime_correlation module so multi-item/multi-part
        responses get real, monotonically increasing indices.
        """
        needs_output = event_type in self._EVENTS_NEEDING_OUTPUT_INDEX
        needs_content = event_type in self._EVENTS_NEEDING_CONTENT_INDEX
        if not needs_output and not needs_content:
            return event, state

        response_id = event.get("response_id")
        item_id = self._event_item_id(event, event_type)
        if not isinstance(response_id, str) or not isinstance(item_id, str):
            # Can't resolve a real index without both ids; leave the event
            # unpatched rather than guessing.
            return event, state

        patch: dict[str, Any] = {}
        if needs_output and "output_index" not in event:
            state, output_index = track_output_index(state, response_id, item_id)
            patch["output_index"] = output_index
        if needs_content and "content_index" not in event:
            content_part_key = self._CONTENT_PART_MODALITY.get(event_type, "content")
            state, content_index = track_content_index(state, response_id, item_id, content_part_key)
            patch["content_index"] = content_index
        if not patch:
            return event, state
        return {**event, **patch}, state

    # ---------------------------------------------------------------------------
    # Pass 4: response usage normalisation
    # ---------------------------------------------------------------------------

    @staticmethod
    def _default_ga_usage() -> dict[str, Any]:
        default_details: dict[str, Any] = {
            "cached_tokens": 0,
            "text_tokens": 0,
            "audio_tokens": 0,
        }
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_token_details": default_details.copy(),
            "output_token_details": default_details.copy(),
        }

    @staticmethod
    def _normalize_usage(usage: object, *, empty_as_null: bool) -> Optional[dict[str, Any]]:
        """Coerce a usage object into the full OpenAI GA shape.

        ``empty_as_null=True`` for ``response.created`` (usage optional).
        ``empty_as_null=False`` for ``response.done`` (e2e tests assert non-null).
        """
        if not isinstance(usage, dict):
            return None
        if not usage:
            return None if empty_as_null else XAIRealtimeNormalizer._default_ga_usage()
        default_details: dict[str, Any] = {
            "cached_tokens": 0,
            "text_tokens": 0,
            "audio_tokens": 0,
        }
        normalized: dict[str, Any] = {
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "input_token_details": default_details.copy(),
            "output_token_details": default_details.copy(),
        }
        for key in ("input_token_details", "output_token_details"):
            details = usage.get(key)
            if isinstance(details, dict):
                normalized[key] = {**default_details, **details}
        return normalized

    def _normalize_response_usage_event(self, event: dict, event_type: str) -> dict:
        if event_type not in ("response.created", "response.done"):
            return event
        response = event.get("response")
        if not isinstance(response, dict) or "usage" not in response:
            return event
        normalized_usage = self._normalize_usage(
            response.get("usage"),
            empty_as_null=event_type == "response.created",
        )
        if normalized_usage is response.get("usage"):
            return event
        return {**event, "response": {**response, "usage": normalized_usage}}
