"""OpenAI-realtime-shaped event-dict builders.

The only place that constructs the wire-format event dicts for the correlation
lifecycle. Every builder is typed against the existing ``OpenAIRealtime*``
TypedDicts in ``litellm/types/llms/openai.py`` — this module never redefines
those shapes, only assembles them from ``state.py`` types.
"""

import uuid
from typing import Dict, List, Literal, Optional, TypedDict

from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeConversationItemAdded,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeFunctionCallArgumentsDone,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseContentPart,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamResponseOutputItem,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamResponseOutputItemContent,
    ResponseAPIUsage,
)

from .state import ClosedItem, ContentDeltaType, ItemType, OpenContentPart, OpenItem


class SpeechStartedEvent(TypedDict):
    type: Literal["input_audio_buffer.speech_started"]
    event_id: str
    audio_start_ms: int
    item_id: str


def _event_id() -> str:
    return "event_{}".format(uuid.uuid4())


def build_response_created(
    response_id: str,
    conversation_id: str,
    modalities: List[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> OpenAIRealtimeStreamResponseBaseObject:
    response: Dict[str, object] = {
        "object": "realtime.response",
        "id": response_id,
        "status": "in_progress",
        "status_details": None,
        "output": [],
        "conversation_id": conversation_id,
        "modalities": modalities,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    return OpenAIRealtimeStreamResponseBaseObject(
        type="response.created",
        event_id=_event_id(),
        response=response,
    )


def _text_content_entry(text: str) -> OpenAIRealtimeStreamResponseOutputItemContent:
    return {"type": "text", "text": text}


def _audio_content_entry() -> OpenAIRealtimeStreamResponseOutputItemContent:
    return {"type": "audio", "transcript": ""}


def _output_item_dict(
    item_id: str,
    item_type: ItemType,
    role: Literal["assistant"],
    status: Literal["completed", "incomplete", "in_progress"],
    content: List[OpenAIRealtimeStreamResponseOutputItemContent],
) -> OpenAIRealtimeStreamResponseOutputItem:
    return OpenAIRealtimeStreamResponseOutputItem(
        id=item_id,
        object="realtime.item",
        type=item_type,
        status=status,
        role=role,
        content=content,
    )


def build_output_item_added(
    response_id: str,
    output_index: int,
    item_id: str,
    item_type: ItemType,
    role: Literal["assistant"],
) -> OpenAIRealtimeStreamResponseOutputItemAdded:
    return OpenAIRealtimeStreamResponseOutputItemAdded(
        type="response.output_item.added",
        event_id=_event_id(),
        response_id=response_id,
        output_index=output_index,
        item=_output_item_dict(item_id, item_type, role, "in_progress", []),
    )


def build_conversation_item_added(
    item_id: str,
    item_type: ItemType,
    role: Literal["assistant"],
) -> OpenAIRealtimeConversationItemAdded:
    # Pipecat requires "conversation.item.added" (not ".created"); sending
    # ".created" raises "Unimplemented server event type" and kills the receive
    # task handler on some GA clients.
    return OpenAIRealtimeConversationItemAdded(
        type="conversation.item.added",
        event_id=_event_id(),
        previous_item_id=None,
        item=_output_item_dict(item_id, item_type, role, "in_progress", []),
    )


def build_content_part_added(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta_type: ContentDeltaType,
) -> OpenAIRealtimeResponseContentPartAdded:
    part: OpenAIRealtimeResponseContentPart = (
        {"type": "text", "text": ""} if delta_type == "text" else {"type": "audio", "transcript": ""}
    )
    return OpenAIRealtimeResponseContentPartAdded(
        type="response.content_part.added",
        event_id=_event_id(),
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part,
    )


def build_content_delta(
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    delta_type: ContentDeltaType,
    delta: str,
) -> OpenAIRealtimeResponseDelta:
    event_type: Literal["response.output_text.delta", "response.output_audio.delta"] = (
        "response.output_text.delta" if delta_type == "text" else "response.output_audio.delta"
    )
    return OpenAIRealtimeResponseDelta(
        type=event_type,
        event_id=_event_id(),
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def build_content_part_done(
    response_id: str,
    item_id: str,
    output_index: int,
    content_part: OpenContentPart,
) -> OpenAIRealtimeContentPartDone:
    part: OpenAIRealtimeResponseContentPart = (
        {"type": "text", "text": content_part.accumulated_text}
        if content_part.delta_type == "text"
        else {"type": "audio", "transcript": ""}
    )
    return OpenAIRealtimeContentPartDone(
        type="response.content_part.done",
        event_id=_event_id(),
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_part.content_index,
        part=part,
    )


def build_output_item_done(
    response_id: str,
    item: OpenItem,
    status: Literal["completed", "incomplete"],
) -> OpenAIRealtimeOutputItemDone:
    content: List[OpenAIRealtimeStreamResponseOutputItemContent] = [
        (_text_content_entry(part.accumulated_text) if part.delta_type == "text" else _audio_content_entry())
        for part in item.content_parts
    ]
    return OpenAIRealtimeOutputItemDone(
        type="response.output_item.done",
        event_id=_event_id(),
        response_id=response_id,
        output_index=item.output_index,
        item=_output_item_dict(item.item_id, item.item_type, item.role, status, content),
    )


def build_response_done(
    response_id: str,
    conversation_id: str,
    closed_items: List[ClosedItem],
    modalities: List[str],
    usage: Optional[ResponseAPIUsage],
) -> OpenAIRealtimeDoneEvent:
    output: List[OpenAIRealtimeStreamResponseOutputItem] = [
        _output_item_dict(
            item.item_id,
            item.item_type,
            item.role,
            item.status,
            [_text_content_entry(item.text)] if item.text else [],
        )
        for item in closed_items
    ]
    response: OpenAIRealtimeResponseDoneObject = OpenAIRealtimeResponseDoneObject(
        object="realtime.response",
        id=response_id,
        status="completed",
        output=output,
        conversation_id=conversation_id,
        modalities=modalities,
    )
    if usage is not None:
        response["usage"] = usage.model_dump()
    return OpenAIRealtimeDoneEvent(
        type="response.done",
        event_id=_event_id(),
        response=response,
    )


def build_speech_started() -> SpeechStartedEvent:
    return SpeechStartedEvent(
        type="input_audio_buffer.speech_started",
        event_id=_event_id(),
        audio_start_ms=0,
        item_id="item_{}".format(uuid.uuid4()),
    )


def build_function_call_arguments_done(
    response_id: str,
    item_id: str,
    output_index: int,
    call_id: str,
    name: str,
    arguments: str,
) -> OpenAIRealtimeFunctionCallArgumentsDone:
    return OpenAIRealtimeFunctionCallArgumentsDone(
        type="response.function_call_arguments.done",
        event_id=_event_id(),
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
