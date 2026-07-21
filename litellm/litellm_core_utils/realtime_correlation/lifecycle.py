"""Pure state-transition functions for the realtime correlation-id lifecycle.

Every function is ``(state, ...) -> (new_state, events)``. This is the *only*
place `output_index`/`content_index` get allocated and the *only* place
OpenAI-realtime event dicts get built for these lifecycle points — no provider
should hand-construct these once migrated onto this module.

The lifecycle is: ``open_response -> open_item -> open_content_part ->
append_content_delta -> close_item -> close_response``, with `close_response`
guaranteeing every still-open item gets an "incomplete" close first (the
barge-in case) and being a safe no-op when called on an already-closed state.
"""

from typing import List, Literal, Optional, Sequence, Tuple, Union, cast

from litellm.types.llms.openai import OpenAIRealtimeEvents, ResponseAPIUsage

from . import events
from .events import SpeechStartedEvent
from .state import (
    ClosedItem,
    ContentDeltaType,
    ItemType,
    OpenContentPart,
    OpenItem,
    OpenResponse,
    RealtimeCorrelationError,
    RealtimeCorrelationState,
)

CorrelationEvent = Union[OpenAIRealtimeEvents, SpeechStartedEvent]
EventTuple = Tuple[CorrelationEvent, ...]


class ToolCallRequest:
    """One function-call request to be emitted as a fully-formed tool-call turn."""

    __slots__ = ("call_id", "name", "arguments")

    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


def open_response(
    state: RealtimeCorrelationState,
    response_id: str,
    conversation_id: str,
    modalities: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is not None and state.response.response_id == response_id:
        return state, ()
    new_response = OpenResponse(response_id=response_id, conversation_id=conversation_id)
    event = events.build_response_created(
        response_id=response_id,
        conversation_id=conversation_id,
        modalities=modalities or ["audio"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    return RealtimeCorrelationState(response=new_response), (event,)


def open_item(
    state: RealtimeCorrelationState,
    item_id: str,
    item_type: ItemType = "message",
    role: Literal["assistant"] = "assistant",
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is None:
        raise RealtimeCorrelationError(
            f"open_item({item_id!r}) called with no response open; call open_response() first"
        )
    response = state.response
    output_index = response.next_output_index
    new_item = OpenItem(item_id=item_id, output_index=output_index, role=role, item_type=item_type)
    new_response = OpenResponse(
        response_id=response.response_id,
        conversation_id=response.conversation_id,
        open_items=response.open_items + (new_item,),
        closed_items=response.closed_items,
        next_output_index=output_index + 1,
    )
    added_event = events.build_output_item_added(
        response_id=response.response_id,
        output_index=output_index,
        item_id=item_id,
        item_type=item_type,
        role=role,
    )
    conversation_added_event = events.build_conversation_item_added(
        item_id=item_id,
        item_type=item_type,
        role=role,
    )
    return RealtimeCorrelationState(response=new_response), (added_event, conversation_added_event)


def _find_open_item(response: OpenResponse, item_id: str) -> Optional[OpenItem]:
    for item in response.open_items:
        if item.item_id == item_id:
            return item
    return None


def _replace_open_item(response: OpenResponse, updated: OpenItem) -> OpenResponse:
    new_open_items = tuple(updated if item.item_id == updated.item_id else item for item in response.open_items)
    return OpenResponse(
        response_id=response.response_id,
        conversation_id=response.conversation_id,
        open_items=new_open_items,
        closed_items=response.closed_items,
        next_output_index=response.next_output_index,
    )


def open_content_part(
    state: RealtimeCorrelationState,
    item_id: str,
    delta_type: ContentDeltaType,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is None:
        raise RealtimeCorrelationError(f"open_content_part({item_id!r}) called with no response open")
    item = _find_open_item(state.response, item_id)
    if item is None:
        raise RealtimeCorrelationError(f"open_content_part({item_id!r}) called for an item that is not open")
    content_index = len(item.content_parts)
    new_part = OpenContentPart(content_index=content_index, delta_type=delta_type)
    updated_item = OpenItem(
        item_id=item.item_id,
        output_index=item.output_index,
        role=item.role,
        item_type=item.item_type,
        content_parts=item.content_parts + (new_part,),
    )
    new_response = _replace_open_item(state.response, updated_item)
    event = events.build_content_part_added(
        response_id=new_response.response_id,
        item_id=item_id,
        output_index=item.output_index,
        content_index=content_index,
        delta_type=delta_type,
    )
    return RealtimeCorrelationState(response=new_response), (event,)


def append_content_delta(
    state: RealtimeCorrelationState,
    item_id: str,
    content_index: int,
    delta_text: str,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is None:
        raise RealtimeCorrelationError(f"append_content_delta({item_id!r}) called with no response open")
    item = _find_open_item(state.response, item_id)
    if item is None:
        raise RealtimeCorrelationError(f"append_content_delta({item_id!r}) called for an item that is not open")
    part = next((p for p in item.content_parts if p.content_index == content_index), None)
    if part is None:
        raise RealtimeCorrelationError(
            f"append_content_delta({item_id!r}, content_index={content_index}) called for a content part "
            "that was never opened"
        )
    updated_part = OpenContentPart(
        content_index=part.content_index,
        delta_type=part.delta_type,
        accumulated_text=part.accumulated_text + delta_text if part.delta_type == "text" else "",
    )
    new_parts = tuple(updated_part if p.content_index == content_index else p for p in item.content_parts)
    updated_item = OpenItem(
        item_id=item.item_id,
        output_index=item.output_index,
        role=item.role,
        item_type=item.item_type,
        content_parts=new_parts,
    )
    new_response = _replace_open_item(state.response, updated_item)
    event = events.build_content_delta(
        response_id=new_response.response_id,
        item_id=item_id,
        output_index=item.output_index,
        content_index=content_index,
        delta_type=part.delta_type,
        delta=delta_text,
    )
    return RealtimeCorrelationState(response=new_response), (event,)


def close_item(
    state: RealtimeCorrelationState,
    item_id: str,
    status: Literal["completed", "incomplete"] = "completed",
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is None:
        return state, ()
    item = _find_open_item(state.response, item_id)
    if item is None:
        # Already closed, or never opened: idempotent no-op by design — this is
        # what replaces every provider-local "did I already close this?" flag.
        return state, ()
    response = state.response
    content_done_events: List[OpenAIRealtimeEvents] = [
        events.build_content_part_done(
            response_id=response.response_id,
            item_id=item_id,
            output_index=item.output_index,
            content_part=part,
        )
        for part in item.content_parts
    ]
    accumulated_text = "".join(part.accumulated_text for part in item.content_parts if part.delta_type == "text")
    closed_item = ClosedItem(
        item_id=item.item_id,
        output_index=item.output_index,
        status=status,
        item_type=item.item_type,
        role=item.role,
        text=accumulated_text,
    )
    output_item_done_event = events.build_output_item_done(
        response_id=response.response_id,
        item=item,
        status=status,
    )
    new_response = OpenResponse(
        response_id=response.response_id,
        conversation_id=response.conversation_id,
        open_items=tuple(i for i in response.open_items if i.item_id != item_id),
        closed_items=response.closed_items + (closed_item,),
        next_output_index=response.next_output_index,
    )
    return RealtimeCorrelationState(response=new_response), tuple(content_done_events) + (output_item_done_event,)


def close_response(
    state: RealtimeCorrelationState,
    usage: Optional[ResponseAPIUsage] = None,
    modalities: Optional[List[str]] = None,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    if state.response is None:
        # Already closed by a prior call — this is the generalization of
        # provider-local "suppress the second empty response.done" flags: calling
        # close_response twice in a row is simply safe by construction.
        return state, ()

    current_state = state
    all_events: List[CorrelationEvent] = []
    for item in state.response.open_items:
        current_state, close_events = close_item(current_state, item.item_id, status="incomplete")
        all_events.extend(close_events)

    assert current_state.response is not None  # every open item was just closed above
    response = current_state.response
    done_event = events.build_response_done(
        response_id=response.response_id,
        conversation_id=response.conversation_id,
        closed_items=list(response.closed_items),
        modalities=modalities or ["audio"],
        usage=usage,
    )
    all_events.append(done_event)
    return RealtimeCorrelationState(response=None), tuple(all_events)


def track_output_index(
    state: RealtimeCorrelationState,
    response_id: str,
    item_id: str,
) -> Tuple[RealtimeCorrelationState, int]:
    """Idempotent query: return the `output_index` already assigned to `item_id`,
    or allocate the next one if this is the first time it's been seen.

    Unlike `open_item`, this never emits events — it's for providers (e.g. xAI)
    whose backend already sent the real wire event and only needs the shared
    module's index bookkeeping, not synthesized lifecycle events. Also unlike
    `open_item`, a missing response is opened silently rather than raising,
    since the caller here is reacting to an already-arrived backend event, not
    driving the lifecycle forward itself.
    """
    if state.response is None or state.response.response_id != response_id:
        state = RealtimeCorrelationState(response=OpenResponse(response_id=response_id, conversation_id=response_id))
    assert state.response is not None
    existing = _find_open_item(state.response, item_id)
    if existing is not None:
        return state, existing.output_index
    new_state, _ = open_item(state, item_id)
    assert new_state.response is not None
    tracked = _find_open_item(new_state.response, item_id)
    assert tracked is not None
    return new_state, tracked.output_index


def track_content_index(
    state: RealtimeCorrelationState,
    response_id: str,
    item_id: str,
    content_part_key: str,
) -> Tuple[RealtimeCorrelationState, int]:
    """Idempotent query: return the `content_index` already assigned to
    `content_part_key` within `item_id`, or allocate the next one (scoped to
    that item) if this is the first time it's been seen.

    `content_part_key` disambiguates multiple content parts on the same item
    when the backend doesn't provide its own content_index (e.g. keyed by
    modality — "text"/"audio" — for providers that emit at most one part per
    modality per item). Never emits events, mirroring `track_output_index`.
    """
    state, _ = track_output_index(state, response_id, item_id)
    assert state.response is not None
    item = _find_open_item(state.response, item_id)
    assert item is not None
    for index, part in enumerate(item.content_parts):
        if part.delta_type == content_part_key:
            return state, index
    new_state, _ = open_content_part(state, item_id, cast(ContentDeltaType, content_part_key))
    assert new_state.response is not None
    updated_item = _find_open_item(new_state.response, item_id)
    assert updated_item is not None
    return new_state, len(updated_item.content_parts) - 1


def cancel_response(
    state: RealtimeCorrelationState,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    speech_started = events.build_speech_started()
    new_state, close_events = close_response(state)
    return new_state, (speech_started,) + close_events


def tool_call_events(
    state: RealtimeCorrelationState,
    response_id: str,
    conversation_id: str,
    calls: Sequence[ToolCallRequest],
    usage: Optional[ResponseAPIUsage] = None,
) -> Tuple[RealtimeCorrelationState, EventTuple]:
    current_state, created_events = open_response(state, response_id=response_id, conversation_id=conversation_id)
    all_events: List[CorrelationEvent] = list(created_events)

    for call in calls:
        item_id = f"item_{call.call_id}"
        current_state, opened_events = open_item(current_state, item_id=item_id, item_type="function_call")
        all_events.extend(opened_events)
        assert current_state.response is not None
        item = _find_open_item(current_state.response, item_id)
        assert item is not None
        all_events.append(
            events.build_function_call_arguments_done(
                response_id=response_id,
                item_id=item_id,
                output_index=item.output_index,
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
        )
        current_state, closed_events = close_item(current_state, item_id=item_id, status="completed")
        all_events.extend(closed_events)

    current_state, done_events = close_response(current_state, usage=usage)
    all_events.extend(done_events)
    return current_state, tuple(all_events)
