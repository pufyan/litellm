from typing import List, Mapping, Optional, Sequence, cast

import pytest

from litellm.litellm_core_utils.realtime_correlation import (
    RealtimeCorrelationState,
    ToolCallRequest,
    append_content_delta,
    cancel_response,
    close_item,
    close_response,
    open_content_part,
    open_item,
    open_response,
    tool_call_events,
)
from litellm.litellm_core_utils.realtime_correlation.lifecycle import CorrelationEvent
from litellm.litellm_core_utils.realtime_correlation.state import RealtimeCorrelationError
from litellm.types.llms.openai import (
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeStreamResponseOutputItem,
    OpenAIRealtimeStreamResponseOutputItemAdded,
)


def _get(mapping: Mapping[str, object], key: str) -> Optional[object]:
    """Read a key off a ``total=False`` TypedDict without fighting pyright's
    per-literal overloaded ``.get()`` signature; these builders always populate
    the fields under test, this only avoids re-deriving that overload here."""
    return mapping.get(key)


def _opened_response() -> RealtimeCorrelationState:
    state, _ = open_response(RealtimeCorrelationState(), "resp_1", "conv_1")
    return state


def _find_response_done(events: Sequence[CorrelationEvent]) -> OpenAIRealtimeDoneEvent:
    for event in events:
        if event["type"] == "response.done":
            return cast(OpenAIRealtimeDoneEvent, event)
    raise AssertionError("no response.done event found")


def _done_output(events: Sequence[CorrelationEvent]) -> List[OpenAIRealtimeStreamResponseOutputItem]:
    output = _get(_find_response_done(events)["response"], "output")
    assert isinstance(output, list)
    return cast(List[OpenAIRealtimeStreamResponseOutputItem], output)


def _output_item_added_events(events: Sequence[CorrelationEvent]) -> List[OpenAIRealtimeStreamResponseOutputItemAdded]:
    return [
        cast(OpenAIRealtimeStreamResponseOutputItemAdded, e)
        for e in events
        if e["type"] == "response.output_item.added"
    ]


def test_open_item_allocates_strictly_incrementing_output_index_across_concurrent_items():
    """Regression test for the xAI/Bedrock/Gemini hardcoded output_index=0 bug
    class: two concurrently open items on one response must get distinct,
    increasing output_index values, not both 0."""
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = open_item(state, "item_2")

    assert state.response is not None
    indices = [item.output_index for item in state.response.open_items]
    assert indices == [0, 1]


def test_open_content_part_allocates_strictly_incrementing_content_index():
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = open_content_part(state, "item_1", "text")
    state, _ = open_content_part(state, "item_1", "audio")

    assert state.response is not None
    item = state.response.open_items[0]
    indices = [part.content_index for part in item.content_parts]
    assert indices == [0, 1]


def test_open_item_raises_when_no_response_open():
    with pytest.raises(RealtimeCorrelationError):
        open_item(RealtimeCorrelationState(), "item_1")


def test_close_response_synthesizes_incomplete_close_for_every_still_open_item():
    """Regression test generalizing the Gemini barge-in fix: any item that was
    opened but never explicitly closed must be closed as "incomplete" by
    close_response, so response.done.output never silently drops it."""
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = open_item(state, "item_2")

    state, events = close_response(state)

    statuses = {_get(item, "id"): _get(item, "status") for item in _done_output(events)}
    assert statuses == {"item_1": "incomplete", "item_2": "incomplete"}


def test_close_response_called_twice_in_a_row_is_a_noop_on_second_call():
    """Regression test replacing Gemini's `_turn_closed_by_interrupt` flag: a
    second close_response call (state.response already None) must produce zero
    events, not a spurious empty response.done."""
    state = _opened_response()
    state, first_events = close_response(state)
    assert any(e["type"] == "response.done" for e in first_events)

    state, second_events = close_response(state)

    assert second_events == ()


def test_close_response_output_includes_both_normally_closed_and_incomplete_items():
    """Regression test for the Bedrock output=[] bug class: response.done.output
    must reflect every item this response ever closed, whether closed
    explicitly (completed) or synthesized by close_response (incomplete)."""
    state = _opened_response()
    state, _ = open_item(state, "item_completed")
    state, _ = open_content_part(state, "item_completed", "text")
    state, _ = append_content_delta(state, "item_completed", 0, "hello")
    state, _ = close_item(state, "item_completed", status="completed")

    state, _ = open_item(state, "item_left_open")

    state, events = close_response(state)

    output = _done_output(events)
    ids_and_status = [(_get(item, "id"), _get(item, "status")) for item in output]
    assert ("item_completed", "completed") in ids_and_status
    assert ("item_left_open", "incomplete") in ids_and_status
    assert len(output) == 2


def test_close_item_on_unknown_item_id_is_a_noop():
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = close_item(state, "item_1")  # closes it for real

    state, events = close_item(state, "item_1")  # already closed

    assert events == ()


def test_close_item_never_opened_is_a_noop():
    state = _opened_response()

    state, events = close_item(state, "never_opened")

    assert events == ()


def test_tool_call_events_allocates_distinct_output_index_per_call():
    state = RealtimeCorrelationState()
    calls = [
        ToolCallRequest(call_id="call_1", name="get_weather", arguments='{"city": "Moscow"}'),
        ToolCallRequest(call_id="call_2", name="get_time", arguments="{}"),
    ]

    state, events = tool_call_events(state, "resp_1", "conv_1", calls)

    output_indices = [e["output_index"] for e in _output_item_added_events(events)]
    assert output_indices == [0, 1]


def test_tool_call_events_produces_exactly_one_closing_response_done():
    state = RealtimeCorrelationState()
    calls = [
        ToolCallRequest(call_id="call_1", name="get_weather", arguments="{}"),
        ToolCallRequest(call_id="call_2", name="get_time", arguments="{}"),
    ]

    state, events = tool_call_events(state, "resp_1", "conv_1", calls)

    done_events = [e for e in events if e["type"] == "response.done"]
    assert len(done_events) == 1
    output_ids = [_get(item, "id") for item in _done_output(events)]
    assert output_ids == ["item_call_1", "item_call_2"]
    assert state.response is None


def test_cancel_response_emits_speech_started_then_closes_open_items_incomplete():
    state = _opened_response()
    state, _ = open_item(state, "item_1")

    state, events = cancel_response(state)

    types = [e["type"] for e in events]
    assert types[0] == "input_audio_buffer.speech_started"
    assert "response.done" in types
    output = _done_output(events)
    assert _get(output[0], "status") == "incomplete"
    assert state.response is None


def test_open_response_is_idempotent_for_same_response_id():
    state = _opened_response()

    state2, events = open_response(state, "resp_1", "conv_1")

    assert events == ()
    assert state2 is state


def test_append_content_delta_accumulates_text_across_multiple_deltas():
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = open_content_part(state, "item_1", "text")

    state, _ = append_content_delta(state, "item_1", 0, "hello ")
    state, _ = append_content_delta(state, "item_1", 0, "world")

    assert state.response is not None
    part = state.response.open_items[0].content_parts[0]
    assert part.accumulated_text == "hello world"


def test_append_content_delta_does_not_accumulate_audio_text():
    state = _opened_response()
    state, _ = open_item(state, "item_1")
    state, _ = open_content_part(state, "item_1", "audio")

    state, _ = append_content_delta(state, "item_1", 0, "base64chunk1")
    state, _ = append_content_delta(state, "item_1", 0, "base64chunk2")

    assert state.response is not None
    part = state.response.open_items[0].content_parts[0]
    assert part.accumulated_text == ""


def test_full_lifecycle_matches_single_item_happy_path():
    """End-to-end sanity check of the full open->content->close->response.done
    sequence for a single-item response, mirroring the normal Gemini flow we
    verified live."""
    state = RealtimeCorrelationState()
    state, created = open_response(state, "resp_1", "conv_1")
    assert [e["type"] for e in created] == ["response.created"]

    state, opened = open_item(state, "item_1")
    assert [e["type"] for e in opened] == ["response.output_item.added", "conversation.item.added"]

    state, part_added = open_content_part(state, "item_1", "text")
    assert [e["type"] for e in part_added] == ["response.content_part.added"]

    state, delta = append_content_delta(state, "item_1", 0, "hi")
    assert [e["type"] for e in delta] == ["response.output_text.delta"]

    state, closed = close_item(state, "item_1", status="completed")
    assert [e["type"] for e in closed] == ["response.content_part.done", "response.output_item.done"]

    state, done = close_response(state)
    output = _done_output(done)
    assert len(output) == 1
    assert _get(output[0], "id") == "item_1"
    assert _get(output[0], "status") == "completed"
    assert _get(output[0], "content") == [{"type": "text", "text": "hi"}]
    assert state.response is None
