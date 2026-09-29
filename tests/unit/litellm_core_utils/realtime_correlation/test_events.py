from typing import List, Mapping, Optional, cast

from litellm.litellm_core_utils.realtime_correlation import events
from litellm.litellm_core_utils.realtime_correlation.state import ClosedItem, OpenContentPart, OpenItem


def _get(mapping: Mapping[str, object], key: str) -> Optional[object]:
    """Read a key off a ``total=False`` TypedDict without fighting pyright's
    per-literal overloaded ``.get()`` signature; these builders always populate
    the fields under test, this only avoids re-deriving that overload here."""
    return mapping.get(key)


def test_build_response_created_shape():
    event = events.build_response_created(
        response_id="resp_1",
        conversation_id="conv_1",
        modalities=["audio"],
        temperature=0.8,
        max_output_tokens=100,
    )
    assert event["type"] == "response.created"
    response = cast(Mapping[str, object], event["response"])
    assert _get(response, "id") == "resp_1"
    assert _get(response, "conversation_id") == "conv_1"
    assert _get(response, "status") == "in_progress"


def test_build_output_item_added_shape():
    event = events.build_output_item_added(
        response_id="resp_1",
        output_index=2,
        item_id="item_1",
        item_type="message",
        role="assistant",
    )
    assert event["type"] == "response.output_item.added"
    assert event["response_id"] == "resp_1"
    assert event["output_index"] == 2
    assert _get(event["item"], "id") == "item_1"
    assert _get(event["item"], "status") == "in_progress"


def test_build_conversation_item_added_shape():
    event = events.build_conversation_item_added(item_id="item_1", item_type="message", role="assistant")
    assert event["type"] == "conversation.item.added"
    item = _get(event, "item")
    assert isinstance(item, dict)
    assert _get(cast(Mapping[str, object], item), "id") == "item_1"


def test_build_content_part_added_shape():
    event = events.build_content_part_added(
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        content_index=1,
        delta_type="text",
    )
    assert event["type"] == "response.content_part.added"
    assert event["content_index"] == 1
    assert event["output_index"] == 0
    assert _get(event["part"], "type") == "text"


def test_build_content_delta_text_uses_ga_event_name():
    event = events.build_content_delta(
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        content_index=0,
        delta_type="text",
        delta="hello",
    )
    assert event["type"] == "response.output_text.delta"
    assert event["delta"] == "hello"


def test_build_content_delta_audio_uses_ga_event_name():
    event = events.build_content_delta(
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        content_index=0,
        delta_type="audio",
        delta="abcd",
    )
    assert event["type"] == "response.output_audio.delta"


def test_build_content_part_done_includes_accumulated_text():
    part = OpenContentPart(content_index=0, delta_type="text", accumulated_text="hello world")
    event = events.build_content_part_done(
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        content_part=part,
    )
    assert event["type"] == "response.content_part.done"
    assert _get(event["part"], "text") == "hello world"


def test_build_output_item_done_completed_status():
    item = OpenItem(
        item_id="item_1",
        output_index=0,
        content_parts=(OpenContentPart(content_index=0, delta_type="text", accumulated_text="hi"),),
    )
    event = events.build_output_item_done(response_id="resp_1", item=item, status="completed")
    assert event["type"] == "response.output_item.done"
    assert _get(event["item"], "status") == "completed"
    assert _get(event["item"], "content") == [{"type": "text", "text": "hi"}]


def test_build_output_item_done_incomplete_status():
    item = OpenItem(item_id="item_1", output_index=0)
    event = events.build_output_item_done(response_id="resp_1", item=item, status="incomplete")
    assert _get(event["item"], "status") == "incomplete"


def test_build_response_done_includes_all_closed_items_in_order():
    closed_items = [
        ClosedItem(
            item_id="item_1", output_index=0, status="completed", item_type="message", role="assistant", text="hi"
        ),
        ClosedItem(
            item_id="item_2", output_index=1, status="incomplete", item_type="message", role="assistant", text=""
        ),
    ]
    event = events.build_response_done(
        response_id="resp_1",
        conversation_id="conv_1",
        closed_items=closed_items,
        modalities=["audio"],
        usage=None,
    )
    assert event["type"] == "response.done"
    output = _get(event["response"], "output")
    assert isinstance(output, list)
    items = cast(List[Mapping[str, object]], output)
    output_ids = [_get(item, "id") for item in items]
    assert output_ids == ["item_1", "item_2"]
    statuses = [_get(item, "status") for item in items]
    assert statuses == ["completed", "incomplete"]


def test_build_response_done_empty_closed_items_gives_empty_output():
    event = events.build_response_done(
        response_id="resp_1",
        conversation_id="conv_1",
        closed_items=[],
        modalities=["audio"],
        usage=None,
    )
    output = _get(event["response"], "output")
    assert output == []


def test_build_speech_started_shape():
    event = events.build_speech_started()
    assert event["type"] == "input_audio_buffer.speech_started"
    assert "item_id" in event


def test_build_function_call_arguments_done_shape():
    event = events.build_function_call_arguments_done(
        response_id="resp_1",
        item_id="item_1",
        output_index=0,
        call_id="call_1",
        name="get_weather",
        arguments='{"city": "Moscow"}',
    )
    assert event["type"] == "response.function_call_arguments.done"
    assert event["call_id"] == "call_1"
    assert event["name"] == "get_weather"
