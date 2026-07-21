from typing import List

from litellm.litellm_core_utils.realtime_correlation import RealtimeCorrelationState
from litellm.llms.xai.realtime.transformation import XAIRealtimeNormalizer


def _normalizer_state():
    return XAIRealtimeNormalizer(), RealtimeCorrelationState()


def test_multi_item_output_index_increments_instead_of_staying_zero():
    """Regression test for the hardcoded output_index=0 bug: two distinct items
    on the same response must get distinct, increasing output_index values."""
    normalizer, state = _normalizer_state()

    event_a, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    event_b, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}, state
    )

    assert event_a["output_index"] == 0
    assert event_b["output_index"] == 1


def test_multi_item_full_sequence_output_index_stable_and_content_index_scoped_per_item():
    normalizer, state = _normalizer_state()

    added_a, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    part_a, state = normalizer.normalize(
        {"type": "response.content_part.added", "response_id": "resp_1", "item_id": "item_A"}, state
    )
    added_b, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}, state
    )
    part_b, state = normalizer.normalize(
        {"type": "response.content_part.added", "response_id": "resp_1", "item_id": "item_B"}, state
    )
    # item A must keep its original output_index even after item B is opened.
    # A second response.output_text.delta on item A must reuse the SAME
    # content_index as the first (same modality == same content part).
    delta_a_1, state = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"}, state
    )
    delta_a_2, state = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": " there"}, state
    )

    assert added_a["output_index"] == 0
    assert part_a["output_index"] == 0
    assert part_a["content_index"] == 0
    assert added_b["output_index"] == 1
    assert part_b["output_index"] == 1
    assert part_b["content_index"] == 0  # scoped per item, not global
    assert delta_a_1["output_index"] == delta_a_2["output_index"] == 0
    assert delta_a_1["content_index"] == delta_a_2["content_index"]


def test_audio_and_its_transcript_share_the_same_content_index():
    """output_audio and output_audio_transcript are two facets of the same
    content part (bytes + transcript) — they must resolve to the same
    content_index, mirroring Gemini's reference behavior."""
    normalizer, state = _normalizer_state()

    normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    _, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    audio_delta, state = normalizer.normalize(
        {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "abc"}, state
    )
    transcript_delta, state = normalizer.normalize(
        {"type": "response.output_audio_transcript.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"},
        state,
    )

    assert audio_delta["content_index"] == transcript_delta["content_index"] == 0


def test_multi_part_single_item_content_index_increments():
    normalizer, state = _normalizer_state()
    _, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )

    text_delta, state = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"}, state
    )
    audio_delta, state = normalizer.normalize(
        {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "abc"}, state
    )

    assert text_delta["content_index"] == 0
    assert audio_delta["content_index"] == 1


def test_idempotency_output_item_done_gets_same_index_as_earlier_added():
    normalizer, state = _normalizer_state()
    added, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    # Open a second item so a naive re-allocation would visibly diverge (index 1)
    # instead of staying pinned at the first item's index (0).
    _, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}, state
    )
    done, state = normalizer.normalize(
        {"type": "response.output_item.done", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )

    assert added["output_index"] == done["output_index"] == 0


def test_fields_already_present_pass_through_unchanged():
    normalizer, state = _normalizer_state()

    event, state = normalizer.normalize(
        {
            "type": "response.content_part.added",
            "response_id": "resp_1",
            "item_id": "item_A",
            "output_index": 5,
            "content_index": 7,
        },
        state,
    )

    assert event["output_index"] == 5
    assert event["content_index"] == 7


def test_event_types_outside_index_sets_are_untouched():
    normalizer, state = _normalizer_state()

    event, state = normalizer.normalize({"type": "response.created", "response": {}}, state)

    assert "output_index" not in event
    assert "content_index" not in event


def test_nested_item_id_output_item_added_and_done_resolve_correct_index():
    """Regression test for the nested-id case: response.output_item.added/.done
    carry their id at event["item"]["id"], not a top-level item_id field. If
    that lookup branch is missing/wrong, these two event types silently keep
    getting output_index=0 regardless of how many items were already open."""
    normalizer, state = _normalizer_state()

    _, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}, state
    )
    added_b, state = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}, state
    )
    done_b, state = normalizer.normalize(
        {"type": "response.output_item.done", "response_id": "resp_1", "item": {"id": "item_B"}}, state
    )

    assert added_b["output_index"] == 1
    assert done_b["output_index"] == 1


def test_state_threads_correctly_across_repeated_normalize_calls():
    """Full-pipeline test proving state threads correctly across multiple
    normalize() calls simulating one connection, as RealTimeStreaming does."""
    normalizer = XAIRealtimeNormalizer()
    state = RealtimeCorrelationState()

    events_in = [
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}},
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}},
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_C"}},
    ]
    output_indices: List[int] = []
    for raw_event in events_in:
        normalized, state = normalizer.normalize(raw_event, state)
        output_indices.append(normalized["output_index"])

    assert output_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Untouched-behavior guards: the signature change must not disturb the other
# three normalization passes.
# ---------------------------------------------------------------------------


def test_should_drop_still_drops_ping():
    normalizer, _ = _normalizer_state()
    assert normalizer.should_drop({"type": "ping"}) is True


def test_patch_outgoing_session_still_defaults_create_response():
    normalizer, _ = _normalizer_state()
    patched = normalizer.patch_outgoing_session({"turn_detection": {"type": "server_vad"}})
    assert patched["turn_detection"]["create_response"] is True


def test_content_part_backfill_still_works():
    normalizer, state = _normalizer_state()
    normalizer.normalize(
        {
            "type": "response.content_part.added",
            "response_id": "resp_1",
            "item_id": "item_A",
            "content_index": 0,
            "part": {"type": "audio", "transcript": ""},
        },
        state,
    )
    event, _ = normalizer.normalize(
        {"type": "response.content_part.done", "response_id": "resp_1", "item_id": "item_A", "content_index": 0},
        state,
    )
    assert event["part"]["type"] == "audio"


def test_conversation_item_added_role_remap_still_works():
    normalizer, state = _normalizer_state()
    event, _ = normalizer.normalize(
        {"type": "conversation.item.added", "item": {"role": "tool", "type": "function_call"}}, state
    )
    assert event["item"]["role"] == "assistant"


def test_usage_normalization_still_works():
    normalizer, state = _normalizer_state()
    event, _ = normalizer.normalize({"type": "response.done", "response": {"usage": {}}}, state)
    assert event["response"]["usage"]["total_tokens"] == 0
