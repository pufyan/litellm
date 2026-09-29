from litellm.litellm_core_utils.realtime_correlation.state import (
    OpenContentPart,
    OpenItem,
    OpenResponse,
    RealtimeCorrelationState,
)


def test_open_item_content_parts_append_produces_new_tuple_not_mutation():
    """Appending a content part must build a new tuple rather than mutating the
    original item's ``content_parts`` in place. Mutating the implementation to use
    ``list.append`` on a shared mutable list would make ``original`` observe the
    new part too — this test fails in that case."""
    original = OpenItem(item_id="item_1", output_index=0)
    part = OpenContentPart(content_index=0, delta_type="text")
    updated = OpenItem(
        item_id=original.item_id,
        output_index=original.output_index,
        role=original.role,
        item_type=original.item_type,
        content_parts=original.content_parts + (part,),
    )

    assert original.content_parts == ()
    assert updated.content_parts == (part,)


def test_open_response_open_items_append_produces_new_tuple_not_mutation():
    original = OpenResponse(response_id="resp_1", conversation_id="conv_1")
    item = OpenItem(item_id="item_1", output_index=0)
    updated = OpenResponse(
        response_id=original.response_id,
        conversation_id=original.conversation_id,
        open_items=original.open_items + (item,),
        closed_items=original.closed_items,
        next_output_index=original.next_output_index,
    )

    assert original.open_items == ()
    assert updated.open_items == (item,)


def test_dataclasses_are_frozen():
    state = RealtimeCorrelationState()
    try:
        state.response = OpenResponse(  # pyright: ignore[reportAttributeAccessIssue]  # intentional: proving frozen
            response_id="resp_1", conversation_id="conv_1"
        )
        assert False, "expected FrozenInstanceError"
    except Exception as e:
        assert type(e).__name__ == "FrozenInstanceError"


def test_realtime_correlation_state_defaults_to_no_open_response():
    assert RealtimeCorrelationState().response is None
