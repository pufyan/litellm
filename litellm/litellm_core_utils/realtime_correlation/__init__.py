"""Provider-agnostic realtime correlation-id lifecycle.

Public surface for building an OpenAI-realtime-compatible event stream with a
guaranteed-consistent `(response_id, item_id, output_index, content_index)`
correlation key across every realtime provider.
"""

from .lifecycle import (
    ToolCallRequest,
    append_content_delta,
    cancel_response,
    close_item,
    close_response,
    open_content_part,
    open_item,
    open_response,
    tool_call_events,
    track_content_index,
    track_output_index,
)
from .state import (
    ClosedItem,
    ContentDeltaType,
    ItemStatus,
    ItemType,
    OpenContentPart,
    OpenItem,
    OpenResponse,
    RealtimeCorrelationError,
    RealtimeCorrelationState,
)

__all__ = [
    "ToolCallRequest",
    "append_content_delta",
    "cancel_response",
    "close_item",
    "close_response",
    "open_content_part",
    "open_item",
    "open_response",
    "tool_call_events",
    "track_content_index",
    "track_output_index",
    "ClosedItem",
    "ContentDeltaType",
    "ItemStatus",
    "ItemType",
    "OpenContentPart",
    "OpenItem",
    "OpenResponse",
    "RealtimeCorrelationError",
    "RealtimeCorrelationState",
]
