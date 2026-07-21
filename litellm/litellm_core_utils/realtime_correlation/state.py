"""Immutable state types for the provider-agnostic realtime correlation-id lifecycle.

These types are the single source of truth for `(response_id, item_id, output_index,
content_index)` correlation across every realtime provider. All updates are
non-mutating: every transition in ``lifecycle.py`` takes one of these and returns a
new one; nothing here is ever mutated in place.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

ContentDeltaType = Literal["text", "audio"]
ItemType = Literal["message", "function_call"]
ItemStatus = Literal["completed", "incomplete"]


@dataclass(frozen=True, slots=True)
class OpenContentPart:
    content_index: int
    delta_type: ContentDeltaType
    accumulated_text: str = ""


@dataclass(frozen=True, slots=True)
class OpenItem:
    item_id: str
    output_index: int
    role: Literal["assistant"] = "assistant"
    item_type: ItemType = "message"
    content_parts: Tuple[OpenContentPart, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ClosedItem:
    item_id: str
    output_index: int
    status: ItemStatus
    item_type: ItemType
    role: Literal["assistant"]
    text: str = ""


@dataclass(frozen=True, slots=True)
class OpenResponse:
    response_id: str
    conversation_id: str
    open_items: Tuple[OpenItem, ...] = field(default_factory=tuple)
    closed_items: Tuple[ClosedItem, ...] = field(default_factory=tuple)
    next_output_index: int = 0


@dataclass(frozen=True, slots=True)
class RealtimeCorrelationState:
    response: Optional[OpenResponse] = None


class RealtimeCorrelationError(ValueError):
    """Raised when a caller violates the lifecycle contract (e.g. opening an item
    with no response open). This is a programming-contract violation, not a
    modelable runtime value — callers must open a response before an item."""
