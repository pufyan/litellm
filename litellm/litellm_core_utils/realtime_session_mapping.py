"""Shared machinery for resolving canonical realtime session fields.

Per ``docs/realtime_session_contract.md`` every provider implementation resolves
each canonical field into exactly one of: map it, adapt it to the active model,
or drop it. Dropping is a normal outcome and is what guarantees that a canonical
payload never breaks a session -- but an unlogged drop is indistinguishable from
a working configuration, so this module makes the drop reason mandatory rather
than optional.
"""

from enum import Enum
from typing import Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

from litellm._logging import verbose_logger
from litellm.types.realtime_session import CANONICAL_SESSION_KEYS


class DropReason(str, Enum):
    """Why a canonical field did not reach the backend."""

    UNSUPPORTED_BY_PROVIDER = "unsupported_by_provider"
    UNSUPPORTED_BY_MODEL = "unsupported_by_model"
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"
    NOT_CANONICAL = "not_canonical"


def log_dropped_fields(
    provider: str,
    model: str,
    dropped: Mapping[str, DropReason],
) -> None:
    """Emit one warning naming every canonical field that was discarded.

    A dropped field is silent to the client by contract, so this log is the only
    signal an operator gets that a requested behavior never took effect.
    """
    if not dropped:
        return
    detail = ", ".join(f"{field} ({reason.value})" for field, reason in sorted(dropped.items()))
    verbose_logger.warning(
        "realtime session.update: dropped %d field(s) for %s/%s: %s",
        len(dropped),
        provider,
        model,
        detail,
    )


def split_canonical_session(
    session: Mapping[str, object],
    supported: FrozenSet[str],
) -> Tuple[Dict[str, object], Dict[str, DropReason]]:
    """Partition a canonical session into the fields to map and the ones to drop.

    Keys outside the canonical schema are dropped as ``NOT_CANONICAL``: the
    contract gives every field exactly one client-facing address, so a backend's
    native spelling is not a second way to say the same thing.
    """
    kept: Dict[str, object] = {}
    dropped: Dict[str, DropReason] = {}
    for key, value in session.items():
        if key not in CANONICAL_SESSION_KEYS:
            dropped[key] = DropReason.NOT_CANONICAL
        elif key not in supported:
            dropped[key] = DropReason.UNSUPPORTED_BY_PROVIDER
        else:
            kept[key] = value
    return kept, dropped


def drop_unsupported_by_model(
    session: Dict[str, object],
    unsupported: Iterable[str],
) -> Dict[str, DropReason]:
    """Remove fields the *active model* cannot honor, mutating ``session``.

    Capability varies by model, not only by provider, so an implementation calls
    this after ``split_canonical_session`` with the subset its current model
    rejects.
    """
    dropped: Dict[str, DropReason] = {}
    for key in unsupported:
        if key in session:
            del session[key]
            dropped[key] = DropReason.UNSUPPORTED_BY_MODEL
    return dropped


def resolve_mutually_exclusive(
    session: Dict[str, object],
    keep: str,
    discard: Iterable[str],
) -> Dict[str, DropReason]:
    """Keep one of several fields that express the same intent, mutating ``session``.

    Used where a backend accepts only one spelling of a concept -- e.g. a
    reasoning token budget versus a reasoning effort level, which are separate
    canonical fields precisely because they are different concepts.
    """
    dropped: Dict[str, DropReason] = {}
    if keep not in session:
        return dropped
    for key in discard:
        if key in session:
            del session[key]
            dropped[key] = DropReason.MUTUALLY_EXCLUSIVE
    return dropped


def select_reasoning_field(
    session: Mapping[str, object],
    model_uses: str,
) -> Tuple[Optional[str], Dict[str, DropReason]]:
    """Pick the reasoning field the active model expresses effort in.

    ``thinking_budget`` (a token count) and ``thinking_level`` (an effort level)
    are different concepts and no known model accepts both. When a client sends
    both, the one the model does not use is dropped as mutually exclusive.
    """
    other = "thinking_level" if model_uses == "thinking_budget" else "thinking_budget"
    dropped: Dict[str, DropReason] = {}
    if other in session:
        dropped[other] = DropReason.MUTUALLY_EXCLUSIVE if model_uses in session else DropReason.UNSUPPORTED_BY_MODEL
    return (model_uses if model_uses in session else None), dropped
