"""Capability descriptors for realtime provider x model pairs.

The bridge derives its decisions (how a turn completes, which tools are
allowed, whether follow-up setup is forwarded, how a session continues,
what bills it, whose token must be refreshed) from these descriptors
instead of branching on provider/model names.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final


class RealtimeTurnModel(str, Enum):
    """How the backend signals turn boundaries."""

    SINGLE_RESPONSE = "single-response"
    UTTERANCE_PLUS_STATUS = "utterance-plus-status"
    NO_RESPONSE_BOUNDARY = "no-response-boundary"


class RealtimeToolMode(str, Enum):
    """Whether the backend accepts synchronous tool calls."""

    BLOCKING = "blocking"
    NON_BLOCKING_ASYNC = "non-blocking-async"


class RealtimeSetupModel(str, Enum):
    """Whether follow-up session configuration is accepted mid-session."""

    ONCE = "once"
    UPDATES_ALLOWED = "updates-allowed"
    RESTRICTED = "restricted"


class RealtimeContinuation(str, Enum):
    """Session-continuation mechanisms, ordered cheapest-first."""

    NATIVE_HANDLE = "native-handle"
    FORK = "fork"
    SEED_HISTORY = "seed-history"
    TRANSCRIPT_REPLAY = "transcript-replay"


class RealtimeBillingUnit(str, Enum):
    """What the provider bills the voice session in."""

    TOKENS = "tokens"
    VOICE_SECONDS_PLUS_BACKEND_TOKENS = "voice-seconds-plus-backend-tokens"


@dataclass(frozen=True)
class RealtimeCapability:
    turn_model: RealtimeTurnModel
    tool_mode: RealtimeToolMode
    setup_model: RealtimeSetupModel
    continuation: tuple
    billing_unit: RealtimeBillingUnit
    refresh_token_on_reconnect: bool


_OPENAI_COMPATIBLE: Final = RealtimeCapability(
    turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
    tool_mode=RealtimeToolMode.BLOCKING,
    setup_model=RealtimeSetupModel.UPDATES_ALLOWED,
    continuation=(RealtimeContinuation.TRANSCRIPT_REPLAY,),
    billing_unit=RealtimeBillingUnit.TOKENS,
    refresh_token_on_reconnect=False,
)

_GEMINI_FAMILY: Final = RealtimeCapability(
    turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
    tool_mode=RealtimeToolMode.BLOCKING,
    setup_model=RealtimeSetupModel.ONCE,
    continuation=(
        RealtimeContinuation.NATIVE_HANDLE,
        RealtimeContinuation.TRANSCRIPT_REPLAY,
    ),
    billing_unit=RealtimeBillingUnit.TOKENS,
    refresh_token_on_reconnect=False,
)

PROVIDER_CAPABILITIES: Final[Mapping[str, RealtimeCapability]] = {
    "openai": _OPENAI_COMPATIBLE,
    "azure": _OPENAI_COMPATIBLE,
    "azure_ai": _OPENAI_COMPATIBLE,
    "xai": RealtimeCapability(
        turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
        tool_mode=RealtimeToolMode.BLOCKING,
        setup_model=RealtimeSetupModel.UPDATES_ALLOWED,
        continuation=(
            RealtimeContinuation.NATIVE_HANDLE,
            RealtimeContinuation.TRANSCRIPT_REPLAY,
        ),
        billing_unit=RealtimeBillingUnit.TOKENS,
        refresh_token_on_reconnect=False,
    ),
    "yandex": _OPENAI_COMPATIBLE,
    "gemini": _GEMINI_FAMILY,
    "meta": RealtimeCapability(
        turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
        tool_mode=RealtimeToolMode.BLOCKING,
        setup_model=RealtimeSetupModel.ONCE,
        continuation=(RealtimeContinuation.TRANSCRIPT_REPLAY,),
        billing_unit=RealtimeBillingUnit.TOKENS,
        refresh_token_on_reconnect=False,
    ),
    "vertex_ai": RealtimeCapability(
        turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
        tool_mode=RealtimeToolMode.BLOCKING,
        setup_model=RealtimeSetupModel.ONCE,
        continuation=(
            RealtimeContinuation.NATIVE_HANDLE,
            RealtimeContinuation.TRANSCRIPT_REPLAY,
        ),
        billing_unit=RealtimeBillingUnit.TOKENS,
        refresh_token_on_reconnect=True,
    ),
    "bedrock": RealtimeCapability(
        turn_model=RealtimeTurnModel.SINGLE_RESPONSE,
        tool_mode=RealtimeToolMode.BLOCKING,
        setup_model=RealtimeSetupModel.UPDATES_ALLOWED,
        continuation=(),
        billing_unit=RealtimeBillingUnit.TOKENS,
        refresh_token_on_reconnect=False,
    ),
}

MODEL_CAPABILITY_OVERRIDES: Final[Mapping[str, RealtimeCapability]] = {
    "gemini-3.8-live": _GEMINI_FAMILY,
    "gemini-3.8-live-extended-thinking": RealtimeCapability(
        turn_model=RealtimeTurnModel.UTTERANCE_PLUS_STATUS,
        tool_mode=RealtimeToolMode.NON_BLOCKING_ASYNC,
        setup_model=RealtimeSetupModel.ONCE,
        continuation=(
            RealtimeContinuation.NATIVE_HANDLE,
            RealtimeContinuation.TRANSCRIPT_REPLAY,
        ),
        billing_unit=RealtimeBillingUnit.TOKENS,
        refresh_token_on_reconnect=False,
    ),
    "gpt-live-1": RealtimeCapability(
        turn_model=RealtimeTurnModel.NO_RESPONSE_BOUNDARY,
        tool_mode=RealtimeToolMode.NON_BLOCKING_ASYNC,
        setup_model=RealtimeSetupModel.RESTRICTED,
        continuation=(
            RealtimeContinuation.FORK,
            RealtimeContinuation.SEED_HISTORY,
            RealtimeContinuation.TRANSCRIPT_REPLAY,
        ),
        billing_unit=RealtimeBillingUnit.VOICE_SECONDS_PLUS_BACKEND_TOKENS,
        refresh_token_on_reconnect=False,
    ),
}

_DEFAULT_CAPABILITY: Final = _OPENAI_COMPATIBLE


def get_realtime_capability(provider: str, model: str | None = None) -> RealtimeCapability:
    """Resolve the capability descriptor for a provider x model pair.

    Model overrides win over provider defaults; unknown providers fall back
    to the OpenAI-compatible default and never raise.
    """
    if model is not None:
        override = MODEL_CAPABILITY_OVERRIDES.get(model.split("/")[-1])
        if override is not None:
            return override
    return PROVIDER_CAPABILITIES.get(provider, _DEFAULT_CAPABILITY)
