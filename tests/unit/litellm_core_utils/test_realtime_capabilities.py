import dataclasses

import pytest

from litellm.litellm_core_utils.realtime_capabilities import (
    RealtimeBillingUnit,
    RealtimeCapability,
    RealtimeContinuation,
    RealtimeSetupModel,
    RealtimeToolMode,
    RealtimeTurnModel,
    get_realtime_capability,
)


@pytest.mark.parametrize(
    "provider",
    ["openai", "azure", "azure_ai", "xai", "yandex", "gemini", "meta", "vertex_ai", "bedrock"],
)
def test_every_dispatched_provider_resolves_a_descriptor(provider: str):
    capability = get_realtime_capability(provider, model="anything")

    assert isinstance(capability, RealtimeCapability)
    assert capability.turn_model == RealtimeTurnModel.SINGLE_RESPONSE
    assert capability.billing_unit == RealtimeBillingUnit.TOKENS


def test_unknown_provider_falls_back_to_openai_compatible_without_raising():
    capability = get_realtime_capability("not-a-provider-yet")

    assert capability == get_realtime_capability("openai")


def test_vertex_ai_flags_token_refresh_on_reconnect():
    assert get_realtime_capability("vertex_ai").refresh_token_on_reconnect is True
    assert get_realtime_capability("gemini").refresh_token_on_reconnect is False


def test_gemini_family_rejects_follow_up_setup():
    assert get_realtime_capability("gemini").setup_model == RealtimeSetupModel.ONCE
    assert get_realtime_capability("meta").setup_model == RealtimeSetupModel.ONCE
    assert get_realtime_capability("vertex_ai").setup_model == RealtimeSetupModel.ONCE


def test_gemini_family_prefers_native_resumption_handle():
    for provider in ("gemini", "vertex_ai", "xai"):
        assert get_realtime_capability(provider).continuation[0] == RealtimeContinuation.NATIVE_HANDLE


def test_bedrock_has_no_continuation_mechanism():
    assert get_realtime_capability("bedrock").continuation == ()


def test_extended_thinking_model_override_wins_over_gemini_default():
    capability = get_realtime_capability("gemini", model="gemini-3.8-live-extended-thinking")

    assert capability.turn_model == RealtimeTurnModel.UTTERANCE_PLUS_STATUS
    assert capability.tool_mode == RealtimeToolMode.NON_BLOCKING_ASYNC


def test_gpt_live_model_override_wins_over_openai_default():
    capability = get_realtime_capability("openai", model="openai/gpt-live-1")

    assert capability.turn_model == RealtimeTurnModel.NO_RESPONSE_BOUNDARY
    assert capability.setup_model == RealtimeSetupModel.RESTRICTED
    assert capability.billing_unit == RealtimeBillingUnit.VOICE_SECONDS_PLUS_BACKEND_TOKENS
    assert capability.continuation[0] == RealtimeContinuation.FORK


def test_capability_is_immutable_data():
    capability = get_realtime_capability("openai")

    with pytest.raises(dataclasses.FrozenInstanceError):
        capability.refresh_token_on_reconnect = True  # type: ignore[misc]
