import json
import os

import pytest

from litellm.litellm_core_utils.openrouter_catalog import (
    build_openrouter_entries,
    entries_digest,
    parse_openrouter_models,
)

_URL = "https://example.invalid/models"


def _model(**overrides):
    base = {
        "id": "vendor/model-a",
        "context_length": 8000,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        "top_provider": {"max_completion_tokens": 1000},
        "supported_parameters": [],
    }
    return {**base, **overrides}


def _entries(*models):
    return build_openrouter_entries({"data": list(models)}, _URL)


def test_entry_carries_prices_limits_and_provider():
    entry = _entries(_model())["openrouter/vendor/model-a"]

    assert entry["litellm_provider"] == "openrouter"
    assert entry["mode"] == "chat"
    assert entry["input_cost_per_token"] == pytest.approx(1e-6)
    assert entry["output_cost_per_token"] == pytest.approx(2e-6)
    assert entry["max_input_tokens"] == 8000
    assert entry["max_output_tokens"] == 1000
    assert entry["source"] == _URL


def test_cache_and_image_prices_map_to_their_cost_fields():
    pricing = {
        "prompt": "1",
        "completion": "2",
        "input_cache_read": "0.1",
        "input_cache_write": "0.5",
        "image": "0.01",
    }
    entry = _entries(_model(pricing=pricing))["openrouter/vendor/model-a"]

    assert entry["cache_read_input_token_cost"] == pytest.approx(0.1)
    assert entry["cache_creation_input_token_cost"] == pytest.approx(0.5)
    assert entry["input_cost_per_image"] == pytest.approx(0.01)
    assert entry["supports_prompt_caching"] is True


@pytest.mark.parametrize(
    "pricing", [{"prompt": "-1", "completion": "-1"}, {"prompt": "1"}, {"prompt": "x", "completion": "1"}]
)
def test_model_without_a_valid_price_is_skipped(pricing):
    assert _entries(_model(pricing=pricing), _model(id="vendor/ok")).keys() == {"openrouter/vendor/ok"}


def test_missing_output_limit_is_not_written():
    entry = _entries(_model(top_provider={"max_completion_tokens": None}))["openrouter/vendor/model-a"]

    assert "max_output_tokens" not in entry


def test_capabilities_follow_supported_parameters_and_modalities():
    model = _model(
        supported_parameters=["tools", "tool_choice", "structured_outputs", "reasoning"],
        architecture={"input_modalities": ["text", "image", "file", "audio"], "output_modalities": ["text"]},
    )
    entry = _entries(model)["openrouter/vendor/model-a"]

    assert {
        "supports_function_calling",
        "supports_tool_choice",
        "supports_response_schema",
        "supports_reasoning",
        "supports_vision",
        "supports_pdf_input",
        "supports_audio_input",
    } <= entry.keys()


def test_model_without_capabilities_has_no_true_flags():
    entry = _entries(_model())["openrouter/vendor/model-a"]

    assert not [name for name in entry if name.startswith("supports_")]


@pytest.mark.parametrize(
    "outputs, expected",
    [(["image"], "image_generation"), (["text", "image"], "chat"), (["text"], "chat"), (["text", "audio"], "chat")],
)
def test_mode_is_image_generation_only_for_image_only_output(outputs, expected):
    model = _model(architecture={"input_modalities": ["text"], "output_modalities": outputs})

    assert _entries(model)["openrouter/vendor/model-a"]["mode"] == expected


def test_malformed_model_is_skipped_without_dropping_the_rest():
    payload = {"data": [{"context_length": 1}, "nonsense", _model(id="vendor/ok")]}

    assert build_openrouter_entries(payload, _URL).keys() == {"openrouter/vendor/ok"}


@pytest.mark.parametrize("payload", [None, [], {"data": "x"}, {"other": []}])
def test_unexpected_payload_shape_yields_no_models(payload):
    assert parse_openrouter_models(payload) == ()


def test_digest_changes_with_price_and_membership_but_not_with_key_order():
    base = _entries(_model(id="vendor/a"), _model(id="vendor/b"))
    reordered = _entries(_model(id="vendor/b"), _model(id="vendor/a"))
    repriced = _entries(_model(id="vendor/a", pricing={"prompt": "9", "completion": "2"}), _model(id="vendor/b"))
    shrunk = _entries(_model(id="vendor/a"))

    assert entries_digest(base) == entries_digest(reordered)
    assert len({entries_digest(base), entries_digest(repriced), entries_digest(shrunk)}) == 3


def test_real_response_sample_builds_well_formed_entries():
    path = os.path.join(os.path.dirname(__file__), "openrouter_models_sample.json")
    with open(path) as f:
        payload = json.load(f)

    entries = build_openrouter_entries(payload, _URL)

    assert entries
    assert len(entries) <= len(payload["data"])
    for key, entry in entries.items():
        assert key.startswith("openrouter/")
        assert entry["litellm_provider"] == "openrouter"
        assert entry["input_cost_per_token"] >= 0
        assert entry["output_cost_per_token"] >= 0
