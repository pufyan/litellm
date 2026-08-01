from typing import List

import pytest

from litellm.llms.xai.realtime.transformation import XAIRealtimeNormalizer


def test_multi_item_output_index_increments_instead_of_staying_zero():
    """Regression test for the hardcoded output_index=0 bug: two distinct items
    on the same response must get distinct, increasing output_index values."""
    normalizer = XAIRealtimeNormalizer()

    event_a = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}
    )
    event_b = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}
    )

    assert event_a["output_index"] == 0
    assert event_b["output_index"] == 1


def test_multi_item_full_sequence_output_index_stable_and_content_index_scoped_per_item():
    normalizer = XAIRealtimeNormalizer()

    added_a = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}}
    )
    part_a = normalizer.normalize({"type": "response.content_part.added", "response_id": "resp_1", "item_id": "item_A"})
    added_b = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}
    )
    part_b = normalizer.normalize({"type": "response.content_part.added", "response_id": "resp_1", "item_id": "item_B"})
    # item A must keep its original output_index even after item B is opened.
    # A second response.output_text.delta on item A must reuse the SAME
    # content_index as the first (same modality == same content part).
    delta_a_1 = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"}
    )
    delta_a_2 = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": " there"}
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
    normalizer = XAIRealtimeNormalizer()

    normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}})
    normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}})
    audio_delta = normalizer.normalize(
        {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "abc"}
    )
    transcript_delta = normalizer.normalize(
        {"type": "response.output_audio_transcript.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"}
    )

    assert audio_delta["content_index"] == transcript_delta["content_index"] == 0


def test_multi_part_single_item_content_index_increments():
    normalizer = XAIRealtimeNormalizer()
    normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}})

    text_delta = normalizer.normalize(
        {"type": "response.output_text.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "hi"}
    )
    audio_delta = normalizer.normalize(
        {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_A", "delta": "abc"}
    )

    assert text_delta["content_index"] == 0
    assert audio_delta["content_index"] == 1


def test_idempotency_output_item_done_gets_same_index_as_earlier_added():
    normalizer = XAIRealtimeNormalizer()
    added = normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}})
    # Open a second item so a naive re-allocation would visibly diverge (index 1)
    # instead of staying pinned at the first item's index (0).
    normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}})
    done = normalizer.normalize({"type": "response.output_item.done", "response_id": "resp_1", "item": {"id": "item_A"}})

    assert added["output_index"] == done["output_index"] == 0


def test_fields_already_present_pass_through_unchanged():
    normalizer = XAIRealtimeNormalizer()

    event = normalizer.normalize(
        {
            "type": "response.content_part.added",
            "response_id": "resp_1",
            "item_id": "item_A",
            "output_index": 5,
            "content_index": 7,
        }
    )

    assert event["output_index"] == 5
    assert event["content_index"] == 7


def test_event_types_outside_index_sets_are_untouched():
    normalizer = XAIRealtimeNormalizer()

    event = normalizer.normalize({"type": "response.created", "response": {}})

    assert "output_index" not in event
    assert "content_index" not in event


def test_nested_item_id_output_item_added_and_done_resolve_correct_index():
    """Regression test for the nested-id case: response.output_item.added/.done
    carry their id at event["item"]["id"], not a top-level item_id field. If
    that lookup branch is missing/wrong, these two event types silently keep
    getting output_index=0 regardless of how many items were already open."""
    normalizer = XAIRealtimeNormalizer()

    normalizer.normalize({"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}})
    added_b = normalizer.normalize(
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}}
    )
    done_b = normalizer.normalize({"type": "response.output_item.done", "response_id": "resp_1", "item": {"id": "item_B"}})

    assert added_b["output_index"] == 1
    assert done_b["output_index"] == 1


def test_state_threads_correctly_across_repeated_normalize_calls():
    """Full-pipeline test proving state threads correctly across multiple
    normalize() calls simulating one connection, as RealTimeStreaming does."""
    normalizer = XAIRealtimeNormalizer()

    events_in = [
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_A"}},
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_B"}},
        {"type": "response.output_item.added", "response_id": "resp_1", "item": {"id": "item_C"}},
    ]
    output_indices: List[int] = []
    for raw_event in events_in:
        normalized = normalizer.normalize(raw_event)
        output_indices.append(normalized["output_index"])

    assert output_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# Untouched-behavior guards: the signature change must not disturb the other
# three normalization passes.
# ---------------------------------------------------------------------------


def test_should_drop_still_drops_ping():
    normalizer = XAIRealtimeNormalizer()
    assert normalizer.should_drop({"type": "ping"}) is True


def test_patch_outgoing_session_still_defaults_create_response():
    normalizer = XAIRealtimeNormalizer()
    patched = normalizer.patch_outgoing_session({"turn_detection": {"type": "server_vad"}})
    assert patched["turn_detection"]["create_response"] is True


def test_content_part_backfill_still_works():
    normalizer = XAIRealtimeNormalizer()
    normalizer.normalize(
        {
            "type": "response.content_part.added",
            "response_id": "resp_1",
            "item_id": "item_A",
            "content_index": 0,
            "part": {"type": "audio", "transcript": ""},
        }
    )
    event = normalizer.normalize(
        {"type": "response.content_part.done", "response_id": "resp_1", "item_id": "item_A", "content_index": 0}
    )
    assert event["part"]["type"] == "audio"


def test_conversation_item_added_role_remap_still_works():
    normalizer = XAIRealtimeNormalizer()
    event = normalizer.normalize({"type": "conversation.item.added", "item": {"role": "tool", "type": "function_call"}})
    assert event["item"]["role"] == "assistant"


def test_usage_normalization_still_works():
    normalizer = XAIRealtimeNormalizer()
    event = normalizer.normalize({"type": "response.done", "response": {"usage": {}}})
    assert event["response"]["usage"]["total_tokens"] == 0


class TestXaiOnlyFieldRestoration:
    """Canonical fields xAI honors that the OpenAI GA allowlist strips.

    xAI rides the OpenAI-compatible passthrough path, so a client's
    ``session.update`` is filtered against the GA schema before it reaches the
    backend. Anything GA has no field for -- reasoning effort, resumption --
    was therefore unreachable on xAI even though xAI documents both.
    """

    @staticmethod
    def _patched(canonical: dict) -> dict:
        from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

        normalizer = XAIRealtimeNormalizer()
        ga_session = RealTimeStreaming._remap_beta_session_to_ga(dict(canonical))
        return normalizer.patch_outgoing_session(ga_session, dict(canonical))

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "extreme", None])
    def test_reasoning_is_always_forced_off(self, level):
        """Realtime voice sessions need the model answering immediately, not
        deliberating: reasoning is always forced to xAI's lowest rung,
        regardless of what (or whether) the client's thinking_level requests --
        xAI defaults to "high" effort when the field is absent, so it can't be
        left unset either."""
        canonical: dict = {"thinking_level": level} if level is not None else {}
        assert self._patched(canonical)["reasoning"] == {"effort": "none"}

    def test_session_resumption_becomes_resumption(self):
        patched = self._patched({"session_resumption": {"enabled": True}})

        assert patched["resumption"] == {"enabled": True}

    def test_resumption_disabled_is_forwarded_explicitly(self):
        """``False`` is a request to turn xAI's default off, not an omission."""
        patched = self._patched({"session_resumption": {"enabled": False}})

        assert patched["resumption"] == {"enabled": False}

    def test_resumption_without_enabled_is_not_forwarded(self):
        assert "resumption" not in self._patched({"session_resumption": {}})

    def test_absent_fields_add_nothing_but_forced_reasoning(self):
        """resumption stays absent when the client didn't ask for it; reasoning
        is always present because it is forced, not restored from the client."""
        patched = self._patched({"instructions": "hi"})

        assert patched["reasoning"] == {"effort": "none"}
        assert "resumption" not in patched

    def test_ga_fields_still_survive_alongside_the_restored_ones(self):
        """The restoration must not clobber what the GA remap produced."""
        patched = self._patched(
            {"instructions": "be brief", "voice": "eve", "thinking_level": "low"}
        )

        assert patched["instructions"] == "be brief"
        assert patched["audio"]["output"]["voice"] == "eve"
        assert patched["reasoning"] == {"effort": "none"}

    def test_create_response_default_still_applies(self):
        """Pre-existing behavior: xAI does not default create_response for
        server_vad, so the normalizer fills it."""
        patched = self._patched({"turn_detection": {"type": "server_vad"}})

        assert patched["audio"]["input"]["turn_detection"]["create_response"] is True

    def test_without_canonical_session_ga_patch_runs_and_reasoning_still_forced(self):
        """Backends on the beta path pass no canonical session; the restoration
        half is a no-op then, but forcing reasoning off does not depend on it."""
        normalizer = XAIRealtimeNormalizer()

        patched = normalizer.patch_outgoing_session({"turn_detection": {"type": "server_vad"}})

        assert patched["turn_detection"]["create_response"] is True
        assert patched["reasoning"] == {"effort": "none"}


class TestXaiTranscriptionLanguage:
    """xAI biases recognition with ``language_hint`` and takes no transcription
    ``model``, while the GA normalizer drops any transcription config that has
    no model -- so a canonical ``language`` was unreachable on xAI entirely.
    """

    @staticmethod
    def _audio(canonical: dict) -> dict:
        from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

        normalizer = XAIRealtimeNormalizer()
        ga_session = RealTimeStreaming._remap_beta_session_to_ga(dict(canonical))
        return normalizer.patch_outgoing_session(ga_session, dict(canonical)).get("audio", {})

    def test_language_alone_reaches_xai_as_language_hint(self):
        """The GA path drops a bare language for lack of a transcription model;
        xAI needs no model, so it must be rebuilt from the canonical payload."""
        audio = self._audio({"language": "ja"})

        assert audio["input"]["transcription"] == {"language_hint": "ja"}

    def test_transcription_model_is_stripped(self):
        """xAI does not accept a transcription model; forwarding OpenAI's would
        be an unknown field."""
        audio = self._audio(
            {"language": "es-MX", "input_audio_transcription": {"model": "whisper-1"}}
        )

        assert audio["input"]["transcription"] == {"language_hint": "es-MX"}

    def test_canonical_language_key_does_not_leak(self):
        audio = self._audio({"language": "ru-RU"})

        assert "language" not in audio["input"]["transcription"]

    def test_without_language_the_transcription_block_is_untouched(self):
        audio = self._audio({"input_audio_transcription": {"model": "whisper-1"}})

        assert audio["input"]["transcription"] == {"model": "whisper-1"}

    def test_language_coexists_with_other_audio_settings(self):
        audio = self._audio(
            {
                "language": "ru-RU",
                "output_audio_speed": 1.2,
                "turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000},
            }
        )

        assert audio["input"]["transcription"] == {"language_hint": "ru-RU"}
        assert audio["output"]["speed"] == 1.2
        assert audio["input"]["turn_detection"]["idle_timeout_ms"] == 5000

    def test_empty_language_is_ignored(self):
        audio = self._audio({"language": ""})

        assert audio == {}


class TestXaiKeyterms:
    """xAI takes domain terms as a ``keyterms`` array, not a prompt string.

    The GA remap folds the canonical list into ``transcription.prompt`` and
    drops the whole block when no transcription model is named -- neither of
    which suits xAI, so the block is rebuilt from the canonical payload.
    """

    @staticmethod
    def _transcription(canonical: dict) -> dict:
        from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

        normalizer = XAIRealtimeNormalizer()
        ga_session = RealTimeStreaming._remap_beta_session_to_ga(dict(canonical))
        patched = normalizer.patch_outgoing_session(ga_session, dict(canonical))
        return patched.get("audio", {}).get("input", {}).get("transcription", {})

    def test_keyterms_stay_a_list(self):
        transcription = self._transcription({"transcription_keyterms": ["xAI", "Grok"]})

        assert transcription["keyterms"] == ["xAI", "Grok"]

    def test_keyterms_work_without_a_transcription_model(self):
        """xAI needs no ASR model, so the GA drop must not take keyterms with
        it."""
        transcription = self._transcription({"transcription_keyterms": ["Grok"]})

        assert transcription == {"keyterms": ["Grok"]}

    def test_ga_prompt_form_does_not_leak(self):
        """Sending both the joined prompt and the list would state the same
        intent twice, in a field xAI does not define."""
        transcription = self._transcription(
            {"transcription_keyterms": ["xAI", "Grok"], "input_audio_transcription": {"model": "whisper-1"}}
        )

        assert "prompt" not in transcription
        assert transcription["keyterms"] == ["xAI", "Grok"]

    def test_keyterms_and_language_hint_coexist(self):
        transcription = self._transcription({"transcription_keyterms": ["Grok"], "language": "ja"})

        assert transcription == {"language_hint": "ja", "keyterms": ["Grok"]}

    def test_non_string_terms_are_filtered_out(self):
        transcription = self._transcription({"transcription_keyterms": ["Grok", 42, ""]})

        assert transcription["keyterms"] == ["Grok"]

    def test_empty_keyterms_add_nothing(self):
        from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming

        normalizer = XAIRealtimeNormalizer()
        canonical = {"transcription_keyterms": []}
        ga_session = RealTimeStreaming._remap_beta_session_to_ga(dict(canonical))

        assert "audio" not in normalizer.patch_outgoing_session(ga_session, canonical)

    def test_language_alone_still_works(self):
        """Guard for the shared code path: adding keyterms must not disturb the
        language-only case."""
        transcription = self._transcription({"language": "es-MX"})

        assert transcription == {"language_hint": "es-MX"}
