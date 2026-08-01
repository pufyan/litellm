import logging

import pytest

from litellm.litellm_core_utils.realtime_session_mapping import (
    DropReason,
    drop_unsupported_by_model,
    log_dropped_fields,
    resolve_mutually_exclusive,
    select_reasoning_field,
    split_canonical_session,
)
from litellm.types.realtime_session import CANONICAL_SESSION_KEYS


class TestSplitCanonicalSession:
    def test_keeps_supported_and_drops_unsupported_canonical_fields(self):
        session = {"temperature": 0.8, "top_k": 40, "instructions": "hi"}
        kept, dropped = split_canonical_session(session, frozenset({"temperature", "instructions"}))

        assert kept == {"temperature": 0.8, "instructions": "hi"}
        assert dropped == {"top_k": DropReason.UNSUPPORTED_BY_PROVIDER}

    def test_non_canonical_key_is_distinguished_from_unsupported_one(self):
        """A backend-native spelling is not a second address for a canonical
        field; it must be reported as NOT_CANONICAL so the contract violation is
        visible rather than looking like a capability gap."""
        session = {"thinkingBudget": 1024, "top_k": 40}
        kept, dropped = split_canonical_session(session, frozenset({"thinking_budget"}))

        assert kept == {}
        assert dropped == {
            "thinkingBudget": DropReason.NOT_CANONICAL,
            "top_k": DropReason.UNSUPPORTED_BY_PROVIDER,
        }

    def test_supported_key_outside_canonical_schema_is_still_dropped(self):
        """Canonicality is checked before support, so an implementation cannot
        widen the client-facing contract by listing a non-canonical key."""
        kept, dropped = split_canonical_session({"generationConfig": {}}, frozenset({"generationConfig"}))

        assert kept == {}
        assert dropped == {"generationConfig": DropReason.NOT_CANONICAL}

    def test_does_not_mutate_input(self):
        session = {"temperature": 0.8, "top_k": 40}
        split_canonical_session(session, frozenset({"temperature"}))

        assert session == {"temperature": 0.8, "top_k": 40}

    def test_every_canonical_key_is_routable(self):
        session = {key: "x" for key in CANONICAL_SESSION_KEYS}
        kept, dropped = split_canonical_session(session, CANONICAL_SESSION_KEYS)

        assert set(kept) == CANONICAL_SESSION_KEYS
        assert dropped == {}


class TestDropUnsupportedByModel:
    def test_removes_only_listed_fields_and_reports_model_reason(self):
        session = {"thinking_budget": 0, "temperature": 0.8}
        dropped = drop_unsupported_by_model(session, ["thinking_budget"])

        assert session == {"temperature": 0.8}
        assert dropped == {"thinking_budget": DropReason.UNSUPPORTED_BY_MODEL}

    def test_absent_field_is_not_reported_as_dropped(self):
        """Reporting a field the client never sent would make the operator log
        lie about what was discarded."""
        session = {"temperature": 0.8}
        dropped = drop_unsupported_by_model(session, ["thinking_budget", "top_k"])

        assert session == {"temperature": 0.8}
        assert dropped == {}


class TestResolveMutuallyExclusive:
    def test_discards_competing_field_when_kept_one_is_present(self):
        session = {"thinking_level": "low", "thinking_budget": 1024}
        dropped = resolve_mutually_exclusive(session, keep="thinking_level", discard=["thinking_budget"])

        assert session == {"thinking_level": "low"}
        assert dropped == {"thinking_budget": DropReason.MUTUALLY_EXCLUSIVE}

    def test_keeps_competing_field_when_the_kept_one_is_absent(self):
        """With nothing to conflict against there is no exclusion to resolve;
        discarding anyway would silently delete the client's only instruction."""
        session = {"thinking_budget": 1024}
        dropped = resolve_mutually_exclusive(session, keep="thinking_level", discard=["thinking_budget"])

        assert session == {"thinking_budget": 1024}
        assert dropped == {}


class TestSelectReasoningField:
    def test_returns_field_the_model_uses(self):
        selected, dropped = select_reasoning_field({"thinking_level": "low"}, model_uses="thinking_level")

        assert selected == "thinking_level"
        assert dropped == {}

    def test_conflicting_pair_reports_mutual_exclusion(self):
        session = {"thinking_level": "low", "thinking_budget": 1024}
        selected, dropped = select_reasoning_field(session, model_uses="thinking_level")

        assert selected == "thinking_level"
        assert dropped == {"thinking_budget": DropReason.MUTUALLY_EXCLUSIVE}

    def test_wrong_field_alone_is_a_model_gap_not_a_conflict(self):
        """A client that sends only thinking_budget to a level-based model gets
        no reasoning config at all; the reason must say the model cannot honor
        it, not that two fields collided."""
        selected, dropped = select_reasoning_field({"thinking_budget": 1024}, model_uses="thinking_level")

        assert selected is None
        assert dropped == {"thinking_budget": DropReason.UNSUPPORTED_BY_MODEL}

    def test_neither_field_sent_yields_nothing_to_report(self):
        selected, dropped = select_reasoning_field({"temperature": 0.8}, model_uses="thinking_budget")

        assert selected is None
        assert dropped == {}

    @pytest.mark.parametrize(
        "model_uses, other",
        [("thinking_budget", "thinking_level"), ("thinking_level", "thinking_budget")],
    )
    def test_selection_is_symmetric_across_model_families(self, model_uses, other):
        session = {model_uses: "x", other: "y"}
        selected, dropped = select_reasoning_field(session, model_uses=model_uses)

        assert selected == model_uses
        assert dropped == {other: DropReason.MUTUALLY_EXCLUSIVE}

    def test_does_not_mutate_input(self):
        session = {"thinking_level": "low", "thinking_budget": 1024}
        select_reasoning_field(session, model_uses="thinking_level")

        assert session == {"thinking_level": "low", "thinking_budget": 1024}


class TestLogDroppedFields:
    def test_names_every_dropped_field_with_its_reason(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            log_dropped_fields(
                "gemini",
                "gemini-3.1-flash-live-preview",
                {
                    "thinking_budget": DropReason.UNSUPPORTED_BY_MODEL,
                    "top_k": DropReason.UNSUPPORTED_BY_PROVIDER,
                },
            )

        message = caplog.text
        assert "thinking_budget (unsupported_by_model)" in message
        assert "top_k (unsupported_by_provider)" in message
        assert "gemini-3.1-flash-live-preview" in message

    def test_silent_when_nothing_was_dropped(self, caplog):
        """A warning on every clean session.update would train operators to
        ignore the log that exists to surface real drops."""
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            log_dropped_fields("gemini", "gemini-3.1-flash-live-preview", {})

        assert caplog.text == ""
