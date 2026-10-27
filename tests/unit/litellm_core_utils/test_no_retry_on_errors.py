"""Matching and validation rules for Router.no_retry_on_errors.

The retry path and the proxy schema share these functions, so the tests pin
the rules themselves rather than one caller's interpretation of them.
"""

from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.no_retry_on_errors import (
    error_matches_no_retry_entry,
    normalize_no_retry_on_errors,
    sorted_no_retry_entries,
)


def _rate_limit_error(status_code: object = 429) -> litellm.RateLimitError:
    error: Final = litellm.RateLimitError(message="rate limit error", model="gpt-4o", llm_provider="openai")
    error.status_code = status_code  # mutable-ok: the matching rule is what is under test
    return error


class TestNormalizeNoRetryOnErrors:
    def test_none_and_empty_list_mean_nothing_is_terminal(self):
        assert normalize_no_retry_on_errors(None) == frozenset()
        assert normalize_no_retry_on_errors([]) == frozenset()

    def test_accepts_class_names_status_codes_and_normalizes_quoted_status_codes(self):
        assert normalize_no_retry_on_errors(["RateLimitError", 429, " 500 ", "litellm.Timeout"]) == frozenset(
            {"RateLimitError", "litellm.Timeout", 429, 500}
        )

    def test_duplicates_collapse(self):
        assert normalize_no_retry_on_errors(["RateLimitError", "RateLimitError", 429, "429"]) == frozenset(
            {"RateLimitError", 429}
        )

    @pytest.mark.parametrize(
        "entries",
        [
            "RateLimitError",
            b"RateLimitError",
            429,
            {"RateLimitError": True},
            ["RateLimitError", None],
            [True],
            [{"RateLimitError": True}],
            [429.5],
            ["700"],
            [1000],
            [0],
            [99],
            ["99"],
            [""],
            ["Not A Class"],
            ["RateLimit-Error"],
        ],
        ids=[
            "bare-string",
            "bytes",
            "bare-int",
            "bare-dict",
            "none-entry",
            "boolean",
            "dict-entry",
            "float",
            "status-above-range",
            "int-out-of-range",
            "status-zero",
            "status-below-range",
            "quoted-status-below-range",
            "empty-name",
            "spaces-in-name",
            "hyphen-in-name",
        ],
    )
    def test_rejects_anything_that_can_never_match(self, entries: object) -> None:
        """A typo must fail where the operator can see it, not silently never match."""
        with pytest.raises(ValueError):
            normalize_no_retry_on_errors(entries)


class TestSortedNoRetryEntries:
    def test_status_codes_come_first_and_order_is_deterministic(self):
        entries: Final = frozenset({500, "Timeout", 429, "RateLimitError", 404})
        assert sorted_no_retry_entries(entries) == [404, 429, 500, "RateLimitError", "Timeout"]

    def test_empty(self):
        assert sorted_no_retry_entries(frozenset()) == []


class TestErrorMatchesNoRetryEntry:
    def test_class_name_matches(self):
        assert error_matches_no_retry_entry(_rate_limit_error(), frozenset({"RateLimitError"})) is True

    def test_dotted_class_name_matches(self):
        assert error_matches_no_retry_entry(_rate_limit_error(), frozenset({"litellm.RateLimitError"})) is True

    def test_superclass_name_matches_through_the_mro(self):
        assert error_matches_no_retry_entry(_rate_limit_error(), frozenset({"openai.APIError"})) is True

    def test_unrelated_class_name_does_not_match(self):
        error: Final = litellm.Timeout(message="timeout", model="gpt-4o", llm_provider="openai")
        assert error_matches_no_retry_entry(error, frozenset({"RateLimitError"})) is False
        assert error_matches_no_retry_entry(error, frozenset({"Timeout"})) is True

    def test_status_code_matches(self):
        assert error_matches_no_retry_entry(_rate_limit_error(status_code=429), frozenset({429})) is True

    def test_status_code_configured_as_int_matches_error_exposing_a_string_status(self):
        error: Final = _rate_limit_error(status_code="429")
        assert error_matches_no_retry_entry(error, frozenset({429})) is True

    def test_other_status_code_does_not_match(self):
        error: Final = _rate_limit_error(status_code=500)
        assert error_matches_no_retry_entry(error, frozenset({429})) is False

    def test_boolean_status_code_is_ignored(self):
        error: Final = _rate_limit_error(status_code=True)
        assert error_matches_no_retry_entry(error, frozenset({429})) is False
        assert error_matches_no_retry_entry(error, frozenset({"RateLimitError"})) is True

    def test_empty_configuration_never_matches(self):
        assert error_matches_no_retry_entry(_rate_limit_error(), frozenset()) is False
