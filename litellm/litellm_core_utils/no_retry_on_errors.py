"""Validation and matching for the Router's `no_retry_on_errors` setting.

The setting is declared on `Router` and on `UpdateRouterConfig`, so the rules
live here rather than on either side: the proxy has to reject a bad list before
it persists it, and the retry path has to apply exactly the same rules.
"""

from collections.abc import Sequence
from typing import Final

_MIN_HTTP_STATUS: Final = 100
_MAX_HTTP_STATUS: Final = 599


def _normalize_entry(entry: object) -> str | int:
    if isinstance(entry, bool):
        raise ValueError("no_retry_on_errors entries must be exception class names or HTTP status codes")
    if isinstance(entry, int):
        if not _MIN_HTTP_STATUS <= entry <= _MAX_HTTP_STATUS:
            raise ValueError(f"no_retry_on_errors status code out of range: {entry}")
        return entry
    if not isinstance(entry, str):
        raise ValueError("no_retry_on_errors entries must be exception class names or HTTP status codes")
    name: Final = entry.strip()
    if not name:
        raise ValueError("no_retry_on_errors entries must be non-empty exception class names or HTTP status codes")
    if name.isdigit():
        status_code: Final = int(name)
        if not _MIN_HTTP_STATUS <= status_code <= _MAX_HTTP_STATUS:
            raise ValueError(f"no_retry_on_errors status code out of range: {name}")
        return status_code
    if not all(segment.isidentifier() for segment in name.split(".")):
        raise ValueError(f"no_retry_on_errors entry is not an exception class name or HTTP status code: {name!r}")
    return name


def normalize_no_retry_on_errors(entries: object) -> frozenset[str | int]:
    """The configured entries as a frozen set of class names and HTTP status codes.

    Rejects anything that is not a status code in 100-599 or a dotted exception
    class name, so a typo fails at configuration time instead of silently never
    matching at request time.
    """
    if entries is None:
        return frozenset()
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise ValueError("no_retry_on_errors must be a list of exception class names or HTTP status codes")
    return frozenset(_normalize_entry(entry) for entry in entries)


def sorted_no_retry_entries(entries: frozenset[str | int]) -> list[str | int]:
    """Status codes first, then class names, so the settings snapshot is deterministic."""
    return sorted(entries, key=lambda entry: (1, entry) if isinstance(entry, str) else (0, str(entry)))


def error_matches_no_retry_entry(error: Exception, configured: frozenset[str | int]) -> bool:
    """True when a configured class name (optionally dotted) or status code identifies this error."""
    if not configured:
        return False
    status_code: Final = getattr(error, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool) and status_code in configured:
        return True
    if isinstance(status_code, str) and status_code.strip().isdigit() and int(status_code.strip()) in configured:
        return True
    class_names: Final = frozenset(cls.__name__ for cls in type(error).__mro__)
    for entry in configured:
        if not isinstance(entry, str):
            continue
        if entry in class_names or entry.rsplit(".", 1)[-1] in class_names:
            return True
    return False
