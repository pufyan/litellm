"""Provider-neutral normalization of nested realtime session structures.

The realtime contract (docs/realtime_session_contract.md) requires every nested
structure a client sends to be delivered to each backend in that backend's
expected shape. The top-level GA allowlist cannot protect nested payloads, so
these helpers normalize them at every depth: JSON-schema tool parameters,
turn-detection blocks, input-audio transcription, and voice values.

Kept free of heavy litellm imports so both the shared OpenAI-compatible remap
and provider transformations (e.g. Bedrock) can use it without import cycles.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union

MAX_SCHEMA_DEPTH = 32

_JSON_SCHEMA_TYPES: FrozenSet[str] = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)
_SCHEMA_FOREIGN_KEYS: FrozenSet[str] = frozenset({"behavior", "propertyOrdering", "property_ordering"})
_SCHEMA_MAP_OF_SCHEMAS_KEYS: Tuple[str, ...] = ("properties", "$defs", "definitions", "patternProperties")
_SCHEMA_SINGLE_SCHEMA_KEYS: Tuple[str, ...] = ("items", "additionalProperties", "contains", "not")
_SCHEMA_LIST_OF_SCHEMAS_KEYS: Tuple[str, ...] = ("anyOf", "oneOf", "allOf", "prefixItems")

_SERVER_VAD_KEYS: FrozenSet[str] = frozenset(
    {
        "type",
        "create_response",
        "idle_timeout_ms",
        "interrupt_response",
        "prefix_padding_ms",
        "silence_duration_ms",
        "threshold",
    }
)
_SEMANTIC_VAD_KEYS: FrozenSet[str] = frozenset({"type", "create_response", "eagerness", "interrupt_response"})
_ANY_VAD_KEYS: FrozenSet[str] = _SERVER_VAD_KEYS | _SEMANTIC_VAD_KEYS

_TRANSCRIPTION_KEYS: FrozenSet[str] = frozenset({"language", "model", "prompt"})

# Canonical built-in tool types. Function tools are always available; these are
# backend-provided capabilities, expressed as typed ``tools[]`` entries rather
# than boolean session flags so there is exactly one way to declare a tool.
CANONICAL_BUILTIN_TOOL_TYPES: FrozenSet[str] = frozenset(
    {"web_search", "code_execution", "file_search", "x_search", "mcp"}
)


def _normalize_schema_type_value(value: object) -> object:
    if isinstance(value, str) and value.lower() in _JSON_SCHEMA_TYPES:
        return value.lower()
    return value


def normalize_tool_json_schema(schema: object, depth: int = 0) -> object:
    """Normalize a JSON-schema tool ``parameters`` object at every depth.

    Lowercases schema ``type`` values (Gemini emits ``"STRING"``/``"OBJECT"``)
    and strips provider-only keys such as ``behavior``. Non-dict input and
    schemas nested deeper than ``MAX_SCHEMA_DEPTH`` are returned unchanged.
    """
    if depth > MAX_SCHEMA_DEPTH or not isinstance(schema, dict):
        return schema

    def _child(value: object) -> object:
        return normalize_tool_json_schema(value, depth + 1)

    def _entry(key: str, value: object) -> object:
        if key == "type":
            if isinstance(value, list):
                return [_normalize_schema_type_value(item) for item in value]
            return _normalize_schema_type_value(value)
        if key in _SCHEMA_MAP_OF_SCHEMAS_KEYS and isinstance(value, dict):
            return {name: _child(sub) for name, sub in value.items()}
        if key in _SCHEMA_SINGLE_SCHEMA_KEYS:
            if isinstance(value, dict):
                return _child(value)
            if isinstance(value, list):
                return [_child(item) for item in value]
            return value
        if key in _SCHEMA_LIST_OF_SCHEMAS_KEYS and isinstance(value, list):
            return [_child(item) for item in value]
        return value

    return {key: _entry(key, value) for key, value in schema.items() if key not in _SCHEMA_FOREIGN_KEYS}


def _function_tool_from_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    named: Dict[str, Any] = {
        "name": fields.get("name"),
        "description": fields.get("description"),
        "parameters": (
            normalize_tool_json_schema(fields["parameters"]) if isinstance(fields.get("parameters"), dict) else None
        ),
    }
    return {"type": "function", **{key: value for key, value in named.items() if value is not None}}


def normalize_tools_to_canonical(tools: object) -> object:
    """Normalize a ``session.tools`` list into the flat GA function-tool form.

    Handles three inbound shapes: Gemini ``{"functionDeclarations": [...]}``
    (expands to one flat tool per declaration), chat-completions
    ``{"type": "function", "function": {...}}`` (flattened), and already-flat
    GA tools (parameters normalized in place). Non-function tools (e.g. MCP)
    and unrecognized entries pass through untouched.
    """
    if not isinstance(tools, list):
        return tools

    def _expand(tool: object) -> List[object]:
        if not isinstance(tool, dict):
            return [tool]
        declarations = tool.get("functionDeclarations")
        if isinstance(declarations, list):
            return [_function_tool_from_fields(decl) for decl in declarations if isinstance(decl, dict)]
        nested_function = tool.get("function")
        if tool.get("type") == "function" and isinstance(nested_function, dict):
            return [_function_tool_from_fields(nested_function)]
        if isinstance(tool.get("parameters"), dict):
            return [{**tool, "parameters": normalize_tool_json_schema(tool["parameters"])}]
        return [tool]

    return [expanded for tool in tools for expanded in _expand(tool)]


def clamp_numeric(
    value: object,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Tuple[object, bool]:
    """Bring a numeric value into a backend's accepted range.

    Returns the value and whether it was changed, so the caller can log the
    substitution -- a silently altered value is indistinguishable from one the
    client chose. Non-numeric input (including ``bool``, which is an ``int`` in
    Python) passes through untouched: guessing at a wrong type would hide a real
    client error, and the field's own validation should reject it.

    Clamping rather than dropping matters because backends reject the entire
    ``session.update`` on an out-of-range value, taking the system prompt and
    tools down with it -- so an unusable number must not cost a whole session.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value, False
    clamped: float = value
    if minimum is not None:
        clamped = max(clamped, minimum)
    if maximum is not None:
        clamped = min(clamped, maximum)
    if clamped == value:
        return value, False
    return (int(clamped) if isinstance(value, int) else clamped), True


def keep_enum(value: object, allowed: FrozenSet[str]) -> Tuple[Optional[str], bool]:
    """Keep an enum value the backend defines, or report it for dropping.

    Unlike a number, an out-of-range enum has no nearest valid neighbour to fall
    back to, so it is dropped and the backend's own default applies.
    """
    if isinstance(value, str) and value in allowed:
        return value, False
    return None, True


def filter_builtin_tools(tools: object, supported: FrozenSet[str]) -> Tuple[object, List[str]]:
    """Keep only the built-in tools a backend provides; report what was dropped.

    Function tools always pass through. A built-in tool the backend cannot
    provide is removed rather than forwarded: an unknown tool type is not
    ignored by backends, it rejects the whole ``session.update`` and takes the
    system prompt and every other tool down with it.

    Returns the filtered list plus the dropped type names, so the caller can log
    them -- a dropped tool is otherwise indistinguishable from one that works.
    """
    if not isinstance(tools, list):
        return tools, []

    kept: List[object] = []
    dropped: List[str] = []
    for tool in tools:
        tool_type = tool.get("type") if isinstance(tool, dict) else None
        if isinstance(tool_type, str) and tool_type in CANONICAL_BUILTIN_TOOL_TYPES and tool_type not in supported:
            dropped.append(tool_type)
            continue
        kept.append(tool)
    return kept, dropped


def normalize_turn_detection_for_ga(turn_detection: object) -> object:
    """Keep only GA-valid keys in a ``turn_detection`` block.

    Gemini-only knobs (``start_sensitivity`` / ``end_sensitivity``) are
    stripped; the allowlist follows the GA server_vad / semantic_vad unions
    from the openai SDK. Non-dict values pass through.
    """
    if not isinstance(turn_detection, dict):
        return turn_detection
    vad_type = turn_detection.get("type")
    allowed = (
        _SERVER_VAD_KEYS
        if vad_type == "server_vad"
        else _SEMANTIC_VAD_KEYS if vad_type == "semantic_vad" else _ANY_VAD_KEYS
    )
    return {key: value for key, value in turn_detection.items() if key in allowed}


def normalize_input_audio_transcription_for_ga(transcription: object) -> Optional[object]:
    """Return a GA-valid transcription config, or None when it must be dropped.

    OpenAI rejects an empty ``{}`` (Gemini-style "just enable it"), so a dict
    without a ``model`` is dropped entirely; unknown keys are stripped.
    """
    if not isinstance(transcription, dict):
        return transcription
    if not transcription.get("model"):
        return None
    return {key: value for key, value in transcription.items() if key in _TRANSCRIPTION_KEYS}


def normalize_voice_for_ga(voice: object) -> Optional[Union[str, object]]:
    """Coerce a voice value to the GA string form.

    A Gemini-style ``{"name": ..., "language_code": ...}`` dict collapses to
    its ``name``; a dict without a usable name is dropped (None).
    """
    if isinstance(voice, dict):
        name = voice.get("name") or voice.get("voice")
        return name if isinstance(name, str) and name else None
    return voice
