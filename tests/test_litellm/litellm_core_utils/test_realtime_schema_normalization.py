from litellm.litellm_core_utils.realtime_schema_normalization import (
    normalize_input_audio_transcription_for_ga,
    normalize_tool_json_schema,
    normalize_tools_to_canonical,
    normalize_turn_detection_for_ga,
    normalize_voice_for_ga,
)


class TestNormalizeToolJsonSchema:
    def test_lowercases_types_at_every_depth(self):
        schema = {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING"},
                "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
                "meta": {
                    "type": "OBJECT",
                    "properties": {"count": {"type": "INTEGER"}},
                    "additionalProperties": {"type": "BOOLEAN"},
                },
            },
        }
        out = normalize_tool_json_schema(schema)
        assert out["type"] == "object"
        assert out["properties"]["city"]["type"] == "string"
        assert out["properties"]["tags"]["items"]["type"] == "string"
        assert out["properties"]["meta"]["properties"]["count"]["type"] == "integer"
        assert out["properties"]["meta"]["additionalProperties"]["type"] == "boolean"

    def test_strips_gemini_only_keys_recursively(self):
        schema = {
            "type": "object",
            "behavior": "BLOCKING",
            "properties": {"a": {"type": "string", "behavior": "BLOCKING", "propertyOrdering": ["a"]}},
        }
        out = normalize_tool_json_schema(schema)
        assert "behavior" not in out
        assert "behavior" not in out["properties"]["a"]
        assert "propertyOrdering" not in out["properties"]["a"]

    def test_property_named_type_is_not_treated_as_schema_type(self):
        schema = {"type": "object", "properties": {"type": {"type": "STRING", "enum": ["A", "B"]}}}
        out = normalize_tool_json_schema(schema)
        assert out["properties"]["type"]["type"] == "string"
        assert out["properties"]["type"]["enum"] == ["A", "B"]

    def test_type_arrays_and_union_keywords(self):
        schema = {
            "anyOf": [{"type": "STRING"}, {"type": ["NUMBER", "NULL"]}],
            "items": [{"type": "OBJECT"}, {"type": "ARRAY"}],
        }
        out = normalize_tool_json_schema(schema)
        assert out["anyOf"][0]["type"] == "string"
        assert out["anyOf"][1]["type"] == ["number", "null"]
        assert [i["type"] for i in out["items"]] == ["object", "array"]

    def test_non_dict_passthrough(self):
        assert normalize_tool_json_schema(None) is None
        assert normalize_tool_json_schema("x") == "x"


class TestNormalizeToolsToCanonical:
    def test_expands_gemini_function_declarations(self):
        tools = [
            {
                "functionDeclarations": [
                    {"name": "get_weather", "description": "d", "parameters": {"type": "OBJECT"}},
                    {"name": "get_time"},
                ]
            }
        ]
        out = normalize_tools_to_canonical(tools)
        assert out == [
            {"type": "function", "name": "get_weather", "description": "d", "parameters": {"type": "object"}},
            {"type": "function", "name": "get_time"},
        ]

    def test_flattens_chat_completions_shape(self):
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "OBJECT"}}}]
        out = normalize_tools_to_canonical(tools)
        assert out == [{"type": "function", "name": "f", "parameters": {"type": "object"}}]

    def test_flat_ga_tool_parameters_normalized_in_place(self):
        tools = [{"type": "function", "name": "f", "parameters": {"type": "OBJECT", "behavior": "BLOCKING"}}]
        out = normalize_tools_to_canonical(tools)
        assert out[0]["parameters"] == {"type": "object"}

    def test_non_function_tools_pass_through(self):
        mcp_tool = {"type": "mcp", "server_label": "x"}
        assert normalize_tools_to_canonical([mcp_tool]) == [mcp_tool]

    def test_non_list_passthrough(self):
        assert normalize_tools_to_canonical(None) is None


class TestNormalizeTurnDetection:
    def test_strips_gemini_sensitivity_keys(self):
        td = {
            "type": "server_vad",
            "threshold": 0.5,
            "start_sensitivity": "high",
            "end_sensitivity": "low",
        }
        out = normalize_turn_detection_for_ga(td)
        assert out == {"type": "server_vad", "threshold": 0.5}

    def test_keeps_all_ga_server_vad_keys(self):
        td = {
            "type": "server_vad",
            "create_response": False,
            "idle_timeout_ms": 5000,
            "interrupt_response": True,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
            "threshold": 0.5,
        }
        assert normalize_turn_detection_for_ga(dict(td)) == td

    def test_semantic_vad_keeps_eagerness_drops_threshold(self):
        td = {"type": "semantic_vad", "eagerness": "high", "threshold": 0.5}
        assert normalize_turn_detection_for_ga(td) == {"type": "semantic_vad", "eagerness": "high"}

    def test_typeless_guardrail_injection_survives(self):
        assert normalize_turn_detection_for_ga({"create_response": False}) == {"create_response": False}


class TestNormalizeTranscriptionAndVoice:
    def test_empty_transcription_dropped(self):
        assert normalize_input_audio_transcription_for_ga({}) is None

    def test_transcription_without_model_dropped(self):
        assert normalize_input_audio_transcription_for_ga({"language": "ru"}) is None

    def test_transcription_unknown_keys_stripped(self):
        out = normalize_input_audio_transcription_for_ga({"model": "gpt-realtime-whisper", "foo": 1})
        assert out == {"model": "gpt-realtime-whisper"}

    def test_voice_dict_collapses_to_name(self):
        assert normalize_voice_for_ga({"name": "Puck", "language_code": "ru-RU"}) == "Puck"

    def test_voice_dict_without_name_dropped(self):
        assert normalize_voice_for_ga({"language_code": "ru-RU"}) is None

    def test_voice_string_passthrough(self):
        assert normalize_voice_for_ga("marin") == "marin"


class TestFilterBuiltinTools:
    """Built-in tools are typed ``tools[]`` entries, so a backend that cannot
    provide one must have it removed: an unknown tool type is not ignored, it
    rejects the whole session.update along with the system prompt.
    """

    def test_unsupported_builtin_is_dropped_and_reported(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        tools, dropped = filter_builtin_tools([{"type": "web_search"}], frozenset({"mcp"}))

        assert tools == []
        assert dropped == ["web_search"]

    def test_supported_builtin_passes_through(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        mcp = {"type": "mcp", "server_label": "x", "server_url": "https://y"}
        tools, dropped = filter_builtin_tools([mcp], frozenset({"mcp"}))

        assert tools == [mcp]
        assert dropped == []

    def test_function_tools_are_never_filtered(self):
        """Function tools are the one capability every backend has; gating them
        on a support set would break every session."""
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        fn = {"type": "function", "name": "f", "parameters": {"type": "object"}}
        tools, dropped = filter_builtin_tools([fn], frozenset())

        assert tools == [fn]
        assert dropped == []

    def test_unknown_tool_types_are_left_alone(self):
        """Only canonical built-ins are gated; anything else is a provider's own
        shape whose handling belongs to that provider."""
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        tools, dropped = filter_builtin_tools([{"type": "computer_use"}], frozenset({"mcp"}))

        assert tools == [{"type": "computer_use"}]
        assert dropped == []

    def test_mixed_list_keeps_order_of_survivors(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        fn = {"type": "function", "name": "f"}
        tools, dropped = filter_builtin_tools(
            [fn, {"type": "web_search"}, {"type": "mcp"}, {"type": "code_execution"}],
            frozenset({"mcp"}),
        )

        assert tools == [fn, {"type": "mcp"}]
        assert sorted(dropped) == ["code_execution", "web_search"]

    def test_non_list_input_passes_through(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import filter_builtin_tools

        tools, dropped = filter_builtin_tools(None, frozenset({"mcp"}))

        assert tools is None
        assert dropped == []


class TestClampNumeric:
    """Out-of-range values reject a backend's whole session.update, so numbers
    are brought to the nearest bound instead of forwarded."""

    def test_above_maximum_is_clamped(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric(9.9, 0.0, 1.0) == (1.0, True)

    def test_below_minimum_is_clamped(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric(-5, 1, 4096) == (1, True)

    def test_in_range_reports_no_change(self):
        """The changed flag drives the log line; reporting a change that did not
        happen would cry wolf on every valid session."""
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric(0.5, 0.0, 1.0) == (0.5, False)

    def test_int_stays_int(self):
        """Sending a float where the backend expects a token count would be a
        different kind of invalid."""
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        clamped, _ = clamp_numeric(9999, 1, 4096)
        assert clamped == 4096 and isinstance(clamped, int)

    def test_open_ended_range_only_clamps_the_given_side(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric(10_000_000, 1, None) == (10_000_000, False)
        assert clamp_numeric(-1, 0, None) == (0, True)

    def test_strings_pass_through(self):
        """GA's "inf" is a valid value, not a number to be clamped."""
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric("inf", 1, 4096) == ("inf", False)

    def test_bool_is_not_treated_as_a_number(self):
        """``True`` is an int in Python; clamping it would invent a setting."""
        from litellm.litellm_core_utils.realtime_schema_normalization import clamp_numeric

        assert clamp_numeric(True, 1, 4096) == (True, False)


class TestKeepEnum:
    """An out-of-range enum has no nearest valid neighbour, so it is dropped and
    the backend's own default applies."""

    def test_known_value_is_kept(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import keep_enum

        assert keep_enum("high", frozenset({"low", "high"})) == ("high", False)

    def test_unknown_value_is_reported_for_dropping(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import keep_enum

        assert keep_enum("insane", frozenset({"low", "high"})) == (None, True)

    def test_non_string_is_reported_for_dropping(self):
        from litellm.litellm_core_utils.realtime_schema_normalization import keep_enum

        assert keep_enum(42, frozenset({"low", "high"})) == (None, True)
