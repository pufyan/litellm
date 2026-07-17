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
