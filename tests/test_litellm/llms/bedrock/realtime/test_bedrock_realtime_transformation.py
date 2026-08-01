import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path

import base64

from litellm.llms.bedrock.realtime.transformation import (
    TRIGGER_LEADING_SILENCE,
    TRIGGER_TRAILING_SILENCE,
    BedrockRealtimeConfig,
)
from litellm.llms.bedrock.realtime.trigger_audio import ready_trigger_pcm
from litellm.types.llms.openai import OpenAIRealtimeEventTypes


class TestBedrockRealtimeConfig:
    """Test suite for BedrockRealtimeConfig class"""

    def test_initialization(self):
        """Test that BedrockRealtimeConfig initializes with correct defaults"""
        config = BedrockRealtimeConfig()

        assert config is not None
        assert config.max_tokens == 1024
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.voice_id == "matthew"
        assert config.output_sample_rate_hertz == 24000
        assert config.input_sample_rate_hertz == 16000
        assert config.text_media_type == "text/plain"

    def test_session_configuration_request(self):
        """Test session configuration request generation"""
        config = BedrockRealtimeConfig()

        session_config = config.session_configuration_request("amazon.nova-sonic-v1:0")
        session_dict = json.loads(session_config)

        assert "session_start" in session_dict
        assert "prompt_start" in session_dict

        # Check session start
        session_start = session_dict["session_start"]["event"]["sessionStart"]
        assert session_start["inferenceConfiguration"]["maxTokens"] == 1024
        assert session_start["inferenceConfiguration"]["temperature"] == 0.7

        # Check prompt start
        prompt_start = session_dict["prompt_start"]["event"]["promptStart"]
        assert prompt_start["audioOutputConfiguration"]["voiceId"] == "matthew"
        assert prompt_start["audioOutputConfiguration"]["sampleRateHertz"] == 24000

    def test_session_configuration_with_tools(self):
        """Test session configuration with tools"""
        config = BedrockRealtimeConfig()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        session_config = config.session_configuration_request("amazon.nova-sonic-v1:0", tools=tools)
        session_dict = json.loads(session_config)

        prompt_start = session_dict["prompt_start"]["event"]["promptStart"]
        assert "toolConfiguration" in prompt_start
        assert "tools" in prompt_start["toolConfiguration"]
        assert len(prompt_start["toolConfiguration"]["tools"]) == 1
        assert prompt_start["toolConfiguration"]["tools"][0]["toolSpec"]["name"] == "get_weather"

    def test_transform_tools_to_bedrock_format(self):
        """Test OpenAI tool format to Bedrock format transformation"""
        config = BedrockRealtimeConfig()

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string", "description": "City name"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        bedrock_tools = config._transform_tools_to_bedrock_format(openai_tools)

        assert len(bedrock_tools) == 1
        assert bedrock_tools[0]["toolSpec"]["name"] == "get_weather"
        assert bedrock_tools[0]["toolSpec"]["description"] == "Get current weather"
        assert "inputSchema" in bedrock_tools[0]["toolSpec"]

        # Verify the schema is properly JSON stringified
        schema = json.loads(bedrock_tools[0]["toolSpec"]["inputSchema"]["json"])
        assert schema["type"] == "object"
        assert "location" in schema["properties"]

    def test_transform_tools_accepts_flat_ga_shape(self):
        """GA flat tools previously produced empty toolSpec names."""
        config = BedrockRealtimeConfig()

        ga_tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            }
        ]

        bedrock_tools = config._transform_tools_to_bedrock_format(ga_tools)

        assert len(bedrock_tools) == 1
        assert bedrock_tools[0]["toolSpec"]["name"] == "get_weather"
        schema = json.loads(bedrock_tools[0]["toolSpec"]["inputSchema"]["json"])
        assert schema["type"] == "object"

    def test_transform_tools_normalizes_uppercase_schema_types(self):
        config = BedrockRealtimeConfig()

        tools = [
            {
                "type": "function",
                "name": "f",
                "parameters": {
                    "type": "OBJECT",
                    "behavior": "BLOCKING",
                    "properties": {"a": {"type": "STRING"}},
                },
            }
        ]

        bedrock_tools = config._transform_tools_to_bedrock_format(tools)
        schema = json.loads(bedrock_tools[0]["toolSpec"]["inputSchema"]["json"])
        assert schema["type"] == "object"
        assert schema["properties"]["a"]["type"] == "string"
        assert "behavior" not in schema

    def test_audio_format_mapping(self):
        """Test audio format to sample rate mapping"""
        config = BedrockRealtimeConfig()

        # Test PCM16 format
        assert config._map_audio_format_to_sample_rate("pcm16", is_output=True) == 24000
        assert config._map_audio_format_to_sample_rate("pcm16", is_output=False) == 16000

        # Test G.711 formats
        assert config._map_audio_format_to_sample_rate("g711_ulaw", is_output=True) == 8000
        assert config._map_audio_format_to_sample_rate("g711_alaw", is_output=False) == 8000

    def test_transform_session_update_event(self):
        """Test session.update event transformation"""
        config = BedrockRealtimeConfig()

        session_update = {
            "type": "session.update",
            "session": {
                "temperature": 0.9,
                "voice": "joanna",
                "max_response_output_tokens": 2048,
                "output_audio_format": "pcm16",
            },
        }

        messages = config.transform_session_update_event(session_update)

        assert len(messages) >= 2  # At least session start and prompt start

        # Verify attributes were updated
        assert config.temperature == 0.9
        assert config.voice_id == "joanna"
        assert config.max_tokens == 2048

        # Verify session start message
        session_start = json.loads(messages[0])
        assert session_start["event"]["sessionStart"]["inferenceConfiguration"]["temperature"] == 0.9

    def test_transform_session_update_with_tools(self):
        """Test session.update with tools"""
        config = BedrockRealtimeConfig()

        session_update = {
            "type": "session.update",
            "session": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_time",
                            "description": "Get current time",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]
            },
        }

        messages = config.transform_session_update_event(session_update)

        # Find prompt start message
        prompt_start = json.loads(messages[1])
        assert "toolConfiguration" in prompt_start["event"]["promptStart"]

    def test_transform_conversation_item_create_text(self):
        """Test conversation.item.create with text"""
        config = BedrockRealtimeConfig()

        item_create = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello, how are you?"}],
            },
        }

        messages = config.transform_conversation_item_create_event(item_create)

        # Should have content start, text input, and content end
        assert len(messages) == 3

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "TEXT"
        assert content_start["event"]["contentStart"]["role"] == "USER"

        text_input = json.loads(messages[1])
        assert text_input["event"]["textInput"]["content"] == "Hello, how are you?"

    def test_transform_conversation_item_create_tool_result(self):
        """Test conversation.item.create with tool result"""
        config = BedrockRealtimeConfig()

        tool_result = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": json.dumps({"temperature": 72, "conditions": "sunny"}),
            },
        }

        messages = config.transform_conversation_item_create_event(tool_result)

        # Should have content start, tool result, and content end
        assert len(messages) == 3

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "TOOL"
        assert content_start["event"]["contentStart"]["role"] == "TOOL"
        assert content_start["event"]["contentStart"]["toolResultInputConfiguration"]["toolUseId"] == "call_123"

    def test_transform_input_audio_buffer_append(self):
        """Test input_audio_buffer.append transformation"""
        config = BedrockRealtimeConfig()

        audio_append = {
            "type": "input_audio_buffer.append",
            "audio": "base64_audio_data_here",
        }

        messages = config.transform_input_audio_buffer_append_event(audio_append)

        # First call should include content start
        assert len(messages) == 2

        content_start = json.loads(messages[0])
        assert content_start["event"]["contentStart"]["type"] == "AUDIO"
        assert content_start["event"]["contentStart"]["audioInputConfiguration"]["sampleRateHertz"] == 16000

        audio_input = json.loads(messages[1])
        assert audio_input["event"]["audioInput"]["content"] == "base64_audio_data_here"

    def test_transform_input_audio_buffer_commit(self):
        """Test input_audio_buffer.commit transformation"""
        config = BedrockRealtimeConfig()

        # First append to set the flag
        config._audio_content_started = True

        commit = {"type": "input_audio_buffer.commit"}

        messages = config.transform_input_audio_buffer_commit_event(commit)

        assert len(messages) == 1
        content_end = json.loads(messages[0])
        assert "contentEnd" in content_end["event"]


class TestBedrockRealtimeResponseCreate:
    """response.create must trigger Nova Sonic generation (LIT-2239 regression)"""

    def _start_session(self, config):
        config.transform_realtime_request(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"instructions": "You are a helpful assistant."},
                }
            ),
            "amazon.nova-sonic-v1:0",
        )

    def test_response_create_before_session_update_is_noop(self):
        config = BedrockRealtimeConfig()

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert messages == []

    def test_response_create_emits_spoken_trigger_audio(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert len(messages) > 1

        content_start = json.loads(messages[0])["event"]["contentStart"]
        assert content_start["promptName"] == config.prompt_name
        assert content_start["contentName"] == config.audio_content_name
        assert content_start["type"] == "AUDIO"
        assert content_start["interactive"] is True
        assert content_start["role"] == "USER"
        assert content_start["audioInputConfiguration"]["sampleRateHertz"] == 16000

        audio_events = [json.loads(message)["event"]["audioInput"] for message in messages[1:]]
        assert all(event["promptName"] == config.prompt_name for event in audio_events)
        assert all(event["contentName"] == config.audio_content_name for event in audio_events)

        sent_pcm = b"".join(base64.b64decode(event["content"]) for event in audio_events)
        assert sent_pcm == TRIGGER_LEADING_SILENCE + ready_trigger_pcm() + TRIGGER_TRAILING_SILENCE

    def test_second_response_create_reuses_open_audio_content(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)

        first = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        second = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert len(second) == len(first) - 1
        assert all("audioInput" in json.loads(message)["event"] for message in second)

    def test_response_create_is_noop_when_client_streams_audio(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        messages = config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        assert messages == []

    def test_client_audio_after_trigger_reopens_block_at_client_sample_rate(self):
        config = BedrockRealtimeConfig()
        config.transform_realtime_request(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "You are a helpful assistant.",
                        "input_audio_format": "g711_ulaw",
                    },
                }
            ),
            "amazon.nova-sonic-v1:0",
        )
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        trigger_content_name = config.audio_content_name

        messages = config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        events = [json.loads(message)["event"] for message in messages]
        assert [next(iter(event)) for event in events] == [
            "contentEnd",
            "contentStart",
            "audioInput",
        ]
        assert events[0]["contentEnd"]["contentName"] == trigger_content_name
        new_content_start = events[1]["contentStart"]
        assert new_content_start["contentName"] == config.audio_content_name
        assert new_content_start["contentName"] != trigger_content_name
        assert new_content_start["audioInputConfiguration"]["sampleRateHertz"] == 8000
        assert events[2]["audioInput"]["contentName"] == config.audio_content_name

    def test_client_audio_after_trigger_reuses_block_at_matching_sample_rate(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")
        trigger_content_name = config.audio_content_name

        messages = config.transform_realtime_request(
            json.dumps({"type": "input_audio_buffer.append", "audio": "c2lsZW5jZQ=="}),
            "amazon.nova-sonic-v1:0",
        )

        assert len(messages) == 1
        audio_input = json.loads(messages[0])["event"]["audioInput"]
        assert audio_input["contentName"] == trigger_content_name

    def test_session_close_messages_close_audio_prompt_and_session(self):
        config = BedrockRealtimeConfig()
        self._start_session(config)
        config.transform_realtime_request(json.dumps({"type": "response.create"}), "amazon.nova-sonic-v1:0")

        close_messages = [json.loads(message)["event"] for message in config.session_close_messages()]

        assert [next(iter(event)) for event in close_messages] == [
            "contentEnd",
            "promptEnd",
            "sessionEnd",
        ]
        assert close_messages[0]["contentEnd"]["contentName"] == config.audio_content_name
        assert close_messages[1]["promptEnd"]["promptName"] == config.prompt_name
        assert config.session_close_messages() == []

    def test_session_close_messages_before_session_update_is_empty(self):
        config = BedrockRealtimeConfig()

        assert config.session_close_messages() == []


class TestBedrockRealtimeResponseTransformation:
    """Test suite for response transformation"""

    def test_bedrock_session_start_does_not_emit_duplicate_session_created(self):
        """A Bedrock output sessionStart must not forward a second session.created to the
        client; session.created is sent exactly once on connect (LIT-4655)"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        bedrock_message = {
            "event": {"sessionStart": {"inferenceConfiguration": {"maxTokens": 1024, "temperature": 0.7}}}
        }

        result = config.transform_realtime_response(
            json.dumps(bedrock_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": None,
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        assert result["response"] == []
        assert result["session_configuration_request"] == json.dumps({"configured": True})

    def test_transform_text_output_response(self):
        """Test textOutput response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # First create a content start to initialize IDs
        content_start_message = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}

        result1 = config.transform_realtime_response(
            json.dumps(content_start_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        # Now send text output
        text_output_message = {"event": {"textOutput": {"content": "Hello, world!"}}}

        result2 = config.transform_realtime_response(
            json.dumps(text_output_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": result1["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )

        # Check for text delta
        text_deltas = [msg for msg in result2["response"] if msg["type"] == "response.text.delta"]
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"] == "Hello, world!"

        # Check that delta chunks are accumulated
        assert len(result2["current_delta_chunks"]) == 1

    def test_transform_audio_output_response(self):
        """Test audioOutput response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # First create a content start for audio
        content_start_message = {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}}

        result1 = config.transform_realtime_response(
            json.dumps(content_start_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        # Now send audio output
        audio_output_message = {"event": {"audioOutput": {"content": "base64_audio_content"}}}

        result2 = config.transform_realtime_response(
            json.dumps(audio_output_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )

        # Check for audio delta
        audio_deltas = [msg for msg in result2["response"] if msg["type"] == "response.audio.delta"]
        assert len(audio_deltas) == 1
        assert audio_deltas[0]["delta"] == "base64_audio_content"

    def test_transform_tool_use_response(self):
        """Test toolUse response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        tool_use_message = {
            "event": {
                "toolUse": {
                    "toolUseId": "tool_call_123",
                    "toolName": "get_weather",
                    "input": json.dumps({"location": "San Francisco"}),
                }
            }
        }

        result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        # Check for function call event
        assert len(result["response"]) == 1
        function_call = result["response"][0]
        assert function_call["type"] == "response.function_call_arguments.done"
        assert function_call["call_id"] == "tool_call_123"
        assert function_call["name"] == "get_weather"

        # Verify arguments are properly formatted
        args = json.loads(function_call["arguments"])
        assert args["location"] == "San Francisco"

    def test_tool_use_item_is_recorded_for_response_done_output(self):
        """toolUse must contribute a function_call item to response.done.output,
        same as text/audio items already do via transform_content_end_event —
        regression for the case where response.done.output silently dropped
        tool calls because transform_tool_use_event never touched
        current_item_chunks."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Text item already open in the response (contentStart -> textOutput),
        # then a toolUse arrives referencing the SAME still-open output item,
        # then contentEnd/END_TURN closes the whole response.
        content_start = config.transform_realtime_response(
            json.dumps({"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        tool_use_message = {
            "event": {
                "toolUse": {
                    "toolUseId": "tool_call_123",
                    "toolName": "get_weather",
                    "input": json.dumps({"location": "San Francisco"}),
                }
            }
        }
        tool_result = config.transform_realtime_response(
            json.dumps(tool_use_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": content_start["current_output_item_id"],
                "current_response_id": content_start["current_response_id"],
                "current_conversation_id": content_start["current_conversation_id"],
                "current_delta_chunks": content_start["current_delta_chunks"],
                "current_item_chunks": content_start["current_item_chunks"],
                "current_delta_type": content_start["current_delta_type"],
            },
        )

        function_call_event = [
            msg for msg in tool_result["response"] if msg["type"] == "response.function_call_arguments.done"
        ][0]
        assert function_call_event["output_index"] == 0

        content_end_message = {"event": {"contentEnd": {"stopReason": "END_TURN", "type": "TEXT"}}}
        final_result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": tool_result["current_output_item_id"],
                "current_response_id": tool_result["current_response_id"],
                "current_conversation_id": tool_result["current_conversation_id"],
                "current_delta_chunks": tool_result["current_delta_chunks"],
                "current_item_chunks": tool_result["current_item_chunks"],
                "current_delta_type": tool_result["current_delta_type"],
            },
        )

        response_done = [msg for msg in final_result["response"] if msg["type"] == "response.done"][0]
        output = response_done["response"]["output"]
        function_call_items = [item for item in output if item.get("type") == "function_call"]
        assert len(function_call_items) == 1
        assert function_call_items[0]["call_id"] == "tool_call_123"
        assert function_call_items[0]["name"] == "get_weather"

    def test_transform_content_end_text(self):
        """Test contentEnd for text response"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create some delta chunks first
        delta_chunks = [
            {"delta": "Hello, ", "type": "response.text.delta"},
            {"delta": "world!", "type": "response.text.delta"},
        ]

        content_end_message = {"event": {"contentEnd": {}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": delta_chunks,
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        # Should have text.done, content_part.done, and output_item.done
        assert len(result["response"]) == 3

        text_done = [msg for msg in result["response"] if msg["type"] == "response.text.done"][0]
        assert text_done["text"] == "Hello, world!"

        # Delta chunks should be reset
        assert result["current_delta_chunks"] is None

    def test_content_end_end_turn_emits_response_done(self):
        """END_TURN contentEnd must produce response.done (LIT-2239 regression)"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_end_message = {"event": {"contentEnd": {"stopReason": "END_TURN", "type": "AUDIO"}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "audio",
            },
        )

        response_done_events = [msg for msg in result["response"] if msg["type"] == "response.done"]
        assert len(response_done_events) == 1
        assert response_done_events[0]["response"]["status"] == "completed"
        assert result["current_output_item_id"] is None
        assert result["current_response_id"] is None
        assert result["current_delta_type"] is None

    def test_content_end_partial_turn_does_not_emit_response_done(self):
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_end_message = {"event": {"contentEnd": {"stopReason": "PARTIAL_TURN", "type": "TEXT"}}}

        result = config.transform_realtime_response(
            json.dumps(content_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        assert all(msg["type"] != "response.done" for msg in result["response"])
        assert result["current_response_id"] == "resp_123"

    def test_transform_prompt_end_response(self):
        """Test promptEnd response transformation"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        prompt_end_message = {"event": {"promptEnd": {}}}

        result = config.transform_realtime_response(
            json.dumps(prompt_end_message),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": "item_123",
                "current_response_id": "resp_123",
                "current_conversation_id": "conv_123",
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": "text",
            },
        )

        # Should have response.done
        assert len(result["response"]) == 1
        assert result["response"][0]["type"] == "response.done"
        assert result["response"][0]["response"]["status"] == "completed"

        # State should be reset
        assert result["current_output_item_id"] is None
        assert result["current_response_id"] is None
        assert result["current_delta_type"] is None

    def test_event_id_uniqueness(self):
        """Test that all event_ids are unique"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create a sequence of messages
        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        text_output1 = {"event": {"textOutput": {"content": "Hello"}}}
        text_output2 = {"event": {"textOutput": {"content": " world"}}}

        all_events = []
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        # Process all messages
        for msg in [content_start, text_output1, text_output2]:
            result = config.transform_realtime_response(
                json.dumps(msg),
                "amazon.nova-sonic-v1:0",
                logging_obj,
                realtime_response_transform_input=state,
            )
            all_events.extend(result["response"])
            # Update state for next iteration
            state.update(
                {
                    "current_output_item_id": result["current_output_item_id"],
                    "current_response_id": result["current_response_id"],
                    "current_conversation_id": result["current_conversation_id"],
                    "current_delta_chunks": result["current_delta_chunks"],
                    "current_delta_type": result["current_delta_type"],
                }
            )

        # Check all event_ids are unique
        event_ids = [event["event_id"] for event in all_events if "event_id" in event]
        assert len(event_ids) == len(set(event_ids)), "Event IDs should be unique"

    def test_response_id_consistency(self):
        """Test that response_id remains consistent across related events"""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        # Create a sequence of messages
        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        text_output = {"event": {"textOutput": {"content": "Hello"}}}

        all_events = []
        state = {
            "session_configuration_request": json.dumps({"configured": True}),
            "current_output_item_id": None,
            "current_response_id": None,
            "current_conversation_id": None,
            "current_delta_chunks": [],
            "current_item_chunks": [],
            "current_delta_type": None,
        }

        # Process messages
        for msg in [content_start, text_output]:
            result = config.transform_realtime_response(
                json.dumps(msg),
                "amazon.nova-sonic-v1:0",
                logging_obj,
                realtime_response_transform_input=state,
            )
            all_events.extend(result["response"])
            state.update(
                {
                    "current_output_item_id": result["current_output_item_id"],
                    "current_response_id": result["current_response_id"],
                    "current_conversation_id": result["current_conversation_id"],
                    "current_delta_chunks": result["current_delta_chunks"],
                    "current_delta_type": result["current_delta_type"],
                }
            )

        # Check all response_ids are the same
        response_ids = [event["response_id"] for event in all_events if "response_id" in event]
        assert len(set(response_ids)) == 1, "Response IDs should be consistent"

    def test_output_index_stable_for_repeated_events_on_same_item(self):
        """output_index for the same (response_id, item_id) must not drift across
        multiple events referencing it (contentStart -> textOutput -> contentEnd)."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        result1 = config.transform_realtime_response(
            json.dumps(content_start),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )
        output_item_added = [msg for msg in result1["response"] if msg["type"] == "response.output_item.added"][0]

        text_output = {"event": {"textOutput": {"content": "Hello"}}}
        result2 = config.transform_realtime_response(
            json.dumps(text_output),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": result1["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )
        text_delta = [msg for msg in result2["response"] if msg["type"] == "response.text.delta"][0]

        content_end = {"event": {"contentEnd": {}}}
        result3 = config.transform_realtime_response(
            json.dumps(content_end),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": result2["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result2["current_delta_type"],
            },
        )
        output_item_done = [msg for msg in result3["response"] if msg["type"] == "response.output_item.done"][0]

        assert output_item_added["output_index"] == text_delta["output_index"] == output_item_done["output_index"]

    def test_content_index_differs_between_text_and_audio_parts_on_same_item(self):
        """Two content parts of different modality on the same item must get distinct
        content_index values (0 and 1), not both hardcoded to 0."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_start_text = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        result1 = config.transform_realtime_response(
            json.dumps(content_start_text),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )
        text_part_added = [msg for msg in result1["response"] if msg["type"] == "response.content_part.added"][0]

        # Same item_id, but this time an AUDIO content part starts on it too.
        content_start_audio = {"event": {"contentStart": {"role": "ASSISTANT", "type": "AUDIO"}}}
        result2 = config.transform_realtime_response(
            json.dumps(content_start_audio),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )
        audio_part_added = [msg for msg in result2["response"] if msg["type"] == "response.content_part.added"][0]

        assert text_part_added["content_index"] == 0
        assert audio_part_added["content_index"] == 1

    def test_new_response_restarts_output_index_from_zero(self):
        """A new response_id must not inherit output_index allocations from a
        previous, unrelated response."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        first_content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        result1 = config.transform_realtime_response(
            json.dumps(first_content_start),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )
        first_output_item_added = [
            msg for msg in result1["response"] if msg["type"] == "response.output_item.added"
        ][0]
        assert first_output_item_added["output_index"] == 0

        # Brand-new response/item, unrelated to the first one.
        second_content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        result2 = config.transform_realtime_response(
            json.dumps(second_content_start),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )
        second_output_item_added = [
            msg for msg in result2["response"] if msg["type"] == "response.output_item.added"
        ][0]
        assert second_output_item_added["output_index"] == 0

    def test_response_done_output_contains_completed_text_item(self):
        """response.done.response.output must not always be [] (LIT bug fix): the
        text produced during contentEnd should show up as a completed output item."""
        config = BedrockRealtimeConfig()
        logging_obj = MagicMock()
        logging_obj.litellm_trace_id = "trace_123"

        content_start = {"event": {"contentStart": {"role": "ASSISTANT", "type": "TEXT"}}}
        result1 = config.transform_realtime_response(
            json.dumps(content_start),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": None,
                "current_response_id": None,
                "current_conversation_id": None,
                "current_delta_chunks": [],
                "current_item_chunks": [],
                "current_delta_type": None,
            },
        )

        text_output = {"event": {"textOutput": {"content": "Hello, world!"}}}
        result2 = config.transform_realtime_response(
            json.dumps(text_output),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result1["current_output_item_id"],
                "current_response_id": result1["current_response_id"],
                "current_conversation_id": result1["current_conversation_id"],
                "current_delta_chunks": result1["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result1["current_delta_type"],
            },
        )

        content_end = {"event": {"contentEnd": {"stopReason": "END_TURN"}}}
        result3 = config.transform_realtime_response(
            json.dumps(content_end),
            "amazon.nova-sonic-v1:0",
            logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": json.dumps({"configured": True}),
                "current_output_item_id": result2["current_output_item_id"],
                "current_response_id": result2["current_response_id"],
                "current_conversation_id": result2["current_conversation_id"],
                "current_delta_chunks": result2["current_delta_chunks"],
                "current_item_chunks": [],
                "current_delta_type": result2["current_delta_type"],
            },
        )

        response_done = [msg for msg in result3["response"] if msg["type"] == "response.done"][0]
        output = response_done["response"]["output"]
        assert len(output) == 1
        assert output[0]["status"] == "completed"
        assert output[0]["content"][0]["text"] == "Hello, world!"

        # Accumulated item chunks must be reset for the next response.
        assert result3["current_item_chunks"] is None


class TestBedrockRealtimeSessionEvents:
    """session.created / session.updated builders produce spec-shaped events (LIT-4655)"""

    @staticmethod
    def _logging():
        from types import SimpleNamespace

        return SimpleNamespace(litellm_trace_id="trace_123")

    def test_session_created_event_shape(self):
        event = BedrockRealtimeConfig().session_created_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["type"] == "session.created"
        assert event["session"]["id"] == "trace_123"
        assert event["session"]["model"] == "amazon.nova-sonic-v1:0"
        assert event["session"]["modalities"] == ["text", "audio"]
        assert event["event_id"]

    def test_session_updated_event_shape(self):
        event = BedrockRealtimeConfig().session_updated_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["type"] == "session.updated"
        assert event["session"]["id"] == "trace_123"
        assert event["session"]["model"] == "amazon.nova-sonic-v1:0"
        assert event["event_id"]

    def test_created_and_updated_have_distinct_event_ids(self):
        config = BedrockRealtimeConfig()
        logging_obj = self._logging()
        created = config.session_created_event("amazon.nova-sonic-v1:0", logging_obj)
        updated = config.session_updated_event("amazon.nova-sonic-v1:0", logging_obj)
        assert created["event_id"] != updated["event_id"]

    def test_session_updated_reflects_requested_modalities(self):
        event = BedrockRealtimeConfig().session_updated_event(
            "amazon.nova-sonic-v1:0", self._logging(), modalities=["text"]
        )
        assert event["session"]["modalities"] == ["text"]

    def test_session_updated_defaults_modalities_when_unspecified(self):
        event = BedrockRealtimeConfig().session_updated_event("amazon.nova-sonic-v1:0", self._logging())
        assert event["session"]["modalities"] == ["text", "audio"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestSessionInferenceAndTurnDetection:
    """Canonical session fields Bedrock previously ignored.

    ``top_p`` was never read from the session, so the hardcoded 0.9 always won;
    Nova 2's ``turnDetectionConfiguration`` was not mapped at all, and the
    implementation did not distinguish Nova 1 from Nova 2.
    """

    NOVA_1 = "amazon.nova-sonic-v1:0"
    NOVA_2 = "amazon.nova-2-sonic-v1:0"

    @staticmethod
    def _session_start(model: str, session: dict) -> dict:
        config = BedrockRealtimeConfig()
        messages = config.transform_session_update_event(
            {"type": "session.update", "session": session}, model=model
        )
        return json.loads(messages[0])["event"]["sessionStart"]

    def test_top_p_from_session_overrides_the_protocol_default(self):
        """``inferenceConfiguration`` is required by the Bedrock event schema,
        so a default must be sent -- but a client value must win over it."""
        session_start = self._session_start(self.NOVA_2, {"top_p": 0.5})

        assert session_start["inferenceConfiguration"]["topP"] == 0.5

    def test_inference_defaults_are_kept_when_client_omits_them(self):
        session_start = self._session_start(self.NOVA_2, {})

        assert session_start["inferenceConfiguration"] == {
            "maxTokens": 1024,
            "topP": 0.9,
            "temperature": 0.7,
        }

    def test_all_inference_params_can_be_set_together(self):
        session_start = self._session_start(
            self.NOVA_2,
            {"top_p": 0.3, "temperature": 0.2, "max_response_output_tokens": 512},
        )

        assert session_start["inferenceConfiguration"] == {
            "maxTokens": 512,
            "topP": 0.3,
            "temperature": 0.2,
        }

    def test_nova_2_maps_end_sensitivity_to_endpointing(self):
        session_start = self._session_start(
            self.NOVA_2, {"turn_detection": {"type": "server_vad", "end_sensitivity": "high"}}
        )

        assert session_start["turnDetectionConfiguration"] == {"endpointingSensitivity": "HIGH"}

    @pytest.mark.parametrize(
        "canonical, expected", [("high", "HIGH"), ("medium", "MEDIUM"), ("low", "LOW")]
    )
    def test_every_sensitivity_level_maps(self, canonical, expected):
        session_start = self._session_start(
            self.NOVA_2, {"turn_detection": {"end_sensitivity": canonical}}
        )

        assert session_start["turnDetectionConfiguration"]["endpointingSensitivity"] == expected

    def test_nova_1_drops_end_sensitivity_rather_than_sending_it(self):
        """Nova 1 has no turnDetectionConfiguration; forwarding one would be
        rejected by the backend and take the whole session down."""
        session_start = self._session_start(
            self.NOVA_1, {"turn_detection": {"end_sensitivity": "high"}}
        )

        assert "turnDetectionConfiguration" not in session_start

    def test_nova_1_drop_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            self._session_start(self.NOVA_1, {"turn_detection": {"end_sensitivity": "low"}})

        assert any("end_sensitivity" in record.message for record in caplog.records)

    def test_unknown_sensitivity_is_not_forwarded(self):
        session_start = self._session_start(
            self.NOVA_2, {"turn_detection": {"end_sensitivity": "aggressive"}}
        )

        assert "turnDetectionConfiguration" not in session_start

    def test_turn_detection_without_end_sensitivity_sends_no_config(self):
        """Other turn_detection fields have no Nova equivalent; an empty
        turnDetectionConfiguration would be noise."""
        session_start = self._session_start(
            self.NOVA_2, {"turn_detection": {"type": "server_vad", "threshold": 0.5}}
        )

        assert "turnDetectionConfiguration" not in session_start

    def test_null_turn_detection_sends_no_config(self):
        session_start = self._session_start(self.NOVA_2, {"turn_detection": None})

        assert "turnDetectionConfiguration" not in session_start


class TestAudioFormatSampleRate:
    """Canonical audio formats resolve to Nova sample rates.

    The codec name implies a rate, but Nova accepts 8000/16000/24000 on either
    direction, so "pcm16 at 8kHz" is only expressible through the object form of
    the canonical field. Two non-canonical keys used to carry this instead,
    which gave the setting a second address the contract does not define.
    """

    MODEL = "amazon.nova-2-sonic-v1:0"

    @staticmethod
    def _rates(session: dict) -> tuple:
        config = BedrockRealtimeConfig()
        config.transform_session_update_event(
            {"type": "session.update", "session": session}, model=TestAudioFormatSampleRate.MODEL
        )
        return config.input_sample_rate_hertz, config.output_sample_rate_hertz

    def test_string_format_keeps_the_implied_rates(self):
        assert self._rates({"input_audio_format": "pcm16"}) == (16000, 24000)

    def test_g711_implies_8k(self):
        input_rate, _ = self._rates({"input_audio_format": "g711_ulaw"})

        assert input_rate == 8000

    def test_object_form_sets_an_explicit_input_rate(self):
        """The whole point of the object form: pcm at a rate the codec name does
        not imply."""
        input_rate, _ = self._rates({"input_audio_format": {"type": "audio/pcm", "rate": 8000}})

        assert input_rate == 8000

    def test_object_form_sets_an_explicit_output_rate(self):
        _, output_rate = self._rates({"output_audio_format": {"type": "audio/pcm", "rate": 16000}})

        assert output_rate == 16000

    def test_unsupported_rate_falls_back_instead_of_being_forwarded(self):
        """Nova rejects rates outside its set, so forwarding one would fail the
        session rather than degrade it."""
        input_rate, _ = self._rates({"input_audio_format": {"type": "audio/pcm", "rate": 44100}})

        assert input_rate == 16000

    def test_unsupported_rate_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            self._rates({"input_audio_format": {"type": "audio/pcm", "rate": 44100}})

        assert any("44100" in record.message for record in caplog.records)

    def test_object_without_rate_falls_back_to_the_codec_default(self):
        input_rate, _ = self._rates({"input_audio_format": {"type": "audio/pcm"}})

        assert input_rate == 16000

    def test_boolean_rate_is_rejected(self):
        """``True`` is an int in Python but never a sample rate."""
        input_rate, _ = self._rates({"input_audio_format": {"type": "audio/pcm", "rate": True}})

        assert input_rate == 16000

    def test_non_canonical_sample_rate_keys_have_no_effect(self):
        """These were a second address for a setting the canonical schema
        already covers; the contract gives every field exactly one."""
        assert self._rates({"input_sample_rate_hertz": 8000, "output_sample_rate_hertz": 8000}) == (16000, 24000)

    def test_rate_reaches_the_audio_input_configuration(self):
        """End-to-end: the resolved rate must appear in the Bedrock event, not
        just on the config object."""
        config = BedrockRealtimeConfig()
        config.transform_session_update_event(
            {"type": "session.update", "session": {"input_audio_format": {"type": "audio/pcm", "rate": 8000}}},
            model=self.MODEL,
        )
        messages = config.transform_input_audio_buffer_append_event(
            {"type": "input_audio_buffer.append", "audio": base64.b64encode(b"pcm").decode()}
        )

        content_start = next(
            json.loads(message)["event"]["contentStart"]
            for message in messages
            if "contentStart" in json.loads(message)["event"]
        )
        assert content_start["audioInputConfiguration"]["sampleRateHertz"] == 8000


class TestInferenceRangeClamping:
    """``inferenceConfiguration`` is required by the Bedrock event schema, so a
    bad value cannot be omitted -- it is clamped to the nearest bound."""

    MODEL = "amazon.nova-2-sonic-v1:0"

    def _inference(self, session: dict) -> dict:
        config = BedrockRealtimeConfig()
        messages = config.transform_session_update_event(
            {"type": "session.update", "session": session}, model=self.MODEL
        )
        return json.loads(messages[0])["event"]["sessionStart"]["inferenceConfiguration"]

    def test_top_p_above_one_is_clamped(self):
        assert self._inference({"top_p": 5.0})["topP"] == 1.0

    def test_temperature_above_one_is_clamped(self):
        assert self._inference({"temperature": 99})["temperature"] == 1.0

    def test_negative_max_tokens_is_raised_to_one(self):
        assert self._inference({"max_response_output_tokens": -5})["maxTokens"] == 1

    def test_in_range_values_are_untouched(self):
        inference = self._inference({"top_p": 0.8, "temperature": 0.3, "max_response_output_tokens": 2048})

        assert inference == {"maxTokens": 2048, "topP": 0.8, "temperature": 0.3}

    def test_session_still_carries_tools_with_out_of_range_values(self):
        config = BedrockRealtimeConfig()
        messages = config.transform_session_update_event(
            {
                "type": "session.update",
                "session": {
                    "temperature": 99,
                    "tools": [{"type": "function", "name": "attended_transfer", "parameters": {}}],
                },
            },
            model=self.MODEL,
        )

        prompt_start = json.loads(messages[1])["event"]["promptStart"]
        assert prompt_start["toolConfiguration"]["tools"][0]["toolSpec"]["name"] == "attended_transfer"
