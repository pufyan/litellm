import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.yandex.chat.transformation import YandexChatConfig, YandexError
from litellm.types.utils import ModelResponse


class TestYandexTransformRequest:
    """folder_id must be resolved and embedded in modelUri, or the request must fail loudly."""

    def test_builds_model_uri_from_litellm_params_folder_id(self):
        config = YandexChatConfig()

        request = config.transform_request(
            model="yandexgpt/latest",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={"folder_id": "b1gfolder"},
            headers={},
        )

        assert request["modelUri"] == "gpt://b1gfolder/yandexgpt/latest"

    def test_falls_back_to_env_var_folder_id(self, monkeypatch):
        monkeypatch.setenv("YANDEX_FOLDER_ID", "b1gfromenv")
        config = YandexChatConfig()

        request = config.transform_request(
            model="yandexgpt/latest",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert request["modelUri"] == "gpt://b1gfromenv/yandexgpt/latest"

    def test_raises_when_folder_id_missing_everywhere(self, monkeypatch):
        monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
        config = YandexChatConfig()

        with pytest.raises(ValueError, match="folder_id"):
            config.transform_request(
                model="yandexgpt/latest",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={},
                litellm_params={},
                headers={},
            )

    def test_maps_openai_messages_to_yandex_text_field(self):
        config = YandexChatConfig()

        request = config.transform_request(
            model="yandexgpt/latest",
            messages=[{"role": "user", "content": "hello there"}],
            optional_params={},
            litellm_params={"folder_id": "b1gfolder"},
            headers={},
        )

        assert request["messages"] == [{"role": "user", "text": "hello there"}]


class TestYandexTransformResponse:
    """Yandex wraps the reply in result.alternatives[] and reports usage as strings."""

    @staticmethod
    def _raw_response(body: dict) -> httpx.Response:
        return httpx.Response(status_code=200, json=body, request=httpx.Request("POST", "https://example.com"))

    def test_extracts_message_text_and_stop_reason(self):
        config = YandexChatConfig()
        raw = self._raw_response(
            {
                "result": {
                    "alternatives": [
                        {
                            "message": {"role": "assistant", "text": "Привет!"},
                            "status": "ALTERNATIVE_STATUS_FINAL",
                        }
                    ],
                    "usage": {
                        "inputTextTokens": "14",
                        "completionTokens": "2",
                        "totalTokens": "16",
                    },
                }
            }
        )

        result = config.transform_response(
            model="yandexgpt/latest",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "Привет!"
        assert result.choices[0].finish_reason == "stop"

    def test_converts_stringified_usage_counters_to_ints(self):
        config = YandexChatConfig()
        raw = self._raw_response(
            {
                "result": {
                    "alternatives": [
                        {"message": {"role": "assistant", "text": "hi"}, "status": "ALTERNATIVE_STATUS_FINAL"}
                    ],
                    "usage": {
                        "inputTextTokens": "14",
                        "completionTokens": "2",
                        "totalTokens": "16",
                    },
                }
            }
        )

        result = config.transform_response(
            model="yandexgpt/latest",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.usage.prompt_tokens == 14
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 16

    def test_uses_first_alternative_when_multiple_present(self):
        config = YandexChatConfig()
        raw = self._raw_response(
            {
                "result": {
                    "alternatives": [
                        {"message": {"role": "assistant", "text": "first"}, "status": "ALTERNATIVE_STATUS_FINAL"},
                        {"message": {"role": "assistant", "text": "second"}, "status": "ALTERNATIVE_STATUS_FINAL"},
                    ],
                    "usage": {"inputTextTokens": "1", "completionTokens": "1", "totalTokens": "2"},
                }
            }
        )

        result = config.transform_response(
            model="yandexgpt/latest",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=None,
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.choices[0].message.content == "first"

    def test_raises_yandex_error_when_no_alternatives_returned(self):
        config = YandexChatConfig()
        raw = self._raw_response({"result": {"alternatives": [], "usage": {}}})

        with pytest.raises(YandexError):
            config.transform_response(
                model="yandexgpt/latest",
                raw_response=raw,
                model_response=ModelResponse(),
                logging_obj=None,
                request_data={},
                messages=[],
                optional_params={},
                litellm_params={},
                encoding=None,
            )


class TestYandexAuth:
    def test_uses_api_key_scheme_not_bearer(self):
        config = YandexChatConfig()

        headers = config.validate_environment(
            headers={},
            model="yandexgpt/latest",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="secret-123",
        )

        assert headers["Authorization"] == "Api-Key secret-123"

    def test_raises_when_api_key_missing_everywhere(self, monkeypatch):
        monkeypatch.delenv("YANDEX_API_KEY", raising=False)
        config = YandexChatConfig()

        with pytest.raises(ValueError, match="API key"):
            config.validate_environment(
                headers={},
                model="yandexgpt/latest",
                messages=[],
                optional_params={},
                litellm_params={},
            )
