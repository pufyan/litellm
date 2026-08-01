"""
Yandex Foundation Models chat transformation.

Transforms OpenAI-format requests to the Yandex Foundation Models
`/foundationModels/v1/completion` format and back.
"""

import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

YANDEX_FOUNDATION_MODELS_BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"


class YandexError(BaseLLMException):
    """Yandex Foundation Models API error."""

    pass


class YandexChatConfig(BaseConfig):
    """
    Configuration class for the Yandex Foundation Models chat API.

    Model URI format: gpt://<folder_id>/<model>, where folder_id is a
    Yandex Cloud folder id passed via `folder_id=` kwarg or the
    YANDEX_FOLDER_ID env var.
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = api_base or get_secret_str("YANDEX_API_BASE") or YANDEX_FOUNDATION_MODELS_BASE_URL
        return f"{base}/completion"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_key = api_key or get_secret_str("YANDEX_API_KEY")
        if api_key is None:
            raise ValueError("Missing Yandex API key. Pass api_key= or set YANDEX_API_KEY env var.")

        headers["Authorization"] = f"Api-Key {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_supported_openai_params(self, model: str) -> List[str]:
        # Streaming not implemented yet: Yandex sends newline-delimited JSON
        # chunks that are cumulative (full text so far), not OpenAI-style
        # deltas, which requires a dedicated response iterator.
        return ["temperature", "max_tokens", "max_completion_tokens"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            elif param in ("max_tokens", "max_completion_tokens"):
                optional_params["maxTokens"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        folder_id = litellm_params.get("folder_id") or get_secret_str("YANDEX_FOLDER_ID")
        if not folder_id:
            raise ValueError(
                "Missing Yandex folder_id. Pass folder_id= or set YANDEX_FOLDER_ID env var. "
                "It is required to build the modelUri gpt://<folder_id>/<model>."
            )

        yandex_messages = [
            {"role": message.get("role", "user"), "text": message.get("content") or ""} for message in messages
        ]

        return {
            "modelUri": f"gpt://{folder_id}/{model}",
            "completionOptions": {
                "stream": False,
                "temperature": optional_params.get("temperature", 0.6),
                "maxTokens": optional_params.get("maxTokens"),
            },
            "messages": yandex_messages,
        }

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        try:
            response_json = raw_response.json()
        except Exception:
            raise YandexError(
                status_code=raw_response.status_code,
                message=f"Invalid JSON response: {raw_response.text}",
            )

        result = response_json.get("result", {})
        alternatives = result.get("alternatives", [])
        if not alternatives:
            raise YandexError(
                status_code=raw_response.status_code,
                message=f"No alternatives in response: {raw_response.text}",
            )

        alternative = alternatives[0]
        message_data = alternative.get("message", {})
        finish_reason = "stop" if alternative.get("status") == "ALTERNATIVE_STATUS_FINAL" else None

        model_response.choices = [
            Choices(
                index=0,
                message=Message(role=message_data.get("role", "assistant"), content=message_data.get("text")),
                finish_reason=finish_reason,
            )
        ]

        usage_json = result.get("usage", {})
        usage = Usage(
            prompt_tokens=int(usage_json.get("inputTextTokens", 0)),
            completion_tokens=int(usage_json.get("completionTokens", 0)),
            total_tokens=int(usage_json.get("totalTokens", 0)),
        )

        model_response.id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model_response.created = int(time.time())
        model_response.model = model
        setattr(model_response, "usage", usage)

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return YandexError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
