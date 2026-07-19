"""
Handler for Yandex AI Studio Realtime API `/v1/realtime/openai` endpoint.

Yandex's Realtime API is OpenAI-compatible, so we inherit from OpenAIRealtime
and override only the configuration differences:
- Endpoint host and path: wss://rest-assistant.api.cloud.yandex.net/v1/realtime/openai
- Authorization uses the "Api-Key" scheme, not "Bearer"
- The model is passed as a Yandex model URI: gpt://<folder_id>/<model>

This requires websockets, and is currently only supported on LiteLLM Proxy.
"""

from litellm.constants import YANDEX_REALTIME_API_BASE
from litellm.types.realtime import RealtimeQueryParams

from ...openai.realtime.handler import OpenAIRealtime


class YandexRealtime(OpenAIRealtime):
    """
    Handler for Yandex AI Studio Realtime API (voice agents).

    Uses the same WebSocket event protocol as OpenAI Realtime GA, so all
    streaming logic is inherited from OpenAIRealtime.
    """

    def _get_default_api_base(self) -> str:
        return YANDEX_REALTIME_API_BASE

    def _get_additional_headers(
        self,
        api_key: str,
        *,
        openai_beta_realtime: bool = False,
    ) -> dict:
        return {"Authorization": f"Api-Key {api_key}"}

    def _construct_url(self, api_base: str, query_params: RealtimeQueryParams) -> str:
        from httpx import URL

        api_base = api_base.replace("https://", "wss://").replace("http://", "ws://")
        url = URL(api_base).copy_with(path="/v1/realtime/openai")
        if query_params:
            url = url.copy_with(params=query_params)
        return str(url)
