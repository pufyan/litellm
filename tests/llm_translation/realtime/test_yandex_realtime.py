"""
Yandex AI Studio Realtime API tests.

Unit tests assert the connection configuration (endpoint, path, auth scheme,
model URI passthrough) that differs from the OpenAI base handler. These run
without network access. The E2E class exercises the live API when
YANDEX_API_KEY is set and is skipped otherwise.
"""

import os
import sys
from typing import Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.llms.yandex.realtime.handler import YandexRealtime
from tests.llm_translation.realtime.base_realtime_tests import BaseRealtimeTest


def test_default_api_base_is_yandex_endpoint():
    assert YandexRealtime()._get_default_api_base() == "https://rest-assistant.api.cloud.yandex.net"


def test_auth_header_uses_api_key_scheme_not_bearer():
    headers = YandexRealtime()._get_additional_headers("secret-123")
    assert headers == {"Authorization": "Api-Key secret-123"}
    assert "Bearer" not in headers["Authorization"]


def test_no_openai_beta_header_even_when_requested():
    headers = YandexRealtime()._get_additional_headers("k", openai_beta_realtime=True)
    assert "OpenAI-Beta" not in headers


def test_construct_url_targets_realtime_openai_path_over_wss():
    url = YandexRealtime()._construct_url(
        "https://rest-assistant.api.cloud.yandex.net",
        {"model": "gpt://b1gfolder/speech-realtime-250923"},
    )
    assert url.startswith("wss://rest-assistant.api.cloud.yandex.net/v1/realtime/openai")
    assert "model=" in url


def test_construct_url_preserves_yandex_model_uri():
    from urllib.parse import parse_qs, urlparse

    url = YandexRealtime()._construct_url(
        "https://rest-assistant.api.cloud.yandex.net",
        {"model": "gpt://b1gfolder/speech-realtime-250923"},
    )
    model = parse_qs(urlparse(url).query)["model"][0]
    assert model == "gpt://b1gfolder/speech-realtime-250923"


class TestYandexRealtime(BaseRealtimeTest):
    """
    E2E tests for Yandex Realtime API (voice agents).

    OpenAI-compatible event protocol:
    - Endpoint: wss://rest-assistant.api.cloud.yandex.net/v1/realtime/openai
    - Model URI: gpt://<folder_id>/speech-realtime-250923
    - Auth: Authorization: Api-Key <key>
    """

    def get_model(self) -> str:
        folder_id = os.getenv("YANDEX_FOLDER_ID", "")
        return f"yandex/gpt://{folder_id}/speech-realtime-250923"

    def get_api_key_env_var(self) -> str:
        return "YANDEX_API_KEY"

    def get_initial_event_type(self) -> Tuple[str, ...]:
        return ("session.created",)
