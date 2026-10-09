# Реестр тестов realtime (задача 1.5, инвентаризация от 2026-10-03)

Обновлять по мере фаз ченджа. Команды: `make test-unit-core-utils`
(`pytest tests/unit/litellm_core_utils --tb=short -vv -n 2`),
`make test-unit-llms` (`pytest tests/unit/llms --tb=short -vv -n 4`).

## Ядро моста (фазы 1–3 затронут сильнее всего)

- `tests/unit/litellm_core_utils/test_realtime_streaming.py` — ~100+ тестов:
  порядок/буфер/flush (`deferred_setup_*`, `flush_paces_*`,
  `clear_drops_buffered_appends`, `client_messages_buffer_while_reconnecting`),
  reconnect (`go_away_triggers_native_resumption`,
  `unexpected_backend_drop_reconnects`, `reconnect_gives_up/resets_counter`),
  транскрипт/реплей (`accumulates_turns`, `replays_transcript`,
  `native_does_not_replay`), beta/GA-перемаппинг (`remap_beta_session_to_ga_*`,
  `translate_event_to_beta_*`), shutdown (`shutdown_close_notifies_1012`),
  spend (`log_messages_sets_input_messages_on_logging_obj`).
- `tests/unit/litellm_core_utils/test_realtime_backend_connector.py` — коннектор бэкенда.
- `tests/unit/litellm_core_utils/test_realtime_session_mapping.py` — маппинг сессий.
- `tests/unit/litellm_core_utils/test_realtime_schema_normalization.py` — нормализация схем.
- `tests/unit/litellm_core_utils/test_realtime_errors.py` — ошибки.
- `tests/unit/realtime_api/test_main.py` — роутинг/credentials/health-check точки входа.
- `tests/test_litellm/proxy/shutdown/test_realtime_session_registry.py` —
  семантика register/unregister/drain/force-close, `on_signal`.
- `tests/unit/proxy/common_utils/test_realtime_cache.py` — кэш realtime.
- `tests/test_litellm/proxy/realtime_endpoints/test_realtime_webrtc_endpoints.py` — webrtc-эндпоинты.

## Провайдерские трансформации (фаза 4 — адаптеры)

- `tests/unit/llms/openai/realtime/` (`test_openai_realtime_handler.py` incl.
  beta-header forwarding, `test_transcription_sessions.py`).
- `tests/unit/llms/azure/realtime/` (`test_azure_realtime_handler.py`,
  `test_handler.py` — URL beta/GA, `default_maintains_backwards_compatibility`).
- `tests/unit/llms/gemini/realtime/test_gemini_realtime_transformation.py`.
- `tests/unit/llms/vertex_ai/realtime/test_vertex_ai_realtime_transformation.py`.
- `tests/unit/llms/meta/realtime/test_meta_realtime_transformation.py`.
- `tests/unit/llms/bedrock/realtime/` (handler: lifecycle/failure/auth;
  transformation) — вне скоупа фаз 1–6, затронется фазой 7.
- `tests/test_litellm/llms/xai/realtime/test_xai_realtime_transformation.py`
  (index/idempotency/usage, `TestXaiNativeResumption`).
- `tests/unit/llms/base_llm/realtime/test_transcription_protocol.py`.

## Биллинг (фаза 6)

- `tests/unit/test_cost_calculator.py` — формулы
  `handle_realtime_stream_cost_calculation`,
  `handle_realtime_transcription_cost_calculation` (per-second).
- `tests/integration/cost_calculation/test_batch_realtime_cost.py` +
  `cost_tracking_cases.json` (`realtime_costs.*`, zero-spend кейсы).
- `tests/integration/pricing/test_realtime_cached_audio_pricing.py`.

## Живые (для звонков по фазам)

- `tests/llm_translation/realtime/` (openai/xai/yandex/guardrails, base).
- `tests/e2e/llm_translation/realtime/` (e2e, bedrock, pipecat audio).
- `tests/llm_translation/realtime/test_realtime_guardrails_openai.py` — guardrails фаза 3.5.
