# Proposal

## Why

На провайдере `xai` голос замедлен и понижен, настройка сэмплрейта не работает; на `gemini`/`openai` чисто. Байтовый путь `response.output_audio.delta` — passthrough без ресемплинга, поэтому любое расхождение частоты потока и часов плеера превращается в замедление + падение питча. По доке xAI `audio/pcm 16000` валиден на вход и выход, дефолт обеих сторон — `24000`.

## What Changes

- xAI-нормалайзер форсит `audio.input.format` и `audio.output.format` в `{"type": "audio/pcm", "rate": 16000}` в обе стороны: в `patch_outgoing_session` (что уходит в xAI) и в `normalize` для `session.created`/`session.updated` (что видит клиент в сетапе).
- Соседние ключи блоков (`turn_detection`, `transcription`, `voice`, `speed`, `transport`) сохраняются; снап `_snap_sample_rate_logged` не удаляется (после форса no-op для xAI-пути).

## Capabilities

### New Capabilities
- Нет новых capability: фиксация значений внутри существующего xAI-маппинга.

### Modified Capabilities
- `realtime/session-mapping`: аудио-форматы xAI-сессии фиксируются на `pcm 16k` в обе стороны (тот же delta-путь, что у `g711-format-fix` и `voice-audio-preserve`).

## Impact

- Затрагивает: `litellm/llms/xai/realtime/transformation.py` + тесты `tests/test_litellm/llms/xai/realtime/test_xai_realtime_transformation.py` (класс `TestXaiSampleRateValidation` пинит старое поведение passthrough/fallback-to-24000 — обновляется осознанно).
- Остальные провайдеры не трогаем. Spend-формат не меняется. Клиентский `/v1/realtime` контракт не меняется (GA-формат тот же).
- Откат: revert нормалайзера и тестов.
