## Why

Realtime-мост держит состояние всей сессии в памяти до конца звонка: `RealTimeStreaming.messages`, `input_messages`, `tool_calls`, чанки транскриптов, буфер досылки, реестр сессий на сильных ссылках. Капы есть только у pending-буфера (200 сообщений / 10 МБ) и транскрипта (200 записей). Остальные накопители растут линейно с длительностью звонка. На часовых звонках и висящих сессиях это дает рост RSS без возврата, плюс зависшие задачи и незакрытые сокеты при reconnect удерживают объекты после конца сессии.

## What Changes

- Вводятся границы памяти для всех накопителей сессии: ring-buffer или дроп старых с счетчиком вместо бесконечного append.
- Реестр сессий остается на сильных ссылках, но освобождение становится детерминированным: done-callback задач плюс reaper orphan-сессий, drain-дефолт не меняется.
- Закрытие сокетов и отмена задач делаются детерминированными: retry-цикл reconnect закрывает промежуточные сокеты, зависший `recv` прерывается таймаутом.
- LRU/evict вводится везде, где сейчас словарь только растет (`xai _content_part_by_key`, Chirp `_turn`, чанки Gemini).
- Добавляется наблюдаемость: счетчики дропов и размеров накопителей в debug-лог, способ прогнать звонок под `tracemalloc`.

## Capabilities

### New Capabilities
- `realtime/resource-bounds`: границы памяти, дроп-политики, bounded-реестр, детерминированное закрытие, наблюдаемость утечек.

### Modified Capabilities
- Нет: `openspec/specs/` пуст, поведение клиентского диалекта и spend-лога не меняется.

## Impact

- Затрагивает: `litellm/litellm_core_utils/realtime_streaming.py`, `litellm/proxy/shutdown/realtime_session_registry.py`, `litellm/llms/custom_httpx/llm_http_handler.py`, `litellm/llms/xai/realtime/transformation.py`, `litellm/llms/gemini/realtime/transformation.py`, `litellm/llms/vertex_ai/audio_transcription/realtime_backend.py`, `litellm/llms/bedrock/realtime/handler.py`.
- Клиентский `/v1/realtime` не меняется. Формат spend-записей не меняется, добавляются только debug-счетчики.
- Каждая фаза проверяется юнит-тестами на капы и долгим staging-звонком с плоским RSS.
