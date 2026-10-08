## Why

Мост клиент ↔ провайдер в `/v1/realtime` собран из трёх независимых реализаций с разными возможностями, а ядро (`RealTimeStreaming`, 2271 строка) управляется восемью неявными булевыми флагами и двумя пишущими задачами без очереди. Новые модели — OpenAI `gpt-live-1` (отдельный API `/v1/live`, делегация, посекундный биллинг, нет события конца ответа) и Gemini 3.8 Live Extended Thinking (`turnComplete` закрывает реплику, а не взаимодействие; только `NON_BLOCKING`-тулзы) — ломают базовые допущения ядра, а не отдельные строки. Точечные фиксы из аудита `openspec/reports/realtime-audit-2026-10-03.md` лечат симптомы, добавляя 9-й и 10-й флаги. Нужен новый каркас с запасом под Gen2-модели и сохранением совместимости с текущими клиентами.

## What Changes

- Вводится `SessionBridge`: две транспортные ноги (клиент/бэкенд) с независимыми явными машинами состояний, один writer в бэкенд через очередь с двумя линиями (управление/аудио), канонические типизированные события (разбор один раз).
- Вводится capability-дескриптор провайдер×модель и реестр провайдеров вместо if/elif-диспетчера в `realtime_api/main.py`.
- Состояние хода становится capability-driven (`LISTENING / SPEAKING / WORKING / IDLE`, сигнал завершения — плагин из дескриптора), вместо `current_*` и допущения «конец ответа = `response.done`».
- Продолжение сессии — лестница: нативный resume-handle → fork stored-сессии → сидирование истории → реплей транскрипта; reconnect с лимитом и эскалацией вместо бесконечного resume.
- Биллинг: посекундный учёт голоса (`usage.seconds`) и раздельный учёт делегированного бэкенда рядом с токенным.
- Graceful close — протокол (слушатель → close → drain → финальный usage + reason), единый для всех провайдеров.
- Клиентская сторона `/v1/realtime` (диалект OpenAI Realtime) не меняется: Gen2-бэкенды транслируются адаптерами в привычные клиентам события.

## Capabilities

### New Capabilities
- `realtime/session-bridge`: ноги транспорта, writer с очередью, явные состояния ног, graceful close, таймауты сессии.
- `realtime/provider-capabilities`: схема дескриптора и реестр провайдеров, замена диспетчера.
- `realtime/turn-tracking`: capability-driven состояние хода, закрытие оборванного хода, сидирование истории при старте.
- `realtime/session-continuation`: лестница продолжения, политика reconnect (backoff, лимит, эскалация), обновление токена при reconnect.
- `realtime/voice-billing`: посекундный учёт голоса, раздельный учёт бэкенда, финальный usage при close.

### Modified Capabilities
- Нет: `openspec/specs/` пуст, все спецификации новые. Поведение существующих клиентов `/v1/realtime` сохраняется как требование внутри `realtime/session-bridge`.

## Impact

- Затрагивает: `litellm/litellm_core_utils/realtime_streaming.py`, `litellm/realtime_api/main.py`, адаптеры `litellm/llms/*/realtime/`, spend-логирование realtime-сессий.
- Новые провайдерские режимы: OpenAI GPT-Live (`/v1/live`, делегация), Gemini 3.8 Live / Extended Thinking (`interaction_status`, `NON_BLOCKING`-тулзы).
- Bedrock-мост (`litellm/llms/bedrock/realtime/`) и Vertex/STT-мосты (`litellm/llms/vertex_ai/realtime/`, `audio_transcription/realtime_*`) на общий мост в этом чендже НЕ переезжают (отдельный чендж-продолжение).
- Каждая фаза проверяется живым звонком через прокси с `LITELLM_LOG=DEBUG` (правило из аудита).
