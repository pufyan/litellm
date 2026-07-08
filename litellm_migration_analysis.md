# Анализ миграции ai_voip → LiteLLM (Gemini Live API)

Дата: 2026-07-06. Итог исследовательской сессии: сравнение того, что проект шлёт в Gemini Live API напрямую, что умеет LiteLLM realtime-прокси, и план доработки LiteLLM.

## Контекст

- `ai_voip` использует `GeminiRealtimeProvider` (`apps/ai_voip/src/ai/gemini_provider.cpp`, ~2700 строк) — прямое WebSocket-соединение к `wss://generativelanguage.googleapis.com/ws/...BidiGenerateContent` на нативном Gemini Live протоколе.
- Цель: работать через LiteLLM-прокси (унифицированный биллинг/ключи), модель в LiteLLM уже заведена: `gemini-live`.
- **Ключевой факт**: LiteLLM НЕ проксирует Gemini WebSocket 1:1. Его эндпоинт `/v1/realtime` говорит с клиентом протоколом **OpenAI Realtime API** и сам транслирует его в Gemini Live (`litellm/llms/gemini/realtime/transformation.py`, класс `GeminiRealtimeConfig`). Значит C++-провайдер придётся переписать под OpenAI Realtime протокол.

## Что реально используется в проде (config.json + дефолты кода)

- model=`gemini-3.1-flash-live-preview`, temperature=0.8, maxOutputTokens=8192, modalities=AUDIO, voice=Leda/ru-RU
- VAD: enabled, start_sensitivity=3, end_sensitivity=0, prefixPadding=100ms, silence=300ms, activityHandling=START_OF_ACTIVITY_INTERRUPTS, turnCoverage=TURN_INCLUDES_ONLY_ACTIVITY
- tools: 4 функции из конфига + программный transfer tool
- input/output транскрипция: обе включены (output критична: на ней transfer protection — детекция «сейчас соединю» и dialog logger)
- sessionResumption + goAway + авто-реконнект: включено дефолтом кода (~400 строк логики в gemini_provider.cpp)
- contextWindowCompression: включено дефолтом (64k/102400)
- НЕ используется: thinking mode, affective dialog, proactive audio (все false; affective/proactive вообще требуют v1alpha и Gemini 2.5, а у нас 3.1)

## Сравнение: LiteLLM support matrix

| Возможность | Проект использует | LiteLLM транслирует |
|---|---|---|
| model/temperature/maxOutputTokens/modalities | да | ✅ |
| voice | да | ✅ (но languageCode — нет) |
| systemInstruction, tools (двусторонне) | да | ✅ |
| input транскрипция | да | ✅ |
| **output транскрипция** | да, критично | ❌ (закомментировано в session_configuration_request, обратный transform уже есть) |
| VAD on/off + padding + silence | да | ✅ через turn_detection |
| VAD sensitivity | да | ❌ (тип AutomaticActivityDetection поля имеет, маппинга нет) |
| activityHandling/turnCoverage | да (= дефолты API) | ❌, но дефолты совпадают — некритично |
| **interrupted как отдельное событие** | да (очистка playback-буфера) | ❌ схлопывается в response.done |
| **sessionResumption/goAway/реконнект** | да | ❌ не пробрасывается вообще |
| contextWindowCompression | да | ❌ (поле в типе есть, маппинга нет) |
| topP/topK/candidateCount/mediaResolution | дефолты кода | ❌ |

Блокирующие: output-транскрипция, interrupted, session resumption. Остальное некритично.

## Находки в самом проекте (независимо от LiteLLM)

1. **Захардкоженный API-ключ Google** в `gemini_provider.hpp:20` (дефолт `api_key`) и открытым текстом в `config.json`. Вынести в секрет/env.
2. **VAD sensitivity, вероятно, не работает и сейчас**: проект шлёт числа 3/0, а API ожидает enum-строки `START_SENSITIVITY_HIGH/LOW` (валидные числа только 0/1/2; комментарий в hpp «0-3, where 3 is highest» не соответствует API). Починить на enum-строки.
3. Greeting/transfer protection реализованы клиентским дропом аудио (send_audio_frame), НЕ через Gemini activityHandling — при миграции на LiteLLM конфликтов нет.

## План доработки LiteLLM

Файлы: `litellm/llms/gemini/realtime/transformation.py`, `litellm/types/llms/gemini.py`, `litellm/llms/base_llm/realtime/transformation.py` (BaseRealtimeConfig), общий realtime-цикл прокси (realtime_streaming).

1. **outputAudioTranscription** — СДЕЛАНО (2026-07-06, ветка `litellm_pufyan`, коммит 49cd91e27b): запрашивается всегда в обоих setup-путях (eager `session_configuration_request()` и deferred `_handle_session_update()`).
2. **interrupted** — СДЕЛАНО cherry-pick'ом diff'а открытого PR BerriAI/litellm#31709 (коммит dae76bcc55). PR upstream всё ещё open — при merge наш коммит совпадёт по содержимому.
3. **Параметры** — СДЕЛАНО частично (коммит 34cb1fc1f9): VAD sensitivity (`turn_detection.start_sensitivity/end_sensitivity`: "high"/"low" → enum-строки Gemini), voice как dict `{"name", "language_code"}` → speechConfig.languageCode, `context_window_compression` (Gemini-shape passthrough в setup). НЕ делал: activityHandling/turnCoverage (дефолты совпадают), topP/topK/candidateCount/mediaResolution (некритично). Важно: параметры применяются через session.update (deferred/первый session.update); статический eager-setup их не включает.
4. **Session resumption — generic-механизм** (основная работа). Общий WebSocket-цикл прокси: `litellm/litellm_core_utils/realtime_streaming.py` (класс RealTimeStreaming) — именно его учить пересоздавать backend-сокет. В main (на 2026-07-06, ~v1.92.0-rc.1) никаких следов sessionResumption/goAway нет. Принятые решения:
   - Делать провайдер-агностично: в `BaseRealtimeConfig` capability-хуки (`supports_session_resumption()`, `extract_resumption_state(msg)`, `build_resume_request(state, original_setup)`), общий цикл прокси реализует удержание клиентского сокета, ретраи/backoff, буферизацию аудио на время реконнекта. Gemini — первая реализация (handle из `sessionResumptionUpdate`, проактивный реконнект на `goAway`).
   - Наружу — пассивные события `litellm.session.reconnecting` / `litellm.session.reconnected` (+ поле resumed: по handle или с нуля) и error при исчерпании ретраев. Механика — целиком внутри LiteLLM. Клиенту события нужны для: метрик, подавления silence-nudge на время дыры в аудио, обработки фатального обрыва.
   - Выигрыш на стороне C++: ~400 строк reconnect-логики заменяются подпиской на 2 события.

Пункты 1–3 — кандидаты на PR в upstream BerriAI/litellm. Пункт 4 — сначала у себя, потом отдельным PR.

### Дизайн пункта 4 (session resumption, по коду на 2026-07-06)

Ключевой факт из кода: backend-сокет создаётся в `llm_http_handler.py` (`async_realtime`, ~строка 5651) и живёт в `async with backend_ws:` — `RealTimeStreaming` получает уже готовый сокет и не может его пересоздать. Отсюда план:

1. **Refactor владения сокетом**: вынести создание backend ws в фабрику (`RealtimeBackendConnector` с методом `connect() -> ClientConnection`), передавать её в `RealTimeStreaming` (DI); `async with` заменить на явный lifecycle. Без изменения поведения — чистый refactor, отдельный коммит.
2. **Capability-хуки в `BaseRealtimeConfig`** (default: резюмирование не поддерживается):
   - `supports_session_resumption()`
   - `initial_resumption_fields()` — что добавить в setup для включения (Gemini: `"sessionResumption": {}`)
   - `extract_resumption_state(event)` — handle из `sessionResumptionUpdate` (+ resumable)
   - `extract_go_away(event)` — `goAway.timeLeft`
   - `build_resume_setup(state, original_setup)` — setup с `sessionResumption.handle`; служебные события клиенту не пробрасываются
3. **Реконнект-логика в `RealTimeStreaming`**: хранить последний handle; на goAway — проактивное пересоздание сокета, на неожиданный `ConnectionClosed` бэкенда — ретраи с backoff (3 попытки, 0.5/1/2s). На время дыры клиентские сообщения буферизовать, переиспользуя готовую механику `_pending_messages_until_setup` / `_collapse_buffered_audio_messages` / `_flush_pending_messages_until_setup` (лимит байт уже реализован); flush после нового setupComplete.
4. **События клиенту**: `litellm.session.reconnecting` / `litellm.session.reconnected` (`resumed: "native" | "replayed" | "fresh"`), при исчерпании ретраев — error-событие и закрытие клиентского сокета.
5. **Тесты**: unit на хуки Gemini-конфига + интеграционные на цикл с fake backend ws (goAway, обрыв, исчерпание ретраев, целостность буфера аудио).

Решение (2026-07-06, обсуждено): восстановление двухуровневое, провайдер-независимое ядро.
- **Уровень 1 — нативный** (хук провайдера): Gemini handle. Точно и бесплатно.
- **Уровень 2 — реплей из буфера** (общий код, итерация 2): `RealTimeStreaming.messages` уже накапливает транскрипции обеих сторон, function-call'ы и barge-in-события. При отсутствии/отказе handle: новая сессия + кешированный session.update + реплей истории стандартными `conversation.item.create` через существующий трансформ (работает для любого провайдера автоматически). При реплее добавлять служебные текстовые вставки, сохраняющие нюансы голосового диалога: префикс «далее — транскрипт голосового разговора, восстановленного после обрыва», маркеры прерываний ([абонент перебил]) и дыр в связи. Кап на размер буфера; для коротких голосовых сессий стоимость токенов незначительна.
- Итерация 1 = ядро (фабрика, хуки, реконнект, события) + уровень 1 — СДЕЛАНО 2026-07-06 (коммиты 1460da75f5 фабрика RealtimeBackendConnector, e820c1eb6f хуки + Gemini, afa6e8a478 реконнект/буферизация/события litellm.session.*). setup Gemini теперь всегда включает sessionResumption. Итерация 2 (реплей) — СДЕЛАНА (ba9bb659e2): RealTimeStreaming копит транскрипт (user/assistant/notes, кап 200 записей), при fresh-реконнекте провайдерский хук build_history_replay_messages инжектит его; Gemini — один clientContent с ролями user/model и turnComplete:false; событие reconnected отдаёт resumed=replayed. e2e-проверка на живом прокси ПРОЙДЕНА (2026-07-06, локальный прокси вместо docker, та же БД): output-транскрипция работает (transcript delta пришёл клиенту); goAway приходит на ~541-й секунде; реконнект ~1 с; resumed=native при живом диалоге (handle шлётся Gemini только при содержательной активности, при чистой тишине handle не выдаётся — тогда fresh/replayed, для реальных звонков не проблема); после native-реконнекта сессия продолжает отвечать. Локальный запуск: uv run python litellm/proxy/proxy_cli.py --config /home/puf/Work/LiteLLM/config.yaml --port 4000, env из .env.docker-import (DATABASE_URL на 172.30.0.2 — docker litellm_db_1), docker-контейнер litellm_litellm_1 остановлен.

Порядок: 1 → 3 (быстрый прототип; 2 закрывается чужим PR #31709), затем 4.

Ветки/релизы LiteLLM: разработка только в `main` (HEAD 2026-07-05, тег v1.92.0-rc.1); фиче-ветки для PR — от `main`, прод-деплой — от последнего stable-тега (`v1.x.y-stable`) + cherry-pick своих коммитов.

## Механика перебивания (interrupted / barge-in) через LiteLLM

Физика: модель генерирует аудио быстрее реального времени, у клиента в playback-буфере лежат секунды ещё не проигранного звука. Перебивание = два действия: Gemini обрывает генерацию (шлёт `serverContent.interrupted`), клиент выбрасывает уже полученный буфер. «Отозвать» отправленные байты сервер не может.

Наша ветка (dae76bcc55, из PR #31709) транслирует `interrupted` в ДВА события клиенту в гарантированном порядке: `input_audio_buffer.speech_started`, затем `response.done`. Upstream main этого НЕ умеет (interrupted схлопывается в голый `response.done`) — пока PR #31709 не смержен, деплой только со своего форка.

Правила для нового pjcpp-провайдера:
- чистить playback сразу по `speech_started` (аналог текущего `handle_interruption`: clear_playback_queue + interruption_callback), НЕ ждать `response.done`;
- различение «оборвали vs закончил сам»: `response.done` с предшествующим `speech_started` = interrupted (`on_interrupted`), голый `response.done` = turnComplete (`on_turn_complete`). Детектить ФЛАГОМ, выставленным на `speech_started`, а не таймингом;
- окна между двумя событиями практически нет (рождаются из одного Gemini-кадра, уходят подряд по одному сокету). Реальная задержка barge-in — VAD Gemini (десятки-сотни мс, регулируется start_sensitivity) + сеть; транзит LiteLLM добавляет миллисекунды;
- `speech_started` на Gemini-пути LiteLLM эмитится ТОЛЬКО при interrupted (других источников нет) — событие однозначно;
- нюанс протокола: `audio_start_ms`/`item_id` в событии синтетические, status в `response.done` обычный (не "cancelled" как у OpenAI) — на них не опираться;
- transfer protection («interrupted при активной защите не сбрасывает буфер фраз») остаётся клиентской логикой как есть;
- при реплей-восстановлении сессии прерывания сохраняются в истории вставкой `[note: the user interrupted the assistant's previous answer]`.

НЕ проверено вживую: сквозной сценарий с реальным перебивающим аудио (текстом не спровоцировать) — проверить первым тестовым звонком через pjcpp.

## Аудиоформаты и частоты (принято 2026-07-07)

Принцип: LiteLLM — тупой транзит, аудио НЕ конвертирует (и не конвертировал), только маркирует mime и сообщает частоты; ресемплинг — целиком забота клиента.

Вход и выход — независимые потоки в разные стороны, частоты НЕ обязаны совпадать. Gemini Live: вход нативно 16 кГц (распознавание речи; API ресемплит любую входную частоту), выход всегда 24 кГц (синтез голоса, звучит богаче). Официальная дока: «Audio output always uses 24kHz. Input audio is natively 16kHz, but the Live API will resample if needed».

| Провайдер | Вход | Выход |
|---|---|---|
| OpenAI / Azure / xAI | pcm16 = 24 кГц (passthrough) | pcm16 = 24 кГц |
| Gemini (AI Studio) / Vertex | нативно 16 кГц (API ресемплит любую) | 24 кГц |
| Bedrock (Nova Sonic) | 16 кГц | 24 кГц |

- Частоты определяются ПОД КАПОТОМ по модели (`_audio_sample_rate(model, is_output)` читает `input/output_audio_sample_rate` из model_cost, дефолт 16k-вход/24k-выход). Новая модель с другими частотами = запись в cost-map, без изменений кода. Публичная сигнатура `get_audio_mime_type(input_audio_format)` НЕ менялась — model не течёт в интерфейс
- Контракт с клиентом стабилен (форма `session.audio.input/output.format` та же), меняются только числовые значения rate в зависимости от модели. `session.created`/`session.updated` несут `audio.input.format={"type":"audio/pcm","rate":16000}`, `audio.output.format={...,"rate":24000}` + flat `input/output_audio_format="pcm16"`. Коммит dd10acaf8d (был баг: обе частоты хардкожены в 24000, что мислейблит вход)
- Разделение: клиент<->LiteLLM — стабильный OpenAI-контракт; LiteLLM<->провайдер — частота подстраивается под модель прозрачно для клиента
- Для pjcpp: два независимых ресемплера — восходящий линия 8k -> 16k (в модель), нисходящий 24k -> 8k (в линию). Целевые частоты брать из `session.audio.*.format.rate`, не хардкодить
- Убран дублирующий Vertex-override `get_audio_mime_type` (хардкодил 24k вход — неверно)
- G.711-вход через Gemini-путь НЕ работает: маппинг g711_ulaw/alaw -> audio/pcmu/pcma есть, но формат клиента не прокинут в get_audio_mime_type() — мёртвый код. Не полагаться

## Изменения на стороне ai_voip (второй этап, после LiteLLM)

- Переписать `GeminiRealtimeProvider` под OpenAI Realtime протокол (session.update, input_audio_buffer.append, conversation.item.create, function_call события) — фактически новый `OpenAIRealtimeProvider` с endpoint на LiteLLM.
- Выкинуть: session resumption/goAway/reconnect (уходит в LiteLLM), нативную сборку Gemini setup JSON.
- Сохранить: greeting/transfer protection (клиентский дроп аудио), sender/keepalive-потоки, транскрипт-логгер (события транскрипции приходят в OpenAI-формате).
- Endpoint: `ws(s)://<litellm-host>/v1/realtime?model=gemini-live`, авторизация виртуальным ключом LiteLLM вместо Google-ключа.

## Открытые вопросы

- Критичность потери VAD sensitivity не решена (возможно, не работает и сейчас — см. находку 2; проверить реальное влияние enum-фикса до миграции).
- Инфраструктура патчей LiteLLM (форк vs patch поверх pip) — решено начать с форка GitHub + upstream remote.
