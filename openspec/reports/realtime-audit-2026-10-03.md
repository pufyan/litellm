# Аудит realtime-кода

Дата: 2026-10-03. Ветка `litellm_pufyan`, состояние на коммите `24615a44bb`

Анализ сделан по коду, комментарии и docstring при чтении вырезались. Номера строк даны по исходным файлам на момент аудита и со временем поплывут, поэтому у каждого пункта есть описание, по которому место находится поиском

Статусы: **исправлено**, **открыто**, **решено не трогать**

## Архитектура

```
                         /v1/realtime (proxy_server.py, realtime_websocket_endpoint)
                                      |
                    realtime_api/main.py: if/elif по провайдеру
        +--------------+--------------+---------------+--------------+
        v              v              v               v              v
   provider_config  OpenAI/xAI/    Azure          Bedrock      /vertex_ai/live,
   Gemini/Vertex/   Yandex         handler        своя         responses-ws:
   Meta/Chirp       handler                       реализация    ещё свои пути
        |              |              |               |
        +--------------+------+-------+               |
                              v                       v
                     RealTimeStreaming         _bridge: 2 задачи,
                     (реестр, guardrails,      wait/cancel/gather,
                      reconnect, буфер)        без реестра, reconnect, guardrails
```

Главная системная проблема: единого жизненного цикла сессии нет. Есть минимум три независимые реализации моста клиент <-> провайдер, и возможности у них разные

| Возможность | RealTimeStreaming | Bedrock | /vertex_ai/live |
|---|---|---|---|
| Drain при остановке (реестр) | есть | нет | нет |
| Reconnect / resumption | есть | нет | нет |
| Guardrails | есть | нет | не проверено |
| Остановка при уходе клиента | есть (с `24615a44bb`) | есть | не проверено |

Внутри `RealTimeStreaming` две задачи (`client_ack_messages` и `backend_to_client_send_messages`) делят около 35 изменяемых полей без синхронизации. Состояние сессии нигде не описано явно: оно складывается из восьми булевых флагов (`_backend_setup_complete`, `_flushing_pending_messages_until_setup`, `_content_sent_after_setup`, `_reconnecting_backend`, `_session_created_sent_to_client`, `_guardrail_turn_detection_update_sent`, `_is_transcription_session`, `_shutting_down`). Это неявная машина состояний, и допустимые комбинации флагов никто не определил

## Исправлено

| № | Проблема | Коммит | Как проверено |
|---|---|---|---|
| F1 | Backend-сокет, открытый при reconnect, не закрывался после конца звонка | `aa2b271161` | звонки 3.1 и 3.8, юнит-тест |
| F2 | При неудачном reconnect или отмене посреди него новый сокет не закрывался | `aa2b271161` | юнит-тесты |
| F3 | Каждый `session.update` добавлял полный `instructions` в `input_messages`, все копии уходили в лог | `7299bd23c2` | юнит-тест (промпты в spend log у нас не хранятся) |
| F4 | `pre_call` вызывался на каждом аудиокадре, в логе оставался base64 | `7299bd23c2` | звонки 3.1 и 3.8, юнит-тест |
| F5 | После ухода клиента цикл backend продолжал слать в мёртвый сокет, пачка `Cannot call "send"` | `24615a44bb` | юнит-тест, звонком отдельно не проверялось |

История с F1: коммит откатывали, потому что сразу после выкатки `gemini-3.1-flash-live-preview` начала пачками отдавать 1011. Бисект звонками его оправдал: каждая половина и полный коммит позже работали чисто, а 1011 шли окнами по времени и не зависели от версии кода. Debug-логи прогонов лежали в scratchpad сессии

## Открыто: ядро (`litellm_core_utils/realtime_streaming.py`)

### C1. После reconnect аудио уходит в backend вперемешку

`_flush_pending_messages_until_setup` сразу обнуляет `_pending_messages_until_setup`. После reconnect `_on_backend_setup_acknowledged_after_reconnect` снимает `_reconnecting_backend` и досылает буфер, но в отличие от первичного setup не ставит `_flushing_pending_messages_until_setup`. Клиентская задача видит пустой буфер и шлёт живые кадры напрямую, пока задача backend ещё досылает старые с паузами по 20 мс. Провайдер получает поток не по порядку. Вероятная причина симптома «после обрыва модель не слышит или слышит кашу»

### C2. Досыл буфера блокирует чтение провайдера

Flush идёт внутри задачи backend (обработка `session.created` в `_handle_provider_config_message` и `_on_backend_setup_acknowledged_after_reconnect`) с `asyncio.sleep(0.02)` между аудиокадрами. Полный буфер в 200 кадров это около 4 с, когда ответы провайдера не читаются и аудио модели не идёт клиенту

### C3. Защита от зацикливания reconnect не срабатывает

`_consecutive_immediate_reconnect_closes` обнуляется на любом полученном кадре в `_recv_from_backend_reconnecting_on_drop`, включая `setupComplete`. Сценарий «подключились, setupComplete, через 14 с 1011, resume тем же handle» повторяется бесконечно, как и было видно 2026-10-02. Если resume раз за разом заканчивается 1011, после N попыток надо переходить на fresh-сессию с реплеем транскрипта

### C4. Состояние хода не сбрасывается при reconnect

При reconnect не обнуляются `current_response_id`, `current_item_chunks` и остальные `current_*`, не сбрасывается `_correlation_state` в конфиге провайдера, и для оборванного ответа клиенту не отправляется закрывающий `response.done`. Если обрыв случится, пока модель говорит, клиент останется с `response.created` без завершения

### C5. Два одновременных reconnect

`_reconnect_backend` вызывается из задачи backend (обрыв, goAway) и из клиентской задачи (`session.update` после setup в `_send_to_backend`). Блокировки нет: при совпадении откроются два сокета, один потеряется. Сейчас редкость, потому что voip-клиент шлёт `session.update` один раз

### C6. Любая ошибка отправки складывается в буфер

В `client_ack_messages` любое исключение из `_send_to_backend` при поддержке reconnect кладёт кадр в буфер, даже если это не обрыв, а ошибка трансформации. Пока буфер не пуст, все следующие кадры тоже уходят в буфер и застревают

### C7. Лишняя работа на каждом аудиокадре

Один `input_audio_buffer.append` проходит 4-5 `json.loads` (цикл клиента, `_collect_user_input_from_client_event`, `_is_audio_append`, иногда `_should_buffer_client_message_until_setup`, трансформация провайдера), плюс `json.dumps` и base64-декодирование. Событие надо разбирать один раз и передавать дальше типизированным

### C8. Нет тайм-аутов сессии

Нет ни тайм-аута простоя клиента, ни максимальной длительности. Зависший клиент держит сокет провайдера, пока провайдер сам не закроет его

### C9. Одно состояние хода у двух владельцев

`current_*` хранятся в `RealTimeStreaming` и гоняются в трансформацию и обратно через dict из 7 ключей. При этом `GeminiRealtimeConfig` сам хранит `_correlation_state`, `_turn_closed_by_interrupt`, `_pending_usage_metadata`. Конфиг создаётся на сессию, то есть это stateful-объект сессии под видом stateless-конфига. Из-за этого C4 нельзя решить в одном месте

### C10. Мелочи

После последней неудачной попытки reconnect всё равно выполняется `sleep(2)`. `_transcript_entries` при 200 записях перестаёт пополняться, поэтому в долгом звонке при fresh-reconnect реплеится начало разговора, а не конец. Лимиты чужого API (`GA_MAX_OUTPUT_TOKENS_MAX = 4096` и подобные) захардкожены

## Открыто: провайдеры и общий код

### R1. xAI: словарь частей контента растёт всю сессию (утечка)

`XAIRealtimeNormalizer._content_part_by_key` в `llms/xai/realtime/transformation.py` получает запись на каждую часть каждого ответа (`_remember_content_part`, `_update_content_part_field`) и никогда не чистится. Записи хранят полный текст или транскрипт. Надо удалять ключ на `response.content_part.done` или при закрытии ответа

### R2. Задача drain может быть собрана сборщиком мусора

`RealtimeSessionRegistry._on_signal` делает `loop.create_task(cls._drain_then_delegate(signum))` без сохранения ссылки. asyncio держит на задачи только слабые ссылки, поэтому drain при SIGTERM может молча не выполниться

### R3. Reconnect после истечения токена получает 401

`RealtimeBackendConnector` фиксирует заголовки при старте сессии. Azure AD токен берётся один раз в `realtime_api/main.py`, Gemini через Vertex фиксирует `access_token` в конструкторе `VertexAIRealtimeConfig`, хотя резолвер обновления (`resolve_vertex_access_token`) рядом есть и используется только для Chirp. С goAway примерно каждые 10 минут и токеном на час первый reconnect после истечения токена провалится

### R4. `SSL_VERIFY=False` обрабатывается тремя способами

OpenAI/xAI/Yandex (`OpenAIRealtime._get_ssl_config`) превращают `False` в `True`, то есть оставляют проверку сертификата включённой. Azure передаёт `False` как есть. Общий путь в `llm_http_handler.async_realtime` создаёт контекст `CERT_NONE`. Одна настройка, три разных поведения

### B1. Bedrock прячет состояние сессии в ASGI scope

`bedrock/realtime/handler.py` пишет в `websocket.scope` флаги ожидающего `session.update` и закоммиченной сессии, чтобы протащить их между повторными вызовами при fallback роутера. Скрытая связь, не видна из сигнатур

### B2. Bedrock вызывает `await_output()` из обеих задач

Из `_ack_session_update` в клиентской задаче и из `_forward_bedrock_to_client`. Корректность держится на идемпотентности SDK, которая нигде не зафиксирована

### B3. Bedrock без reconnect, drain и guardrails

Собственный `_bridge` не регистрирует сессию в `RealtimeSessionRegistry`, не умеет reconnect (у Nova Sonic лимит длительности соединения) и не запускает guardrails

### E1. Три обработчика, три политики закрытия

OpenAI сравнивает строки (`"already completed" in str(close_error)`) и пробрасывает голый `Exception`. Azure глотает всё. Общий путь шлёт клиенту событие ошибки и закрывает 1011. Клиент видит одну и ту же ошибку по-разному в зависимости от провайдера

### E2. Мёртвый код в Azure

`forward_messages` в `llms/azure/realtime/handler.py` нигде не вызывается

### D1. Диспетчер на if/elif

`_arealtime` в `realtime_api/main.py` это семь веток с копированием параметров. Часть провайдеров идёт через `provider_config`, часть через свои обработчики, правила нет. Нужен реестр провайдеров по образцу `ProviderConfigManager`

### D2. Мутация чужого dict

`headers.update(extra_headers)` в `_arealtime` правит словарь, пришедший из `kwargs`

### D3. Захардкожена старая версия API Azure

`api_version` по умолчанию `"2024-10-01-preview"`

## Решено не трогать

| Пункт | Почему |
|---|---|
| Рост `self.messages` и `tool_calls` за сессию | около 3 небольших событий на реплику, освобождаются в конце звонка, нужны для стоимости |
| Буфер до setup до 10 МБ на сессию | при текущей нагрузке не критично |
| `current_item_chunks` сбрасывается только на `response.done` | растёт только в пределах хода; вернуться, если C4 будет делаться |

## Что сделано хорошо

Новый код заметно лучше старого, и на него стоит ориентироваться при переделке ядра. Meta (Muse) и Vertex Chirp используют ограниченные очереди, лимит ходов, резолвер токена, неизменяемые dataclass и явные состояния (`_StreamFailure`, `_Closed`, `_TurnDiscarded`). Модуль `realtime_correlation` построен на неизменяемом состоянии и чистых функциях. Мост Bedrock правильно останавливается при неудачной отправке клиенту

## План

Первая очередь, точечные правки по одной, каждая со звонком до коммита: R1, R2, C3, C1, R3

Вторая очередь, единый жизненный цикл для всех провайдеров:

```
SessionBridge
  - один writer в backend через asyncio.Queue: порядок кадров гарантирован, flush не блокирует чтение
  - reconnect только через writer, у одного владельца; сбрасывает состояние хода и закрывает оборванный ответ
  - asyncio.TaskGroup и типизированный исход сессии (ClientGone | BackendClosed(code) | ProxyError)
  - событие клиента разбирается один раз
  - явная машина состояний вместо восьми флагов:
    CONNECTING -> AWAIT_SETUP -> LIVE -> RECONNECTING -> AWAIT_SETUP -> LIVE -> CLOSED
```

Bedrock переезжает на общий мост и получает реестр и тайм-ауты. Обработчики OpenAI/Azure/xAI/Yandex сводятся к «URL и заголовки». Политика ошибок и кодов закрытия становится одной. После этого диспетчер переходит на реестр провайдеров

Правила работы, выученные на этом аудите: realtime-правку перед коммитом проверять звонком через прокси с `LITELLM_LOG=DEBUG`; при регрессии сначала зафиксировать все переменные (клиент, модель, время, окружение) и бисектить звонками, а не спорить с результатом
