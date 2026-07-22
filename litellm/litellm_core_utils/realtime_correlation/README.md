# Realtime Correlation Layer

## Зачем это существует

LiteLLM proxy транслирует нативный WebSocket-протокол каждого realtime-провайдера
(OpenAI, Azure, xAI, Gemini/Vertex, Bedrock Nova Sonic, Yandex) в единый поток
событий, совместимый с протоколом **OpenAI Realtime API**. Клиент (Pipecat,
собственный voice-агент и т.п.) видит один и тот же формат событий независимо от
того, какой провайдер реально стоит за прокси.

Ключевая проблема, которую решает этот модуль: у каждого события в потоке
(дельта или "done") должен быть согласованный correlation-key —

```
(response_id, item_id, output_index, content_index)
```

— чтобы клиент понимал, к какой именно фразе/элементу относится сигнал
завершения. Это особенно критично при **barge-in** (пользователь перебивает
модель): в этот момент может существовать одновременно несколько логических
"нитей" — старый ответ ещё не до конца закрыт, новый уже начинается.

До появления этого модуля каждый провайдер реализовывал эту логику
самостоятельно: OpenAI/Azure получают её готовой от бэкенда, xAI и Gemini/Bedrock
собирают её вручную — и делают это с разными, независимо всплывающими багами
(см. раздел "История и мотивация" ниже).

## Соответствие стандарту OpenAI Realtime

Мы **не изобретаем свой протокол** — модуль строго следует форме событий
актуального (GA) OpenAI Realtime API, как она описана в
`litellm/types/llms/openai.py` (`OpenAIRealtimeEvents` union,
`OpenAIRealtimeEventTypes` enum). Два важных нюанса:

1. **GA-именование, не Beta.** OpenAI Realtime существует в двух версиях
   именования событий:
   - Beta (легаси): `response.text.delta`, `response.audio.delta`,
     `response.text.done`, `response.audio.done`
   - GA (текущий): `response.output_text.delta`, `response.output_audio.delta`,
     `response.output_audio_transcript.delta`, `response.output_text.done`,
     `response.output_audio.done`, `response.output_audio_transcript.done`

   Наш модуль **всегда** генерирует GA-имена. Beta-имена в кодовой базе
   существуют только как совместимость для клиентов, которые явно шлют
   заголовок `OpenAI-Beta: realtime=v1` — это отдельный слой трансляции
   (`realtime_streaming.py`, GA↔Beta remap), не относящийся к этому модулю.

2. **`conversation.item.added`, а не `.created`.** OpenAI Beta API использовал
   `conversation.item.created`. GA API переименовал это в `.added`. Некоторые
   клиенты (в частности Pipecat 1.3.x) обрабатывают только `.added` и роняют
   соединение с ошибкой `"Unimplemented server event type"`, встретив
   `.created`. Модуль эмитит **только `.added`** — это сознательное решение,
   продиктованное реальной совместимостью с клиентами, а не отклонение от
   спецификации (GA-спецификация тоже предписывает `.added`).

Итог: клиент, написанный строго под официальный OpenAI Realtime GA API, будет
работать с любым провайдером за LiteLLM proxy без каких-либо провайдер-специфичных
доработок.

## Полная последовательность событий (к чему мы приходим)

Ниже — исчерпывающий список событий, которые эмитит каждая функция модуля, в
порядке их появления в потоке. Это тот набор, который в итоге увидит клиент,
независимо от того, Gemini это, Bedrock или OpenAI passthrough.

### 1. Начало ответа — `open_response`

```
→ response.created
```

Идемпотентно: повторный вызов с тем же `response_id` не эмитит ничего второй
раз (`state, ()`).

### 2. Начало элемента ответа — `open_item`

```
→ response.output_item.added      (output_index = следующий свободный, монотонно растёт)
→ conversation.item.added
```

`output_index` берётся из счётчика внутри текущего `OpenResponse` и **не может
повториться** для двух одновременно открытых элементов одного ответа — это
прямой фикс бага "все элементы получают `output_index=0`", который был у xAI,
Gemini и Bedrock независимо.

### 3. Начало содержимого элемента — `open_content_part`

```
→ response.content_part.added     (content_index = следующий свободный внутри item)
```

### 4. Потоковые куски содержимого — `append_content_delta`

```
→ response.output_text.delta                  (delta_type="text")
→ response.output_audio.delta                  (delta_type="audio")
```

Текст накапливается внутри состояния (`accumulated_text`); аудио-дельты не
накапливаются как текст (аудио остаётся сырыми байтами на стороне клиента).

### 5. Закрытие элемента — `close_item`

```
→ response.content_part.done       (для каждой открытой content part элемента)
→ response.output_item.done        (status: "completed" | "incomplete")
```

Идемпотентно: если `item_id` уже закрыт или никогда не открывался — `(state, ())`,
без исключений. Это единая точка "закрытия", которую переиспользует и обычный
путь, и путь barge-in.

### 6. Завершение ответа — `close_response`

```
→ [для каждого ещё открытого item: response.content_part.done + response.output_item.done, status="incomplete"]
→ response.done      (response.output = ВСЕ закрытые items, в порядке закрытия)
```

Это самое важное место с точки зрения корректности:

- **Любой не закрытый явно элемент закрывается автоматически** со статусом
  `incomplete` — при barge-in клиент никогда не получит "response.done" без
  соответствующего "output_item.done" для каждого item, который был объявлен
  через `.added`.
- **Повторный вызов на уже закрытом ответе — чистый no-op**
  (`state.response is None` → `(state, ())`). Это заменяет провайдер-специфичные
  флаги вроде `_turn_closed_by_interrupt`, которые мы писали вручную для Gemini:
  вместо "запомнить, что уже закрывали" модуль просто не может закрыть то, чего
  уже нет в состоянии.
- **`response.done.output` всегда полон** — содержит каждый item, который этот
  ответ когда-либо закрыл, независимо от того, закрылся ли он штатно
  (`completed`) или был синтетически закрыт при обрыве (`incomplete`). Это прямой
  фикс бага, который был в Bedrock (`output=[]` захардкожен) и был в Gemini до
  нашего ручного фикса.

### 7. Barge-in / отмена — `cancel_response`

```
→ input_audio_buffer.speech_started
→ [всё то же самое, что close_response]
```

Единая точка входа для "пользователь начал говорить, обрывай текущий ответ".

### 8. Tool calls — `tool_call_events`

```
→ response.created                              (если ответ ещё не открыт)
→ [на каждый вызов функции:]
    response.output_item.added                   (item_type="function_call", свой output_index)
    conversation.item.added
    response.function_call_arguments.done
    response.content_part.done + response.output_item.done   (status="completed")
→ response.done                                  (output = все вызовы функций)
```

Один вызов `tool_call_events` = ровно один `response.done`, закрывающий все
функции этого раунда — устраняет дублирование логики "response.done для
tool-call" и "response.done для обычного текста/аудио", которое раньше жило как
два почти одинаковых куска кода в каждом провайдере.

## Полная карта событий (справочная таблица)

| Событие | Кто эмитит | Обязательные поля |
|---|---|---|
| `response.created` | `open_response` | `response.id`, `response.status="in_progress"` |
| `response.output_item.added` | `open_item` | `response_id`, `output_index`, `item.id`, `item.status="in_progress"` |
| `conversation.item.added` | `open_item` | `item.id`, `item.status="in_progress"` |
| `response.content_part.added` | `open_content_part` | `response_id`, `item_id`, `output_index`, `content_index`, `part.type` |
| `response.output_text.delta` | `append_content_delta` (text) | `response_id`, `item_id`, `output_index`, `content_index`, `delta` |
| `response.output_audio.delta` | `append_content_delta` (audio) | то же, `delta` = base64 audio chunk |
| `response.content_part.done` | `close_item` | `response_id`, `item_id`, `output_index`, `content_index`, `part` (с накопленным текстом) |
| `response.output_item.done` | `close_item` / `close_response` | `response_id`, `output_index`, `item.status` (`completed`/`incomplete`) |
| `response.done` | `close_response` | `response.id`, `response.status="completed"`, `response.output` = все закрытые items |
| `input_audio_buffer.speech_started` | `cancel_response` | `item_id` (синтетический, маркер barge-in) |
| `response.function_call_arguments.done` | `tool_call_events` | `call_id`, `name`, `arguments`, `item_id`, `output_index` |

## Пример полного потока (один текстовый ответ)

```
response.created                     (response_id=resp_1)
response.output_item.added           (item_id=item_1, output_index=0)
conversation.item.added              (item_id=item_1)
response.content_part.added          (item_id=item_1, content_index=0, type=text)
response.output_text.delta           ("Привет")
response.output_text.delta           (", мир")
response.content_part.done           (item_id=item_1, content_index=0, text="Привет, мир")
response.output_item.done            (item_id=item_1, status=completed)
response.done                        (output=[{id:item_1, status:completed, content:[{type:text,text:"Привет, мир"}]}])
```

## Пример с barge-in (перебивание на середине фразы)

```
response.created                     (response_id=resp_1)
response.output_item.added           (item_id=item_1, output_index=0)
conversation.item.added
response.content_part.added          (item_id=item_1, content_index=0, type=audio)
response.output_audio.delta          (кусок звука 1)
response.output_audio.delta          (кусок звука 2)
   ⚡ пользователь начинает говорить
input_audio_buffer.speech_started
response.content_part.done           (синтезировано close_response, item_id=item_1)
response.output_item.done            (item_id=item_1, status=incomplete  ← не "completed"!)
response.done                        (output=[{id:item_1, status:incomplete, ...}])
```

Ключевой момент: `item_1` **обязательно** появляется в `response.done.output`,
даже несмотря на то, что фраза была прервана — просто со статусом `incomplete`.
Раньше (до этого модуля) в Bedrock и в Gemini-до-фикса этот item мог тихо
исчезнуть из `response.done`, и клиент не узнавал бы, что модель вообще начинала
что-то говорить.

## Что модуль намеренно НЕ делает

- **Не решает, когда открывать/закрывать элемент** — эта логика остаётся у
  вызывающего провайдера (Gemini решает по `generationComplete`/`interrupted`,
  Bedrock — по `contentStart`/`contentEnd`, и т.д.). Модуль — это гарантия
  *корректности* лексикона и порядка событий, а не автоопределение границ фраз.
- **Не занимается session-level конфигурацией** (`session.update`,
  `turn_detection`, voice-параметры) — это отдельный слой
  (`realtime_schema_normalization.py`).
- **Не занимается session resumption / reconnect** — отдельная подсистема
  (`RealtimeBackendConnector`).
- **Не навязывает единственный способ интеграции для tool calls.** Полная
  последовательность событий из раздела 8 (`tool_call_events`) используется
  целиком только у Gemini — она предполагает, что один tool call = один
  самостоятельный logical turn. Это не универсально: у Bedrock `toolUse`
  структурно приходит внутри уже открытого content-цикла, поэтому его
  tool-call путь осознанно переиспользует модуль только частично (индексация
  через `track_output_index`, без `open_item`/`close_item`/`close_response`).
  Это архитектурное решение под конкретный протокол провайдера, не
  недоделанная миграция — см. ниже "Кто на что реально мигрирован".

## Кто на что реально мигрирован

Все три провайдера (кроме OpenAI/Azure, которым это не нужно — см. ниже) уже
подключены к модулю, но с разной глубиной интеграции:

- **xAI** (`litellm/llms/xai/realtime/transformation.py`,
  `XAIRealtimeNormalizer`) — использует только query-функции
  `track_output_index`/`track_content_index` для инъекции пропущенных индексов
  в уже готовые события бэкенда (xAI structurally OpenAI-compatible, событий
  сам не строит). Нормализатор владеет своим `self._correlation_state` сам
  (не получает его параметром снаружи). Отдельного tool-call пути нет — xAI не
  выделяет tool calls в отдельный поток обработки, они идут через тот же
  passthrough + normalize.
- **Gemini/Vertex** (`litellm/llms/gemini/realtime/transformation.py`,
  `GeminiRealtimeConfig`) — самая полная интеграция. Обычный text/audio-путь
  использует `track_output_index`/`track_content_index`; tool-call путь
  использует полноценную `tool_call_events()` (раздел 8), которая сама строит
  `response.created`/`response.output_item.added`/
  `response.function_call_arguments.done`/`response.output_item.done`/
  `response.done` — Gemini лишь патчит обратно `call_id`/`name`/`arguments` на
  уже построенные события (эти поля вне контракта generic item-shape модуля).
- **Bedrock** (`litellm/llms/bedrock/realtime/transformation.py`,
  `BedrockRealtimeConfig`) — обычный text/audio-путь использует
  `track_output_index`/`track_content_index` так же, как Gemini/xAI.
  **Tool-call путь — сознательно НЕ через `tool_call_events()`, но полностью
  учтён.** Структурная причина: у Bedrock `toolUse` приходит внутри уже
  открытого `contentStart`/`contentEnd` content-цикла (тот же
  `output_item_id`, что у text/audio), а `response.done` строится позже,
  отдельным `contentEnd`/`END_TURN`-событием — `tool_call_events()` не
  подходит напрямую, потому что она сама закрывает весь response
  (`close_response()`) за один вызов, а Bedrock response должен остаться
  открытым до END_TURN. Более того, `open_item`/`close_item` тоже не
  используются для регистрации самого tool-call item: `open_item` безусловно
  добавляет новый `OpenItem` даже при повторном `item_id` (не idempotent, в
  отличие от `track_output_index`), а `close_item` сохраняет `item_type`
  уже открытого item — если тот же `output_item_id` был открыт как
  `"message"` предшествующим text/audio `contentStart`, `close_item` дал бы
  `type: "message"` в `response.done.output` для tool-call turn вместо
  `"function_call"`. Поэтому `transform_tool_use_event` берёт `output_index`
  через `track_output_index` (не трогая `open_items`/`closed_items` в
  `self._correlation_state` вообще для tool calls) и вручную добавляет
  корректно типизированный `function_call` item в `current_item_chunks` —
  тот же паттерн, которым `transform_content_end_event` уже накапливает
  text/audio items. Итог для клиента: `response.done.output` полон и
  единообразен (правильный `type` на каждом item, независимо от того,
  tool call это или текст/аудио), даже если внутренний механизм получения
  этого результата не идентичен Gemini дословно.
- **OpenAI/Azure** — не используют модуль и не должны. У них нет
  transform-слоя вообще (raw passthrough): backend уже присылает корректные
  `output_index`/`content_index`/`response.done.output`, пересобирать их
  заново не нужно и негде — класс бага, который чинил этот модуль, там
  структурно не может существовать.

## История и мотивация (кратко)

Модуль вырос из ручной отладки двух реальных багов в Gemini:

1. `response.done` строился с `output_items=None` вместо накопленных закрытых
   items — итоговое событие теряло уже подтверждённую фразу.
2. При barge-in элемент, открытый через `outputTranscription` до прихода первой
   аудио-дельты, оставался никогда не закрытым, если пользователь перебивал
   модель в этом узком окне — `response.done` не содержал этот item вообще.

Оба бага были исправлены точечно в Gemini и подтверждены живыми звонками через
прокси. При обзоре всех провайдеров выяснилось, что тот же класс багов
присутствует (не всегда исправлен) у xAI и Bedrock независимо — что и стало
поводом вынести один общий, протестированный алгоритм вместо N частных
реализаций.
