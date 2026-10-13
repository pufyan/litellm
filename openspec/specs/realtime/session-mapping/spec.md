# Session Mapping Specification

## Purpose

Мостинг аудио-конфигурации realtime-сессии между канонической формой клиента и провайдером: без потерь, без невалидных типов и без рассинхрона частоты потока с часами плеера.

## Requirements

### Requirement: Форс upstream-форматов xAI на pcm 16k
Исходящий `session.update` в xAI SHALL нести `audio.input.format` и `audio.output.format` как `{"type": "audio/pcm", "rate": 16000}` независимо от запрошенного клиентом. Реализуется задачей 1.1.

#### Scenario: Клиент просил 24k на выход
- **WHEN** `session.update` несет `audio.output.format: {"type": "audio/pcm", "rate": 24000}`
- **THEN** в xAI уходит `audio.output.format: {"type": "audio/pcm", "rate": 16000}`

#### Scenario: Клиент просил не-pcm кодек
- **WHEN** `session.update` несет `audio.output.format: {"type": "audio/pcmu", "rate": 8000}`
- **THEN** в xAI уходит `audio.output.format: {"type": "audio/pcm", "rate": 16000}`

#### Scenario: Клиент просил 24k на вход
- **WHEN** `session.update` несет `audio.input.format: {"type": "audio/pcm", "rate": 24000}`
- **THEN** в xAI уходит `audio.input.format: {"type": "audio/pcm", "rate": 16000}`

#### Scenario: Соседние ключи сохраняются
- **WHEN** `audio.input` несет `turn_detection`/`transcription`, `audio.output` — `voice`/`speed`
- **THEN** после форса форматов эти ключи на месте, меняется только `format`

### Requirement: Форс downstream-сетапа xAI на pcm 16k
`session.created` / `session.updated` от xAI SHALL доходить до клиента с `session.audio.input.format` и `session.audio.output.format` как `{"type": "audio/pcm", "rate": 16000}`. Реализуется задачей 1.2.

#### Scenario: Бэкенд прислал 24k в session.created
- **WHEN** xAI прислал `session.created` с `audio.output.format.rate = 24000`
- **THEN** клиент получает `session.created` с `audio.output.format = {"type": "audio/pcm", "rate": 16000}` (и так же для input)

#### Scenario: Не-сессионные события не трогаем
- **WHEN** событие не `session.created`/`session.updated`
- **THEN** аудио-пасс `normalize` его не меняет (индексы/usage/роли как раньше)

### Requirement: Канонический G.711-маппинг
Плоские `g711_ulaw` / `g711_alaw` преобразуются в `audio/pcmu` / `audio/pcma` с rate 8000. Реализуется задачей 1.1.

#### Scenario: G.711 ulaw плоским полем
- **WHEN** `session.update` несет `output_audio_format: "g711_ulaw"` (или input)
- **THEN** провайдер получает `format: {"type": "audio/pcmu", "rate": 8000}`

#### Scenario: G.711 alaw плоским полем
- **WHEN** `session.update` несет `output_audio_format: "g711_alaw"` (или input)
- **THEN** провайдер получает `format: {"type": "audio/pcma", "rate": 8000}`

#### Scenario: pcm16 не меняется
- **WHEN** `session.update` несет `output_audio_format: "pcm16"`
- **THEN** провайдер получает `format: {"type": "audio/pcm", "rate": 24000}` как раньше

### Requirement: Сохранение вложенного audio без плоских замен
`session.update` со вложенным `audio` и без плоских аудио-полей доходит до провайдера с тем же блоком (после существующей нормализации вложенных структур). Реализуется задачами 1.1, 1.2.

#### Scenario: GA-клиент с вложенным audio 8кГц
- **WHEN** апдейт несет только `audio.input/output` без плоских полей
- **THEN** провайдер получает форматы и голос клиента, а не пустой апдейт

#### Scenario: Пустой или не-словарный вложенный блок
- **WHEN** вложенный `audio` пуст (`{}`) или не словарь
- **THEN** блок дропается как раньше — пустой блок невалиден для GA, инвариант `test_no_empty_audio_block_when_every_field_was_dropped` сохраняется

#### Scenario: Beta-клиент с плоскими полями
- **WHEN** апдейт несет плоские поля (возможно вместе со вложенным блоком)
- **THEN** выход бит-в-бит как раньше: вложенное дропается, GA собирается из плоских

### Requirement: Алиасы дропаются как раньше
`output_modalities` / `max_output_tokens` без плоских аналогов по-прежнему дропаются (тесты `test_..._drops_lone_ga_aliases` и `test_..._canonical_name_wins_over_ga_alias` не меняются). Вне скоупа бага. Реализуется задачей 1.2 (сьют зеленым без правок этих тестов).

#### Scenario: Клиент с готовым GA-блоком
- **WHEN** апдейт уже в целевой GA-форме
- **THEN** мост его не разбирает на запчасти; алиасы без плоских аналогов дропаются как раньше
