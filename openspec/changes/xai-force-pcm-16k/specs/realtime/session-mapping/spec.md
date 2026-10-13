# Spec Delta

## Purpose

Аудио-форматы xAI-сессии фиксированы на PCM 16кГц в обе стороны моста, чтобы частота потока совпадала с часами плеера клиента.

## ADDED Requirements

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
