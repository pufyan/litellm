## Purpose

Единое машиночитаемое описание возможностей связки провайдер×модель и реестр провайдеров вместо if/elif-диспетчера. Все решения моста («как завершать ход», «как продолжать сессию», «чем биллить») выводятся из дескриптора, а не из ветвлений по имени модели.

## ADDED Requirements

### Requirement: Схема дескриптора
Дескриптор описывает: модель хода (`single-response` | `utterance-plus-status` | `no-response-boundary`); режим тулзов (`blocking` | `non-blocking-async`); модель setup (`once` | `updates-allowed` | `restricted`); механизмы продолжения по приоритету; единицу биллинга (`tokens` | `voice-seconds-plus-backend-tokens`); требует ли reconnect обновления токена.

#### Scenario: Новая модель без правок ядра
- **WHEN** добавляется модель с уже покрытой комбинацией возможностей
- **THEN** достаточно нового дескриптора и адаптера трансляции событий, ядро моста не меняется

### Requirement: Реестр вместо диспетчера
Выбор обработчика, URL и заголовков бэкенда выполняется по реестру провайдеров. Ветки if/elif по имени провайдера в точке входа не допускаются.

#### Scenario: Добавление провайдера
- **WHEN** регистрируется новый провайдер
- **THEN** точка входа находит его в реестре; существующие провайдеры не затрагиваются

### Requirement: Дескрипторы Gen2-моделей из коробки
Дескрипторы покрывают OpenAI GPT-Live (делегация `client`/`responses`, `session.update` только для настроек делегации, immutable остального) и Gemini 3.8 Live / Extended Thinking (завершение хода по `interaction_status`, только `NON_BLOCKING`-тулзы, `thinking_level` low/medium/high).

#### Scenario: Запрещённая операция Gen2
- **WHEN** клиент просит сменить голос или режим делегации посреди GPT-Live-сессии
- **THEN** мост отвечает ошибкой с требованием новой сессии, а не пробрасывает заведомо отклоняемую команду

### Requirement: Привязка известных моделей
Дескрипторы из коробки фиксируют: `gpt-realtime-*`, `gemini-live-2.5-*`, xAI и аналоги — `single-response`; `gemini-3.8-live` — `single-response` (`turnComplete` закрывает ход); `gemini-3.8-live-extended-thinking` — `utterance-plus-status` (завершение только по `interaction_status == IDLE`); `gpt-live-1` — `no-response-boundary`.

#### Scenario: Проверка привязки
- **WHEN** резолвится дескриптор `gemini-3.8-live-extended-thinking`
- **THEN** модель хода — `utterance-plus-status`, тулзы — только `non-blocking-async`
