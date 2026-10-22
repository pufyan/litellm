## 1. Построение записей

- [ ] 1.1 Добавить `litellm/litellm_core_utils/openrouter_catalog.py`: frozen pydantic-модели ответа и чистая функция «список моделей из API -> записи каталога» по отображению из design.md (цены, лимиты, режим, возможности, пропуск моделей без валидной цены, пропуск повреждённых записей). Проверка: юнит-тесты `tests/unit/litellm_core_utils/test_openrouter_catalog.py` на цены, лимиты, режим `image_generation`, возможности, цену `-1`, отсутствие `max_completion_tokens`, повреждённую запись; тест на реальном срезе ответа API, закреплённом как фикстура.

## 2. Релоад каталога

- [ ] 2.1 Добавить `litellm.openrouter_models_url` (`LITELLM_OPENROUTER_MODELS_URL`, пустое значение отключает), константу `OPENROUTER_CATALOG_MIN_MODEL_COUNT` (50) и параметры `openrouter_url` и `openrouter_client` в `refetch_model_cost_map`: один запрос с таймаутом, проверка числа записей (минимум и половина от вшитых), замена всех записей `openrouter` при успехе, при любой неудаче поведение `_restore_providers_dropped_upstream` без изменений. В `tests/unit/conftest.py` задать пустой адрес рядом с `LITELLM_LOCAL_MODEL_COST_MAP`. Проверка: юнит-тесты на успех, недоступный источник, неверный JSON, урезанный ответ, пустой адрес (запроса нет), и что ключи `openrouter`, которых нет в ответе, исчезли.
- [ ] 2.2 Включить отпечаток списка в `revision` (`<основная ревизия>+openrouter:<12 hex>`) при применённом наложении. Проверка: юнит-тесты: тот же ответ даёт ту же ревизию, изменение цены или состава даёт другую; дефолтный путь `_check_and_reload_model_cost_map` публикует ревизию при изменении только списка OpenRouter (мок `refetch`).

## 3. Проверка режима в форме «Add Model»

- [ ] 3.1 Добавить в `ui/litellm-dashboard/src/components/add_model/model_mode_filter.ts` чистые функции `filterModelsByMode(models, modelMap, mode)` и `findModeMismatches(selectedModels, modelMap, mode)` по правилам спеки (совместимость `chat` и `responses`, режимы без сужения, модель без режима подходит, пустой результат даёт полный список, имена не из каталога и значения `custom`/`all-wildcard` игнорируются). Проверка: `model_mode_filter.test.ts` в `npm run test:unit`: озвучка сужает до своих и безрежимных, `chat` принимает `responses`, режим `realtime` не сужает, пустой результат возвращает всё, вручную введённое имя не даёт несовпадения.
- [ ] 3.2 Подключить в `AddModelForm.tsx`: прокинуть `modelCostMap` из `AddModelPanel`, смотреть значение поля `mode` через `useWatch`, передавать в `LiteLLMModelNameField` отфильтрованный список и показывать предупреждение под полем режима (выбранные модели из поля `model`), не блокируя сохранение. Проверка: `AddModelForm.integration.test.tsx`: выбор `audio_speech` убирает chat-модель из списка; выбранная chat-модель при смене режима на `audio_speech` показывает предупреждение, кнопка сохранения активна; без выбранного режима список полный.

## 4. Проверки

- [ ] 4.1 Прогнать питоновские `tests/unit/litellm_core_utils/test_get_model_cost_map.py`, `tests/unit/litellm_core_utils/test_openrouter_catalog.py`, `tests/unit/proxy/test_proxy_server_endpoints_and_startup.py`, `tests/unit/proxy/proxy_server/test_routes_model_cost_map.py`. Проверка: зелёные, кроме двух падающих до изменения `test_token_counter_loads_a_custom_tokenizer*`.
- [ ] 4.2 Прогнать `npm run test:unit` и `npm run test:integration` для файлов `model_mode_filter.test.ts` и `AddModelForm.integration.test.tsx` в `ui/litellm-dashboard`. Проверка: зелёные.
- [ ] 4.3 `openspec validate openrouter-live-catalog --strict` и `python3 openspec/check_change.py openrouter-live-catalog`. Проверка: без ошибок.
- [ ] 4.4 Живая проверка на реальном API: релоад через `refetch_model_cost_map` против `https://openrouter.ai/api/v1/models`, число записей `openrouter/*` и пара цен сверены с ответом. Проверка: вывод команды приложен к ченджу.
- [ ] 4.5 После выкатки (выполняет владелец): `GET /public/litellm_model_cost_map` содержит записи `openrouter/*`, выпадашка «Add Model» для OpenRouter не пуста, `source_revision` совпадает у всех подов, при режиме `audio_speech` в списке нет chat-моделей. Проверка: ответы эндпоинтов.
