# Tasks

## 1. Форс pcm 16k в xAI-нормалайзере

- [x] 1.1 Форс upstream: в `litellm/llms/xai/realtime/transformation.py` добавить константы типа/рейта (`audio/pcm`, `16000`) и применение в `patch_outgoing_session` БЕЗУСЛОВНО (в т.ч. при `canonical_session=None` на beta-пути, вне `_apply_xai_only_fields`) и ДО существующего `_clamp_xai_audio_ranges` (снап на 16000 — no-op, ложного варнинга `falling back to 24000` нет) — `audio.input.format` и `audio.output.format` выставляются в форс-значение с сохранением соседних ключей (создавать блоки при отсутствии, строковый `format` заменять словарем); логировать варнинг при переписывании запрошенного значения; verify: `python3 -m pytest tests/test_litellm/llms/xai/realtime/test_xai_realtime_transformation.py -q -p no:cacheprovider`
- [x] 1.2 Форс downstream: в `normalize` добавить пасс для `session.created`/`session.updated` — `session.audio.input/output.format` выставляются в то же форс-значение, остальные события untouched; verify: та же команда, что в 1.1
- [x] 1.3 Обновить `TestXaiSampleRateValidation` под новое поведение (passthrough-ожидания и fallback-to-24000 заменяются форс-ожиданиями `== 16000` вверх и вниз, плюс соседние-ключи-целые и не-сессионные-события-untouched); verify: `python3 openspec/check_change.py xai-force-pcm-16k`
