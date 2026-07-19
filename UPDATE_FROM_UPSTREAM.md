# Обновление из официального репозитория LiteLLM

## Git-структура форка

**Remotes:**
- `origin` = git@git.aitomaton.online:aitomaton/LiteLLM_AiTomaton_fork.git (корпоративный)
- `origin` (push) = git@github.com:pufyan/litellm.git (GitHub зеркало)
- `upstream` = https://github.com/BerriAI/litellm.git (официальный репозиторий)

**Ветки:**
- `litellm_pufyan` — личная рабочая ветка (здесь разработка)
- `litellm_internal_staging` — зеркало upstream (коммиты «official_litellm_update»)
- `master` — резервная ветка

## Процесс обновления

```bash
cd /home/puf/Work/ForkLiteLLM/litellm

# 1. Получи свежие изменения с upstream
git fetch upstream

# 2. Обнови staging ветку
git checkout litellm_internal_staging
git merge upstream/main

# 3. Вернись на рабочую ветку и мержи изменения
git checkout litellm_pufyan
git merge litellm_internal_staging
```

## Если есть конфликты

```bash
# Разреши конфликты в редакторе, потом:
git add .
git commit -m "Merge upstream updates into litellm_pufyan"
```

## Проверка перед мержем (опционально)

```bash
# Посмотри какие изменения придут из upstream
git fetch upstream
git log litellm_pufyan..upstream/main --oneline
```

## Примечания

- Proxy запускается в Docker контейнере `litellm-litellm-1` на localhost:4000
- БД `llmproxy` в контейнере `litellm-db-1`
- Модели добавляются через Admin API `/model/new` (store_model_in_db: True)
