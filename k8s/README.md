# Деплой LiteLLM в кластер aitomaton

Прокси разворачивается в namespace `production` кластера k3s (`aitomaton-prod`, 88.218.71.116)
одним монолитным подом: data plane, management API и Admin UI живут в одном процессе на порту 4000.

## Состав

| Файл | Что делает |
|------|------------|
| [deployment.yaml](deployment.yaml) | Deployment `litellm`, 1 реплика, образ из Gitea registry |
| [service.yaml](service.yaml) | ClusterIP с закреплённым адресом 10.43.100.100 |
| [proxy-config.yaml](proxy-config.yaml) | Статическая конфигурация прокси → ConfigMap `litellm-config` |
| [../.gitea/workflows/deploy.yml](../.gitea/workflows/deploy.yml) | Сборка образа и выкатка |

Имя `proxy-config.yaml` вместо `config.yaml` — вынужденное: апстримный `.gitignore` глушит любой
`config.yaml`, чтобы в историю не попадали локальные конфиги с ключами. В ConfigMap файл всё равно
кладётся под именем `config.yaml`.

## Доступ

| Что | Как |
|-----|-----|
| API из кластера | `http://litellm.production.svc.cluster.local:4000` |
| Realtime из кластера | `ws://litellm.production.svc.cluster.local:4000/v1/realtime?model=<модель>` |
| Admin UI | `http://10.43.100.100:4000/ui/` — только из WireGuard VPN (internal-клиент) |
| Метрики | `http://litellm.production.svc.cluster.local:4000/metrics/` (слэш обязателен) |

Ingress нет и не планируется: снаружи кластера прокси недоступен. Логин в UI — `admin` и пароль из
секрета `LITELLM_UI_PASSWORD`, либо master key.

## Где что хранится

Модели, виртуальные ключи, команды, бюджеты и лимиты живут в PostgreSQL (`litellm_db`) и заводятся
через Admin UI — в репозитории их нет. Всё остальное поведение прокси описано в
[proxy-config.yaml](proxy-config.yaml) и применяется деплоем.

Ключи провайдеров в БД зашифрованы `LITELLM_SALT_KEY`. **Менять этот секрет нельзя**: после ротации
модели перестают расшифровываться и исчезают из роутера. Master key ротировать безопасно.

Схему БД создаёт сам под при старте (`prisma migrate deploy`); CLI и движки присма зашиты в образ
под `/opt/prisma`, интернет для этого не нужен, отдельная миграционная Job не требуется.

## Секреты и переменные

Секреты Gitea (организация `aitomaton`): `LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`,
`LITELLM_DATABASE_URL`, `LITELLM_UI_PASSWORD`, `LITELLM_DRAIN_TOKEN`, а также существующий
`GEMINI_LIVE_API_KEY` — тот же ключ Google AI Studio, которым pjcpp сейчас ходит в Gemini Live напрямую.

Переопределяемые переменные (`vars`, у всех есть значения по умолчанию): `LITELLM_UI_USERNAME`,
`LITELLM_LOG_LEVEL`, `LITELLM_REDIS_HOST`, `LITELLM_REDIS_PORT`, `LITELLM_WS_DRAIN_TIMEOUT`,
`LITELLM_GRACEFUL_SHUTDOWN_TIMEOUT`, `LITELLM_STORE_MODEL_IN_DB`.

В `litellm-secrets` не должно появиться `GOOGLE_API_KEY`: он имеет приоритет над `GEMINI_API_KEY`
и молча перебьёт ключ на всех путях, включая realtime.

## Выкатка

Push в `master` собирает образ и катит его; ручной запуск — через workflow_dispatch. Деплой ставит
иммутабельный тег по SHA коммита, поэтому откат делается через `kubectl rollout undo`.

Сборка тяжёлая (npm build админки + `uv sync` с расширениями), первый прогон занимает около 15 минут;
дальше выручает слоевой кэш на runner-хосте. Нужен BuildKit — в Dockerfile используются cache-mounts;
в workflow он включается переменной `DOCKER_BUILDKIT=1`.

## Выкатка и активные звонки

Через прокси идут realtime-сессии длиной в телефонный разговор, поэтому завершение пода растянуто:

1. `preStop` дёргает `/health/drain`; проба готовности сразу отдаёт 503, и под уходит из Service —
   новые соединения идут на новый под.
2. По SIGTERM активные realtime-сессии доживают до `WS_DRAIN_TIMEOUT` (420 с), остаток закрывается
   кодом 1012 Service Restart, после которого клиент переподключается уже на живой под.
3. `terminationGracePeriodSeconds: 480` даёт этому окну отработать; если сделать его меньше
   `WS_DRAIN_TIMEOUT`, SIGKILL прилетит посреди дренажа.

Счётчик in-flight в LiteLLM считает только HTTP-запросы, WebSocket-сессии в него не попадают, поэтому
дренаж звонков обеспечивает именно эта связка таймаутов, а не `/health/drain` сам по себе.

## Нюансы, на которые легко наступить

- `/metrics` монтируется, только если `prometheus` указан в `litellm_settings.callbacks` **в YAML**.
  Включение того же коллбэка через Admin UI эндпоинт не создаёт — метрики будут считаться в память.
- В этом форке `/metrics` по умолчанию требует авторизации; мы снимаем её через
  `require_auth_for_metrics_endpoint: false`, что безопасно только пока нет Ingress.
- `/health/readiness` отдаёт 200 даже при недоступной БД, так что проба готовности не поймает аварию
  PostgreSQL. При этом `allow_requests_on_db_unavailable: false` — запросы в такой ситуации отклоняются.
- Не задавать `general_settings.enforced_params` — без коммерческой лицензии это роняет старт пода.
  Ключи с полями `tags` и `model_max_budget` тоже требуют лицензии.
- `--num_workers` держать равным 1: при нескольких воркерах preStop-дренаж дойдёт лишь до одного из
  них, а prometheus потребует `PROMETHEUS_MULTIPROC_DIR`.
- Раздел `environment_variables` в Admin UI не использовать: он подменяет переменные в работающем
  поде, и конфигурация из Secret перестаёт быть источником правды.

## Обновление из upstream

Порядок мержа описан в [../UPDATE_FROM_UPSTREAM.md](../UPDATE_FROM_UPSTREAM.md). После мержа стоит
проверить, что в апстриме не изменились дефолты, на которые опирается наш конфиг — прежде всего
`require_auth_for_metrics_endpoint` и набор ключей `general_settings`.
