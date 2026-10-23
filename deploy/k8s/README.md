# LiteLLM в Kubernetes (cluster01)

```
deploy/k8s/
├── base/          # Deployment + Service, PDB, конфиг прокси (proxy-config*.yaml)
└── overlays/
    ├── stage/     # namespace aitomaton-stage, 1 копия
    └── prod/      # namespace aitomaton-prod, 2 копии
```

Старый прод (namespace `production`) — по-прежнему `k8s/` и
`.gitea/workflows/deploy.yml`; правки `deploy/k8s/**` его не запускают.

## Выкладка

`.gitea/workflows/deploy-k8s.yml`:

| Событие | Окружение |
|---|---|
| push в `master` | stage (и старый прод — `deploy.yml`) |
| ручной запуск (Actions → Deploy LiteLLM (cluster01) → Run workflow) | выбранная ветка → stage или prod |
| prod не из `master` | запуск падает первым шагом |

Образ `git.aitomaton.online/aitomaton/litellm:<полный хеш коммита>` — тот же
тег, что у `deploy.yml`: собранный им образ не пересобирается. Job миграции
нет — схему ставит шлюз при старте.

Откат: `kubectl -n <ns> rollout undo deployment/litellm` или ручной запуск
на нужном коммите.

## Секреты

Gitea, секреты репозитория: `KUBECONFIG_STAGE`, `KUBECONFIG_PROD`
(`ci-deployer`), `DOCKER_USERNAME`, `REGISTRY_TOKEN`, `REPO_TOKEN`.

Кластер, Secret `litellm-keys` в каждом namespace — руками, один раз, CI
не трогает:

```bash
NS=aitomaton-stage
kubectl -n $NS create secret generic litellm-keys \
  --from-literal=LITELLM_MASTER_KEY="sk-$(openssl rand -hex 24)" \
  --from-literal=LITELLM_SALT_KEY="$(openssl rand -hex 32)" \
  --from-literal=UI_PASSWORD="$(openssl rand -hex 16)" \
  --from-literal=OPENROUTER_API_KEY='<ключ OpenRouter>'
```

Prod: `LITELLM_SALT_KEY` — **с текущего прода, точно** (им зашифрованы ключи
провайдеров в базе; другой — портит их), `LITELLM_MASTER_KEY` — проще тоже
с прода.

## Админка и ключи

```bash
kubectl -n aitomaton-stage port-forward svc/litellm 4000:4000
# http://localhost:4000/ui  (admin / UI_PASSWORD из litellm-keys)
```

Там выдаются виртуальные ключи для crm (`Crm__Ai__Key`,
`Crm__Ops__LiteLlmKey` — роль `proxy_admin_viewer`, `allowed_routes`
`["/spend/logs/v2"]`) и rag.
