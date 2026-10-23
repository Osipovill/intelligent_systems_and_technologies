# Практическая работа 3

## Запуск локальной среды

Docker socket даёт Jupyter права управления Docker-хостом и по уровню риска близок к root-доступу.

Из корня репозитория задайте токен и подготовьте сеть и Linux volume моделей:

```sh
export JUPYTER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DOCKER_GID="$(getent group docker | cut -d: -f3)"
docker network create practice3-private 2>/dev/null || true
docker volume create practice3-ollama-models
docker compose -f practice_3/compose.yaml build
docker compose -f practice_3/compose.yaml up -d
```

Откройте `http://127.0.0.1:8890` и введите `JUPYTER_TOKEN`. 

