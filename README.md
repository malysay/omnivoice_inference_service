# OmniVoice Triton Service

Triton-first сервис для запуска `OmniVoice` через Triton Python backend.

Основная идея:
- Triton является основным runtime для inference
- код в `app/` используется как shared inference layer
- веса модели живут отдельно от Docker image и монтируются в контейнер как volume

## Что В Репозитории

- `docker-compose.yml` — локальный запуск Triton
- `Dockerfile` — образ Triton + Python runtime
- `triton_model_repo/omnivoice/` — Triton model repository
- `app/service.py` — загрузка OmniVoice и синтез
- `app/triton_client.py` — HTTP-клиент для тестового inference
- `scripts/download_model.sh` — скачивание модели из Hugging Face
- `scripts/run_triton.sh` — запуск Triton
- `scripts/check_triton.sh` — проверка live/ready/model endpoints

## Важное Ограничение

Triton ориентирован на Linux deployment.

Если вы работаете на macOS:
- контейнер можно использовать для smoke-check
- `mps` внутри Triton-контейнера недоступен
- локальный запуск на Mac не равен production GPU-среде

Если целевой прод у вас на Linux + NVIDIA GPU, текущая структура подходит хорошо.

## Быстрый Старт

### 1. Подготовьте `.env`

```bash
cp .env.example .env
```

Минимально нужно заполнить:

```dotenv
OMNIVOICE_MODEL_DIR=/absolute/path/to/local/OmniVoice-runtime
OMNIVOICE_DEVICE=auto
OMNIVOICE_PRELOAD_MODEL=true
OMNIVOICE_DEFAULT_LANGUAGE=sah
OMNIVOICE_MAX_CONCURRENCY=1
```

Важно:
- `OMNIVOICE_MODEL_DIR` указывается как путь на хост-машине
- `docker-compose.yml` монтирует эту папку в контейнер как `/models/omnivoice`
- сами веса не запекаются в Docker image

### 2. Скачайте модель

Рекомендуемый путь для новых разработчиков: сначала скачать весь Hugging Face snapshot, а уже потом при желании оптимизировать скачивание.

Команда по умолчанию:

```bash
./scripts/download_model.sh
```

Что делает скрипт:
- устанавливает `.[download]`, если это ещё не сделано
- скачивает модель `k2-fsa/OmniVoice`
- кладёт её в `./models/omnivoice-runtime`
- по умолчанию использует `full` mode, то есть скачивает весь snapshot

После скачивания пропишите абсолютный путь в `.env`:

```dotenv
OMNIVOICE_MODEL_DIR=/absolute/path/to/omnivoice_inference_service/models/omnivoice-runtime
```

### 3. Поднимите Triton

```bash
./scripts/run_triton.sh
```

или:

```bash
docker compose up --build
```

Будут открыты порты:
- `8000` — HTTP
- `8001` — gRPC
- `8002` — metrics

### 4. Проверьте Health

В отдельном терминале:

```bash
./scripts/check_triton.sh
```

Проверяются:
- `/v2/health/live`
- `/v2/health/ready`
- `/v2/models/omnivoice/ready`
- `/v2/models/omnivoice`

### 5. Сделайте Первый Inference

```bash
python -m app.triton_client \
  --url http://127.0.0.1:8000 \
  --model omnivoice \
  --text "Мин аатым Айаал. Бу Triton көмөтүнэн тургутуллар тест." \
  --language sah \
  --num-step 32 \
  --output ./outputs/triton-test.wav
```

Если всё прошло успешно:
- появится файл `./outputs/triton-test.wav`
- клиент выведет JSON с `output`, `sample_rate`, `elapsed_ms`

## Скачивание Модели

### Вариант 1. Готовый Скрипт

Публичный репозиторий:

```bash
./scripts/download_model.sh
```

Другой репозиторий:

```bash
HF_REPO_ID=owner/other-model ./scripts/download_model.sh
```

Другая ревизия:

```bash
HF_REPO_ID=k2-fsa/OmniVoice HF_REVISION=main ./scripts/download_model.sh
```

Закрытый или gated репозиторий:

```bash
HF_REPO_ID=owner/private-model HF_TOKEN=hf_xxx ./scripts/download_model.sh
```

Скачать не весь snapshot, а только минимальный runtime subset:

```bash
DOWNLOAD_MODE=minimal ./scripts/download_model.sh
```

Переопределить папку назначения:

```bash
MODEL_DIR=/srv/models/omnivoice-runtime ./scripts/download_model.sh
```

Поддерживаемые переменные:
- `HF_REPO_ID` — Hugging Face repo id, по умолчанию `k2-fsa/OmniVoice`
- `HF_REVISION` — branch, tag или commit SHA, по умолчанию `main`
- `HF_TOKEN` — токен для gated/private модели
- `MODEL_DIR` — папка для скачивания модели
- `DOWNLOAD_MODE` — `full` или `minimal`

### Вариант 2. Python CLI

Полный snapshot:

```bash
python -m pip install '.[download]'

python -m app.download_model \
  --repo-id k2-fsa/OmniVoice \
  --revision main \
  --output-dir ./models/omnivoice-runtime \
  --full-snapshot
```

Минимальный runtime snapshot:

```bash
python -m app.download_model \
  --repo-id k2-fsa/OmniVoice \
  --revision main \
  --output-dir ./models/omnivoice-runtime
```

Закрытый или gated репозиторий:

```bash
python -m app.download_model \
  --repo-id owner/private-model \
  --revision main \
  --output-dir ./models/omnivoice-runtime \
  --full-snapshot \
  --token hf_xxx
```

Для первого запуска лучше использовать `--full-snapshot`: это надёжнее и снижает риск missing-file ошибок.

## Что Нужна Модели Для Работы

Сервис ожидает, что `OMNIVOICE_MODEL_DIR` указывает на корректную папку модели.

Минимально должны присутствовать:
- `config.json`
- один или несколько `*.safetensors`
- файлы текстового tokenizer
- папка `audio_tokenizer/`

Если используете `full` mode, об этом можно не думать: скрипт скачает весь репозиторий модели.

## Диагностика

Если сервис не отвечает:

```bash
docker compose ps
docker compose logs -f triton
```

Если `model ready` не проходит, самые частые причины:
- неправильный `OMNIVOICE_MODEL_DIR`
- скачалась неполная папка модели
- для private/gated repo не был передан `HF_TOKEN`
- контейнер не смог загрузить tokenizer или зависимости
- на локальном Mac есть ограничения по сравнению с Linux GPU runtime

Health-check вручную:

```bash
curl -fsS http://127.0.0.1:8000/v2/health/live
curl -fsS http://127.0.0.1:8000/v2/health/ready
curl -fsS http://127.0.0.1:8000/v2/models/omnivoice/ready
```

Проверка содержимого модели:

```bash
find "$OMNIVOICE_MODEL_DIR" -maxdepth 2 | sort | sed -n '1,120p'
```

## Production Подход

Для production удобнее держать отдельно:
- Docker image с Triton и кодом сервиса
- директорию с весами модели на сервере

Обычно flow такой:
1. Скачать модель на машине с доступом к Hugging Face
2. Передать папку на сервер через `rsync`, артефакт или object storage
3. На сервере выставить `OMNIVOICE_MODEL_DIR=/srv/omnivoice/weights/omnivoice-runtime`
4. Выполнить `docker compose up -d --build`

Это удобнее, чем запекать веса внутрь образа: rebuild образа остаётся быстрым, а модель можно обновлять отдельно от кода.
