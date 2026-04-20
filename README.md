# OmniVoice Inference Service

Отдельный production-oriented проект для локального инференса `OmniVoice`, не меняющий исходный репозиторий модели.

Сервис рассчитан на такой сценарий:
- у вас уже есть локально скачанный checkpoint OmniVoice;
- при необходимости у вас есть локальный исходный репозиторий `OmniVoice`;
- вы поднимаете отдельный HTTP API и/или CLI для синтеза речи;
- дефолтный язык сервиса настроен на якутский (`sah`, `Yakut`).

## Что умеет

- HTTP API на `FastAPI`
- `GET /healthz` и `GET /readyz`
- `POST /v1/audio/speech` c JSON
- `POST /v1/tts` c `multipart/form-data`
- Ограничение конкурентности через `OMNIVOICE_MAX_CONCURRENCY`
- Lazy/preload загрузка модели
- Валидация входа и безопасные дефолты
- CLI для локального вызова без HTTP
- Опциональное сохранение результата в `outputs/`

## Важная оговорка

Этот проект **не содержит веса модели**. Нужно указать путь к локально скачанной модели через `OMNIVOICE_MODEL_DIR`.

Также сам Python-пакет `omnivoice` должен быть доступен одним из двух способов:
- либо установлен в окружение, например `pip install omnivoice`
- либо указан путь к локальному исходному репозиторию через `OMNIVOICE_SOURCE_DIR`

Сервис умеет автоматически пытаться импортировать `omnivoice` из соседней папки `../OmniVoice`, но в проде лучше явно задать `OMNIVOICE_SOURCE_DIR`.

## Структура

- `app/main.py` — HTTP API
- `app/service.py` — управление моделью и инференсом
- `app/bootstrap.py` — импорт локального OmniVoice без правок upstream
- `app/cli.py` — локальный CLI
- `tests/` — базовые smoke/unit tests

## Быстрый старт

### 1. Создать окружение и установить зависимости

```bash
cd omnivoice_inference_service
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Если вы хотите использовать локальный исходный репозиторий `OmniVoice`, убедитесь, что его зависимости тоже установлены. Минимально вам нужен рабочий импорт `omnivoice`.

### 2. Настроить переменные окружения

```bash
cp .env.example .env
```

Минимально нужно выставить:

```dotenv
OMNIVOICE_MODEL_DIR=/absolute/path/to/your/local/model
OMNIVOICE_SOURCE_DIR=/absolute/path/to/your/local/OmniVoice
OMNIVOICE_DEFAULT_LANGUAGE=sah
OMNIVOICE_DEVICE=auto
OMNIVOICE_PRELOAD_MODEL=true
```

### 3. Запустить сервис

```bash
omnivoice-service
```

или

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Проверка готовности

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/readyz
```

Если `readyz` возвращает `503`, обычно причина одна из двух:
- не задан `OMNIVOICE_MODEL_DIR`
- не найден импорт `omnivoice`

## Использование API

### JSON endpoint

```bash
curl -X POST "http://127.0.0.1:8080/v1/audio/speech" \
  -H "Content-Type: application/json" \
  --output yakut.wav \
  -d '{
    "text": "Мин аатым Айаал. Бүгүҥҥү күн бу сервис якут тылын тургутарга бэлэм.",
    "language": "sah",
    "speed": 1.0,
    "num_step": 32
  }'
```

### Voice cloning через multipart

```bash
curl -X POST "http://127.0.0.1:8080/v1/tts" \
  -F 'text=Мин аатым Айаал. Бу кэпсээн якут тылынан айанар.' \
  -F 'language=sah' \
  -F 'ref_text=Мин аатым Айаал.' \
  -F 'ref_audio=@./reference.wav' \
  --output clone.wav
```

## CLI

```bash
omnivoice-synth \
  --text "Мин аатым Айаал. Бу якут тылынан локальнай тест." \
  --language sah \
  --output ./tmp/yakut.wav
```

Для cloning:

```bash
omnivoice-synth \
  --text "Мин аатым Айаал. Бу якут тылынан локальнай тест." \
  --language sah \
  --ref-audio ./reference.wav \
  --ref-text "Мин аатым Айаал." \
  --output ./tmp/yakut-clone.wav
```

## Production notes

- По умолчанию `OMNIVOICE_MAX_CONCURRENCY=1`. Для тяжёлой TTS модели это безопасный старт.
- Для GPU production лучше держать один процесс на одну модель и масштабировать горизонтально.
- `GET /readyz` форсирует проверку реальной загрузки модели.
- Авто-ASR для `ref_text` по умолчанию отключён. Для продакшена это правильно: меньше зависимостей, меньше сюрпризов, предсказуемее latency.
- Для якутского лучше всегда явно передавать `language="sah"`, даже несмотря на дефолт.
- Референсное аудио для cloning желательно держать коротким, примерно 3-10 секунд.

## Почему `sah`

В upstream OmniVoice якутский язык поддерживается и обозначается как `Yakut` с language id `sah`.

## Что ещё можно добавить потом

- очередь задач через Redis/Celery
- Prometheus метрики
- bearer auth / API key
- persistent voice profile cache
- OpenAI-compatible слой поверх текущего API
