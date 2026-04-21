# OmniVoice Triton Service

Triton-first проект для production-oriented инференса `OmniVoice`.

Главная идея теперь такая:
- `Triton Inference Server` является основным runtime;
- текущий Python-код используется как shared inference layer для Triton Python backend;
- локальный CLI остается как вспомогательный инструмент для отладки;
- старый `FastAPI` код можно считать legacy/dev-only, а не основным способом деплоя.

## Что есть в репозитории

- `triton_model_repo/omnivoice/config.pbtxt` — Triton model config
- `triton_model_repo/omnivoice/1/model.py` — Triton Python backend entrypoint
- `app/service.py` — общая логика загрузки OmniVoice и синтеза
- `app/triton_client.py` — клиент для Triton HTTP API с сохранением WAV
- `docker-compose.yml` — основной локальный запуск Triton
- `scripts/run_triton.sh` — быстрый старт сервера
- `scripts/check_triton.sh` — проверка live/ready/model metadata

## Важное ограничение

Triton как production-формат ориентирован на Linux deployment.

Если ты работаешь на macOS:
- полноценный production-сценарий с Triton нужно проверять на Linux-хосте;
- Apple `mps` внутри Triton-контейнера недоступен;
- локально на Mac можно сделать smoke-check через Docker, но это не будет эквивалентом продовой GPU-конфигурации.

Если целевой прод у тебя Linux + NVIDIA GPU, то текущая структура подходит гораздо лучше, чем отдельный FastAPI-процесс.

## Что нужно заранее

Проект не содержит веса модели. Нужна локальная папка checkpoint `OmniVoice`.

Минимально должны быть доступны:
- корневой `config.json`
- корневой `model.safetensors`
- подпапка `audio_tokenizer/`

Также в контейнере должен импортироваться Python-пакет `omnivoice`. Это уже учтено в `Dockerfile`: при сборке устанавливается `omnivoice` и сам проект.

## Быстрый старт

### 1. Настрой `.env`

```bash
cp .env.example .env
```

Заполни минимум:

```dotenv
OMNIVOICE_MODEL_DIR=/absolute/path/to/local/OmniVoice-model
OMNIVOICE_DEVICE=auto
OMNIVOICE_PRELOAD_MODEL=true
OMNIVOICE_DEFAULT_LANGUAGE=sah
OMNIVOICE_MAX_CONCURRENCY=1
```

Важно:
- `OMNIVOICE_MODEL_DIR` в `.env` — это путь на хост-машине;
- `docker-compose.yml` сам примонтирует эту папку в контейнер как `/models/omnivoice`;
- внутри контейнера Triton будет работать уже с `/models/omnivoice`.

### 2. Подними Triton

```bash
./scripts/run_triton.sh
```

или напрямую:

```bash
docker compose up --build
```

Сервис откроет стандартные Triton порты:
- `8000` — HTTP
- `8001` — gRPC
- `8002` — metrics

### 3. Проверь health

В отдельном терминале:

```bash
./scripts/check_triton.sh
```

Это проверит:
- `/v2/health/live`
- `/v2/health/ready`
- `/v2/models/omnivoice/ready`
- `/v2/models/omnivoice`

Если `model ready` не проходит, почти всегда причина одна из этих:
- неправильный `OMNIVOICE_MODEL_DIR`
- в контейнере не загрузилась модель OmniVoice
- модель загрузилась, но уперлась в device/dependency issue

## Первый реальный inference test

Самый удобный способ проверить end-to-end:

```bash
python -m app.triton_client \
  --url http://127.0.0.1:8000 \
  --model omnivoice \
  --text "Мин аатым Айаал. Бу Triton көмөтүнэн тургутуллар тест." \
  --language sah \
  --num-step 32 \
  --output ./outputs/triton-test.wav
```

На выходе клиент:
- отправит Triton infer request;
- получит `AUDIO_WAV`;
- соберет байты обратно в WAV;
- сохранит файл локально;
- выведет JSON с `output`, `sample_rate`, `elapsed_ms`.

## Пример прямого Triton HTTP запроса

Если хочется проверить API вручную:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v2/models/omnivoice/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "REQUEST_JSON",
        "shape": [1],
        "datatype": "BYTES",
        "data": [
          "{\"text\":\"Мин аатым Айаал.\",\"language\":\"sah\",\"num_step\":32}"
        ]
      }
    ],
    "outputs": [
      {"name": "AUDIO_WAV"},
      {"name": "SAMPLE_RATE"},
      {"name": "ELAPSED_MS"},
      {"name": "FILENAME"},
      {"name": "SAVED_TO"}
    ]
  }'
```

Ответ будет JSON, где:
- `AUDIO_WAV.data` — массив байтов WAV
- `SAMPLE_RATE.data[0]` — sample rate
- `ELAPSED_MS.data[0]` — время инференса

Для обычной ручной проверки лучше использовать `python -m app.triton_client`, потому что он уже сохраняет результат в файл.

## Как устроен вход модели

Triton модель принимает один input tensor:

- `REQUEST_JSON` (`TYPE_STRING`, shape `[1]`)

Внутри него JSON, совместимый с `SynthesisRequest`.

Поддерживаются те же поля, что и раньше:
- `text`
- `language`
- `instruct`
- `ref_text`
- `reference_audio_b64`
- `num_step`
- `guidance_scale`
- `speed`
- `duration`
- `t_shift`
- `denoise`
- `postprocess_output`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`

## Что именно проверять после запуска

Минимальный checklist:

1. `docker compose up --build` завершился без model load errors.
2. `curl http://127.0.0.1:8000/v2/health/live` возвращает success.
3. `curl http://127.0.0.1:8000/v2/health/ready` возвращает success.
4. `curl http://127.0.0.1:8000/v2/models/omnivoice/ready` возвращает success.
5. `python -m app.triton_client ... --output ./outputs/test.wav` реально создает WAV.
6. WAV открывается и звучит ожидаемо.

Если нужен voice cloning, дополнительно проверь:
- короткий референс 3-10 секунд;
- корректный `ref_text`;
- что `reference_audio_b64` действительно соответствует WAV/decodable audio.

## Файлы, которые теперь важны для деплоя

- [Dockerfile](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/Dockerfile:1)
- [docker-compose.yml](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/docker-compose.yml:1)
- [triton_model_repo/omnivoice/config.pbtxt](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/triton_model_repo/omnivoice/config.pbtxt:1)
- [triton_model_repo/omnivoice/1/model.py](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/triton_model_repo/omnivoice/1/model.py:1)
- [app/service.py](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/app/service.py:1)
- [app/triton_client.py](/Users/malysay/Apros/Work/Code/omnivoice_inference_service/app/triton_client.py:1)

## Production замечания

- Для реального GPU production лучше держать один model instance на одну GPU allocation и масштабировать горизонтально.
- `OMNIVOICE_MAX_CONCURRENCY=1` — хороший безопасный старт для тяжелой TTS-модели.
- Если deployment идет на Linux + NVIDIA, обычно имеет смысл выставлять `OMNIVOICE_DEVICE=cuda`.
- Если нужен CPU-only Triton, он тоже возможен, но latency будет существенно хуже.
- На macOS текущий стек полезен для подготовки репозитория и smoke-тестов, но не как финальная prod-среда.

## Локальный CLI

Для отладки без Triton по-прежнему можно использовать:

```bash
omnivoice-synth \
  --text "Мин аатым Айаал. Бу локальнай тест." \
  --language sah \
  --output ./outputs/debug.wav
```

Это полезно, если нужно отделить “проблема в Triton runtime” от “проблема в самой модели/весах”.
