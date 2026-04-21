FROM nvcr.io/nvidia/tritonserver:24.10-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/omnivoice_inference_service

COPY pyproject.toml README.md ./
COPY app ./app
COPY triton_model_repo ./triton_model_repo
COPY .env.example ./
COPY examples ./examples
COPY scripts ./scripts

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir omnivoice && \
    python3 -m pip install --no-cache-dir .

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/workspace/omnivoice_inference_service/triton_model_repo"]
