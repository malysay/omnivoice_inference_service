from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import JSONResponse

from app.audio import decode_audio_bytes
from app.config import Settings, get_settings
from app.logging_utils import configure_logging
from app.schemas import HealthResponse, ReadinessResponse, SynthesisRequest
from app.service import OmniVoiceManager

logger = logging.getLogger(__name__)
settings = get_settings()
configure_logging(settings.log_level)
manager = OmniVoiceManager(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.preload_model and settings.model_path is not None:
        try:
            await manager.ensure_model_loaded()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model preload failed: %s", exc)
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(**manager.health_payload())


@app.get("/readyz", response_model=ReadinessResponse)
async def readyz() -> JSONResponse:
    payload = await manager.readiness_payload()
    status_code = 200 if payload["ready"] else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/v1/audio/speech")
async def create_speech(request: SynthesisRequest) -> Response:
    try:
        result = await manager.synthesize(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {
        "X-Sample-Rate": str(result.sample_rate),
        "X-Elapsed-Ms": str(result.elapsed_ms),
    }
    if result.saved_to:
        headers["X-Saved-To"] = result.saved_to
    return Response(content=result.audio_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/tts")
async def create_tts_form(
    text: str = Form(...),
    language: str | None = Form(default=None),
    ref_text: str | None = Form(default=None),
    instruct: str | None = Form(default=None),
    output_filename: str | None = Form(default=None),
    num_step: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    speed: float | None = Form(default=None),
    duration: float | None = Form(default=None),
    t_shift: float | None = Form(default=None),
    denoise: bool | None = Form(default=None),
    postprocess_output: bool | None = Form(default=None),
    layer_penalty_factor: float | None = Form(default=None),
    position_temperature: float | None = Form(default=None),
    class_temperature: float | None = Form(default=None),
    ref_audio: UploadFile | None = File(default=None),
) -> Response:
    audio_b64 = None
    if ref_audio is not None:
        import base64

        decoded, sr = decode_audio_bytes(await ref_audio.read())
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, decoded, sr, format="WAV", subtype="PCM_16")
        audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request = SynthesisRequest(
        text=text,
        language=language,
        ref_text=ref_text,
        instruct=instruct,
        reference_audio_b64=audio_b64,
        output_filename=output_filename,
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        duration=duration,
        t_shift=t_shift,
        denoise=denoise,
        postprocess_output=postprocess_output,
        layer_penalty_factor=layer_penalty_factor,
        position_temperature=position_temperature,
        class_temperature=class_temperature,
    )
    return await create_speech(request)


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=False,
    )


if __name__ == "__main__":
    run()
