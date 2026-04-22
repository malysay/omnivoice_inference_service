from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.audio import decode_audio_b64, encode_wav_bytes, persist_wav, sanitize_filename
from app.bootstrap import OmniVoiceImportError, ensure_omnivoice_importable
from app.config import Settings
from app.model_assets import validate_model_dir
from app.schemas import SynthesisRequest

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    audio_bytes: bytes
    sample_rate: int
    filename: str
    elapsed_ms: int
    saved_to: str | None = None


class OmniVoiceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._device: str | None = None
        self._load_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._last_error: str | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _resolve_torch_device(self) -> tuple[str, Any]:
        import torch

        requested = self.settings.device.lower()
        if requested != "auto":
            if requested == "cpu":
                return "cpu", torch.float32
            if requested.startswith("cuda"):
                return requested, torch.float16
            if requested == "mps":
                return "mps", torch.float16
            raise ValueError(f"Unsupported OMNIVOICE_DEVICE={self.settings.device}")

        if torch.cuda.is_available():
            return "cuda", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _load_model_sync(self) -> None:
        if self.settings.model_path is None:
            raise RuntimeError(
                "OMNIVOICE_MODEL_DIR is not configured. Point it to the local OmniVoice model directory."
            )
        validate_model_dir(self.settings.model_path)

        ensure_omnivoice_importable(self.settings)
        from omnivoice.models.omnivoice import OmniVoice

        device, dtype = self._resolve_torch_device()
        logger.info("Loading OmniVoice model from %s on %s", self.settings.model_path, device)
        self._model = OmniVoice.from_pretrained(
            str(self.settings.model_path),
            device_map=device,
            dtype=dtype,
            load_asr=False,
        )
        self._device = device
        self._last_error = None

    async def ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            try:
                await asyncio.to_thread(self._load_model_sync)
            except Exception as exc:  # noqa: BLE001
                self._last_error = str(exc)
                raise

    def _build_generate_kwargs(self, request: SynthesisRequest) -> dict[str, Any]:
        language = request.language or self.settings.default_language
        kwargs: dict[str, Any] = {
            "text": request.text,
            "language": language,
            "ref_text": request.ref_text,
            "instruct": request.instruct,
            "num_step": request.num_step or self.settings.default_num_step,
            "guidance_scale": request.guidance_scale
            if request.guidance_scale is not None
            else self.settings.default_guidance_scale,
            "speed": request.speed if request.speed is not None else self.settings.default_speed,
            "duration": request.duration,
            "t_shift": request.t_shift if request.t_shift is not None else self.settings.default_t_shift,
            "denoise": request.denoise if request.denoise is not None else self.settings.default_denoise,
            "postprocess_output": request.postprocess_output
            if request.postprocess_output is not None
            else self.settings.default_postprocess_output,
            "layer_penalty_factor": request.layer_penalty_factor
            if request.layer_penalty_factor is not None
            else self.settings.default_layer_penalty_factor,
            "position_temperature": request.position_temperature
            if request.position_temperature is not None
            else self.settings.default_position_temperature,
            "class_temperature": request.class_temperature
            if request.class_temperature is not None
            else self.settings.default_class_temperature,
        }
        if request.reference_audio_b64:
            kwargs["ref_audio"] = decode_audio_b64(request.reference_audio_b64)
        return kwargs

    def _synthesize_sync(self, request: SynthesisRequest) -> SynthesisResult:
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        started = time.perf_counter()
        generate_kwargs = self._build_generate_kwargs(request)
        logger.info(
            "Starting synthesis text_len=%s language=%s mode=%s",
            len(request.text),
            generate_kwargs.get("language"),
            "voice-clone" if request.reference_audio_b64 else ("voice-design" if request.instruct else "auto"),
        )
        audios = self._model.generate(**generate_kwargs)
        sample_rate = int(self._model.sampling_rate)
        audio_bytes = encode_wav_bytes(audios[0], sample_rate)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        filename = sanitize_filename(request.output_filename, fallback_stem="omnivoice")
        saved_to = None
        if request.output_filename:
            saved_to = str(persist_wav(audio_bytes, self.settings.output_path, filename))
        logger.info("Synthesis finished elapsed_ms=%s sample_rate=%s", elapsed_ms, sample_rate)
        return SynthesisResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            filename=filename,
            elapsed_ms=elapsed_ms,
            saved_to=saved_to,
        )

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        await self.ensure_model_loaded()
        async with self._semaphore:
            return await asyncio.to_thread(self._synthesize_sync, request)

    def health_payload(self) -> dict[str, Any]:
        degraded = self.settings.model_path is None or self.last_error is not None
        return {
            "status": "degraded" if degraded else "ok",
            "model_loaded": self.model_loaded,
            "model_dir": str(self.settings.model_path) if self.settings.model_path else None,
            "device": self.device,
            "default_language": self.settings.default_language,
            "detail": self.last_error,
        }

    async def readiness_payload(self) -> dict[str, Any]:
        if self.settings.model_path is None:
            return {
                "ready": False,
                "detail": "OMNIVOICE_MODEL_DIR is not configured.",
            }
        try:
            await self.ensure_model_loaded()
        except OmniVoiceImportError as exc:
            return {"ready": False, "detail": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"ready": False, "detail": str(exc)}
        return {"ready": True, "detail": "model loaded"}
