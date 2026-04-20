from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str | None = Field(default=None, max_length=64)
    ref_text: str | None = Field(default=None, max_length=5000)
    instruct: str | None = Field(default=None, max_length=256)
    reference_audio_b64: str | None = None
    response_format: Literal["wav"] = "wav"
    num_step: int | None = Field(default=None, ge=1, le=128)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=20.0)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)
    duration: float | None = Field(default=None, gt=0.0, le=600.0)
    t_shift: float | None = Field(default=None, ge=0.0, le=1.0)
    denoise: bool | None = None
    postprocess_output: bool | None = None
    layer_penalty_factor: float | None = Field(default=None, ge=0.0, le=20.0)
    position_temperature: float | None = Field(default=None, ge=0.0, le=20.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=5.0)
    output_filename: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_prompt_mode(self) -> "SynthesisRequest":
        if self.ref_text and not self.reference_audio_b64:
            raise ValueError("`ref_text` requires `reference_audio_b64`.")
        return self


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_dir: str | None
    device: str | None
    default_language: str | None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    detail: str
