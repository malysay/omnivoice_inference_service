from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(slots=True)
class Settings:
    app_name: str = "omnivoice-inference-service"
    environment: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"

    model_dir: str | None = None
    source_dir: str | None = None
    output_dir: str = "./outputs"
    device: str = "auto"
    dtype: str = "auto"
    preload_model: bool = True
    allow_auto_asr: bool = False
    default_language: str | None = "sah"
    max_concurrency: int = 1

    default_num_step: int = 32
    default_guidance_scale: float = 2.0
    default_t_shift: float = 0.1
    default_speed: float = 1.0
    default_denoise: bool = True
    default_postprocess_output: bool = True
    default_layer_penalty_factor: float = 5.0
    default_position_temperature: float = 5.0
    default_class_temperature: float = 0.0

    def __post_init__(self) -> None:
        self.app_name = _env_str("OMNIVOICE_APP_NAME", self.app_name)
        self.environment = _env_str("OMNIVOICE_ENVIRONMENT", self.environment)
        self.host = _env_str("OMNIVOICE_HOST", self.host)
        self.port = _env_int("OMNIVOICE_PORT", self.port)
        self.log_level = _env_str("OMNIVOICE_LOG_LEVEL", self.log_level)

        self.model_dir = _env_optional_str("OMNIVOICE_MODEL_DIR", self.model_dir)
        self.source_dir = _env_optional_str("OMNIVOICE_SOURCE_DIR", self.source_dir)
        self.output_dir = _env_str("OMNIVOICE_OUTPUT_DIR", self.output_dir)
        self.device = _env_str("OMNIVOICE_DEVICE", self.device)
        self.dtype = _env_str("OMNIVOICE_DTYPE", self.dtype)
        self.preload_model = _env_bool("OMNIVOICE_PRELOAD_MODEL", self.preload_model)
        self.allow_auto_asr = _env_bool("OMNIVOICE_ALLOW_AUTO_ASR", self.allow_auto_asr)
        self.default_language = _env_optional_str(
            "OMNIVOICE_DEFAULT_LANGUAGE", self.default_language
        )
        self.max_concurrency = max(1, min(32, _env_int("OMNIVOICE_MAX_CONCURRENCY", self.max_concurrency)))

        self.default_num_step = max(1, min(128, _env_int("OMNIVOICE_DEFAULT_NUM_STEP", self.default_num_step)))
        self.default_guidance_scale = _env_float(
            "OMNIVOICE_DEFAULT_GUIDANCE_SCALE", self.default_guidance_scale
        )
        self.default_t_shift = _env_float("OMNIVOICE_DEFAULT_T_SHIFT", self.default_t_shift)
        self.default_speed = _env_float("OMNIVOICE_DEFAULT_SPEED", self.default_speed)
        self.default_denoise = _env_bool("OMNIVOICE_DEFAULT_DENOISE", self.default_denoise)
        self.default_postprocess_output = _env_bool(
            "OMNIVOICE_DEFAULT_POSTPROCESS_OUTPUT", self.default_postprocess_output
        )
        self.default_layer_penalty_factor = _env_float(
            "OMNIVOICE_DEFAULT_LAYER_PENALTY_FACTOR", self.default_layer_penalty_factor
        )
        self.default_position_temperature = _env_float(
            "OMNIVOICE_DEFAULT_POSITION_TEMPERATURE", self.default_position_temperature
        )
        self.default_class_temperature = _env_float(
            "OMNIVOICE_DEFAULT_CLASS_TEMPERATURE", self.default_class_temperature
        )

    @property
    def model_path(self) -> Path | None:
        return Path(self.model_dir).expanduser().resolve() if self.model_dir else None

    @property
    def source_path(self) -> Path | None:
        return Path(self.source_dir).expanduser().resolve() if self.source_dir else None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_optional_str(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_present() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv_if_present()
    return Settings()
