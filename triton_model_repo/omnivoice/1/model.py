from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


LOGGER = logging.getLogger("omnivoice_triton")


def _is_project_root(candidate: Path) -> bool:
    app_dir = candidate / "app"
    return (app_dir / "service.py").exists() and (app_dir / "schemas.py").exists()


def _ensure_project_on_path() -> Path:
    current = Path(__file__).resolve()
    configured_root = os.getenv("OMNIVOICE_PROJECT_ROOT")
    candidates: list[Path] = []

    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve())

    candidates.extend(
        [
            Path.cwd().resolve(),
            Path("/workspace/omnivoice_inference_service"),
            current.parent,
            *current.parents,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_project_root(candidate):
            sys.path.insert(0, str(candidate))
            return candidate
    raise RuntimeError(
        "Unable to locate project root for OmniVoice service. "
        "Set OMNIVOICE_PROJECT_ROOT or place the Triton model repository inside the project."
    )


PROJECT_ROOT = _ensure_project_on_path()

from app.config import get_settings
from app.logging_utils import configure_logging
from app.request_parsing import parse_synthesis_request_payload
from app.service import OmniVoiceManager


def _extract_request_json(request: pb_utils.InferenceRequest) -> str:
    tensor = pb_utils.get_input_tensor_by_name(request, "REQUEST_JSON")
    if tensor is None:
        raise ValueError("Missing required Triton input tensor: REQUEST_JSON")

    value = tensor.as_numpy()
    if value.size != 1:
        raise ValueError("REQUEST_JSON must contain exactly one JSON payload.")

    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, str):
        return item
    return str(item)


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        self.model_config = json.loads(args["model_config"])
        self.settings = get_settings()
        configure_logging(self.settings.log_level)
        self.manager = OmniVoiceManager(self.settings)
        self._load_lock = threading.Lock()

        if self.settings.preload_model and self.settings.model_path is not None:
            try:
                with self._load_lock:
                    self.manager._load_model_sync()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Triton model preload failed: %s", exc)

    def execute(
        self, requests: list[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                request_json = _extract_request_json(request)
                synthesis_request = parse_synthesis_request_payload(request_json)
                if not self.manager.model_loaded:
                    with self._load_lock:
                        if not self.manager.model_loaded:
                            self.manager._load_model_sync()
                result = self.manager._synthesize_sync(synthesis_request)

                outputs = [
                    pb_utils.Tensor(
                        "AUDIO_WAV",
                        np.frombuffer(result.audio_bytes, dtype=np.uint8),
                    ),
                    pb_utils.Tensor(
                        "SAMPLE_RATE",
                        np.asarray([result.sample_rate], dtype=np.int32),
                    ),
                    pb_utils.Tensor(
                        "ELAPSED_MS",
                        np.asarray([result.elapsed_ms], dtype=np.int32),
                    ),
                    pb_utils.Tensor(
                        "FILENAME",
                        np.asarray([result.filename.encode("utf-8")], dtype=object),
                    ),
                    pb_utils.Tensor(
                        "SAVED_TO",
                        np.asarray(
                            [(result.saved_to or "").encode("utf-8")],
                            dtype=object,
                        ),
                    ),
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
            except Exception as exc:  # noqa: BLE001
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(exc)))
                )
        return responses

    def finalize(self) -> None:
        LOGGER.info("Finalizing Triton OmniVoice model from %s", PROJECT_ROOT)
