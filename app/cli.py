from __future__ import annotations

import argparse
import base64
from pathlib import Path

from app.config import get_settings
from app.logging_utils import configure_logging
from app.schemas import SynthesisRequest
from app.service import OmniVoiceManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local OmniVoice synthesis CLI")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--ref-audio", dest="ref_audio", default=None)
    parser.add_argument("--ref-text", dest="ref_text", default=None)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--num-step", dest="num_step", type=int, default=None)
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    manager = OmniVoiceManager(settings)

    reference_audio_b64 = None
    if args.ref_audio:
        reference_audio_b64 = base64.b64encode(Path(args.ref_audio).read_bytes()).decode("utf-8")

    request = SynthesisRequest(
        text=args.text,
        language=args.language,
        instruct=args.instruct,
        ref_text=args.ref_text,
        reference_audio_b64=reference_audio_b64,
        output_filename=Path(args.output).name,
        speed=args.speed,
        duration=args.duration,
        num_step=args.num_step,
    )
    result = await manager.synthesize(request)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result.audio_bytes)
    print(out_path)


def main() -> None:
    import asyncio

    asyncio.run(_main())


if __name__ == "__main__":
    main()
