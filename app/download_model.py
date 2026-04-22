from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.model_assets import minimal_model_allow_patterns, validate_model_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download an OmniVoice model snapshot from Hugging Face for local inference."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repository id, for example openai/omnivoice",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision, branch, tag, or commit SHA",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the model snapshot will be stored",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private models",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected download mode without downloading anything",
    )
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help="Download the full repository snapshot instead of the filtered runtime subset",
    )
    return parser


def _import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised outside test env
        raise SystemExit(
            "huggingface_hub is not installed. Install it with `python -m pip install .[download]` "
            "or `python -m pip install huggingface_hub`."
        ) from exc
    return snapshot_download


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    allow_patterns = minimal_model_allow_patterns()
    mode = "full snapshot" if args.full_snapshot else "minimal runtime snapshot"
    if args.dry_run:
        print(f"Selected mode: {mode}")
        if args.full_snapshot:
            print("All repository files will be downloaded.")
        else:
            print("Will download only these patterns:")
            for pattern in allow_patterns:
                print(pattern)
        return 0

    snapshot_download = _import_snapshot_download()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_kwargs = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "local_dir": str(output_dir),
        "local_dir_use_symlinks": False,
        "token": args.token,
        "cache_dir": args.cache_dir,
    }
    if not args.full_snapshot:
        snapshot_kwargs["allow_patterns"] = allow_patterns

    snapshot_download(**snapshot_kwargs)
    validate_model_dir(output_dir)

    print(f"{mode.capitalize()} is ready at: {output_dir}")
    print("Mount this directory into the container as /models/omnivoice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
