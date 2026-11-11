"""Command-line entry point: download the pinned model or run local OCR."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .model_store import DEFAULT_MODEL_PATH, download_model, ensure_model
from .runtime import DEFAULT_LOG, PSM_MODES, recognize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inference",
        description="Download the pinned Tamil Tesseract model and run local OCR.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download", help="Download and verify the pinned model release")
    download.add_argument("--dest", type=Path, default=DEFAULT_MODEL_PATH,
                          help="Destination .traineddata path")
    download.add_argument("--force", action="store_true",
                          help="Replace an existing file after hash verification")

    run = sub.add_parser("run", help="Recognize text in one image")
    run.add_argument("--image", type=Path, required=True, help="PNG, TIFF or other Tesseract-readable image")
    run.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                     help="Full path to a .traineddata model (defaults to the pinned release)")
    run.add_argument("--psm", type=int, choices=PSM_MODES, default=7,
                     help="7=line, 6=block, 3=page, 11=sparse, 13=raw line")
    run.add_argument("--output", type=Path, help="Optional UTF-8 text output; existing files are not overwritten")
    run.add_argument("--log", type=Path, default=DEFAULT_LOG, help="Append command and OCR diagnostics")
    run.add_argument("--download", action="store_true", help="Download the pinned model if it is missing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            path = download_model(args.dest, force=args.force)
            print(f"Model ready: {path}")
            return 0
        model = ensure_model(args.model, download=args.download)
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"Output already exists; choose a new path: {args.output}")
            if args.output.resolve() in (args.image.resolve(), model.resolve(), args.log.resolve()):
                raise ValueError("Output must differ from the input image, model and log.")
        text = recognize(args.image, model, args.psm, args.log)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text + "\n")
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
