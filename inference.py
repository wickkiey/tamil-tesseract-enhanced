"""Run local Tamil OCR using an explicitly selected Tesseract traineddata model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unicodedata

from tamil_ocr import ROOT, run_tool, tool


PSM_MODES = (3, 6, 7, 11, 13)


def recognize(image: Path, model: Path, psm: int = 7, log: Path | None = None) -> str:
    if psm not in PSM_MODES:
        raise ValueError(f"PSM must be one of {PSM_MODES}.")
    if not image.is_file():
        raise FileNotFoundError(f"Input image missing: {image}")
    if not model.is_file() or model.suffix != ".traineddata":
        raise ValueError(f"Provide an existing .traineddata model: {model}")
    if image.resolve() == model.resolve():
        raise ValueError("Input image and model must be different files.")
    if log is not None and log.resolve() in (image.resolve(), model.resolve()):
        raise ValueError("The log must not overwrite an input image or model.")
    raw = run_tool([
        tool("tesseract"), str(image.resolve()), "stdout",
        "--tessdata-dir", str(model.resolve().parent), "-l", model.stem,
        "--oem", "1", "--psm", str(psm),
    ], log)
    text = unicodedata.normalize("NFC", raw.strip())
    if not text:
        raise RuntimeError("Tesseract returned no text. Check the image, margins, contrast, model and PSM.")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="PNG, TIFF or other Tesseract-readable image")
    parser.add_argument("--model", type=Path, required=True, help="Full path to the selected .traineddata file")
    parser.add_argument("--psm", type=int, choices=PSM_MODES, default=7, help="7=line, 6=block, 3=page, 11=sparse, 13=raw line")
    parser.add_argument("--output", type=Path, help="Optional UTF-8 text output; existing files are not overwritten")
    parser.add_argument("--log", type=Path, default=ROOT / "runs" / "inference.log", help="Append command and OCR diagnostics")
    args = parser.parse_args()
    try:
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"Output already exists; choose a new path: {args.output}")
            if args.output.resolve() in (args.image.resolve(), args.model.resolve(), args.log.resolve()):
                raise ValueError("Output must differ from the input image, model and log.")
        text = recognize(args.image, args.model, args.psm, args.log)
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


if __name__ == "__main__":
    raise SystemExit(main())
