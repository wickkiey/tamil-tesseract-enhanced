"""Core OCR runtime: run Tesseract on one image with an explicit model."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = PROJECT_ROOT / "runs" / "inference.log"
PSM_MODES = (3, 6, 7, 11, 13)


def run_tool(args: list[str], log: Path | None = None, include_stderr: bool = False) -> str:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            encoding="utf-8", errors="replace", check=False)
    diagnostics = result.stdout + result.stderr
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nCOMMAND: {args!r}\n{diagnostics}\n")
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args!r}\n{diagnostics[-5000:]}")
    return diagnostics if include_stderr else result.stdout


def tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"Missing {name}. Install Tesseract 5 and add it to PATH.")
    return found


def recognize(image: Path, model: Path, psm: int = 7, log: Path | None = None) -> str:
    """Recognize one image with the given .traineddata model and return NFC text."""
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
