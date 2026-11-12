"""Shared constants and small helpers used by every training-pipeline stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import unicodedata
import urllib.request


TRAIN_DIR = Path(__file__).resolve().parent
ROOT = TRAIN_DIR.parent  # repository root
SPLITS = ("train", "val", "test")
TARGETS = {
    "ai": ("லை", "ளை", "ணை", "னை"),
    "nedil": ("ணா", "றா", "னா"),
    "older_i": ("கி", "கீ", "சி", "சீ", "தி", "தீ"),
}
PROBES = tuple(dict.fromkeys(t for group in TARGETS.values() for t in group)) + (
    "ணொ", "ணோ", "றொ", "றோ", "னொ", "னோ", "ஔ", "ஊ",
)
DIAGNOSTICS = (
    "மாலை வளையல் அணை நினைவு காணாத கூறாத வினா.",
    "காணாத கூறாத வினா நினைவு அணை வளையல் மாலை.",
    "வினா அணை மாலை காணாத வளையல் கூறாத நினைவு.",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    if not text or any(unicodedata.category(c).startswith("C") for c in text):
        raise ValueError("Transcription must be a nonempty single line without control characters.")
    return text


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


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
        raise RuntimeError(f"Missing {name}. Install Tesseract 5 with training tools and add it to PATH.")
    return found


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == 0:
            raise ValueError(f"Empty asset: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Tamil-OCR-dataset/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
