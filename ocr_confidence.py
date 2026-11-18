"""Tesseract word-level confidence from TSV output, for the labelling UI."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
import subprocess
import tempfile

from PIL import Image


def parse_words(tsv: str) -> list[tuple[str, float]]:
    reader = csv.DictReader(StringIO(tsv), delimiter="\t")
    words: list[tuple[str, float]] = []
    for row in reader:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf", ""))
        except (TypeError, ValueError):
            continue
        if confidence < 0:
            continue
        words.append((text, confidence))
    return words


def confidence_from_words(words: list[tuple[str, float]]) -> float | None:
    if not words:
        return None
    total = sum(len(text) for text, _ in words)
    return sum(confidence * len(text) for text, confidence in words) / total


def parse_confidence(tsv: str) -> float | None:
    return confidence_from_words(parse_words(tsv))


def tesseract_words(image: Image.Image, model_path: str) -> list[tuple[str, float]]:
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Tesseract model not found: {model}")
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "line.png"
        image.convert("L").save(image_path)
        config_path = Path(tmp) / "line_tsv"
        config_path.write_text("tessedit_create_tsv 1\n", encoding="utf-8")
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "--tessdata-dir", str(model.parent),
             "-l", model.stem, "--oem", "1", "--psm", "7", str(config_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "tesseract failed")
    return parse_words(result.stdout)


def tesseract_confidence(image: Image.Image, model_path: str) -> float | None:
    return confidence_from_words(tesseract_words(image, model_path))
