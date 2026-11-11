"""Local Tamil OCR inference with an explicitly selected Tesseract model."""

from .cli import main
from .model_store import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RELEASE,
    MODEL_SHA256,
    download_model,
    ensure_model,
)
from .runtime import DEFAULT_LOG, PSM_MODES, recognize, run_tool, tool

__all__ = [
    "DEFAULT_LOG",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_RELEASE",
    "MODEL_SHA256",
    "PSM_MODES",
    "download_model",
    "ensure_model",
    "main",
    "recognize",
    "run_tool",
    "tool",
]
