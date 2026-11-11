"""Resolve, download and verify the pinned Tesseract model release.

The release is intentionally hard-pinned. Bump ``DEFAULT_RELEASE`` and
``MODEL_SHA256`` together when promoting a new model to the repository.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE = "approved-v1"
MODEL_FILENAME = "tam_historical.traineddata"
MODEL_SHA256 = "0f79f56753ec82641c4f7fd108df466d63a421a2de087d363180b14b826600f1"
RELEASE_BASE_URL = "https://raw.githubusercontent.com/wickkiey/tamil-tesseract-enhanced/main/models"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / DEFAULT_RELEASE / MODEL_FILENAME


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def model_url(release: str = DEFAULT_RELEASE) -> str:
    return f"{RELEASE_BASE_URL}/{release}/{MODEL_FILENAME}"


def download_model(destination: Path | None = None, force: bool = False,
                   expected_sha256: str = MODEL_SHA256) -> Path:
    """Download the pinned model, verify its hash, and return the local path."""
    target = Path(destination) if destination is not None else DEFAULT_MODEL_PATH
    if target.is_file() and not force:
        actual = sha256(target)
        if actual == expected_sha256:
            return target
        raise ValueError(f"Existing model hash mismatch at {target}; re-run with --force to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(model_url(), headers={"User-Agent": "Tamil-OCR-inference/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        actual = sha256(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"Downloaded model hash mismatch: expected {expected_sha256}, got {actual}. "
                "The pinned release constants may be out of date."
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def ensure_model(path: Path | None = None, download: bool = False) -> Path:
    """Return an existing model path, optionally downloading the pinned model."""
    target = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if target.is_file():
        return target
    if download:
        return download_model(target)
    raise FileNotFoundError(
        f"Model not found: {target}. Run `python -m inference download` first, "
        "or pass --download."
    )
