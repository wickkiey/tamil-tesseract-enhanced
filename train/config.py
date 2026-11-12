"""Load ``config.toml`` and resolve values, with command-line flags taking priority."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from .common import ROOT, TRAIN_DIR

DEFAULT_CONFIG = TRAIN_DIR / "config.toml"


class Config:
    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self.data = data
        self.source = source

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = (path or DEFAULT_CONFIG).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("rb") as handle:
            try:
                return cls(tomllib.load(handle), path)
            except tomllib.TOMLDecodeError as exc:
                raise ValueError(f"Invalid TOML in {path}: {exc}") from exc

    def get(self, section: str, key: str, override: Any = None) -> Any:
        """Command-line ``override`` wins; otherwise the config value."""
        if override is not None:
            return override
        try:
            return self.data[section][key]
        except KeyError:
            raise ValueError(f"Missing [{section}] {key} in {self.source}; pass it on the command line.") from None

    def path(self, section: str, key: str, override: Path | str | None = None) -> Path:
        value = self.get(section, key, override)
        if value == "":
            raise ValueError(f"[{section}] {key} is empty in {self.source}; set it or pass it on the command line.")
        resolved = Path(value)
        return resolved if resolved.is_absolute() else ROOT / resolved

    def optional_path(self, section: str, key: str, override: Path | str | None = None) -> Path | None:
        value = self.get(section, key, override)
        if value == "":
            return None
        resolved = Path(value)
        return resolved if resolved.is_absolute() else ROOT / resolved
