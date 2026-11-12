"""Font profiles: loading, HarfBuzz/RAQM line rendering, and feature verification."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, features

from .common import PROBES, ROOT, digest, write_json


@dataclass(frozen=True)
class Profile:
    name: str
    font: Path
    features: tuple[str, ...]
    category: str


def load_profiles(path: Path, base_dir: Path = ROOT) -> list[Profile]:
    """Load profiles; relative font paths are resolved against ``base_dir`` (repo root)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Profiles must be a nonempty JSON array.")
    profiles: list[Profile] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each font profile must be an object.")
        name, font, category = (item.get(k) for k in ("name", "font", "category"))
        tags = item.get("features")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
                or not isinstance(font, str) or not isinstance(category, str)
                or category not in ("modern", "historical", "archival")
                or not isinstance(tags, list)
                or any(not isinstance(t, str) or not re.fullmatch(r"-?[a-z0-9]{4}", t) for t in tags)):
            raise ValueError(f"Invalid font profile: {item!r}")
        font_path = Path(font)
        if not font_path.is_absolute():
            font_path = base_dir / font_path
        if not font_path.is_file():
            raise FileNotFoundError(f"Font missing: {font_path}. Run setup first.")
        profiles.append(Profile(name, font_path, tuple(tags), category))
    if len({p.name for p in profiles}) != len(profiles):
        raise ValueError("Font profile names must be unique.")
    return profiles


def render(text: str, profile: Profile, size: int = 48) -> Image.Image:
    if not features.check("raqm"):
        raise RuntimeError("Pillow requires RAQM/HarfBuzz for Tamil shaping. No unshaped fallback is allowed.")
    font = ImageFont.truetype(str(profile.font), size, layout_engine=ImageFont.Layout.RAQM)
    kwargs = {"font": font, "direction": "ltr", "language": "ta", "features": list(profile.features)}
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, **kwargs)
    margin = max(12, size // 3)
    image = Image.new("L", (right - left + margin * 2, bottom - top + margin * 2), 255)
    ImageDraw.Draw(image).text((margin - left, margin - top), text, fill=0, **kwargs)
    if ImageOps.invert(image).getbbox() is None:
        raise ValueError(f"Font rendered no ink: {profile.name}")
    return image


def pixel_changes(a: Image.Image, b: Image.Image) -> int:
    size = (max(a.width, b.width), max(a.height, b.height))
    images = []
    for image in (a, b):
        padded = Image.new("L", size, 255)
        padded.paste(image, (0, 0))
        images.append(padded)
    return sum(count for value, count in enumerate(ImageChops.difference(*images).histogram()) if value)


def verify_profiles(profiles: list[Profile], out: Path, texts: list[str]) -> None:
    report: dict[str, object] = {}
    for profile in profiles:
        with TTFont(profile.font) as font:
            cmap = font.getBestCmap()
            if cmap is None:
                raise ValueError(f"Font has no Unicode cmap: {profile.font}")
            missing = sorted({c for text in texts for c in text if not c.isspace() and ord(c) not in cmap})
            if missing:
                raise ValueError(f"{profile.name} lacks corpus characters: {missing!r}. Font fallback is forbidden.")
            tags = {r.FeatureTag for r in font["GSUB"].table.FeatureList.FeatureRecord} if "GSUB" in font else set()
            enabled = [t for t in profile.features if not t.startswith("-")]
            if any(t not in tags for t in enabled):
                raise ValueError(f"{profile.name}: enabled features absent from GSUB: {enabled}")
        off = Profile(profile.name, profile.font, tuple("-" + t for t in enabled), profile.category)
        changes = {t: pixel_changes(render(t, off, 64), render(t, profile, 64)) for t in PROBES}
        if profile.category == "historical" and (not enabled or not any(changes.values())):
            raise ValueError(f"{profile.name}: historical features change no probe pixels; refusing mislabeled data.")
        report[profile.name] = {"font_sha256": digest(profile.font.read_bytes()),
                                "features": profile.features, "category": profile.category,
                                "changed_pixels_by_cluster": changes,
                                "note": "Pixel changes prove alternates, not historical authenticity; inspect specimens."}
        rows = []
        for t in PROBES:
            a, b = render(t, off, 64), render(t, profile, 64)
            row = Image.new("L", (400, max(a.height, b.height) + 8), 255)
            row.paste(a, (0, 0))
            row.paste(b, (200, 0))
            rows.append(row)
        sheet = Image.new("L", (400, sum(r.height for r in rows)), 255)
        y = 0
        for row in rows:
            sheet.paste(row, (0, y))
            y += row.height
        sheet.save(out / f"probe_{profile.name}.png")
    write_json(out / "font_probe.json", report)
