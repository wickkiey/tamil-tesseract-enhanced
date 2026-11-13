"""Dataset manifest, synthetic generation, and real-data import (with leakage checks)."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
import shutil
import sys

from PIL import Image, ImageFilter

from .common import DIAGNOSTICS, PROBES, SPLITS, digest, normalized, write_json
from .corpus import assign_diagnostics, load_corpus, split_texts
from .profiles import load_profiles, render, verify_profiles

DEFAULT_PROFILES = Path(__file__).resolve().parent / "profiles.json"
DEFAULT_FONT_SIZES = (40, 48, 56)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: str
    gt: str
    split: str
    source_id: str
    profile: str
    category: str
    kind: str
    text_sha256: str
    image_sha256: str


def save_manifest(dataset: Path, samples: list[Sample]) -> None:
    write_json(dataset / "manifest.json", [asdict(s) for s in samples])


def load_manifest(dataset: Path) -> list[Sample]:
    raw = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Dataset manifest must be a nonempty array.")
    samples = []
    for item in raw:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in Sample.__annotations__):
            raise ValueError(f"Invalid sample record: {item!r}")
        samples.append(Sample(**{k: item[k] for k in Sample.__annotations__}))
    ids: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    for sample in samples:
        if sample.sample_id in ids or sample.split not in SPLITS or sample.kind not in ("synthetic", "real"):
            raise ValueError(f"Duplicate ID, invalid split, or kind: {sample.sample_id}")
        ids.add(sample.sample_id)
        for rel in (sample.image, sample.gt):
            path = (dataset / rel).resolve()
            if not path.is_relative_to(dataset.resolve()) or not path.is_file():
                raise ValueError(f"Missing or out-of-dataset sample file: {rel}")
        text = normalized((dataset / sample.gt).read_text(encoding="utf-8-sig"))
        if digest(text.encode("utf-8")) != sample.text_sha256:
            raise ValueError(f"Transcription changed; reimport sample: {sample.sample_id}")
        if digest((dataset / sample.image).read_bytes()) != sample.image_sha256:
            raise ValueError(f"Image changed; reimport sample: {sample.sample_id}")
        for key in (("source", sample.source_id), ("text", sample.text_sha256), ("image", sample.image_sha256)):
            if key in group_splits and group_splits[key] != sample.split:
                raise ValueError(f"Train/test leakage for {key}: {group_splits[key]} versus {sample.split}")
            group_splits[key] = sample.split
    return samples


def augment(image: Image.Image, rng: random.Random, variant: int,
            rotation_deg: float = 0.6, blur_range: tuple[float, float] = (0.1, 0.45)) -> Image.Image:
    if variant == 0:
        return image
    image = image.rotate(rng.uniform(-rotation_deg, rotation_deg), resample=Image.Resampling.BICUBIC,
                         expand=True, fillcolor=255)
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*blur_range)))
    pixels = bytearray(image.tobytes())
    for _ in range(max(1, len(pixels) // 1200)):
        pos = rng.randrange(len(pixels))
        pixels[pos] = rng.randrange(100, 220)
    return Image.frombytes("L", image.size, bytes(pixels))


def generate(dataset: Path, corpus: Path | None, profiles_path: Path, lines: int,
             variants: int, seed: int,
             font_sizes: tuple[int, ...] = DEFAULT_FONT_SIZES,
             rotation_deg: float = 0.6,
             blur_range: tuple[float, float] = (0.1, 0.45)) -> None:
    if lines < 10 or variants < 1:
        raise ValueError("--lines must be >=10 and --variants >=1.")
    if not font_sizes:
        raise ValueError("At least one font size is required.")
    if dataset.exists() and any(dataset.iterdir()):
        raise ValueError(f"Output is nonempty; choose a new dataset directory: {dataset}")
    texts = sorted(set(load_corpus(corpus)))
    if corpus is None:
        texts = [text for text in texts if text not in DIAGNOSTICS]
    random.Random(seed).shuffle(texts)
    available = len(texts) + (len(DIAGNOSTICS) if corpus is None else 0)
    if lines > available:
        raise ValueError(f"Requested {lines} unique lines; corpus only has {len(texts)}. Supply a larger --corpus.")
    texts = texts[:lines - len(DIAGNOSTICS)] + list(DIAGNOSTICS) if corpus is None else texts[:lines]
    assignments = split_texts(texts, seed)
    if corpus is None:
        assign_diagnostics(assignments)
    profiles = load_profiles(profiles_path)
    dataset.mkdir(parents=True, exist_ok=True)
    verify_profiles(profiles, dataset, texts)
    samples: list[Sample] = []
    for text in sorted(texts):
        text_hash = digest(text.encode("utf-8"))
        split = assignments[text]
        for profile in profiles:
            for variant in range(variants):
                identifier = f"{text_hash[:16]}_{profile.name}_{variant}"
                rng = random.Random(f"{seed}:{identifier}")
                image = augment(render(text, profile, rng.choice(tuple(font_sizes))), rng, variant,
                                rotation_deg, blur_range)
                base = dataset / split / identifier
                base.parent.mkdir(parents=True, exist_ok=True)
                image_path, gt_path = base.with_suffix(".png"), base.with_suffix(".gt.txt")
                image.save(image_path, dpi=(300, 300))
                gt_path.write_text(text + "\n", encoding="utf-8", newline="\n")
                source = "diagnostic" if text in DIAGNOSTICS else "text"
                samples.append(Sample(identifier, str(image_path.relative_to(dataset)),
                                      str(gt_path.relative_to(dataset)), split, f"{source}:{text_hash}",
                                      profile.name, profile.category, "synthetic", text_hash,
                                      digest(image_path.read_bytes())))
    save_manifest(dataset, samples)
    write_json(dataset / "generation.json", {"seed": seed, "unique_lines": lines, "variants": variants,
                                             "font_sizes": list(font_sizes),
                                             "profiles": str(profiles_path.resolve()),
                                             "corpus": str(corpus.resolve()) if corpus is not None else "built-in original starter",
                                             "corpus_sha256": digest(corpus.read_bytes()) if corpus is not None else None,
                                             "python": sys.version,
                                             "packages": {p: version(p) for p in ("Pillow", "fonttools", "regex")},
                                             "splits": dict(Counter(s.split for s in samples)),
                                             "target_occurrences": {t: sum(text.count(t) for text in texts)
                                                                    for t in PROBES},
                                             "target_occurrences_by_split": {
                                                 split: {t: sum(text.count(t) for text in texts if assignments[text] == split)
                                                         for t in PROBES} for split in SPLITS},
                                             "diagnostic_lines": list(DIAGNOSTICS) if corpus is None else [],
                                             "diagnostic_note": "Each split contains the full old/new Tamil mapping example set."})
    load_manifest(dataset)
    print(f"Generated {len(samples)} line pairs: {dict(Counter(s.split for s in samples))}")
    print("Synthetic splits hold out text, not fonts. Real-book accuracy is not measured here.")


def import_real(dataset: Path, directory: Path, source: str, split: str) -> None:
    samples = load_manifest(dataset)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", source):
        raise ValueError("--source-id must contain only letters, digits, underscores, and hyphens.")
    if any(s.source_id == f"real:{source}" and s.split != split for s in samples):
        raise ValueError("A real document source cannot span splits.")
    pairs: list[tuple[Path, Path, str]] = []
    for gt in sorted(directory.glob("*.gt.txt")):
        stem = gt.name.removesuffix(".gt.txt")
        images = [directory / (stem + ext) for ext in (".png", ".tif")]
        images = [p for p in images if p.is_file()]
        if len(images) != 1:
            raise ValueError(f"Expected exactly one PNG or TIF for {gt}")
        text = normalized(gt.read_text(encoding="utf-8-sig"))
        text_hash = digest(text.encode("utf-8"))
        image_hash = digest(images[0].read_bytes())
        if any(s.split != split and (s.text_sha256 == text_hash or s.image_sha256 == image_hash) for s in samples):
            raise ValueError(f"Import would leak a transcription or image across splits: {gt}")
        with Image.open(images[0]) as image:
            image.verify()
        identifier = "real_" + source + "_" + digest(stem.encode("utf-8"))[:16]
        if any(s.sample_id == identifier for s in samples):
            raise ValueError(f"Sample already imported: {identifier}")
        pairs.append((images[0], gt, text))
    if not pairs:
        raise ValueError("No *.gt.txt pairs found in import directory.")
    for image, gt, text in pairs:
        stem = gt.name.removesuffix(".gt.txt")
        identifier = "real_" + source + "_" + digest(stem.encode("utf-8"))[:16]
        dest = dataset / split / identifier
        image_path, gt_path = dest.with_suffix(image.suffix), dest.with_suffix(".gt.txt")
        shutil.copyfile(image, image_path)
        gt_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        samples.append(Sample(identifier, str(image_path.relative_to(dataset)), str(gt_path.relative_to(dataset)),
                              split, f"real:{source}", "real", "real", "real", digest(text.encode("utf-8")),
                              digest(image_path.read_bytes())))
    save_manifest(dataset, samples)
    print(f"Imported {len(pairs)} real lines into {split}; keep other pages of this source in the same split.")
