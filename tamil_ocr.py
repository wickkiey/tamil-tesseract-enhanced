"""Synthetic Tamil line data, native Tesseract fine-tuning, and held-out evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
import zipfile

from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps, features
import regex


ROOT = Path(__file__).resolve().parent
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
    "ணா றா னா ணை லை ளை னை ணொ றொ னொ ணோ றோ னோ.",
    "ணோ றோ னோ ணொ றொ னொ ணை லை ளை னை ணா றா னா.",
    "ணை லை ளை னை ணா றா னா ணோ றோ னோ ணொ றொ னொ.",
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


def setup() -> None:
    fonts = ROOT / "assets" / "fonts"
    records: list[dict[str, str]] = []
    for family in ("Sans", "Serif"):
        folder = f"noto{family.lower()}tamil"
        url = (f"https://raw.githubusercontent.com/google/fonts/main/ofl/{folder}/"
               f"Noto{family}Tamil%5Bwdth%2Cwght%5D.ttf")
        path = fonts / f"Noto{family}Tamil.ttf"
        download(url, path)
        download(f"https://raw.githubusercontent.com/google/fonts/main/ofl/{folder}/OFL.txt",
                 fonts / f"Noto{family}Tamil-OFL.txt")
        records.append({"file": str(path.relative_to(ROOT)), "url": url,
                        "sha256": digest(path.read_bytes()), "license": "SIL OFL 1.1"})
    for family, version in (("Korkai", "1_0_0_3"), ("Vaagai", "1_0_0_0")):
        url = f"https://tamilfontstudio.net/downloads/{family.lower()}_v{version}.zip"
        archive = fonts / f"{family.lower()}.zip"
        download(url, archive)
        with zipfile.ZipFile(archive) as package:
            for name, output in ((f"{family}-Regular.ttf", f"{family}-Regular.ttf"),
                                 ("OFL.txt", f"{family}-OFL.txt")):
                matches = [p for p in package.namelist() if Path(p).name == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in {archive}, found {matches}")
                (fonts / output).write_bytes(package.read(matches[0]))
        path = fonts / f"{family}-Regular.ttf"
        records.append({"file": str(path.relative_to(ROOT)), "url": url,
                        "sha256": digest(path.read_bytes()), "license": "SIL OFL 1.1"})
    model = ROOT / "assets" / "models" / "tam.traineddata"
    url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/tam.traineddata"
    download(url, model)
    records.append({"file": str(model.relative_to(ROOT)), "url": url,
                    "sha256": digest(model.read_bytes()), "license": "Apache 2.0"})
    download("https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/LICENSE",
             model.parent / "LICENSE")
    write_json(ROOT / "assets" / "provenance.json", records)
    print("Assets ready. Fonts are OFL-licensed; Tamil base model is from tessdata_best.")


@dataclass(frozen=True)
class Profile:
    name: str
    font: Path
    features: tuple[str, ...]
    category: str


def load_profiles(path: Path) -> list[Profile]:
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
            font_path = path.resolve().parent / font_path
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


def starter_corpus() -> list[str]:
    subjects = (
        "மாலை நேரத்தில்", "காலை நேரத்தில்", "மழை நின்ற பிறகு", "விழா முடிந்த பிறகு",
        "பள்ளி திறந்த போது", "பழைய நூலகத்தில்", "கடற்கரை அருகில்", "திண்ணை அருகில்",
        "அறையில் அமர்ந்து", "நாளை நடைபெறும் விழாவில்", "மலை அடிவாரத்தில்", "நகரின் எல்லையில்",
    )
    actors = ("மாணவன்", "மாணவி", "ஆசிரியர்", "கலைஞர்", "நண்பன்", "அண்ணன்",
              "தந்தை", "தாய்", "இளைஞன்", "கவிஞர்", "விருந்தினர்", "சிறுவன்")
    actions = ("பழைய நூலை வாசித்தார்", "கதையை எடுத்துரைத்தார்", "நல்ல செய்தியை பகிர்ந்தார்",
               "புதிய பாடலை பாடினார்", "கலை நிகழ்ச்சியை பார்த்தார்", "பணிகளை முடித்தார்",
               "கடிதத்தை எழுதினார்", "மணல் சிற்பத்தை பார்த்தார்", "விளக்கை ஏற்றினார்",
               "துணிகளை அடுக்கினார்", "எழுத்து வடிவங்களை ஆராய்ந்தார்", "மரங்களை நட்டார்")
    sentences = [f"{s} {a} {v}." for s in subjects for a in actors for v in actions]
    return sentences + list(DIAGNOSTICS)


def split_texts(texts: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(normalized(t) for t in texts))
    if len(unique) < 10:
        raise ValueError("At least 10 unique corpus lines are required for train/validation/test splits.")
    random.Random(seed).shuffle(unique)
    n_val = n_test = max(1, len(unique) // 10)
    n_train = len(unique) - n_val - n_test
    return {t: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, t in enumerate(unique)}


def assign_diagnostics(assignments: dict[str, str]) -> None:
    fixed: set[str] = set()
    for text, split in zip(DIAGNOSTICS, SPLITS):
        if assignments[text] != split:
            replacement = next(t for t, s in assignments.items() if s == split and t not in fixed)
            assignments[replacement], assignments[text] = assignments[text], split
        fixed.add(text)


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


def augment(image: Image.Image, rng: random.Random, variant: int) -> Image.Image:
    if variant == 0:
        return image
    image = image.rotate(rng.uniform(-0.6, 0.6), resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.45)))
    pixels = bytearray(image.tobytes())
    for _ in range(max(1, len(pixels) // 1200)):
        pos = rng.randrange(len(pixels))
        pixels[pos] = rng.randrange(100, 220)
    return Image.frombytes("L", image.size, bytes(pixels))


def generate(dataset: Path, corpus: Path | None, profiles_path: Path, lines: int,
             variants: int, seed: int) -> None:
    if lines < 10 or variants < 1:
        raise ValueError("--lines must be >=10 and --variants >=1.")
    if dataset.exists() and any(dataset.iterdir()):
        raise ValueError(f"Output is nonempty; choose a new dataset directory: {dataset}")
    texts = starter_corpus() if corpus is None else [
        normalized(t) for t in corpus.read_text(encoding="utf-8-sig").splitlines() if t.strip()
    ]
    texts = sorted(set(texts))
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
                image = augment(render(text, profile, rng.choice((40, 48, 56))), rng, variant)
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


def line_box(text: str, width: int, height: int) -> str:
    # LSTM line boxes repeat whole-line bounds; the tab marks end of line.
    units: list[str] = []
    for char in text:
        if unicodedata.combining(char) and units:
            units[-1] += char
        else:
            units.append(char)
    return "".join(f"{unit} 0 0 {width} {height} 0\n" for unit in units + ["\t"])


def train(dataset: Path, output: Path, base: Path, name: str, iterations: int,
          learning_rate: float) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name) or iterations < 1:
        raise ValueError("Invalid model name or iteration count.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    samples = load_manifest(dataset)
    if any(not any(s.split == split for s in samples) for split in SPLITS):
        raise ValueError("Train, validation, and untouched test samples are all required.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Training output is nonempty; choose a new directory: {output}")
    if not base.is_file():
        raise FileNotFoundError(f"Base model missing: {base}. Run setup.")
    tesseract, trainer, combine, evaluator = (
        tool(t) for t in ("tesseract", "lstmtraining", "combine_tessdata", "lstmeval"))
    output.mkdir(parents=True, exist_ok=True)
    log = output / "training.log"
    prefix = output.resolve() / "base."
    run_tool([combine, "-u", str(base.resolve()), str(prefix)], log)
    network = output / "base.lstm"
    charset = output / "base.lstm-unicharset"
    if not network.is_file() or not charset.is_file():
        raise RuntimeError("Base model must contain an LSTM network and LSTM unicharset.")
    supported = {c for line in charset.read_text(encoding="utf-8").splitlines()[1:]
                 for c in line.split(" ", 1)[0]} | {" "}
    lists: dict[str, list[str]] = {"train": [], "val": []}
    for sample in samples:
        if sample.split == "test":
            continue
        image_path = (dataset / sample.image).resolve()
        text = normalized((dataset / sample.gt).read_text(encoding="utf-8-sig"))
        missing = sorted(set(text) - supported)
        if missing:
            raise ValueError(f"Base Tamil charset lacks {missing!r} in {sample.sample_id}; charset expansion is required.")
        with Image.open(image_path) as image:
            width, height = image.size
        image_path.with_suffix(".box").write_text(line_box(text, width, height), encoding="utf-8", newline="\n")
        run_tool([tesseract, str(image_path), str(image_path.with_suffix("")), "--psm", "13",
                  "--tessdata-dir", str(base.resolve().parent), "-l", base.stem,
                  str(ROOT / "configs" / "lstm.train")], log)
        lstmf = image_path.with_suffix(".lstmf")
        if not lstmf.is_file() or not lstmf.stat().st_size:
            raise RuntimeError(f"Tesseract did not produce training data for {sample.sample_id}")
        lists[sample.split].append(str(lstmf))
    for split, paths in lists.items():
        random.Random(42).shuffle(paths)
        # Native lstmtraining treats a CR in a list entry as part of the filename.
        (output / f"{split}.list").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
    model_prefix = output.resolve() / name
    print(f"Fine-tuning {len(lists['train'])} lines, validating on {len(lists['val'])}; max_iterations={iterations}")
    run_tool([trainer, "--continue_from", str(network.resolve()), "--traineddata", str(base.resolve()),
              "--model_output", str(model_prefix), "--train_listfile", str((output / "train.list").resolve()),
              "--eval_listfile", str((output / "val.list").resolve()), "--max_iterations", str(iterations),
              "--learning_rate", str(learning_rate), "--reset_learning_rate",
              "--target_error_rate", "0.01", "--debug_interval", "0"], log)
    checkpoint = Path(str(model_prefix) + "_checkpoint")
    candidates = sorted(output.glob(name + "_*.checkpoint"))
    if checkpoint.is_file():
        candidates.append(checkpoint)
    if not candidates:
        raise RuntimeError(f"No checkpoint produced. See {log}.")
    scores: list[tuple[float, Path]] = []
    for candidate in candidates:
        result = run_tool([evaluator, "--model", str(candidate.resolve()), "--traineddata", str(base.resolve()),
                           "--eval_listfile", str((output / "val.list").resolve()), "--verbosity", "0"],
                          log, include_stderr=True)
        match = re.search(r"BCER eval=([0-9]+(?:\.[0-9]+)?)", result)
        if match is None:
            raise RuntimeError(f"Could not parse validation score for {candidate}:\n{result}")
        scores.append((float(match.group(1)), candidate))
    score, checkpoint = min(scores, key=lambda item: (item[0], str(item[1])))
    print(f"Selected checkpoint by validation BCER: {score:.3f}% (not test accuracy)")
    model = output / f"{name}.traineddata"
    run_tool([trainer, "--stop_training", "--continue_from", str(checkpoint),
              "--traineddata", str(base.resolve()), "--model_output", str(model.resolve())], log)
    if not model.is_file() or not model.stat().st_size:
        raise RuntimeError("Training did not export a runtime model.")
    write_json(output / "training.json", {"base": str(base.resolve()), "base_sha256": digest(base.read_bytes()),
                                        "dataset": str(dataset.resolve()), "model": str(model.resolve()),
                                        "manifest_sha256": digest((dataset / "manifest.json").read_bytes()),
                                        "tesseract_version": run_tool([tesseract, "--version"], include_stderr=True).strip(),
                                        "iterations_requested": iterations, "learning_rate": learning_rate,
                                        "selection": "Minimum lstmeval BCER on validation lines; ties use filename",
                                        "selected_checkpoint": str(checkpoint.resolve()),
                                        "validation_bcer_percent": score,
                                        "checkpoint_scores": {str(p): v for v, p in scores},
                                        "test_used_for_training": False})
    print(f"Model exported: {model}")
    return model


def alignment(reference: list[str], hypothesis: list[str]) -> tuple[int, list[tuple[str | None, str | None]]]:
    rows = [list(range(len(hypothesis) + 1))]
    for i, a in enumerate(reference, 1):
        row = [i]
        for j, b in enumerate(hypothesis, 1):
            row.append(min(rows[i - 1][j] + 1, row[j - 1] + 1, rows[i - 1][j - 1] + (a != b)))
        rows.append(row)
    i, j = len(reference), len(hypothesis)
    pairs: list[tuple[str | None, str | None]] = []
    while i or j:
        if i and j and rows[i][j] == rows[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j - 1]))
            j -= 1
    return rows[-1][-1], list(reversed(pairs))


def metric_summary(pairs: list[tuple[str, str]]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    targets: dict[str, Counter[str]] = defaultdict(Counter)
    confusions: Counter[tuple[str, str]] = Counter()
    for reference, hypothesis in pairs:
        counts["lines"] += 1
        counts["exact_lines"] += reference == hypothesis
        for label, a, b in (("codepoint", list(reference), list(hypothesis)),
                            ("word", reference.split(), hypothesis.split()),
                            ("grapheme", regex.findall(r"\X", reference), regex.findall(r"\X", hypothesis))):
            edits, aligned = alignment(a, b)
            counts[label + "_reference"] += len(a)
            counts[label + "_edits"] += edits
            if label == "grapheme":
                for expected, actual in aligned:
                    if expected != actual:
                        confusions[(expected if expected is not None else "<insert>",
                                    actual if actual is not None else "<delete>")] += 1
                    if expected is not None:
                        groups = [k for k, values in TARGETS.items() if expected in values]
                        if expected.endswith("்"):
                            groups.append("pulli")
                        for group in groups:
                            for key in (group, "cluster:" + expected):
                                targets[key]["reference"] += 1
                                targets[key]["correct"] += expected == actual
    rates = {label: counts[label + "_edits"] / counts[label + "_reference"]
             if counts[label + "_reference"] else None for label in ("codepoint", "word", "grapheme")}
    return {"lines": counts["lines"], "cer": rates["codepoint"], "wer": rates["word"],
            "grapheme_error_rate": rates["grapheme"],
            "character_accuracy": max(0.0, 1 - rates["codepoint"]) if rates["codepoint"] is not None else None,
            "grapheme_accuracy": max(0.0, 1 - rates["grapheme"]) if rates["grapheme"] is not None else None,
            "exact_line_accuracy": counts["exact_lines"] / counts["lines"] if counts["lines"] else None,
            "counts": dict(counts),
            "target_accuracy": {k: {"reference": v["reference"], "correct": v["correct"],
                                    "accuracy": v["correct"] / v["reference"]} for k, v in sorted(targets.items())},
            "top_confusions": [{"expected": a, "predicted": b, "count": n}
                               for (a, b), n in confusions.most_common(30)]}


def evaluate(dataset: Path, base: Path, model: Path, output: Path, split: str = "test") -> None:
    samples = [s for s in load_manifest(dataset) if s.split == split]
    if not samples:
        raise ValueError(f"No {split} samples.")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Evaluation output is nonempty: {output}")
    for path in (base, model):
        if not path.is_file():
            raise FileNotFoundError(path)
    tesseract = tool("tesseract")
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, str]] = []
    report: dict[str, object] = {
        "split": split, "normalization": "NFC; trim outer whitespace; preserve internal whitespace",
        "dataset": str(dataset.resolve()), "manifest_sha256": digest((dataset / "manifest.json").read_bytes()),
        "accuracy_definition": "max(0, 1 - edit_error_rate); error rates may exceed 1",
        "scope": "Line recognition with OEM 1 / PSM 7; not full-page segmentation accuracy",
        "contains_real_test_data": any(s.kind == "real" for s in samples),
        "models": {},
    }
    models_report: dict[str, object] = {}
    for label, path in (("baseline", base), ("finetuned", model)):
        all_pairs: list[tuple[str, str]] = []
        grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for sample in samples:
            reference = normalized((dataset / sample.gt).read_text(encoding="utf-8-sig"))
            raw = run_tool([tesseract, str((dataset / sample.image).resolve()), "stdout",
                            "--tessdata-dir", str(path.resolve().parent), "-l", path.stem,
                            "--oem", "1", "--psm", "7"], output / "ocr.log")
            hypothesis = unicodedata.normalize("NFC", raw.strip())
            pair = (reference, hypothesis)
            all_pairs.append(pair)
            content = "diagnostic" if sample.source_id.startswith("diagnostic:") else "text"
            for group in ("profile:" + sample.profile, "category:" + sample.category,
                          "kind:" + sample.kind, "content:" + content,
                          "source:" + sample.source_id if sample.kind == "real" else "source:synthetic"):
                grouped[group].append(pair)
            predictions.append({"model": label, "sample_id": sample.sample_id,
                                "reference": reference, "prediction": hypothesis, "profile": sample.profile})
        summary = metric_summary(all_pairs)
        models_report[label] = {"path": str(path.resolve()), "sha256": digest(path.read_bytes()),
                                "overall": summary, "by_group": {k: metric_summary(v) for k, v in sorted(grouped.items())}}
        print(f"{label:10}  CER={summary['cer']:.2%}  WER={summary['wer']:.2%}  "
              f"character accuracy={summary['character_accuracy']:.2%}  "
              f"grapheme accuracy={summary['grapheme_accuracy']:.2%}  "
              f"exact lines={summary['exact_line_accuracy']:.2%}")
        target_scores = summary["target_accuracy"]
        for group in ("ai", "nedil", "pulli"):
            target = target_scores.get(group)
            if target is None:
                print(f"  {group}: no reference occurrences; accuracy not measurable")
            else:
                print(f"  {group}: {target['accuracy']:.2%} ({target['correct']}/{target['reference']} reference clusters)")
    report["models"] = models_report
    write_json(output / "metrics.json", report)
    write_json(output / "predictions.json", predictions)
    print(f"Metrics saved: {output / 'metrics.json'}")
    if not report["contains_real_test_data"]:
        print("WARNING: Synthetic-only results do not establish accuracy on historical scans.")


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Download licensed fonts and floating-point Tamil model")
    gen = sub.add_parser("generate", help="Generate shaped PNG / UTF-8 ground-truth pairs")
    full = sub.add_parser("run", help="Setup, generate, train, compare baseline and fine-tuned model")
    for p in (gen, full):
        p.add_argument("--corpus", type=Path)
        p.add_argument("--profiles", type=Path, default=ROOT / "profiles.json")
        p.add_argument("--lines", type=int, default=100)
        p.add_argument("--variants", type=int, default=2)
        p.add_argument("--seed", type=int, default=42)
    gen.add_argument("--dataset", type=Path, required=True)
    full.add_argument("--output", type=Path, required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--dataset", type=Path, required=True)
    tr.add_argument("--output", type=Path, required=True)
    for p in (tr, full):
        p.add_argument("--base", type=Path, default=ROOT / "assets" / "models" / "tam.traineddata")
        p.add_argument("--model-name", default="tam_historical")
        p.add_argument("--iterations", type=int, default=1000)
        p.add_argument("--learning-rate", type=float, default=0.0001)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--dataset", type=Path, required=True)
    ev.add_argument("--base", type=Path, default=ROOT / "assets" / "models" / "tam.traineddata")
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--split", choices=("val", "test"), default="test")
    imp = sub.add_parser("import-real")
    imp.add_argument("--dataset", type=Path, required=True)
    imp.add_argument("--directory", type=Path, required=True)
    imp.add_argument("--source-id", required=True)
    imp.add_argument("--split", choices=SPLITS, required=True)
    return cli


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "setup":
            setup()
        elif args.command == "generate":
            generate(args.dataset, args.corpus, args.profiles, args.lines, args.variants, args.seed)
        elif args.command == "import-real":
            import_real(args.dataset, args.directory, args.source_id, args.split)
        elif args.command == "train":
            train(args.dataset, args.output, args.base, args.model_name, args.iterations, args.learning_rate)
        elif args.command == "evaluate":
            evaluate(args.dataset, args.base, args.model, args.output, args.split)
        elif args.command == "run":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("Run output is nonempty. Choose a new --output path.")
            setup()
            dataset = args.output / "dataset"
            generate(dataset, args.corpus, args.profiles, args.lines, args.variants, args.seed)
            model = train(dataset, args.output / "model", args.base, args.model_name,
                          args.iterations, args.learning_rate)
            evaluate(dataset, args.base, model, args.output / "evaluation")
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
