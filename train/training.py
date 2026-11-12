"""Stage 3 - train: native Tesseract LSTM fine-tuning from a base traineddata."""

from __future__ import annotations

import math
from pathlib import Path
import random
import re
import unicodedata

from PIL import Image

from .common import SPLITS, TRAIN_DIR, digest, normalized, run_tool, tool, write_json
from .dataset import load_manifest

DEFAULT_LSTM_CONFIG = TRAIN_DIR / "lstm.train"


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
          learning_rate: float, lstm_config: Path = DEFAULT_LSTM_CONFIG,
          target_error_rate: float = 0.01, psm: int = 13) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9_]+", name) or iterations < 1:
        raise ValueError("Invalid model name or iteration count.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if not lstm_config.is_file():
        raise FileNotFoundError(f"Tesseract LSTM training config missing: {lstm_config}")
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
        run_tool([tesseract, str(image_path), str(image_path.with_suffix("")), "--psm", str(psm),
                  "--tessdata-dir", str(base.resolve().parent), "-l", base.stem,
                  str(lstm_config.resolve())], log)
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
              "--target_error_rate", str(target_error_rate), "--debug_interval", "0"], log)
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
                                        "target_error_rate": target_error_rate, "psm": psm,
                                        "selection": "Minimum lstmeval BCER on validation lines; ties use filename",
                                        "selected_checkpoint": str(checkpoint.resolve()),
                                        "validation_bcer_percent": score,
                                        "checkpoint_scores": {str(p): v for v, p in scores},
                                        "test_used_for_training": False})
    print(f"Model exported: {model}")
    return model
