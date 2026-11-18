"""Continuous loop: labels + synthetic data -> fine-tune -> accuracy vs the Tesseract baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

from .common import digest, write_json
from .config import Config
from .dataset import import_real
from .evaluation import evaluate
from .training import train

ITERATION_DIR = re.compile(r"iter-(\d+)$")
PRIMARY_METRIC = "character_accuracy"
METRIC_KEYS = ("cer", "wer", "grapheme_error_rate", "character_accuracy", "grapheme_accuracy",
               "exact_line_accuracy")


@dataclass(frozen=True)
class LabelPair:
    stem: str
    image: Path
    gt: Path
    fingerprint: str


def label_pairs(directory: Path) -> dict[str, LabelPair]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Labels directory missing: {directory}")
    pairs: dict[str, LabelPair] = {}
    for gt in sorted(directory.glob("*.gt.txt")):
        stem = gt.name.removesuffix(".gt.txt")
        images = [directory / (stem + ext) for ext in (".png", ".tif")]
        images = [path for path in images if path.is_file()]
        if len(images) != 1:
            raise ValueError(f"Expected exactly one PNG or TIF for {gt}")
        fingerprint = digest(images[0].read_bytes() + gt.read_bytes())
        pairs[stem] = LabelPair(stem, images[0], gt, fingerprint)
    return pairs


def changed_labels(current: dict[str, LabelPair], previous: dict[str, str]) -> set[str]:
    return {stem for stem, pair in current.items() if previous.get(stem) != pair.fingerprint}


def next_iteration(output: Path) -> int:
    if not output.is_dir():
        return 1
    numbers = [int(match.group(1)) for path in output.iterdir()
               if path.is_dir() and (match := ITERATION_DIR.fullmatch(path.name))]
    return max(numbers, default=0) + 1


def wait_for_labels(directory: Path, previous: dict[str, str], wait_seconds: float,
                    timeout: float) -> tuple[dict[str, LabelPair], set[str]]:
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    while True:
        current = label_pairs(directory)
        changed = changed_labels(current, previous)
        if changed or wait_seconds <= 0 or (deadline is not None and time.monotonic() >= deadline):
            return current, changed
        print(f"Waiting {wait_seconds:g}s for new or edited labels in {directory} ...", flush=True)
        time.sleep(wait_seconds)


def stage_labels(destination: Path, pairs: list[LabelPair] | tuple[LabelPair, ...]) -> Path:
    if destination.exists():
        raise ValueError(f"Staging directory already exists: {destination}")
    destination.mkdir(parents=True)
    for pair in pairs:
        shutil.copyfile(pair.image, destination / pair.image.name)
        shutil.copyfile(pair.gt, destination / pair.gt.name)
    return destination


def comparison(report: dict, margin: float) -> dict:
    models = report["models"]
    scores = {label: {key: models[label]["overall"].get(key) for key in METRIC_KEYS}
              for label in ("baseline", "finetuned")}
    baseline_score = scores["baseline"][PRIMARY_METRIC]
    finetuned_score = scores["finetuned"][PRIMARY_METRIC]
    delta = None if baseline_score is None or finetuned_score is None else finetuned_score - baseline_score
    return {"baseline": scores["baseline"], "finetuned": scores["finetuned"], "delta": delta,
            "margin": margin, "improved": None if delta is None else delta > margin}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def print_summary(iteration: int, label_count: int, split: str, result: dict) -> None:
    print(f"\nIteration {iteration} accuracy on the {split} split ({label_count} labeled lines in training):")
    print(f"  {'model':<10} {'CER':>8} {'WER':>8} {'char acc':>10} {'grapheme acc':>14} {'exact lines':>12}")
    for label in ("baseline", "finetuned"):
        score = result[label]
        print(f"  {label:<10} {_percent(score['cer']):>8} {_percent(score['wer']):>8} "
              f"{_percent(score['character_accuracy']):>10} {_percent(score['grapheme_accuracy']):>14} "
              f"{_percent(score['exact_line_accuracy']):>12}")
    delta = result["delta"]
    delta_text = "n/a" if delta is None else f"{delta:+.2%}"
    print(f"  Delta {PRIMARY_METRIC} (finetuned - baseline): {delta_text} "
          f"(target: > {result['margin']:+.2%})")
    if result["improved"]:
        print("  RESULT: better than the Tesseract baseline.")
    else:
        print("  RESULT: not better than the Tesseract baseline yet.")


def improve(cfg: Config, args: argparse.Namespace) -> None:
    synthetic = cfg.path("improve", "synthetic_dataset", args.synthetic_dataset)
    labels_dir = cfg.path("improve", "labels", args.labels)
    output = cfg.path("improve", "output", args.output)
    baseline_model = cfg.path("paths", "base_model")
    base_override = cfg.optional_path("improve", "base", args.base)
    source = cfg.get("improve", "source_id", args.source_id)
    test_labels = cfg.optional_path("improve", "real_test_labels", args.real_test_labels)
    test_source = cfg.get("improve", "real_test_source_id", args.real_test_source_id)
    name = cfg.get("improve", "model_name", args.model_name)
    iterations = int(cfg.get("improve", "iterations", args.iterations))
    learning_rate = float(cfg.get("improve", "learning_rate", args.learning_rate))
    split = cfg.get("improve", "split", args.split)
    margin = float(cfg.get("improve", "margin", args.margin))
    max_iterations = 1 if args.once else int(cfg.get("improve", "max_iterations", args.max_iterations))
    wait_seconds = 0.0 if args.once else float(cfg.get("improve", "wait_seconds", args.wait_seconds))
    timeout = 0.0 if args.once else float(cfg.get("improve", "timeout", args.timeout))
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and non-negative.")
    if iterations < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Iterations must be >=1 and learning rate finite and positive.")
    if max_iterations < 0 or wait_seconds < 0 or timeout < 0:
        raise ValueError("max-iterations, wait-seconds and timeout must be non-negative.")
    if split not in ("val", "test"):
        raise ValueError("Split must be val or test.")
    if not synthetic.is_dir():
        raise FileNotFoundError(f"Synthetic dataset missing: {synthetic}")

    history_path = output / "accuracy_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    if not isinstance(history, list):
        raise ValueError(f"Invalid accuracy history: {history_path}")
    last = history[-1] if history else {}
    previous = last.get("label_fingerprints", {}) if last.get("labels_dir") == str(labels_dir.resolve()) else {}
    previous_model = Path(last["model"]) if last.get("model") else None
    base_model = previous_model if previous_model is not None and previous_model.is_file() else (base_override or baseline_model)

    start = next_iteration(output)
    iteration = start
    while max_iterations == 0 or iteration < start + max_iterations:
        current = label_pairs(labels_dir)
        changed = changed_labels(current, previous)
        if iteration > start and not changed:
            current, changed = wait_for_labels(labels_dir, previous, wait_seconds, timeout)
            if not changed:
                print("No new or edited labels before the timeout; stopping.", flush=True)
                break
        if not current:
            raise ValueError(f"No labeled PNG/TIF + .gt.txt pairs found in {labels_dir}")

        iteration_dir = output / f"iter-{iteration:03d}"
        if iteration_dir.exists():
            raise ValueError(f"Iteration directory already exists: {iteration_dir}")
        iteration_dir.mkdir(parents=True)
        dataset = iteration_dir / "dataset"
        shutil.copytree(synthetic, dataset)
        train_stage = stage_labels(iteration_dir / "incoming_train", list(current.values()))
        import_real(dataset, train_stage, source, "train")
        if test_labels is not None:
            real_test = label_pairs(test_labels)
            if real_test:
                test_stage = stage_labels(iteration_dir / "incoming_test", list(real_test.values()))
                import_real(dataset, test_stage, test_source, "test")

        model_name = f"{name}_iter{iteration:03d}"
        print(f"\nIteration {iteration}: fine-tuning from {base_model} on "
              f"{len(current)} labeled lines plus synthetic data", flush=True)
        model = train(dataset, iteration_dir / "model", base_model, model_name, iterations, learning_rate,
                      lstm_config=cfg.path("paths", "lstm_config"),
                      target_error_rate=float(cfg.get("train", "target_error_rate")),
                      psm=int(cfg.get("train", "psm")))
        evaluate(dataset, baseline_model, model, iteration_dir / "evaluation", split=split,
                 psm=int(cfg.get("evaluate", "psm")))
        report = json.loads((iteration_dir / "evaluation" / "metrics.json").read_text(encoding="utf-8"))
        result = comparison(report, margin)
        history.append({"iteration": iteration,
                        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "model": str(model.resolve()),
                        "base_model": str(base_model.resolve()),
                        "baseline_model": str(baseline_model.resolve()),
                        "dataset": str(dataset.resolve()),
                        "evaluation": str((iteration_dir / "evaluation").resolve()),
                        "split": split,
                        "labels_dir": str(labels_dir.resolve()),
                        "labeled_lines": len(current),
                        "new_or_edited_labels": sorted(changed),
                        "label_fingerprints": {stem: pair.fingerprint for stem, pair in current.items()},
                        "primary_metric": PRIMARY_METRIC,
                        **result})
        write_json(history_path, history)
        print_summary(iteration, len(current), split, result)
        print(f"Accuracy history: {history_path}", flush=True)
        if result["improved"]:
            print(f"Target reached: {model} beats the Tesseract baseline by "
                  f"{result['delta']:+.2%} {PRIMARY_METRIC}.", flush=True)
            break
        base_model = model
        previous = {stem: pair.fingerprint for stem, pair in current.items()}
        iteration += 1


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main
    return cli_main(["improve", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
