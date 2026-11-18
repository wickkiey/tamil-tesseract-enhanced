"""Promote the best fine-tuned iteration into a release folder and point the labeller at it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from .common import ROOT, digest, write_json
from .config import Config

METRICS = ("cer", "wer", "grapheme_error_rate", "character_accuracy", "grapheme_accuracy",
           "exact_line_accuracy")
ERROR_METRICS = {"cer", "wer", "grapheme_error_rate"}


def load_history(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"No iterations found in {path}")
    return records


def score(record: dict, metric: str) -> float | None:
    value = record.get(metric)
    if value is None:
        value = (record.get("finetuned") or {}).get(metric)
    return value


def select_best(records: list[dict], metric: str, split: str | None = None) -> dict:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose one of {', '.join(METRICS)}")
    candidates = [record for record in records
                  if score(record, metric) is not None
                  and (split is None or record.get("split") == split)
                  and Path(record["model"]).is_file()]
    if not candidates:
        raise ValueError(f"No iteration with a live model for metric {metric!r} and split {split!r}")
    if metric in ERROR_METRICS:
        return min(candidates, key=lambda r: (score(r, metric), -r.get("labeled_lines", 0), -r.get("iteration", 0)))
    return max(candidates, key=lambda r: (score(r, metric), r.get("labeled_lines", 0), r.get("iteration", 0)))


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    if os.path.splitdrive(resolved)[0].lower() != os.path.splitdrive(ROOT)[0].lower():
        return resolved.as_posix()
    return os.path.relpath(resolved, ROOT).replace(os.sep, "/")


def update_labeling_config(path: Path, model: Path) -> None:
    config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid labelling config: {path}")
    models = dict(config["tesseract_models"]) if isinstance(config.get("tesseract_models"), dict) else {}
    models["finetuned"] = relative_path(model)
    config["tesseract_models"] = models
    config["default_tesseract_model"] = "finetuned"
    write_json(path, config)


def promote(cfg: Config, args: argparse.Namespace) -> None:
    history_path = cfg.optional_path("promote", "history", args.history) or \
        cfg.path("improve", "output") / "accuracy_history.json"
    output = cfg.path("promote", "output", args.output)
    metric = cfg.get("promote", "metric", args.metric)
    labeling_config = cfg.path("promote", "labeling_config", args.labeling_config)
    best = select_best(load_history(history_path), metric, args.split)
    source = Path(best["model"])
    release = output / source.stem
    if release.exists() and any(release.iterdir()) and not args.force:
        raise ValueError(f"Release folder is nonempty: {release}; use --force to overwrite.")
    release.mkdir(parents=True, exist_ok=True)
    model = release / source.name
    shutil.copyfile(source, model)

    evaluation = Path(best.get("evaluation") or source.parent.parent / "evaluation")
    metrics_path = evaluation / "metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    if metrics_path.is_file():
        shutil.copyfile(metrics_path, release / "metrics.json")
    base = Path(best["base_model"])
    write_json(release / "accuracy.json", {key: best.get(key) for key in
                                          ("iteration", "evaluated_at", "split", "baseline", "finetuned",
                                           "delta", "margin", "improved", "labeled_lines")} |
               {"primary_metric": metric, "model_sha256": digest(model.read_bytes())})
    write_json(release / "manifest.json", {
        "release": source.stem,
        "model_file": source.name,
        "language": source.stem,
        "sha256": digest(model.read_bytes()),
        "size_bytes": model.stat().st_size,
        "status": "best-from-improve",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_by": {"metric": metric, "value": score(best, metric), "iteration": best["iteration"],
                        "history": str(history_path.resolve())},
        "base_model": {"path": str(base.resolve()),
                       "sha256": digest(base.read_bytes()) if base.is_file() else None},
        "baseline_model": best.get("baseline_model"),
        "dataset": {"path": best.get("dataset"), "split": best.get("split"),
                    "labeled_lines": best.get("labeled_lines"),
                    "contains_real_test_data": report.get("contains_real_test_data")},
        "evaluation": {"split": best.get("split"), "baseline": best.get("baseline"),
                       "finetuned": best.get("finetuned"), "delta": best.get("delta")},
    })
    print(f"Selected iteration {best['iteration']}: {metric}={score(best, metric):.2%} "
          f"(baseline={best.get('baseline', {}).get(metric)}, "
          f"improved={best.get('improved')})")
    print(f"Released: {model}")
    if not args.no_update_config:
        update_labeling_config(labeling_config, model)
        print(f"Labeller now defaults to: {relative_path(model)} ({labeling_config})")


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main
    return cli_main(["promote", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
