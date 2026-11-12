"""Stage 4 - evaluate: held-out CER/WER/grapheme metrics for baseline vs fine-tuned models."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import unicodedata

import regex

from .common import TARGETS, digest, normalized, run_tool, tool, write_json
from .dataset import load_manifest


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


def accuracy_summary(report: dict[str, object]) -> dict[str, object]:
    """Compact, human-readable accuracy record saved next to a model release."""
    keys = ("cer", "wer", "grapheme_error_rate", "character_accuracy", "grapheme_accuracy",
            "exact_line_accuracy", "lines")
    models = {}
    for label, entry in report["models"].items():
        overall = entry["overall"]
        models[label] = {"model_sha256": entry["sha256"],
                         **{k: overall[k] for k in keys},
                         "target_accuracy": overall["target_accuracy"]}
    return {"evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": report["split"], "scope": report["scope"],
            "accuracy_definition": report["accuracy_definition"],
            "contains_real_test_data": report["contains_real_test_data"],
            "dataset_manifest_sha256": report["manifest_sha256"],
            "models": models}


def evaluate(dataset: Path, base: Path, model: Path, output: Path, split: str = "test",
             psm: int = 7, release_dir: Path | None = None) -> None:
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
        "scope": f"Line recognition with OEM 1 / PSM {psm}; not full-page segmentation accuracy",
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
                            "--oem", "1", "--psm", str(psm)], output / "ocr.log")
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
    if release_dir is not None:
        write_json(release_dir / "accuracy.json", accuracy_summary(report))
        write_json(release_dir / "metrics.json", report)
        print(f"Accuracy saved: {release_dir / 'accuracy.json'}")
    if not report["contains_real_test_data"]:
        print("WARNING: Synthetic-only results do not establish accuracy on historical scans.")
