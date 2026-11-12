"""Command-line entry point: ``python -m train <stage>``. Values come from config.toml."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

from .config import DEFAULT_CONFIG, Config
from .dataset import generate, import_real
from .evaluation import evaluate
from .setup_assets import setup
from .training import train


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        prog="python -m train",
        description="Synthetic Tamil line data, native Tesseract fine-tuning, and held-out evaluation. "
                    "Defaults come from the config file; flags override it.")
    cli.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                     help=f"TOML config file (default: {DEFAULT_CONFIG})")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Download licensed fonts and floating-point Tamil model")
    gen = sub.add_parser("generate", help="Generate shaped PNG / UTF-8 ground-truth pairs")
    full = sub.add_parser("run", help="Setup, generate, train, compare baseline and fine-tuned model")
    for p in (gen, full):
        p.add_argument("--corpus", type=Path)
        p.add_argument("--profiles", type=Path)
        p.add_argument("--lines", type=int)
        p.add_argument("--variants", type=int)
        p.add_argument("--seed", type=int)
    gen.add_argument("--dataset", type=Path)
    full.add_argument("--output", type=Path)
    tr = sub.add_parser("train", help="Fine-tune the base model on the dataset")
    tr.add_argument("--dataset", type=Path)
    tr.add_argument("--output", type=Path)
    for p in (tr, full):
        p.add_argument("--base", type=Path)
        p.add_argument("--model-name")
        p.add_argument("--iterations", type=int)
        p.add_argument("--learning-rate", type=float)
    ev = sub.add_parser("evaluate", help="Compare baseline and fine-tuned model on held-out lines")
    ev.add_argument("--dataset", type=Path)
    ev.add_argument("--base", type=Path)
    ev.add_argument("--model", type=Path)
    ev.add_argument("--output", type=Path)
    ev.add_argument("--split", choices=("val", "test"))
    ev.add_argument("--name", help="Save results to <models_dir>/<name>/ (default: the model file name)")
    imp = sub.add_parser("import-real", help="Add real image + .gt.txt pairs to a dataset")
    imp.add_argument("--dataset", type=Path)
    imp.add_argument("--directory", type=Path)
    imp.add_argument("--source-id")
    imp.add_argument("--split", choices=("train", "val", "test"))
    return cli


def _generate(cfg: Config, args: argparse.Namespace, dataset: Path) -> None:
    sizes = cfg.get("generate", "font_sizes")
    generate(dataset,
             cfg.optional_path("generate", "corpus", args.corpus),
             cfg.path("paths", "profiles", args.profiles),
             cfg.get("generate", "lines", args.lines),
             cfg.get("generate", "variants", args.variants),
             cfg.get("generate", "seed", args.seed),
             font_sizes=tuple(int(s) for s in sizes),
             rotation_deg=float(cfg.get("augment", "rotation_deg")),
             blur_range=(float(cfg.get("augment", "blur_min")), float(cfg.get("augment", "blur_max"))))


def _train(cfg: Config, args: argparse.Namespace, dataset: Path, output: Path) -> Path:
    return train(dataset, output,
                 cfg.path("paths", "base_model", args.base),
                 cfg.get("train", "model_name", args.model_name),
                 cfg.get("train", "iterations", args.iterations),
                 cfg.get("train", "learning_rate", args.learning_rate),
                 lstm_config=cfg.path("paths", "lstm_config"),
                 target_error_rate=float(cfg.get("train", "target_error_rate")),
                 psm=int(cfg.get("train", "psm")))


def _release_dir(cfg: Config, name: str) -> Path:
    return cfg.path("paths", "models_dir") / name


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        cfg = Config.load(args.config)
        if args.command == "setup":
            setup(cfg.path("paths", "assets_dir"))
        elif args.command == "generate":
            _generate(cfg, args, cfg.path("generate", "dataset", args.dataset))
        elif args.command == "import-real":
            import_real(cfg.path("import_real", "dataset", args.dataset),
                        cfg.path("import_real", "directory", args.directory),
                        cfg.get("import_real", "source_id", args.source_id),
                        cfg.get("import_real", "split", args.split))
        elif args.command == "train":
            _train(cfg, args, cfg.path("train", "dataset", args.dataset), cfg.path("train", "output", args.output))
        elif args.command == "evaluate":
            evaluate(cfg.path("evaluate", "dataset", args.dataset),
                     cfg.path("paths", "base_model", args.base),
                     cfg.path("evaluate", "model", args.model),
                     cfg.path("evaluate", "output", args.output),
                     cfg.get("evaluate", "split", args.split),
                     psm=int(cfg.get("evaluate", "psm")),
                     release_dir=_release_dir(cfg, cfg.get("evaluate", "name", args.name)
                                              or cfg.path("evaluate", "model", args.model).stem))
        elif args.command == "run":
            output = cfg.path("run", "output", args.output)
            if output.exists() and any(output.iterdir()):
                raise ValueError("Run output is nonempty. Choose a new --output path.")
            setup(cfg.path("paths", "assets_dir"))
            dataset = output / "dataset"
            _generate(cfg, args, dataset)
            model = _train(cfg, args, dataset, output / "model")
            evaluate(dataset, cfg.path("paths", "base_model", args.base), model, output / "evaluation",
                     psm=int(cfg.get("evaluate", "psm")),
                     release_dir=_release_dir(cfg, cfg.get("train", "model_name", args.model_name)))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
