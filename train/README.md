# Training pipeline (`train/`)

Everything needed to build a Tamil line-image dataset, fine-tune a Tesseract LSTM
model and measure accuracy lives in this package. Run every command from the
repository root as `python -m train <stage>`.

| Stage | Command | What it does |
|---|---|---|
| 1. [Setup](#1-setup) | `python -m train setup` | Download fonts and the base Tamil model |
| 2. [Generate data](#2-generate-data) | `python -m train generate` | Render synthetic line images + ground truth |
| 3. [Import real data](#3-import-real-data-optional) | `python -m train import-real` | Add real scanned lines (optional) |
| 4. [Train](#4-train) | `python -m train train` | Fine-tune the base model |
| 5. [Evaluate](#5-evaluate) | `python -m train evaluate` | Measure accuracy and save it to `models/<name>/` |
| All | `python -m train run` | Stages 1, 2, 4 and 5 in one go |

## Configuration

All settings are in [`config.toml`](config.toml). Edit it, then run a stage. A
command-line flag overrides the matching config value:

```bash
python -m train generate --lines 500        # overrides [generate] lines
python -m train --config my.toml train      # use a different config file
```

Relative paths in the config are resolved against the repository root. The config
needs Python 3.11 or later.

| Section | Keys |
|---|---|
| `[paths]` | `assets_dir`, `profiles`, `lstm_config`, `base_model`, `models_dir` |
| `[generate]` | `dataset`, `corpus`, `lines`, `variants`, `seed`, `font_sizes` |
| `[augment]` | `rotation_deg`, `blur_min`, `blur_max` |
| `[import_real]` | `dataset`, `directory`, `source_id`, `split` |
| `[train]` | `dataset`, `output`, `model_name`, `iterations`, `learning_rate`, `target_error_rate`, `psm` |
| `[evaluate]` | `dataset`, `model`, `output`, `split`, `psm`, `name` |
| `[run]` | `output` |

## Requirements

```bash
python -m pip install -r requirements.txt
```

Training and evaluation also need **Tesseract 5 with the training tools**
(`tesseract`, `lstmtraining`, `lstmeval`, `combine_tessdata`) on your `PATH`. Pillow
must be built with RAQM/HarfBuzz so Tamil shaping works.

## Files in this package

| File | Purpose |
|---|---|
| `config.toml` | The one configuration file |
| `profiles.json` | Font profiles: font file, OpenType features, category |
| `lstm.train` | Tesseract config used to build `.lstmf` training files |
| `setup_assets.py` | Stage 1 |
| `corpus.py`, `profiles.py`, `dataset.py` | Stages 2 and 3 |
| `training.py` | Stage 4 |
| `evaluation.py` | Stage 5 |
| `cli.py`, `config.py`, `common.py` | Argument parsing, config loading, shared helpers |

---

## 1. Setup

Downloads the OFL-licensed fonts (Noto Sans/Serif Tamil, Korkai, Vaagai) and the
Tamil base model `tam.traineddata` into `assets_dir`, with licences and a
`provenance.json` of URLs and hashes. Existing files are not downloaded again.

```bash
python -m train setup
```

## 2. Generate data

Renders each corpus line with every font profile in `profiles.json` and writes
`<id>.png` + `<id>.gt.txt` pairs into `train/`, `val/` and `test/` (about 80/10/10,
split by text so a sentence never appears in two splits).

**Corpus format:** a UTF-8 text file, one Tamil line per row, at least 10 unique
lines. Leave `corpus = ""` to use the built-in starter corpus.

| Setting | Meaning |
|---|---|
| `lines` | Unique corpus lines to use (10 or more, no more than the corpus has) |
| `variants` | Images per line per profile. `0` is clean; later ones are rotated, blurred and noisy |
| `font_sizes` | Sizes picked at random (seeded) for each image |
| `seed` | Makes the dataset reproducible |

Number of images = `lines` × number of profiles × `variants`.

```bash
python -m train generate
python -m train generate --corpus my_corpus.txt --lines 200 --variants 2 --dataset runs/my-data/dataset
```

The dataset folder must be new or empty. It also gets `manifest.json` (hashes and
splits), `generation.json` (settings and target-cluster counts) and probe sheets
showing the old-style glyph forms.

## 3. Import real data (optional)

Adds real scanned lines to an existing dataset. Put image and transcription pairs in
one folder as `NAME.png` (or `.tif`) + `NAME.gt.txt`, one line each.

```bash
python -m train import-real --dataset runs/my-data/dataset \
  --directory path/to/book001 --source-id book001 --split train
```

Keep all pages of one source in the same split. Imports that would leak a
transcription or image across splits are rejected.

## 4. Train

Fine-tunes `base_model` with `lstmtraining` on the train split, checks every
checkpoint on the validation split, keeps the one with the lowest validation
character error rate, and exports `<model_name>.traineddata`. The test split is never
used for training.

```bash
python -m train train
python -m train train --dataset runs/my-data/dataset --output runs/my-data/model \
  --iterations 5000 --model-name tam_historical
```

The output folder must be new or empty. It holds the model, `training.json`
(settings, hashes, selected checkpoint) and `training.log`.

## 5. Evaluate

Runs the base model and the fine-tuned model on the held-out split and reports
character error rate (CER), word error rate (WER), grapheme accuracy and exact-line
accuracy, plus accuracy for the target clusters (ai, nedil, pulli).

```bash
python -m train evaluate
python -m train evaluate --dataset runs/my-data/dataset \
  --model runs/my-data/model/tam_historical.traineddata \
  --output runs/my-data/evaluation --name tam_historical
```

Results are written twice:

- `output` folder: `metrics.json` (full detail) and `predictions.json`.
- **`models/<name>/`**: `accuracy.json` (compact baseline vs fine-tuned summary) and
  a copy of `metrics.json`. `<name>` is `--name` / `[evaluate] name`, or the model
  file name if that is empty. For `run` it is `[train] model_name`.

Re-evaluating with the same name overwrites that folder's `accuracy.json` and
`metrics.json`. It never touches a model file or an existing `manifest.json`.

Synthetic-only results do not prove accuracy on real historical scans. Import real
held-out pages (stage 3) before relying on a model.

## Run everything

```bash
python -m train run --output runs/experiment01
```

Runs setup, generate, train and evaluate with the values in `config.toml`. The
dataset, model and evaluation go under the output folder, and the accuracy is
saved to `models/<model_name>/`.

## Tests

```bash
python -m unittest discover -s tests
```
