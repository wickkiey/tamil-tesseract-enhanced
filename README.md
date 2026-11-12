# Tamil typography dataset and Tesseract fine-tuning

**Detailed workflow:** [Adding datasets, training, model export and inference examples](docs/DATASET_TRAINING_INFERENCE.md).

A native-Windows Python CLI that downloads licensed fonts, renders synthetic
Tamil text lines, generates Tesseract training files, fine-tunes the floating-point
Tamil LSTM model, selects a checkpoint using validation BCER, and prints baseline
versus fine-tuned accuracy on an untouched test set.

## Code layout and configuration

All pipeline code is in [`train/`](train/), with one module per stage and a single
config file, [`train/config.toml`](train/config.toml). **See
[train/README.md](train/README.md)** for a short run section for each stage: setup,
generate data, import real data, train and evaluate.

```bash
python -m train generate     # reads train/config.toml
python -m train train
python -m train evaluate     # saves accuracy to models/<name>/
```

Command-line flags override the config (`python -m train generate --lines 500`).

## What the dataset covers

Six rendering profiles are provided in [profiles.json](train/profiles.json):

| Profile | Typeface and shaping | Purpose |
|---|---|---|
| modern_sans | Noto Sans Tamil | Modern sans-serif |
| modern_serif | Noto Serif Tamil | Modern serif |
| korkai_modern | Korkai, stylistic sets disabled | Modern comparison |
| korkai_historical | Korkai, `ss01`, `ss02`, `ss03` enabled | Traditional ai, long-vowel and other alternate forms |
| vaagai_archival | Vaagai, stylistic sets disabled | Archival woodcut revival |
| vaagai_alternates | Vaagai, `ss01`, `ss02` enabled | Additional historical alternates |

The downloaded Noto fonts do **not** have historical stylistic sets. The pipeline
does not pretend that applying `ss01` to them creates old Tamil.

Korkai is a publishing revival based on metal type; Vaagai revives early Tamil
woodcut designs from 1714 and 1723. These are useful historical-type proxies, not
a claim that every old book uses precisely these forms.

Before rendering, the pipeline checks Unicode cmap coverage and enabled GSUB
features. It then compares pixels with features disabled/enabled. A historical
profile that changes no target probe is rejected. Each dataset includes:

- `font_probe.json`: changed-pixel counts per target cluster and font hash.
- `probe_*.png`: comparison sheets; **left = features off, right = configured features**.
  Row order: `லை ளை ணை னை ணா றா னா கி கீ சி சீ தி தீ ணொ ணோ றொ றோ னொ னோ ஔ ஊ`.

Inspect these sheets against your book specimens before scaling up. Pixel changes
prove that alternate shaping happened, not that a particular book's glyphs match.

**Actual Vatteluttu is not included.** It needs a separately verified font and
expert-approved mapping into the desired transcription script. This pipeline
currently preserves the existing Tamil model's charset; it does not implement
charset expansion or legacy-font encoding conversion.

## Requirements

- Python 3.10+.
- Tesseract 5 with `tesseract`, `combine_tessdata`, `lstmtraining`, and `lstmeval\
  on PATH.
- Pillow with RAQM/HarfBuzz support. The pipeline fails explicitly if shaping is
  unavailable; it never silently falls back to drawing unshaped Tamil.
- Internet access for initial asset downloads only. Images and transcriptions
  remain local and are not uploaded.

From the repository root:

```bash
python -m venv .venv
# activate the virtual environment, then:
python -m pip install -r requirements.txt
# Tesseract 5 with training tools must be on PATH
```

Downloads include font/model license files and `assets/provenance.json` with URLs
and SHA-256 hashes. Existing downloads are reused. Upstream URLs can change;
preserve your assets and provenance for reproducibility.

## One-command complete flow

Small smoke test:

```bash
python -m train run \
  --output runs/my-smoke \
  --lines 30 --variants 1 --iterations 50
```

This produces 180 image/text pairs: 144 training, 18 validation, 18 test.
`--lines` counts unique transcriptions; multiply by six profiles and `--variants\
to get the number of images.

Larger first experiment:

```bash
python -m train run \
  --output runs/experiment01 \
  --lines 1000 --variants 3 --iterations 5000
```

This produces 18,000 pairs. Training time depends on the CPU and line lengths.
Fifty iterations verifies plumbing; it is not enough to certify a production model.
More iterations do not guarantee better held-out accuracy.

Always use a **new output directory**. Existing nonempty outputs are rejected to
avoid mixing datasets, checkpoints or metrics from different experiments.

### Expanded local dataset

The generated dataset at `runs/expanded/dataset` contains **18,000 image/text
pairs**: 14,400 training, 1,800 validation and 1,800 test. It uses 1,000 unique
transcriptions, all six profiles, and three variants per profile (seed 42).
The original 180-pair dataset remains at `runs/demo/dataset`.

Reproduce the expanded data without training:

```bash
python -m train setup
python -m train generate \
  --dataset runs/expanded-new/dataset --lines 1000 --variants 3 --seed 42
```

Train and evaluate the expanded data when ready:

```bash
python -m train train \
  --dataset runs/expanded/dataset --output runs/expanded/model --iterations 5000
python -m train evaluate \
  --dataset runs/expanded/dataset --model runs/expanded/model/tam_historical.traineddata \
  --output runs/expanded/evaluation
```

The expanded dataset has not been trained or evaluated yet; the smoke-run
accuracy below applies only to the original demo. The two datasets are separate
experiments and can share transcriptions: do not combine one experiment's
training split with the other's test split.

### Local Git repository

Source code, tests, profiles, training config, dependency requirements, this
documentation and shareable model releases under `models\` are tracked. `assets\`,
`runs\`, virtual environments and Python caches are ignored. Generated images,
transcriptions and training checkpoints remain available locally but are not
stored in Git. Use the setup/generation commands to recreate them, and preserve
asset provenance and hashes for exact reproducibility.

Committed model releases follow the layout documented in [models/README.md](models/README.md).
Each release directory contains the exported `.traineddata` plus a `manifest.json\
with its hash, dataset version, training settings and metrics. `models/approved-v1\
is the current smoke model; keep it for rollback and add future releases alongside
it as `models/candidate-v2`, and so on.

### Corpus and augmentation

The built-in starter has 1,728 composed Tamil sentences and three diagnostic
word-list lines. It is original demonstration text, not a production language
corpus; review grammar and replace it with representative, licensed text.
The maximum built-in `--lines` is 1,731.

One diagnostic line is reserved in each split to ensure `லை ளை ணை னை` and
`ணா றா னா` occur even in a small smoke run. These reorder the same target words:
their accuracy is a **glyph stress test**, not independent language generalization.
Metrics separate diagnostic and ordinary text.

For your own UTF-8 corpus, provide one text line per line:

```bash
python -m train run \
  --output runs/custom01 --corpus corpus.txt \
  --lines 2000 --variants 3 --iterations 5000
```

Custom corpora are not modified with diagnostic lines. Check
`dataset/generation.json` for target occurrence counts **by split**; an absent
target is not measurable. The renderer refuses unsupported characters instead
of using font fallback.

Variant 0 is clean. Additional variants use small rotation, blur, and speckle
noise. Sizes vary between 40, 48 and 56 pixels. Noise is deterministic with
`--seed` (default 42), and rotation expands the canvas rather than clipping marks.
These are modest scan proxies, not simulations of all historical printing defects.

## Individual stages

```bash
python -m train setup
python -m train generate \
  --dataset runs/dataset01 --lines 100 --variants 2
python -m train train \
  --dataset runs/dataset01 --output runs/model01 \
  --iterations 1000 --learning-rate 0.0001
python -m train evaluate \
  --dataset runs/dataset01 --model runs/model01/tam_historical.traineddata \
  --output runs/metrics01
```

Every image has a same-stem `.gt.txt` containing exactly one UTF-8 NFC text line.
Typography changes the image, **not the Unicode label**.

```text
dataset/
  manifest.json
  generation.json
  font_probe.json
  probe_korkai_historical.png
  train/
    <id>.png
    <id>.gt.txt
    <id>.box
    <id>.lstmf
  val/
    ...
  test/
    <id>.png
    <id>.gt.txt
```

The native training stage uses standard whole-line LSTM boxes and the explicit
[training config](train/lstm.train). Training lists use LF, not Windows CRLF:
native `lstmtraining` otherwise includes the CR in the filename.

The original model's full unicharset, recoder and dictionaries are retained. Only
the LSTM is adapted. Use the provided `tessdata_best` floating-point model, not an
integerized `tessdata_fast` model.

## Adding real data later

Manually crop **complete text lines** and provide matching single-line UTF-8
transcriptions. Retain small hooks, pulli and vowel marks with sufficient margin.
Transcribe printed spelling, not modernized spelling; review uncertain lines.

```text
real-book001/
  p012_l003.png
  p012_l003.gt.txt
  p012_l004.tif
  p012_l004.gt.txt
```

Import one book/source at a time. The import directory is nonrecursive and supports
`.png` or `.tif`, exactly one image for each `.gt.txt`.

```bash
python -m train import-real \
  --dataset runs/dataset01 --directory real-book001 \
  --source-id book001 --split train
python -m train import-real \
  --dataset runs/dataset01 --directory real-book002 \
  --source-id book002 --split val
python -m train import-real \
  --dataset runs/dataset01 --directory real-book003 \
  --source-id book003 --split test
```

Keep every page of a real source in one split. Different books/printers in the
test set give a more credible measurement. Source IDs, identical normalized
transcriptions and identical image hashes cannot cross splits. Conservative
duplicate-transcription checks can reject common repeated headings: remove those
duplicates or keep them within a split rather than bypassing the check.

Then train into a fresh output directory:

```bash
python -m train train \
  --dataset runs/dataset01 --output runs/model02 \
  --base runs/model01/tam_historical.traineddata --iterations 5000
python -m train evaluate \
  --dataset runs/dataset01 --model runs/model02/tam_historical.traineddata \
  --output runs/metrics02
```

The baseline for evaluation remains the original downloaded `tam` unless you
explicitly override `--base`. Real and synthetic scores are reported separately.
Do not edit imported files in place: their recorded hashes are validated.

## Accuracy reports

The terminal prints baseline/fine-tuned:

- Unicode code-point CER and character accuracy.
- WER.
- Grapheme-cluster accuracy and exact-line accuracy.
- `ai`, `nedil` and pulli reference-cluster accuracy with denominators.

`evaluation/metrics.json` also contains per-font, per-category, real/synthetic,
ordinary/diagnostic and real-source metrics, cluster-specific scores, confusion
counts, model hashes and dataset-manifest hash. `predictions.json` stores every
reference and hypothesis for inspection.

Error rates use Levenshtein edits divided by reference length and may exceed
100% when there are many insertions. Accuracy is `max(0, 1 - error_rate)`.
Missing target occurrences are unmeasurable, not 100% correct. Output is normalized
to NFC and trimmed at the edges; internal whitespace and punctuation are retained.

Validation checkpoint selection uses `lstmeval` BCER, which is an internal LSTM
metric and is **not interchangeable** with the reported held-out Unicode CER.
Test samples never enter training or checkpoint selection.

Synthetic variants of the same normalized line stay in one split. Fonts are
shared across splits: these scores measure **unseen text in known fonts**, not
unseen-typeface or historical-book generalization. Evaluation uses OEM 1 / PSM 7
on cropped lines; it does not measure full-page layout/line segmentation.

### Verified native Windows smoke run

The implementation was exercised with Tesseract 5.5 and Python 3.12 on 180
synthetic pairs, 50 training iterations, and 18 held-out line images representing
only three unique transcriptions (including one diagnostic word list):

| Metric | Original Tamil | Fine-tuned |
|---|---:|---:|
| Code-point CER | 13.93% | 12.81% |
| Character accuracy | 86.07% | 87.19% |
| Grapheme accuracy | 78.79% | 81.25% |
| WER | 57.02% | 52.63% |
| ai cluster accuracy | 80.56% | 83.33% |
| nedil cluster accuracy | 77.78% | 83.33% |
| Exact-line accuracy | 0% | 0% |

These small synthetic-only numbers establish that the flow works, **not that the
model is ready for real historical documents**. Retain modern and real historical
holdouts before adopting a model. Results can differ with assets and tool versions.

## Using the exported model

Committed releases live under `models\` (see [models/README.md](models/README.md)).
For a step-by-step guide to running the latest committed model on new images, see
[inference/README.md](inference/README.md). For example, the current smoke release
is `models/approved-v1/tam_historical.traineddata` with its metadata in
`models/approved-v1/manifest.json`.

The [inference package](inference/) can download the pinned model, select it
explicitly, print Unicode Tamil, and save UTF-8 output:

```bash
python -m inference download
image=$(ls runs/demo/dataset/test/*.png | head -n 1)
python -m inference run --image "$image" --psm 7 --output runs/inference/demo-line.txt
```

`download` is optional when you already have `models/approved-v1/tam_historical.traineddata`.
Use `--model <path>` to select a different release. Choose a new output filename if
it already exists. For page/block modes, Python API usage and real-data versioning,
see the [detailed guide](docs/DATASET_TRAINING_INFERENCE.md) and the
[inference guide](inference/README.md).

No system model is overwritten. Point Tesseract to the release directory:

```bash
tesseract line.png stdout --tessdata-dir models/approved-v1 -l tam_historical --oem 1 --psm 7
```

Use a suitable page segmentation mode for full-page input, and evaluate that
page-level pipeline separately.

## Tests

```bash
python -m train setup
python -m unittest discover -s tests -v
```

Tests cover Tamil metrics, normalization, empty predictions, subprocess failures,
split leakage, imports, deterministic generation, and **actual pixel changes for
all seven key Korkai ai/nedil clusters**. Font integration tests are skipped if
assets have not been downloaded.

## Sources

- [Official tesstrain dataset format](https://github.com/tesseract-ocr/tesstrain)
- [Tesseract LSTM training](https://tesseract-ocr.github.io/tessdoc/tess5/TrainingTesseract-5.html)
- [tessdata_best models](https://github.com/tesseract-ocr/tessdata_best)
- [Korkai font and license](https://tamilfontstudio.net/fonts/korkai)
- [Vaagai font and license](https://tamilfontstudio.net/fonts/vaagai)
- [Noto Tamil fonts](https://github.com/google/fonts)
