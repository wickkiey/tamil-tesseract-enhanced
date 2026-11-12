# Tamil OCR: dataset growth, training, evaluation and inference

This guide describes the implemented commands, their limitations, and a
repeatable workflow for improving Tamil typography recognition over time.
Run all commands from the repository root.
Adjust absolute paths if you move the project.

## 1. Understand the artifacts

| Artifact | Purpose |
|---|---|
| [train](../train) | Pipeline package (one module per stage) and [config.toml](../train/config.toml) |
| [inference/](../inference/) | Download the pinned model and run it on a new image |
| [profiles.json](../train/profiles.json) | Fonts, OpenType features and typography categories |
| `assets/fonts` | Downloaded fonts and licenses |
| `assets/models/tam.traineddata` | Original floating-point Tamil baseline |
| `assets/provenance.json` | Download URLs, licenses and hashes |
| `runs/demo/dataset` | Original 180-pair smoke dataset |
| `runs/demo/model/tam_historical.traineddata` | Exported smoke model, not a production-certified model |
| `models/approved-v1` | Tracked, shareable release of the smoke model plus `manifest.json` |
| `runs/demo/evaluation` | Original smoke metrics and predictions |
| `runs/expanded/dataset` | 18,000 pairs; generated but not yet trained |

The local Git repository tracks source, tests, documentation and shareable model
releases under `models\`. Assets, datasets, training checkpoints, logs and Python
caches are ignored, not deleted. Back up these ignored files separately if you need
to preserve an exact experiment.

Tracked model releases use this layout:

```text
models/
  README.md
  approved-v1/
    tam_historical.traineddata
    manifest.json
```

Only exported `.traineddata` files and their `manifest.json` belong in `models\`;
training checkpoints and `.lstm` components stay under `runs\`.

The original and expanded experiments can share text. Never train on one dataset's
training split and claim independent accuracy on another dataset's test split
without checking for overlap.

## 2. Prepare the environment

```bash
python -m venv .venv
# activate the virtual environment, then:
python -m pip install -r requirements.txt

tesseract --version
for t in tesseract combine_tessdata lstmtraining lstmeval; do command -v "$t"; done
python -m train setup
python -m unittest discover -s tests -v
```

Use Python 3.10+ and Tesseract 5 with training tools. Tamil generation requires
Pillow RAQM/HarfBuzz; unshaped rendering is rejected. Inference reuses pipeline
helpers, so install the same Python requirements even though Tesseract performs
the actual recognition.

`setup` downloads licensed fonts and `tam` from `tessdata_best`. Do not fine-tune
an integerized `tessdata_fast` model. No input images or text are uploaded.
Preserve downloaded assets, license files, package versions and hashes: upstream
download URLs may change.

The commands below use the virtual-environment interpreter explicitly. If you
already installed the requirements into your current Python, `python` also works.

## 3. Required ground-truth format

Each sample is a **complete text-line image**, not an entire page:

```text
p012_l003.png
p012_l003.gt.txt
```

Example UTF-8 label:

```text
மழை நின்ற பிறகு மாணவி கடிதத்தை எழுதினார்.
```

Rules:

1. Match image and text basenames exactly.
2. Use one nonempty text line. A final newline is fine.
3. Use Unicode Tamil, normalized to NFC; do not use legacy font encodings.
4. Keep printed spelling, digits and punctuation rather than modernizing the text.
5. Traditional `ணா`, `றா`, `னா`, `லை`, `ளை`, `ணை`, `னை` shapes keep the same
   Unicode labels as their modern equivalents.
6. Remove accidental internal control characters, zero-width characters and tabs.
7. Keep all hooks, pulli and vowel marks inside the crop, with a small margin.
8. Exclude illegible lines until a qualified reviewer can label them reliably.

The importer validates files and hashes, but cannot determine whether a human
transcription is linguistically correct. Arrange a second review for historical
forms. Training generates whole-line `.box` and `.lstmf` files automatically.

## 4. Add more synthetic data

### A. More images using the existing starter

```bash
python -m train generate \
  --dataset runs/synthetic-v2/dataset \
  --lines 1500 --variants 4 --seed 42
```

With six profiles, this makes **36,000 pairs**:
`1,500 unique text lines x 6 profiles x 4 variants`.

- `--lines` counts unique normalized transcriptions, not images.
- Variant 0 is clean; other variants add modest rotation, blur and speckle.
- Sizes vary between 40, 48 and 56 pixels.
- The built-in maximum is **1,731 unique lines**, including three diagnostic lines.
- Increasing variants adds visual diversity, not new language content.
- Outputs must be new or empty. Generation does **not** append to an existing dataset.

Inspect generated `probe_*.png` sheets and `font_probe.json` before training.
The historical Korkai profile is verified to change all seven key ai/nedil clusters.
Vaagai is an archival typeface revival. Neither font guarantees a match to every
historical book.

### B. Add your own larger corpus

Create a UTF-8 corpus such as `assets/corpora/tamil-v2.txt`, with one sentence or
printed line per line. Include licensed, representative material:

- Everyday Tamil and historical vocabulary.
- Target vowel combinations in different word positions.
- Similar-looking letters, pulli, numbers and punctuation.
- Clean modern Tamil to help preserve baseline recognition.

Blank corpus lines are skipped; normalized duplicate lines are deduplicated.
Choose `--lines` no greater than the number of unique lines:

```bash
python -m train generate \
  --dataset runs/corpus-v2/dataset \
  --corpus assets/corpora/tamil-v2.txt \
  --lines 5000 --variants 3 --seed 42
```

With six profiles, this makes **90,000 pairs**. A custom corpus is not automatically
augmented with the built-in diagnostic lines. Check `generation.json`, especially
`target_occurrences_by_split`; absent targets cannot have a meaningful accuracy.
Long text becomes wide line images: prefer realistic line lengths rather than
whole paragraphs.

### C. Add another historical font

Acquire a Unicode TTF/OTF font with permission to use it, store it under
`assets/fonts`, and retain its license. Add a profile to [profiles.json](../train/profiles.json):

```json
{
  "name": "new_historical",
  "font": "assets/fonts/MyHistoricalTamil.ttf",
  "features": ["ss01"],
  "category": "historical"
}
```

This is a template: do not assume the font actually supports `ss01`.
Paths in a profile are resolved relative to the JSON file's directory.
Profile names must be unique. Allowed categories are `modern`, `historical`,
and `archival`; OpenType tags may be enabled (`ss01`) or disabled (`-ss01`).

The generator checks corpus cmap coverage and enabled feature presence. A profile
marked `historical` must enable a feature that changes target-probe pixels;
otherwise it is rejected. For a font whose historical shapes are the default
rather than optional features, use `archival` and verify the specimens manually.
Do not relabel modern glyphs as historical merely to bypass checks.

Generate into a new dataset after changing profiles. This workflow does not
convert Bamini/TAB/TAM encoded text or support actual Vatteluttu transcription.
New output characters need charset expansion, which is not implemented.

### D. Dataset versions versus a fixed benchmark

Synthetic splitting is approximately 80/10/10 by normalized text. All font/noise
variants of the same text stay together. Fonts are shared across splits.

Changing the corpus size or seed can change the split assigned to existing text,
even with the same seed. Therefore:

- Treat regeneration as a **new dataset version**, not a safe append.
- Do not continue a previously trained model on regenerated data and reuse an old
  synthetic test set unless you have checked transcription overlap.
- For a rigorous fixed benchmark, reserve entire real books for test, exclude
  their repeated lines from training, and compare candidates on that benchmark.
- Keep validation for tuning. Do not repeatedly tune choices using test scores.

The built-in diagnostic lines reorder the same target words across splits.
Their scores are glyph stress tests, not independent language generalization.

## 5. Add real scanned data later

Create a working dataset version before modifying the existing one:

```bash
New-Item -ItemType Directory -Path runs/mixed-v1
Copy-Item -LiteralPath runs/expanded/dataset -Destination runs/mixed-v1/dataset -Recurse
```

The copied manifest uses relative image paths, so it remains valid. Old generated
training files, if present, will be regenerated during training.

Prepare one nonrecursive import folder per real book:

```text
assets/real/book001/
  p012_l003.png
  p012_l003.gt.txt
  p012_l004.tif
  p012_l004.gt.txt
```

The importer supports lowercase `.png` or `.tif`, exactly one image per label.
Do not place both image formats with the same basename. Full-page segmentation,
PDF conversion and manual annotation are preparation tasks, not automated by this
pipeline.

Import distinct books into distinct splits:

```bash
python -m train import-real \
  --dataset runs/mixed-v1/dataset --directory assets/real/book001 \
  --source-id book001 --split train

python -m train import-real \
  --dataset runs/mixed-v1/dataset --directory assets/real/book002 \
  --source-id book002 --split val

python -m train import-real \
  --dataset runs/mixed-v1/dataset --directory assets/real/book003 \
  --source-id book003 --split test
```

Keep every page of a source in the same split. Source IDs use letters, digits,
hyphens or underscores. New pages of `book001` can later be imported into `train\
from another folder; reuse the same source ID and unique page/line basenames.

The manifest records source IDs and text/image hashes. Cross-split duplicate text
or identical image bytes are rejected, even when source names differ. This is
conservative: common repeated headings may need to be excluded. Do not bypass the
check by changing source IDs.

Do not manually drop pairs into `train` and expect training to discover them;
training uses the manifest. Do not edit already imported labels/images in place:
hash validation will reject the edits. For corrections, prepare corrected source
pairs and rebuild a fresh mixed dataset from the original synthetic version,
then reimport the corrected sources. There is no update/delete command.

Data proportions matter: 50 real lines mixed into 14,400 synthetic training lines
have limited influence. Expand reviewed real data and compare different mixtures
on validation. The current trainer has no real/synthetic weighting parameter.

## 6. Train and export a model

Train the expanded synthetic dataset from the original Tamil model:

```bash
python -m train train \
  --dataset runs/expanded/dataset \
  --output runs/expanded/model \
  --model-name tam_historical \
  --iterations 5000 --learning-rate 0.0001
```

Or train your mixed dataset:

```bash
python -m train train \
  --dataset runs/mixed-v1/dataset \
  --output runs/mixed-v1/model \
  --model-name tam_historical_v1 \
  --iterations 5000 --learning-rate 0.0001
```

What happens:

1. Validate manifest, hashes, and split integrity.
2. Extract the base model's floating-point LSTM and existing charset.
3. Reject training text outside that charset.
4. Generate `.box` and `.lstmf` files for train/validation only.
5. Fine-tune with validation samples; test samples are excluded.
6. Score produced checkpoint candidates with `lstmeval`.
7. Select the lowest validation BCER candidate and export `.traineddata`.

The runtime model for the mixed example is:

```text
runs/mixed-v1/model/tam_historical_v1.traineddata
```

Also preserve `training.json`, `training.log`, checkpoint scores, dataset
manifest, provenance and licenses. The original model is not overwritten.

`--iterations` is a training limit, not an accuracy promise. Early stopping can
occur. The initial learning rate is modest; compare settings on validation rather
than blindly increasing it. Short smoke runs establish plumbing, not convergence.
Training runs synchronously and can take a long time on CPU.

### Continue adaptation from your model

The default export remains floating-point and can be used as the next base:

```bash
python -m train train \
  --dataset runs/mixed-v2/dataset \
  --output runs/mixed-v2/model \
  --base runs/mixed-v1/model/tam_historical_v1.traineddata \
  --model-name tam_historical_v2 --iterations 5000
```

Create `mixed-v2` by copying `mixed-v1` and adding reviewed sources without moving
existing samples between splits. Include representative previous/modern data to
reduce forgetting. This starts a new adaptation run from recognition weights;
it is **not** an optimizer-state resume from an interrupted checkpoint.

For interrupted/failed runs, inspect the log and rerun into a new output directory.
The CLI has no automatic checkpoint-resume command. Do not delete your last
working model while experimenting.

## 7. Evaluate and read accuracy

```bash
python -m train evaluate \
  --dataset runs/mixed-v1/dataset \
  --model runs/mixed-v1/model/tam_historical_v1.traineddata \
  --output runs/mixed-v1/evaluation
```

The default baseline is the original downloaded `tam`. To compare against the
previous custom model instead:

```bash
python -m train evaluate \
  --dataset runs/mixed-v2/dataset \
  --base runs/mixed-v1/model/tam_historical_v1.traineddata \
  --model runs/mixed-v2/model/tam_historical_v2.traineddata \
  --output runs/mixed-v2/previous-model-comparison
```

Use `--split val` for development comparisons; the default is `test`.
Keep a fresh output folder for every evaluation.

| Metric | Interpretation |
|---|---|
| CER | Code-point edit distance / reference code points; lower is better |
| Character accuracy | `max(0, 1 - CER)` |
| WER | Word edits / reference words; lower is better |
| Grapheme accuracy | `max(0, 1 - grapheme edit rate)`; handles visible Tamil clusters |
| Exact-line accuracy | Fraction of completely correct lines |
| ai/nedil/pulli accuracy | Correct aligned reference clusters / target occurrences |

Insertions can make error rates exceed 100%; they are not hidden. Missing targets
are unmeasurable. Target accuracy focuses on reference clusters; inspect overall
edit rates for extra inserted marks too.

Outputs:

- `metrics.json`: overall and per-profile/category/source/real-synthetic/content
  metrics, target counts, confusions and model hashes.
- `predictions.json`: every reference and prediction for review.
- `ocr.log`: commands and Tesseract diagnostics.

Read the important groups with Python:

```bash
python - <<'EOF'
import json
m = json.load(open('runs/mixed-v1/evaluation/metrics.json', encoding='utf-8'))['models']
print(m['baseline']['overall'])
print(m['finetuned']['overall'])
print(m['finetuned']['by_group'].get('kind:real'))
print(m['finetuned']['by_group'].get('category:modern'))
print(m['finetuned']['overall']['target_accuracy'])
EOF
```

Accept a model only if real historical accuracy improves and modern accuracy does
not regress beyond a tolerance you explicitly choose. Scores on known synthetic
fonts are not proof of real-book accuracy. Validation BCER is an internal LSTM
metric, not the same quantity as the reported Unicode CER.

## 8. Run inference on a new image

### Single text line with the inference package

The package is pinned to the currently approved release. Download and verify it
first if `models/approved-v1/tam_historical.traineddata` is missing:

```bash
python -m inference download
```

This example uses a generated test line:

```bash
image=$(ls runs/demo/dataset/test/*.png | head -n 1)
python -m inference run \
  --image "$image" \
  --psm 7 --output runs/inference/demo-line.txt
```

For a real new line, replace `--image` and pass `--model` to select another
release:

```bash
python -m inference run \
  --image assets/incoming/line001.png \
  --model models/approved-v1/tam_historical.traineddata \
  --psm 7 --output runs/inference/line001.txt
```

The command:

- Uses the model's parent folder as `--tessdata-dir`.
- Uses its filename stem as the language/model identifier.
- Runs LSTM inference with `--oem 1`.
- Prints NFC Unicode text and optionally saves UTF-8 text with LF endings.
- Preserves internal line breaks for blocks/pages.
- Logs commands and diagnostics to `runs/inference.log` by default; use `--log\
  to change the path.
- Refuses to overwrite existing output or use an input/model as a log.
- Returns a nonzero exit code for missing inputs, OCR failure or empty output.

Logs may contain OCR text and local paths. They stay local; handle them according
to the sensitivity of your documents.

### Choose a page segmentation mode

| PSM | Use |
|---:|---|
| 7 | One cropped text line; default |
| 13 | Raw single-line recognition; alternative for unusual segmentation |
| 6 | One uniform text block |
| 3 | Page with automatic layout analysis, no orientation detection |
| 11 | Sparse text scattered through an image |

For a page:

```bash
python -m inference run \
  --image assets/incoming/page001.png \
  --psm 3 --output runs/inference/page001.txt
```

The command does not deskew, crop, convert PDFs, correct spelling or detect
orientation. Convert PDFs to images separately. Rotate pages upright, preserve
small marks and choose PSM according to the layout. Multi-column pages may need
separate region/line extraction.

### Python API example

From the project root or with it on your Python import path:

```python
from pathlib import Path
from inference import recognize

text = recognize(
    image=Path(r"assets/incoming/line001.png"),
    model=Path(r"models/approved-v1/tam_historical.traineddata"),
    psm=7,
    log=Path(r"runs/inference/python-api.log"),
)
print(text)
```

`recognize` raises an exception on failure; your application should report it,
not treat it as a successful empty transcription. It launches one Tesseract
process per call. A batch application can iterate over images with this function.

### Direct Tesseract equivalent

```bash
tesseract assets/incoming/line001.png stdout \
  --tessdata-dir models/approved-v1 \
  -l tam_historical --oem 1 --psm 7
```

The `-l` value must match the `.traineddata` filename without its extension.
Do not pass the full model file to `--tessdata-dir`; that parameter takes a folder.
For reliable UTF-8 output on older PowerShell, prefer the Python package's
`--output` rather than `>` redirection.

## 9. Deploy and retain a rollback model

For inference, install Tesseract and use the release directory directly. You do
not need training checkpoints or `.lstmf` files at runtime. The inference package
is self-contained (standard library only) plus the Tesseract binary. Preserve
licenses, `manifest.json` and training provenance with the model release.

Current committed layout (see [models/README.md](../models/README.md)):

```text
models/
  README.md
  approved-v1/
    tam_historical.traineddata
    manifest.json
  candidate-v2/                (future)
    tam_historical_v2.traineddata
    manifest.json
```

Select models explicitly by path. Keep the approved version available for
rollback. Avoid changing the system's original Tamil model. Each `manifest.json\
tracks the model hash, dataset version, validation/test metrics, licenses and
intended document families. For mixed English/Tamil, this script selects only one
model; multilanguage inference requires the corresponding additional models and a
separately tested configuration.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| Missing training executable | Install Tesseract training tools and update PATH |
| RAQM unavailable | Use a Pillow build with RAQM/HarfBuzz; do not disable shaping checks |
| Font lacks corpus characters | Choose a Unicode font with coverage or correct the corpus |
| Historical feature absent/no pixel changes | Verify actual GSUB features and specimen shapes |
| Base charset lacks a character | Use supported labels; charset expansion is currently out of scope |
| Output is nonempty | Choose a new versioned output directory |
| Hash changed | Rebuild/reimport a corrected dataset; do not hand-edit manifest hashes |
| Leakage error | Keep books and duplicate lines/images within one split |
| No OCR text | Check crop, contrast, margins, orientation, PSM and model |
| Model cannot load | Check file existence, folder path, filename stem and Tesseract version |
| Tamil looks corrupted in terminal | Use the UTF-8 output file and a Tamil-capable editor font |
| Good synthetic score, poor real score | Add representative reviewed scans and source-separated real tests |
| Modern text regresses | Retain modern examples, reduce adaptation strength, and compare validation |

The trainer writes LF-only sample lists because native Windows `lstmtraining\
can interpret CRLF's CR as part of a filename. Avoid hand-rewriting those lists.

## 11. Recommended improvement cycle

1. Inspect prediction errors and identify missing typography/document families.
2. Prepare reviewed real lines or correctly shaped synthetic coverage.
3. Version the dataset; retain existing source assignments and untouched test data.
4. Generate/import through the CLI so manifests and hashes stay consistent.
5. Train into a new model directory.
6. Compare original, previous and candidate models on validation.
7. Freeze the candidate, then evaluate on the untouched real test benchmark.
8. Run new-document inference and page-layout checks.
9. Approve only after your accuracy/regression criteria pass; preserve rollback.

There is no universal guaranteed accuracy threshold. Define acceptable historical
CER, target-cluster accuracy, modern regression, and page-level behavior for your
actual documents before adopting a model.
