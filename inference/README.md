# Inference: download the pinned model and run it locally

All inference code lives in this folder:

```text
inference\
  __init__.py      # public API: recognize, download_model, ensure_model
  __main__.py      # python -m inference
  cli.py           # download / run subcommands
  model_store.py   # pinned release, download and SHA-256 verification
  runtime.py       # Tesseract process wrapper and recognize()
  README.md
```

The model is **hard-pinned** to one release in `model_store.py`. To promote a new
model, bump `DEFAULT_RELEASE` and `MODEL_SHA256` together (and update
`DEFAULT_MODEL_PATH` if the filename changes).

## 1. Which model is used?

```python
# inference/model_store.py
DEFAULT_RELEASE = "approved-v1"
MODEL_FILENAME = "tam_historical.traineddata"
MODEL_SHA256 = "0f79f56753ec82641c4f7fd108df466d63a421a2de087d363180b14b826600f1"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / DEFAULT_RELEASE / MODEL_FILENAME
```

The matching release in the repository is `models\approved-v1`, with its hash,
dataset version, training settings, metrics and licenses in
`models\approved-v1\manifest.json`.

> `approved-v1` is a **smoke** model trained on the 180-pair synthetic demo
> dataset for 50 iterations. It proves the pipeline works; it is not certified
> for real historical documents.

## 2. Install runtime requirements

- Tesseract 5 with `tesseract` on `PATH`.
- Python 3.10+. The inference package uses only the standard library, but the
  repository's `requirements.txt` also covers the training pipeline.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# If Tesseract is not already on PATH, adapt this to your installation:
$env:Path = 'D:\Program Files\Tesseract-OCR;' + $env:Path

tesseract --version
```

## 3. Download and verify the model

```powershell
.\.venv\Scripts\python.exe -m inference download
```

This downloads the pinned `.traineddata` to
`models\approved-v1\tam_historical.traineddata`, verifies its SHA-256 and refuses
to keep a mismatched file. Options:

```powershell
python -m inference download --dest path\to\models --force
```

- `--dest` sets the destination `.traineddata` path.
- `--force` replaces an existing file, but only after the new download verifies.

Manual verification:

```powershell
(Get-FileHash models\approved-v1\tam_historical.traineddata -Algorithm SHA256).Hash.ToLower()
# expected: 0f79f56753ec82641c4f7fd108df466d63a421a2de087d363180b14b826600f1
```

## 4. Run inference on a line image

```powershell
.\.venv\Scripts\python.exe -m inference run `
  --image path\to\line001.png `
  --psm 7 `
  --output out\line001.txt
```

- `--model` defaults to the pinned release; pass it to use another `.traineddata`.
- `--psm 7` is correct for a single cropped text line.
- `--output` is optional; omit it to print to the terminal only.
- Existing output files are never overwritten, so choose a new name each run.
- `--download` fetches the pinned model automatically if `--model` is missing.

## 5. Choose a page segmentation mode

| PSM | Use |
|---:|---|
| 7 | One cropped text line (default) |
| 13 | Raw single-line recognition; alternative for unusual segmentation |
| 6 | One uniform text block |
| 3 | Page with automatic layout analysis |
| 11 | Sparse text scattered through an image |

For a full page:

```powershell
.\.venv\Scripts\python.exe -m inference run `
  --image path\to\page001.png --psm 3 --output out\page001.txt
```

The package does not deskew, crop, convert PDFs, correct spelling or detect
orientation. Convert PDFs to images separately and rotate pages upright first.

## 6. Use the model from Python

```python
from pathlib import Path
from inference import download_model, recognize

model = download_model()  # downloads/verifies the pinned release if needed

text = recognize(
    image=Path(r"path\to\line001.png"),
    model=model,
    psm=7,
    log=Path(r"runs\inference\python-api.log"),
)
print(text)
```

`recognize` raises on missing inputs, OCR failure or empty output; treat an
exception as a failure, not an empty transcription. It starts one Tesseract
process per call, so a batch job can simply loop over images.

## 7. Use the model with the Tesseract CLI directly

Point `--tessdata-dir` at the release **folder**, and pass the filename **stem**
to `-l`:

```powershell
tesseract path\to\line001.png stdout `
  --tessdata-dir models\approved-v1 `
  -l tam_historical --oem 1 --psm 7
```

Do not pass the full `.traineddata` file to `--tessdata-dir`; it takes a folder.
For reliable UTF-8 output on older PowerShell, prefer `-m inference run --output`
over `>` redirection.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| `Model not found` | Run `python -m inference download`, or pass `--download` / `--model` |
| `hash mismatch` | Re-run with `--force`; the pinned constants may be out of date |
| `Provide an existing .traineddata model` | Check the `--model` path and that it ends in `.traineddata` |
| `Tesseract returned no text` | Check crop, contrast, margins, orientation, PSM and model |
| Model cannot load | Ensure the folder and `-l` stem match the file name |
| Tamil looks corrupted in the terminal | Write to a UTF-8 file with `--output` and open it in a Tamil-capable editor |
| Output already exists | Choose a new `--output` path |

## 9. What to keep with the model

When publishing or sharing a release, keep the `.traineddata` and its
`manifest.json` together, and preserve the license and provenance information in
the manifest. Keep the previously approved release available for rollback.
