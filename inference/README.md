# Using the latest model for inference

This guide shows how to take the model committed in this repository and run it on
new images. No training data, checkpoints or `.lstm` files are required at runtime.

## 1. Which model is "latest"?

Tracked releases live under `models\`, one versioned directory per release:

```text
models\
  approved-v1\
    tam_historical.traineddata
    manifest.json
```

`models\approved-v1` is the current approved release. Future releases will appear
as `models\candidate-v2`, `models\approved-v2`, and so on. To find the latest:

```powershell
Get-ChildItem models -Directory | Sort-Object Name
```

Always read the chosen `manifest.json` first. It records the model hash, dataset
version, training settings, metrics, intended use and licenses.

> `approved-v1` is a **smoke** model trained on the 180-pair synthetic demo
> dataset for 50 iterations. It demonstrates the pipeline; it is not certified for
> real historical documents. Check metrics and intended use before relying on it.

## 2. Get the model

If you cloned the repository, the model is already present:

```powershell
git clone https://github.com/wickkiey/tamil-tesseract-enhanced.git
Set-Location tamil-tesseract-enhanced
```

If you only need the model file, download it directly:

```powershell
New-Item -ItemType Directory -Force models\approved-v1 | Out-Null
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/wickkiey/tamil-tesseract-enhanced/main/models/approved-v1/tam_historical.traineddata" `
  -OutFile models\approved-v1\tam_historical.traineddata
```

Verify the download against the hash in `manifest.json`:

```powershell
(Get-FileHash models\approved-v1\tam_historical.traineddata -Algorithm SHA256).Hash.ToLower()
# expected: 0f79f56753ec82641c4f7fd108df466d63a421a2de087d363180b14b826600f1
```

## 3. Install runtime requirements

- Tesseract 5 with `tesseract` on `PATH`.
- Python 3.10+ with this project's requirements (the wrapper script imports from
  `tamil_ocr.py`).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# If Tesseract is not already on PATH, adapt this to your installation:
$env:Path = 'D:\Program Files\Tesseract-OCR;' + $env:Path

tesseract --version
```

## 4. Run inference on a line image

Pass the model file from the release directory; the script derives
`--tessdata-dir` and the language id from it automatically.

```powershell
.\.venv\Scripts\python.exe inference.py `
  --image path\to\line001.png `
  --model models\approved-v1\tam_historical.traineddata `
  --psm 7 `
  --output out\line001.txt
```

- `--model` is the full path to the `.traineddata` file.
- `--psm 7` is correct for a single cropped text line.
- `--output` is optional; omit it to print to the terminal only.
- Existing output files are never overwritten, so choose a new name each run.

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
.\.venv\Scripts\python.exe inference.py `
  --image path\to\page001.png `
  --model models\approved-v1\tam_historical.traineddata `
  --psm 3 --output out\page001.txt
```

The script does not deskew, crop, convert PDFs, correct spelling or detect
orientation. Convert PDFs to images separately and rotate pages upright first.

## 6. Use the model from Python

```python
from pathlib import Path
from inference import recognize

text = recognize(
    image=Path(r"path\to\line001.png"),
    model=Path(r"models\approved-v1\tam_historical.traineddata"),
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
For reliable UTF-8 output on older PowerShell, prefer `inference.py --output`
over `>` redirection.

## 8. Troubleshooting

| Symptom | Action |
|---|---|
| `Provide an existing .traineddata model` | Check the `--model` path and that it ends in `.traineddata` |
| `Tesseract returned no text` | Check crop, contrast, margins, orientation, PSM and model |
| Model cannot load | Ensure the folder and `-l` stem match the file name |
| Tamil looks corrupted in the terminal | Write to a UTF-8 file with `--output` and open it in a Tamil-capable editor |
| Output already exists | Choose a new `--output` path |
| Hash mismatch | Re-download the model; do not use a corrupted or modified file |

## 9. What to keep with the model

When publishing or sharing a release, keep the `.traineddata` and its
`manifest.json` together, and preserve the license and provenance information in
the manifest. Keep the previously approved release available for rollback.
