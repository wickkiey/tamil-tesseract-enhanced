# Model releases

This directory holds shareable Tesseract LSTM model releases. Only exported
`.traineddata` files and their metadata are tracked; training checkpoints,
`.lstm` components, sample lists and logs stay under `runs/` (gitignored).

## Layout

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

- `approved-v1` is the currently approved model for rollback.
- Give each new release its own versioned directory; never overwrite an approved
  model in place.
- `python -m train evaluate` saves `accuracy.json` and `metrics.json` into
  `models/<name>/` (see [train/README.md](../train/README.md#5-evaluate)).
- Keep `manifest.json` next to the model. It records the model hash, dataset
  version, training settings, metrics and licenses.

## Using a model

`--tessdata-dir` takes the model's **parent folder** and `-l` takes the filename
**stem**:

```bash
python -m inference download
python -m inference run --image line.png --psm 7

tesseract line.png stdout --tessdata-dir models/approved-v1 \
  -l tam_historical --oem 1 --psm 7
```

`python -m inference download` fetches the pinned release, verifies its SHA-256 and
caches it here. `run` defaults to `models/approved-v1/tam_historical.traineddata`;
pass `--model` to select another release.

Verify a downloaded model before using it:

```bash
python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" models/approved-v1/tam_historical.traineddata
# Must match manifest.json -> sha256
```

## Status of approved-v1

`approved-v1` is an exported **smoke** model trained on the 180-pair synthetic
demo dataset for 50 iterations. It proves the pipeline works; it is not
certified for real historical documents. See `manifest.json` for metrics and
`docs/DATASET_TRAINING_INFERENCE.md` for the full release checklist.
