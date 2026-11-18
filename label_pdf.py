from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

import fitz
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_lines import extract_lines

st.set_page_config(page_title="PDF line labeler", layout="wide")

ROOT = Path(__file__).resolve().parent
_TESS_CANDIDATES = [
    ROOT / "models/approved-v1/tam_historical.traineddata",
    ROOT / "assets/models/tam.traineddata",
]
DEFAULT_TESSERACT = next((p for p in _TESS_CANDIDATES if p.is_file()), _TESS_CANDIDATES[-1])
EASYOCR_TAMIL_MISSING = "வஷஸஹாிீுூெேைொோௌ்"

CONFIG_PATH = Path(os.environ.get("LABELING_CONFIG", str(ROOT / "labeling_config.json")))
DEFAULT_CONFIG = {
    "tesseract_models": {
        "tesseract": "assets/models/tam.traineddata",
        "finetuned": "runs/default/model/tam_historical.traineddata",
    },
    "default_tesseract_model": "finetuned",
    "output_dir": "runs/labeled/dataset",
}


def load_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if CONFIG_PATH.is_file():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                if key == "tesseract_models" and isinstance(value, dict):
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config.get(key), dict):
                    config[key] = {**config[key], **value}
                else:
                    config[key] = value
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path)


@st.cache_data(show_spinner=False)
def count_pages(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc.page_count
    finally:
        doc.close()


@st.cache_data(show_spinner=False)
def render_page(pdf_bytes: bytes, index: int, dpi: int) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc.load_page(index)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def reset(ss) -> None:
    ss.page = 1
    ss.processed = {}
    ss.next_version = 0
    ss.texts = {}


def remember(page: int, version: int, index: int, key: str) -> None:
    st.session_state.texts[(page, version, index)] = st.session_state.get(key, "")


def ocr_with_tesseract(image: Image.Image, model_path: str) -> str:
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Tesseract model not found: {model}")
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "line.png"
        image.convert("L").save(image_path)
        result = subprocess.run(
            [
                "tesseract", str(image_path), "stdout",
                "--tessdata-dir", str(model.parent),
                "-l", model.stem, "--oem", "1", "--psm", "7",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "tesseract failed")
    return result.stdout.strip()


def ocr_with_ollama(image: Image.Image, model: str, url: str) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "prompt": (
            "Transcribe the Tamil text in this image exactly as printed. "
            "Preserve punctuation and spacing. Output only the transcription, "
            "with no explanation, labels or quotes."
        ),
        "images": [encoded],
        "stream": False,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = (data.get("response") or "").strip()
    return text.strip().strip('"').strip("'").strip()


@st.cache_resource(show_spinner=False)
def get_easyocr_reader(languages: tuple[str, ...]):
    import easyocr
    from easyocr.config import recognition_models

    if "ta" in languages:
        entry = recognition_models["gen1"]["tamil_g1"]
        if EASYOCR_TAMIL_MISSING not in entry["characters"]:
            entry["characters"] += EASYOCR_TAMIL_MISSING
    return easyocr.Reader(list(languages), gpu=False, verbose=False)


def ocr_with_easyocr(image: Image.Image, languages: list[str]) -> str:
    reader = get_easyocr_reader(tuple(languages))
    array = np.array(image.convert("RGB"))
    results = reader.readtext(array, detail=0, paragraph=False)
    return " ".join(results).strip()


def extract_line(page: int, version: int, index: int, key: str, method: str, model_path: str | None = None) -> None:
    info = st.session_state.processed.get(page)
    if not info or info["version"] != version or index >= len(info["lines"]):
        return
    image = info["lines"][index]
    try:
        if method == "tesseract":
            model = model_path or str(DEFAULT_TESSERACT)
            text = ocr_with_tesseract(image, model)
        elif method == "ollama":
            model = st.session_state.get("ollama_model", "qwen3-vl:8b")
            url = st.session_state.get("ollama_url", "http://localhost:11434")
            text = ocr_with_ollama(image, model, url)
        elif method == "easyocr":
            raw_langs = st.session_state.get("easyocr_langs", "ta")
            languages = [part.strip() for part in raw_langs.split(",") if part.strip()] or ["ta"]
            text = ocr_with_easyocr(image, languages)
        else:
            raise ValueError(f"Unknown extraction method: {method}")
    except ImportError as exc:
        st.session_state["extract_error"] = (
            f"{method} unavailable: {exc}. For EasyOCR run: pip install easyocr"
        )
        return
    except Exception as exc:
        st.session_state["extract_error"] = f"{method} failed: {exc}"
        return
    if not text:
        st.session_state["extract_error"] = f"{method} returned empty text."
        return
    st.session_state.texts[(page, version, index)] = text
    st.session_state[key] = text
    st.session_state["extract_error"] = ""


def save_dataset(processed: dict, texts: dict, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    for page in sorted(processed):
        info = processed[page]
        version = info["version"]
        for i, image in enumerate(info["lines"]):
            raw = texts.get((page, version, i), "")
            if not raw:
                raw = st.session_state.get(f"txt_{page}_{version}_{i}", "")
            text = unicodedata.normalize("NFC", raw.strip())
            if not text:
                skipped += 1
                continue
            stem = f"p{page + 1:03d}_l{i + 1:03d}"
            image.save(out_dir / f"{stem}.png", dpi=(300, 300))
            (out_dir / f"{stem}.gt.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
            saved += 1
    return saved, skipped


ss = st.session_state
if "page" not in ss:
    reset(ss)

st.title("PDF to Tamil line labels")
st.caption("Full page on the left, cropped lines on the right. Try Tesseract, Ollama or EasyOCR per line, then Save.")

uploaded = st.file_uploader("Upload a scanned PDF", type=["pdf"])

CONFIG = load_config()
tess_models = CONFIG.get("tesseract_models", {})
if tess_models:
    default_name = CONFIG.get("default_tesseract_model")
    ordered_models = list(tess_models.items())
    if default_name in tess_models:
        ordered_models.sort(key=lambda item: item[0] != default_name)
    tesseract_buttons = [
        ("Tesseract" if name in ("tesseract", "baseline") else f"Tesseract ({name})", str(resolve_path(raw)))
        for name, raw in ordered_models
    ]
else:
    tesseract_buttons = [("Tesseract", str(DEFAULT_TESSERACT))]

with st.sidebar:
    st.header("Line split settings")
    dpi = st.slider("Render DPI", 150, 500, 300, step=50)
    padding = st.slider("Crop padding (px)", 0, 20, 4)
    min_height = st.slider("Minimum line height (px)", 4, 80, 10)
    min_gap = st.slider("Merge gap (px)", 0, 20, 0)
    invert = st.checkbox("Light text on dark background", value=False)
    st.divider()
    st.header("OCR engines")
    if tess_models:
        for name, raw in tess_models.items():
            path = resolve_path(raw)
            if path.is_file():
                st.caption(f"{name}: {path}")
            else:
                st.warning(f"{name}: not found at {path}")
    else:
        st.warning("No Tesseract models configured; using default.")
    st.caption(f"Edit {CONFIG_PATH.name} to add models.")

if uploaded is None:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.getvalue()
file_id = hashlib.sha256(pdf_bytes).hexdigest()
if ss.get("file_id") != file_id:
    ss.file_id = file_id
    ss.pdf_bytes = pdf_bytes
    reset(ss)

ss.pdf_bytes = pdf_bytes
total = count_pages(pdf_bytes)

nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
with nav_left:
    if st.button("Previous", disabled=ss.page <= 1):
        ss.page -= 1
        st.rerun()
with nav_mid:
    st.markdown(
        f"<div style='text-align:center'>Page <b>{ss.page}</b> of <b>{total}</b></div>",
        unsafe_allow_html=True,
    )
with nav_right:
    if st.button("Next", type="primary", disabled=ss.page >= total):
        ss.page += 1
        st.rerun()

if total > 1:
    st.select_slider("Jump to page", options=list(range(1, total + 1)), key="page")

page = ss.page - 1
page_image = render_page(pdf_bytes, page, dpi)

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.image(page_image, caption=f"Page {ss.page}", width="stretch")

with right_col:
    process_col, clear_col = st.columns(2)
    with process_col:
        process_clicked = st.button("Process this page", type="primary", width="stretch")
    with clear_col:
        clear_clicked = st.button("Clear lines", disabled=page not in ss.processed, width="stretch")

    if clear_clicked:
        del ss.processed[page]
        st.rerun()

    if process_clicked:
        lines = extract_lines(
            page_image,
            min_gap=min_gap,
            min_height=min_height,
            padding=padding,
            invert=invert,
        )
        ss.next_version += 1
        ss.processed[page] = {"version": ss.next_version, "lines": lines}

    st.divider()

    info = ss.processed.get(page)
    if info:
        version = info["version"]
        lines = info["lines"]
        st.caption(f"{len(lines)} line(s) on this page")
        for i, line_image in enumerate(lines):
            st.markdown(f"**Line {i + 1}**")
            image_col, edit_col = st.columns([1, 2])
            with image_col:
                st.image(line_image, width="stretch")
            with edit_col:
                widget_key = f"txt_{page}_{version}_{i}"
                if widget_key not in ss:
                    ss[widget_key] = ss.texts.get((page, version, i), "")
                st.text_input(
                    f"Line {i + 1} text",
                    key=widget_key,
                    label_visibility="collapsed",
                    placeholder="Type the exact Tamil text",
                    on_change=remember,
                    args=(page, version, i, widget_key),
                )
                buttons = [(label, "tesseract", path) for label, path in tesseract_buttons]
                for row_start in range(0, len(buttons), 3):
                    row = buttons[row_start:row_start + 3]
                    row_cols = st.columns(len(row))
                    for offset, (label, method, model_path) in enumerate(row):
                        with row_cols[offset]:
                            st.button(
                                label,
                                key=f"ext_{page}_{version}_{i}_{row_start + offset}",
                                width="stretch",
                                on_click=extract_line,
                                args=(page, version, i, widget_key, method, model_path),
                            )
            st.divider()
    else:
        st.info("Page not processed yet. Press 'Process this page' to split it into lines.")

if ss.get("extract_error"):
    st.error(ss["extract_error"])
    del ss["extract_error"]

st.divider()
st.subheader("Save")
out_dir = st.text_input("Output folder (image + .gt.txt pairs)", value=CONFIG.get("output_dir", "runs/labeled/dataset"))

processed_count = len(ss.processed)
if processed_count:
    st.caption(f"{processed_count} page(s) processed in this session.")
    if st.button("Save labeled lines", type="primary"):
        saved, skipped = save_dataset(ss.processed, ss.texts, Path(out_dir))
        st.success(f"Saved {saved} line pair(s) to {Path(out_dir).resolve()}. Skipped {skipped} empty line(s).")
        if saved and skipped:
            st.warning("Lines with empty text are not saved; fill them in and save again to add them.")
else:
    st.caption("No pages processed yet.")

with st.expander("How to load the saved folder into training"):
    st.code(
        "python -m train import-real --dataset runs/mixed-v1/dataset "
        f"--directory {out_dir} --source-id book001 --split train",
        language="powershell",
    )
