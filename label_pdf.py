from __future__ import annotations

import hashlib
import sys
import unicodedata
from pathlib import Path

import fitz
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_lines import extract_lines

st.set_page_config(page_title="PDF line labeler", layout="wide")


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
    ss.index = 0
    ss.processed = {}
    ss.next_version = 0
    ss.texts = {}
    if "jump_page" in ss:
        del ss["jump_page"]


def remember(page: int, version: int, index: int, key: str) -> None:
    st.session_state.texts[(page, version, index)] = st.session_state.get(key, "")


def save_dataset(processed: dict, texts: dict, out_dir: Path) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0
    for page in sorted(processed):
        info = processed[page]
        version = info["version"]
        for i, image in enumerate(info["lines"]):
            raw = texts.get((page, version, i), "")
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
if "index" not in ss:
    reset(ss)

st.title("PDF to Tamil line labels")
st.caption("Each page is split into lines only when you press Process. Type the correct text, then Save.")

uploaded = st.file_uploader("Upload a scanned PDF", type=["pdf"])

with st.sidebar:
    st.header("Line split settings")
    dpi = st.slider("Render DPI", 150, 500, 300, step=50)
    padding = st.slider("Crop padding (px)", 0, 20, 4)
    min_height = st.slider("Minimum line height (px)", 4, 80, 10)
    min_gap = st.slider("Merge gap (px)", 0, 20, 0)
    invert = st.checkbox("Light text on dark background", value=False)

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
    if st.button("Previous", disabled=ss.index <= 0):
        ss.index -= 1
        if total > 1:
            ss["jump_page"] = ss.index + 1
        st.rerun()
with nav_mid:
    st.markdown(
        f"<div style='text-align:center'>Page <b>{ss.index + 1}</b> of <b>{total}</b></div>",
        unsafe_allow_html=True,
    )
with nav_right:
    if st.button("Next", type="primary", disabled=ss.index >= total - 1):
        ss.index += 1
        if total > 1:
            ss["jump_page"] = ss.index + 1
        st.rerun()

if total > 1:
    if "jump_page" not in ss:
        ss["jump_page"] = ss.index + 1
    jump = st.select_slider("Jump to page", options=list(range(1, total + 1)), key="jump_page")
    if jump - 1 != ss.index:
        ss.index = jump - 1
        st.rerun()

page_image = render_page(pdf_bytes, ss.index, dpi)
st.image(page_image, caption=f"Page {ss.index + 1}", width="stretch")

action_left, action_right = st.columns([1, 3])
with action_left:
    process_clicked = st.button("Process this page", type="primary")
with action_right:
    if ss.index in ss.processed:
        if st.button("Clear this page's lines"):
            del ss.processed[ss.index]
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
    ss.processed[ss.index] = {"version": ss.next_version, "lines": lines}

st.divider()

info = ss.processed.get(ss.index)
if info:
    version = info["version"]
    lines = info["lines"]
    st.subheader(f"{len(lines)} line(s) on this page")
    for i, line_image in enumerate(lines):
        image_col, text_col = st.columns([1, 2])
        with image_col:
            st.image(line_image, width="stretch")
        with text_col:
            widget_key = f"txt_{ss.index}_{version}_{i}"
            if widget_key not in ss:
                ss[widget_key] = ss.texts.get((ss.index, version, i), "")
            st.text_input(
                f"Line {i + 1}",
                key=widget_key,
                placeholder="Type the exact Tamil text for this line",
                on_change=remember,
                args=(ss.index, version, i, widget_key),
            )
else:
    st.info("Page not processed yet. Press 'Process this page' to split it into lines.")

st.divider()
st.subheader("Save")
out_dir = st.text_input("Output folder (image + .gt.txt pairs)", value="runs/labeled/dataset")

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
