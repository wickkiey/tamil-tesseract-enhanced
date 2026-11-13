from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    best_t = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            best_t = t
    return best_t


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    return runs


def _merge(runs: list[tuple[int, int]], min_gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [[runs[0][0], runs[0][1]]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def detect_line_boxes(
    gray: Image.Image,
    threshold: int | None = None,
    min_gap: int = 0,
    min_height: int = 10,
    padding: int = 4,
    invert: bool = False,
) -> list[tuple[int, int, int, int]]:
    arr = np.asarray(gray)
    height, width = arr.shape

    if threshold is None:
        threshold = otsu_threshold(arr)

    ink = arr > threshold if invert else arr < threshold
    row_ink = ink.sum(axis=1)
    active = row_ink > max(1, int(width * 0.002))

    boxes: list[tuple[int, int, int, int]] = []
    for start, end in _merge(_runs(active), min_gap):
        if end - start < min_height:
            continue
        cols = np.where(ink[start:end].any(axis=0))[0]
        if cols.size == 0:
            continue
        left = max(0, int(cols[0]) - padding)
        right = min(width, int(cols[-1]) + 1 + padding)
        top = max(0, start - padding)
        bottom = min(height, end + padding)
        boxes.append((left, top, right, bottom))
    return boxes


def extract_lines(
    image: Image.Image,
    threshold: int | None = None,
    min_gap: int = 0,
    min_height: int = 10,
    padding: int = 4,
    invert: bool = False,
) -> list[Image.Image]:
    gray = image.convert("L")
    return [gray.crop(box) for box in detect_line_boxes(gray, threshold, min_gap, min_height, padding, invert)]


def split_lines(
    image_path: Path,
    output_dir: Path,
    threshold: int | None = None,
    min_gap: int = 0,
    min_height: int = 10,
    padding: int = 4,
    save_gt: bool = False,
    invert: bool = False,
) -> int:
    with Image.open(image_path) as img:
        gray = img.convert("L")

    boxes = detect_line_boxes(gray, threshold, min_gap, min_height, padding, invert)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for box in boxes:
        crop = gray.crop(box)
        saved += 1
        stem = f"line_{saved:04d}"
        crop.save(output_dir / f"{stem}.png", dpi=(300, 300))
        if save_gt:
            (output_dir / f"{stem}.gt.txt").write_text("", encoding="utf-8")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a scanned page image into per-line images.")
    parser.add_argument("--image", default="image.png", help="Input page image")
    parser.add_argument("--output", default="lines", help="Output folder for line images")
    parser.add_argument("--threshold", type=int, default=None, help="Binarization threshold (default: Otsu)")
    parser.add_argument("--min-gap", type=int, default=0, help="Merge rows separated by <= this many pixels")
    parser.add_argument("--min-height", type=int, default=10, help="Discard bands shorter than this")
    parser.add_argument("--padding", type=int, default=4, help="Pixels of margin around each crop")
    parser.add_argument("--save-gt", action="store_true", help="Also write empty .gt.txt stubs for import-real")
    parser.add_argument("--invert", action="store_true", help="Treat light text on dark background as ink")
    args = parser.parse_args()

    count = split_lines(
        Path(args.image),
        Path(args.output),
        args.threshold,
        args.min_gap,
        args.min_height,
        args.padding,
        args.save_gt,
        args.invert,
    )
    print(f"Saved {count} line image(s) to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
