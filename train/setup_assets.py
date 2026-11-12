"""Stage 1 - setup: download licensed fonts and the floating-point Tamil base model."""

from __future__ import annotations

from pathlib import Path
import zipfile

from .common import ROOT, digest, download, write_json


def setup(assets_dir: Path = ROOT / "assets") -> None:
    fonts = assets_dir / "fonts"
    records: list[dict[str, str]] = []

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(assets_dir.parent))
        except ValueError:
            return str(path)

    for family in ("Sans", "Serif"):
        folder = f"noto{family.lower()}tamil"
        url = (f"https://raw.githubusercontent.com/google/fonts/main/ofl/{folder}/"
               f"Noto{family}Tamil%5Bwdth%2Cwght%5D.ttf")
        path = fonts / f"Noto{family}Tamil.ttf"
        download(url, path)
        download(f"https://raw.githubusercontent.com/google/fonts/main/ofl/{folder}/OFL.txt",
                 fonts / f"Noto{family}Tamil-OFL.txt")
        records.append({"file": rel(path), "url": url,
                        "sha256": digest(path.read_bytes()), "license": "SIL OFL 1.1"})
    for family, version in (("Korkai", "1_0_0_3"), ("Vaagai", "1_0_0_0")):
        url = f"https://tamilfontstudio.net/downloads/{family.lower()}_v{version}.zip"
        archive = fonts / f"{family.lower()}.zip"
        download(url, archive)
        with zipfile.ZipFile(archive) as package:
            for name, output in ((f"{family}-Regular.ttf", f"{family}-Regular.ttf"),
                                 ("OFL.txt", f"{family}-OFL.txt")):
                matches = [p for p in package.namelist() if Path(p).name == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {name} in {archive}, found {matches}")
                (fonts / output).write_bytes(package.read(matches[0]))
        path = fonts / f"{family}-Regular.ttf"
        records.append({"file": rel(path), "url": url,
                        "sha256": digest(path.read_bytes()), "license": "SIL OFL 1.1"})
    model = assets_dir / "models" / "tam.traineddata"
    url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/tam.traineddata"
    download(url, model)
    records.append({"file": rel(model), "url": url,
                    "sha256": digest(model.read_bytes()), "license": "Apache 2.0"})
    download("https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/LICENSE",
             model.parent / "LICENSE")
    write_json(assets_dir / "provenance.json", records)
    print("Assets ready. Fonts are OFL-licensed; Tamil base model is from tessdata_best.")
