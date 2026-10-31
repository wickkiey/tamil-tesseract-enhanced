from dataclasses import replace
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import tamil_ocr as ocr


class MetricsTests(unittest.TestCase):
    def test_perfect_tamil(self):
        result = ocr.metric_summary([("மாலை ணா றா", "மாலை ணா றா")])
        self.assertEqual(result["cer"], 0)
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["grapheme_accuracy"], 1)
        self.assertEqual(result["target_accuracy"]["nedil"]["reference"], 2)

    def test_missing_vowel_mark(self):
        result = ocr.metric_summary([("ணா", "ண")])
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(result["grapheme_error_rate"], 1)
        self.assertEqual(result["target_accuracy"]["nedil"]["accuracy"], 0)

    def test_insertions_are_not_hidden(self):
        result = ocr.metric_summary([("அ", "அ ஆ இ")])
        self.assertGreater(result["cer"], 1)
        self.assertEqual(result["character_accuracy"], 0)

    def test_empty_prediction_is_failure(self):
        result = ocr.metric_summary([("மாலை", "")])
        self.assertEqual(result["cer"], 1)
        self.assertEqual(result["wer"], 1)

    def test_alignment_counts_insert_delete_substitute(self):
        self.assertEqual(ocr.alignment(list("abc"), list("adc"))[0], 1)
        self.assertEqual(ocr.alignment([], list("abc"))[0], 3)
        self.assertEqual(ocr.alignment(list("abc"), [])[0], 3)

    def test_normalization_and_controls(self):
        self.assertEqual(ocr.normalized(" கொ \n"), "கொ")
        self.assertEqual(ocr.normalized("கொ"), "கொ")
        for text in ("", "a\nb", "a\tb", "a\u200bb"):
            with self.assertRaises(ValueError):
                ocr.normalized(text)

    def test_line_boxes(self):
        box = ocr.line_box("ணா றா", 300, 80)
        self.assertTrue(box.endswith("\t 0 0 300 80 0\n"))
        self.assertIn("ண 0 0 300 80 0\n", box)
        self.assertIn("ா 0 0 300 80 0\n", box)

    def test_split_is_deterministic_and_normalizes_duplicates(self):
        texts = [f"மாலை {i}" for i in range(20)] + ["மாலை 0", " மாலை 0 "]
        first = ocr.split_texts(texts, 42)
        self.assertEqual(first, ocr.split_texts(list(reversed(texts)), 42))
        self.assertEqual(len(first), 20)
        self.assertEqual(set(first.values()), set(ocr.SPLITS))

    def test_diagnostic_coverage_assignment_preserves_split_sizes(self):
        texts = ocr.starter_corpus()[:7] + list(ocr.DIAGNOSTICS)
        assignments = ocr.split_texts(texts, 42)
        before = ocr.Counter(assignments.values())
        ocr.assign_diagnostics(assignments)
        self.assertEqual(before, ocr.Counter(assignments.values()))
        for text, split in zip(ocr.DIAGNOSTICS, ocr.SPLITS):
            self.assertEqual(assignments[text], split)
            for cluster in ocr.TARGETS["ai"] + ocr.TARGETS["nedil"]:
                self.assertIn(cluster, text)

    def test_tool_stderr_does_not_become_prediction(self):
        with patch("tamil_ocr.subprocess.run", return_value=subprocess.CompletedProcess(
                ["ocr"], 0, stdout="மாலை\n", stderr="warning\n")):
            self.assertEqual(ocr.run_tool(["ocr"]), "மாலை\n")

    def test_tool_failure_is_explicit(self):
        with patch("tamil_ocr.subprocess.run", return_value=subprocess.CompletedProcess(
                ["ocr"], 1, stdout="", stderr="bad model")):
            with self.assertRaisesRegex(RuntimeError, "bad model"):
                ocr.run_tool(["ocr"])


class DatasetTests(unittest.TestCase):
    def sample(self, root, identifier, text, split, source):
        image = root / (identifier + ".png")
        gt = root / (identifier + ".gt.txt")
        pixels = Image.new("L", (80, 50), 255)
        pixels.putpixel((10, 10), sum(identifier.encode()) % 255)
        pixels.save(image)
        gt.write_text(text + "\n", encoding="utf-8")
        return ocr.Sample(identifier, image.name, gt.name, split, source, "real", "real", "real",
                          ocr.digest(text.encode()), ocr.digest(image.read_bytes()))

    def test_text_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            b = self.sample(root, "b", "மாலை", "test", "book2")
            ocr.save_manifest(root, [a, b])
            with self.assertRaisesRegex(ValueError, "leakage"):
                ocr.load_manifest(root)

    def test_changed_gt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            ocr.save_manifest(root, [a])
            (root / a.gt).write_text("காலை", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Transcription changed"):
                ocr.load_manifest(root)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            ocr.save_manifest(root, [replace(a, image="..\\outside.png")])
            with self.assertRaisesRegex(ValueError, "out-of-dataset"):
                ocr.load_manifest(root)

    def test_import_source_split_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "real:book1")
            ocr.save_manifest(root, [a])
            with self.assertRaisesRegex(ValueError, "cannot span"):
                ocr.import_real(root, root, "book1", "test")

    def test_source_leakage_rejected_even_with_different_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            b = self.sample(root, "b", "காலை", "test", "book1")
            ocr.save_manifest(root, [a, b])
            with self.assertRaisesRegex(ValueError, "leakage"):
                ocr.load_manifest(root)

    def test_identical_image_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            b = self.sample(root, "b", "காலை", "test", "book2")
            (root / b.image).write_bytes((root / a.image).read_bytes())
            b = replace(b, image_sha256=a.image_sha256)
            ocr.save_manifest(root, [a, b])
            with self.assertRaisesRegex(ValueError, "leakage"):
                ocr.load_manifest(root)

    def test_real_import_preserves_valid_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.sample(root, "a", "மாலை", "train", "book1")
            ocr.save_manifest(root, [a])
            incoming = root / "incoming"
            incoming.mkdir()
            self.sample(incoming, "new", "காலை", "train", "book2")
            (root / "train").mkdir()
            ocr.import_real(root, incoming, "book2", "train")
            samples = ocr.load_manifest(root)
            self.assertEqual(len(samples), 2)
            self.assertEqual(samples[1].source_id, "real:book2")
            self.assertEqual((root / samples[1].gt).read_text(encoding="utf-8"), "காலை\n")
            with self.assertRaisesRegex(ValueError, "already imported"):
                ocr.import_real(root, incoming, "book2", "train")


@unittest.skipUnless((ocr.ROOT / "assets" / "fonts" / "Korkai-Regular.ttf").is_file(),
                     "Run setup to enable real font tests")
class ShapingTests(unittest.TestCase):
    def test_historical_features_really_change_nedil_and_ai(self):
        profiles = ocr.load_profiles(ocr.ROOT / "profiles.json")
        old = next(p for p in profiles if p.name == "korkai_historical")
        modern = next(p for p in profiles if p.name == "korkai_modern")
        for text in ("ணா", "றா", "னா", "லை", "ளை", "ணை", "னை"):
            with self.subTest(text=text):
                self.assertGreater(ocr.pixel_changes(ocr.render(text, old), ocr.render(text, modern)), 0)

    def test_false_historical_profile_rejected(self):
        profile = ocr.Profile("false_history", ocr.ROOT / "assets/fonts/NotoSerifTamil.ttf", ("ss01",), "historical")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "absent from GSUB"):
                ocr.verify_profiles([profile], Path(directory), ["மாலை"])

    def test_generation_split_integrity_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a", root / "b"
            for out in (first, second):
                ocr.generate(out, None, ocr.ROOT / "profiles.json", 10, 2, 42)
            a, b = ocr.load_manifest(first), ocr.load_manifest(second)
            self.assertEqual(a, b)
            self.assertEqual(len(a), 120)
            self.assertEqual(set(s.split for s in a), set(ocr.SPLITS))
            self.assertEqual({s.kind for s in a}, {"synthetic"})
            with self.assertRaisesRegex(ValueError, "nonempty"):
                ocr.generate(first, None, ocr.ROOT / "profiles.json", 10, 1, 42)


if __name__ == "__main__":
    unittest.main()
