import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ocr_confidence import parse_confidence, parse_words, tesseract_confidence, tesseract_words


class OcrConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
        self.tsv = "\n".join([
            self.header,
            "4\t1\t1\t1\t1\t0\t0\t0\t0\t0\t-1\t",
            "5\t1\t1\t1\t1\t1\t0\t0\t30\t20\t90.0\tabc",
            "5\t1\t1\t1\t1\t2\t40\t0\t20\t20\t80.0\tde",
            "5\t1\t1\t1\t1\t3\t70\t0\t20\t20\t-1\t",
        ])

    def test_parse_words_skips_non_word_and_unconfident_rows(self):
        self.assertEqual(parse_words(self.tsv), [("abc", 90.0), ("de", 80.0)])

    def test_weighted_line_confidence(self):
        self.assertAlmostEqual(parse_confidence(self.tsv), 86.0)

    def test_no_words_returns_none_or_empty(self):
        self.assertEqual(parse_words(""), [])
        self.assertIsNone(parse_confidence(""))
        self.assertIsNone(parse_confidence("level\tconf\ttext\n5\t-1\t"))

    def test_uses_explicit_tsv_config_for_custom_tessdata_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tam.traineddata"
            model.write_bytes(b"model")
            config_contents = []

            def fake_run(args, **kwargs):
                config_contents.append(Path(args[-1]).read_text(encoding="utf-8"))
                return SimpleNamespace(returncode=0, stdout=self.tsv, stderr="")

            with patch("ocr_confidence.subprocess.run", side_effect=fake_run):
                words = tesseract_words(MagicMock(), str(model))
                confidence = tesseract_confidence(MagicMock(), str(model))
            self.assertEqual(words, [("abc", 90.0), ("de", 80.0)])
            self.assertAlmostEqual(confidence, 86.0)
            self.assertEqual(config_contents, ["tessedit_create_tsv 1\n"] * 2)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            tesseract_words(None, "does-not-exist.traineddata")


if __name__ == "__main__":
    unittest.main()
