import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from train import improve
from train.common import write_json
from train.config import Config


class ImprovementHelpersTests(unittest.TestCase):
    def test_label_pairs_and_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.gt.txt").write_text("மாலை\n", encoding="utf-8")
            (root / "a.png").write_bytes(b"image")
            pairs = improve.label_pairs(root)
            self.assertEqual(list(pairs), ["a"])
            self.assertFalse(improve.changed_labels(pairs, {"a": pairs["a"].fingerprint}))
            (root / "a.gt.txt").write_text("காலை\n", encoding="utf-8")
            edited = improve.label_pairs(root)
            self.assertEqual(improve.changed_labels(edited, {"a": pairs["a"].fingerprint}), {"a"})
            (root / "b.gt.txt").write_text("ணா\n", encoding="utf-8")
            (root / "b.png").write_bytes(b"image")
            added = improve.label_pairs(root)
            self.assertEqual(improve.changed_labels(added, {"a": edited["a"].fingerprint}), {"b"})

    def test_next_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(improve.next_iteration(root), 1)
            (root / "iter-003").mkdir()
            (root / "iter-001").mkdir()
            (root / "notes.txt").write_text("x")
            self.assertEqual(improve.next_iteration(root), 4)

    def test_comparison(self):
        report = {"models": {
            "baseline": {"overall": {"cer": 0.2, "character_accuracy": 0.8}},
            "finetuned": {"overall": {"cer": 0.1, "character_accuracy": 0.9}},
        }}
        result = improve.comparison(report, 0.0)
        self.assertAlmostEqual(result["delta"], 0.1)
        self.assertTrue(result["improved"])
        self.assertFalse(improve.comparison(report, 0.2)["improved"])
        missing = {"models": {"baseline": {"overall": {}}, "finetuned": {"overall": {}}}}
        self.assertIsNone(improve.comparison(missing, 0.0)["improved"])

    def test_stage_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "labels"
            source.mkdir()
            (source / "one.gt.txt").write_text("மாலை\n", encoding="utf-8")
            (source / "one.png").write_bytes(b"image")
            pair = improve.label_pairs(source)["one"]
            staged = improve.stage_labels(root / "incoming", [pair])
            self.assertEqual((staged / "one.png").read_bytes(), b"image")
            self.assertEqual((staged / "one.gt.txt").read_text(encoding="utf-8"), "மாலை\n")
            with self.assertRaises(ValueError):
                improve.stage_labels(staged, [pair])


class ImproveLoopTests(unittest.TestCase):
    def test_prints_and_records_accuracy_comparison(self):
        cfg = Config.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic = root / "synthetic"
            synthetic.mkdir()
            labels = root / "labeled"
            labels.mkdir()
            (labels / "one.gt.txt").write_text("மாலை\n", encoding="utf-8")
            (labels / "one.png").write_bytes(b"image")
            args = argparse.Namespace(synthetic_dataset=synthetic, labels=labels,
                                      real_test_labels=None, output=root / "improve",
                                      source_id="labels", real_test_source_id="labels_test", base=None,
                                      model_name="test", iterations=10, learning_rate=0.0001,
                                      split="test", margin=0.0, max_iterations=2,
                                      wait_seconds=0, timeout=0, once=False)

            def fake_train(dataset, output, base, name, iterations, learning_rate, **kwargs):
                output.mkdir(parents=True)
                model = output / (name + ".traineddata")
                model.write_bytes(b"model")
                return model

            def fake_evaluate(dataset, base, model, output, split, psm):
                write_json(output / "metrics.json", {"models": {
                    "baseline": {"overall": {"cer": 0.2, "wer": 0.3, "character_accuracy": 0.8,
                                             "grapheme_accuracy": 0.7, "exact_line_accuracy": 0.5}},
                    "finetuned": {"overall": {"cer": 0.1, "wer": 0.2, "character_accuracy": 0.9,
                                              "grapheme_accuracy": 0.8, "exact_line_accuracy": 0.6}},
                }})

            with patch.object(improve, "import_real") as imported, \
                    patch.object(improve, "train", side_effect=fake_train), \
                    patch.object(improve, "evaluate", side_effect=fake_evaluate):
                improve.improve(cfg, args)
            imported.assert_called_once()
            history = (root / "improve" / "accuracy_history.json").read_text(encoding="utf-8")
            self.assertIn('"improved": true', history)
            self.assertIn("test_iter001.traineddata", history)
            self.assertTrue((root / "improve" / "iter-001" / "evaluation" / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
