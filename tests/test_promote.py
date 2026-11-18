import argparse
import json
from pathlib import Path
import tempfile
import unittest

from train import promote
from train.common import write_json
from train.config import Config


class PromoteHelpersTests(unittest.TestCase):
    def test_select_best_prefers_higher_accuracy_and_lower_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "a.traineddata", root / "b.traineddata"
            first.write_bytes(b"a")
            second.write_bytes(b"b")
            records = [
                {"iteration": 1, "model": str(first),
                 "finetuned": {"character_accuracy": 0.8, "cer": 0.2},
                 "labeled_lines": 10, "split": "test"},
                {"iteration": 2, "model": str(second),
                 "finetuned": {"character_accuracy": 0.9, "cer": 0.1},
                 "labeled_lines": 20, "split": "val"},
            ]
            self.assertEqual(promote.select_best(records, "character_accuracy")["iteration"], 2)
            self.assertEqual(promote.select_best(records, "cer")["iteration"], 2)
            self.assertEqual(promote.select_best(records, "character_accuracy", "test")["iteration"], 1)
            with self.assertRaises(ValueError):
                promote.select_best(records, "unknown")


class PromoteTests(unittest.TestCase):
    def test_promotes_best_iteration_and_points_labeller(self):
        cfg = Config.load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "accuracy_history.json"
            base = root / "tam.traineddata"
            base.write_bytes(b"base")
            records = []
            for iteration, accuracy in ((1, 0.83), (2, 0.91)):
                model = root / f"model{iteration}" / f"tam_iter{iteration:03d}.traineddata"
                model.parent.mkdir()
                model.write_bytes(f"model{iteration}".encode())
                evaluation = root / f"eval{iteration}"
                write_json(evaluation / "metrics.json", {"contains_real_test_data": iteration == 2})
                records.append({"iteration": iteration, "model": str(model), "base_model": str(base),
                                "baseline_model": str(base), "dataset": str(root),
                                "evaluation": str(evaluation), "split": "test", "labeled_lines": iteration * 10,
                                "baseline": {"character_accuracy": 0.8}, "finetuned": {"character_accuracy": accuracy},
                                "delta": accuracy - 0.8, "margin": 0.0, "improved": accuracy > 0.8})
            write_json(history, records)
            labeling_config = root / "labeling_config.json"
            write_json(labeling_config, {"tesseract_models": {"tesseract": "assets/models/tam.traineddata"},
                                         "output_dir": "runs/labeled/dataset"})
            args = argparse.Namespace(history=history, output=root / "release", metric="character_accuracy",
                                      split=None, labeling_config=labeling_config, force=False,
                                      no_update_config=False)
            promote.promote(cfg, args)
            release = root / "release" / "tam_iter002"
            self.assertEqual((release / "tam_iter002.traineddata").read_bytes(), b"model2")
            self.assertTrue((release / "accuracy.json").is_file())
            self.assertTrue((release / "metrics.json").is_file())
            manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_by"]["iteration"], 2)
            self.assertTrue(manifest["dataset"]["contains_real_test_data"])
            config = json.loads(labeling_config.read_text(encoding="utf-8"))
            self.assertEqual(config["tesseract_models"]["finetuned"],
                             promote.relative_path(release / "tam_iter002.traineddata"))
            self.assertEqual(config["default_tesseract_model"], "finetuned")
            self.assertEqual(config["output_dir"], "runs/labeled/dataset")
            with self.assertRaises(ValueError):
                promote.promote(cfg, args)


if __name__ == "__main__":
    unittest.main()
