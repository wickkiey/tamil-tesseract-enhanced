from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inference


class InferenceTests(unittest.TestCase):
    def files(self, root):
        image, model = root / "line.png", root / "custom_tamil.traineddata"
        image.write_bytes(b"image")
        model.write_bytes(b"model")
        return image, model

    def test_explicit_model_and_unicode_output(self):
        with tempfile.TemporaryDirectory() as directory:
            image, model = self.files(Path(directory))
            with patch("inference.tool", return_value="tesseract"), patch(
                    "inference.run_tool", return_value=" கொ\nமாலை \n") as run:
                self.assertEqual(inference.recognize(image, model, 6), "கொ\nமாலை")
                args = run.call_args.args[0]
                self.assertEqual(args[args.index("-l") + 1], "custom_tamil")
                self.assertEqual(args[args.index("--tessdata-dir") + 1], str(model.resolve().parent))
                self.assertEqual(args[args.index("--oem") + 1], "1")
                self.assertEqual(args[args.index("--psm") + 1], "6")

    def test_empty_output_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            image, model = self.files(Path(directory))
            with patch("inference.tool", return_value="tesseract"), patch(
                    "inference.run_tool", return_value="\n"):
                with self.assertRaisesRegex(RuntimeError, "no text"):
                    inference.recognize(image, model)

    def test_missing_input_and_invalid_model(self):
        with tempfile.TemporaryDirectory() as directory:
            image, model = self.files(Path(directory))
            with self.assertRaises(FileNotFoundError):
                inference.recognize(image.with_name("missing.png"), model)
            with self.assertRaisesRegex(ValueError, "traineddata"):
                inference.recognize(image, image)
            with self.assertRaisesRegex(ValueError, "PSM"):
                inference.recognize(image, model, 0)

    def test_log_cannot_modify_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            image, model = self.files(Path(directory))
            for log in (image, model):
                with self.assertRaisesRegex(ValueError, "log"):
                    inference.recognize(image, model, log=log)

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, model = self.files(root)
            output = root / "output.txt"
            output.write_text("keep", encoding="utf-8")
            with patch("sys.argv", ["inference.py", "--image", str(image), "--model", str(model),
                                    "--output", str(output)]), patch("inference.recognize") as recognize:
                self.assertEqual(inference.main(), 1)
                recognize.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
