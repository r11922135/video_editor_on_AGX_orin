from pathlib import Path
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.cli import build_parser  # noqa: E402


class CliTests(unittest.TestCase):
    def test_process_supports_true_edit_only(self):
        args = build_parser().parse_args(
            ["process", "training.mp4", "--edit-only"]
        )
        self.assertTrue(args.edit_only)

    def test_full_process_is_default(self):
        args = build_parser().parse_args(["process", "training.mp4"])
        self.assertFalse(args.edit_only)

    def test_summarize_supports_model_override(self):
        args = build_parser().parse_args(
            ["summarize", "output/job", "--model", "qwen3.6:27b"]
        )
        self.assertEqual(args.model, "qwen3.6:27b")

    def test_only_cli_commands_remain(self):
        parser = build_parser()
        for removed in ("web", "doctor"):
            with self.assertRaises(SystemExit):
                parser.parse_args([removed])


if __name__ == "__main__":
    unittest.main()
