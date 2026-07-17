from pathlib import Path
import copy
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.config import DEFAULT_CONFIG, validate_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_summary_has_one_fixed_local_format(self):
        summary = DEFAULT_CONFIG["summary"]
        self.assertEqual(summary["model"], "qwen3.6:27b")
        self.assertEqual(summary["ollama_url"], "http://127.0.0.1:11435")
        self.assertEqual(summary["max_output_tokens"], 16384)
        self.assertNotIn("detail_level", summary)
        self.assertNotIn("mode", summary)
        self.assertNotIn("fallback_model", summary)

    def test_rejects_remote_ollama(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["summary"]["ollama_url"] = "https://example.com"
        with self.assertRaisesRegex(ValueError, "loopback"):
            validate_config(config)

    def test_rejects_unverified_context_and_small_output(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["summary"]["context_tokens"] = 65537
        with self.assertRaisesRegex(ValueError, "65536"):
            validate_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["summary"]["max_output_tokens"] = 4096
        with self.assertRaisesRegex(ValueError, "8192"):
            validate_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["summary"]["max_output_tokens"] = 8192
        validate_config(config)
        config["summary"]["max_output_tokens"] = config["summary"]["context_tokens"]
        with self.assertRaisesRegex(ValueError, "smaller"):
            validate_config(config)

    def test_rejects_invalid_silence_and_video_settings(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["silence"]["target_silence"] = 2.0
        with self.assertRaisesRegex(ValueError, "target_silence"):
            validate_config(config)
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["video"]["crf"] = 99
        with self.assertRaisesRegex(ValueError, "crf"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
