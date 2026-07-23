from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.config import (  # noqa: E402
    DEFAULT_CONFIG,
    load_config,
    validate_config,
)


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

    def test_subtitle_defaults_are_bounded_for_local_inference(self):
        subtitles = DEFAULT_CONFIG["subtitles"]
        self.assertNotIn("aligner_model", subtitles)
        self.assertEqual(subtitles["correction_context_tokens"], 65536)
        self.assertEqual(subtitles["correction_output_tokens"], 2048)
        self.assertEqual(subtitles["correction_candidate_limit"], 48)
        self.assertEqual(subtitles["correction_rule_safety_cap"], 32)
        self.assertEqual(subtitles["correction_scope_seconds"], 120)

    def test_rejects_invalid_subtitle_settings(self):
        cases = (
            ("correction_context_tokens", 8191, "correction_context_tokens"),
            ("correction_context_tokens", 65537, "correction_context_tokens"),
            ("correction_output_tokens", 127, "correction_output_tokens"),
            ("correction_output_tokens", 65536, "correction_output_tokens"),
            ("correction_candidate_limit", 7, "correction_candidate_limit"),
            ("correction_candidate_limit", 97, "correction_candidate_limit"),
            ("correction_rule_safety_cap", 0, "correction_rule_safety_cap"),
            ("correction_rule_safety_cap", 65, "correction_rule_safety_cap"),
            ("correction_scope_seconds", 29, "correction_scope_seconds"),
            ("correction_scope_seconds", 151, "correction_scope_seconds"),
        )
        for key, value, message in cases:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(DEFAULT_CONFIG)
                config["subtitles"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_config(config)

    def test_legacy_aligner_settings_are_normalized_before_use(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "subtitles": {
                            "aligner_model": "unused",
                            "alignment_chunk_seconds": 90,
                        }
                    }
                )
            )

            config = load_config(path)

        subtitles = config["subtitles"]
        self.assertEqual(subtitles["correction_scope_seconds"], 90)
        self.assertNotIn("alignment_chunk_seconds", subtitles)
        self.assertNotIn("aligner_model", subtitles)


if __name__ == "__main__":
    unittest.main()
