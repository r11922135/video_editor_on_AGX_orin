from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.summary import (  # noqa: E402
    MIN_ENGLISH_WORDS,
    MIN_OVERVIEW_PARAGRAPHS,
    SummaryError,
    estimate_prompt_tokens,
    render_summary_markdown,
    summary_from_raw,
    summary_schema,
    summarize_oneshot,
    validate_summary_quality,
)


PARAGRAPH = (
    "The detailed overview explains local robotics architecture commands interfaces "
    "dependencies examples tradeoffs constraints timing behavior and implementation "
    "decisions from the complete transcript. " * 8
).strip()
CHINESE_PARAGRAPH = (
    "詳細摘要依照完整逐字稿說明本機機器人架構、指令、介面、相依性、案例、"
    "設計取捨、限制、時序行為與實作決策。"
)


def generated_payload() -> dict:
    return {
        "title_en": "Local Robotics Training",
        "title_zh_tw": "本機機器人訓練",
        "overview_en": ["Unique transcript marker. " + PARAGRAPH]
        + [PARAGRAPH] * (MIN_OVERVIEW_PARAGRAPHS - 1),
        "overview_zh_tw": [CHINESE_PARAGRAPH] * MIN_OVERVIEW_PARAGRAPHS,
        "completion_marker": "complete",
    }


class SummaryTests(unittest.TestCase):
    def test_schema_is_overview_only(self):
        schema = summary_schema()
        self.assertEqual(
            list(schema["properties"]),
            [
                "title_en",
                "title_zh_tw",
                "overview_en",
                "overview_zh_tw",
                "completion_marker",
            ],
        )
        self.assertEqual(schema["required"][-1], "completion_marker")
        self.assertEqual(
            schema["properties"]["overview_en"]["minItems"],
            MIN_OVERVIEW_PARAGRAPHS,
        )

    def test_exact_contract_maps_without_modifying_prose(self):
        raw = generated_payload()
        summary = summary_from_raw(raw)
        self.assertEqual(raw["title_en"], summary["title"]["en"])
        self.assertEqual(raw["overview_en"], summary["overview"]["en"])
        old_shape = dict(raw, key_takeaways_en=["obsolete"])
        with self.assertRaisesRegex(SummaryError, "fields must be exactly"):
            summary_from_raw(old_shape)

    def test_markdown_contains_only_overview(self):
        summary = summary_from_raw(generated_payload())
        english = render_summary_markdown(
            summary, language="en", source_name="training.mp4", model="qwen"
        )
        chinese = render_summary_markdown(
            summary, language="zh_tw", source_name="training.mp4", model="qwen"
        )
        self.assertIn("## Overview", english)
        self.assertIn("## 詳細摘要", chinese)
        for removed in ("Key Takeaways", "Uncertainties", "Action Items"):
            self.assertNotIn(removed, english)

    def test_quality_gate_requires_detail_but_has_no_brittle_maximum(self):
        summary = summary_from_raw(generated_payload())
        quality = validate_summary_quality(summary)
        self.assertEqual(quality["overview_count"], MIN_OVERVIEW_PARAGRAPHS)
        self.assertGreaterEqual(quality["english_visible_word_count"], MIN_ENGLISH_WORDS)

        summary["overview"]["en"].extend([PARAGRAPH] * 12)
        summary["overview"]["zh_tw"].extend([CHINESE_PARAGRAPH] * 12)
        self.assertEqual(validate_summary_quality(summary)["overview_count"], 20)

    def test_quality_gate_rejects_short_misaligned_or_wrong_language(self):
        summary = summary_from_raw(generated_payload())
        summary["overview"]["en"] = summary["overview"]["en"][:7]
        summary["overview"]["zh_tw"] = summary["overview"]["zh_tw"][:7]
        with self.assertRaisesRegex(SummaryError, "at least 8"):
            validate_summary_quality(summary)

        summary = summary_from_raw(generated_payload())
        summary["overview"]["zh_tw"].pop()
        with self.assertRaisesRegex(SummaryError, "aligned"):
            validate_summary_quality(summary)

        summary = summary_from_raw(generated_payload())
        summary["overview"]["en"] = ["Too short."] * 8
        with self.assertRaisesRegex(SummaryError, "1,000"):
            validate_summary_quality(summary)

        summary = summary_from_raw(generated_payload())
        summary["overview"]["zh_tw"][0] = "English only"
        with self.assertRaisesRegex(SummaryError, "Chinese text"):
            validate_summary_quality(summary)

    def test_fresh_oneshot_is_local_and_preserves_raw(self):
        generated = generated_payload()
        raw_content = json.dumps(generated, ensure_ascii=False)
        response = {
            "model": "test-model",
            "message": {"content": raw_content},
            "done_reason": "stop",
            "eval_count": 100,
        }
        with tempfile.TemporaryDirectory() as raw:
            raw_path = Path(raw) / "summary.raw.txt"
            with patch(
                "local_video_editor.summary.ollama_models",
                return_value=["test-model"],
            ), patch(
                "local_video_editor.summary._post_json",
                return_value=response,
            ) as post:
                summary, metrics = summarize_oneshot(
                    [{"start": 0.0, "end": 1.0, "text": "unique transcript marker"}],
                    source_title="Training",
                    ollama_url="http://127.0.0.1:11434",
                    model="test-model",
                    context_tokens=65536,
                    max_output_tokens=8192,
                    raw_response_path=raw_path,
                )
            saved_raw = raw_path.read_text(encoding="utf-8")
        self.assertEqual(summary["overview"]["en"], generated["overview_en"])
        self.assertEqual(saved_raw, raw_content)
        self.assertEqual(metrics["endpoint_scope"], "loopback")
        self.assertFalse(metrics["generation_reused"])
        self.assertFalse(metrics["post_generation_content_modified"])
        self.assertFalse(metrics["format_repair_used"])
        self.assertEqual(metrics["summary_format"], "detailed_overview_v1")
        payload = post.call_args.args[1]
        self.assertEqual(payload["format"], summary_schema())
        self.assertNotIn("key_takeaways_en", payload["format"]["properties"])
        self.assertIn('<window id="1"', str(payload["messages"]))

    def test_missing_marker_and_malformed_json_are_rejected_without_repair(self):
        missing = generated_payload()
        missing.pop("completion_marker")
        responses = [
            json.dumps(missing, ensure_ascii=False),
            '{"title_en":"Training","overview_en":["unfinished"}',
        ]
        for content in responses:
            response = {
                "model": "test-model",
                "message": {"content": content},
                "done_reason": "stop",
            }
            with patch(
                "local_video_editor.summary.ollama_models", return_value=["test-model"]
            ), patch(
                "local_video_editor.summary._post_json", return_value=response
            ) as post:
                with self.assertRaises(SummaryError):
                    summarize_oneshot(
                        [{"start": 0, "end": 1, "text": "Training"}],
                        source_title="Training",
                        ollama_url="http://localhost:11434",
                        model="test-model",
                        context_tokens=65536,
                        max_output_tokens=8192,
                    )
            self.assertEqual(post.call_count, 1)

    def test_output_limit_and_remote_endpoint_are_rejected(self):
        response = {
            "model": "test-model",
            "message": {"content": "unfinished"},
            "done_reason": "length",
        }
        with patch(
            "local_video_editor.summary.ollama_models", return_value=["test-model"]
        ), patch(
            "local_video_editor.summary._post_json", return_value=response
        ):
            with self.assertRaisesRegex(SummaryError, "max_output_tokens"):
                summarize_oneshot(
                    [{"start": 0, "end": 1, "text": "Training"}],
                    source_title="Training",
                    ollama_url="http://localhost:11434",
                    model="test-model",
                    context_tokens=65536,
                    max_output_tokens=8192,
                )
        with self.assertRaisesRegex(SummaryError, "Local-only"):
            summarize_oneshot(
                [{"start": 0, "end": 1, "text": "Training"}],
                source_title="Training",
                ollama_url="https://example.com",
                model="test-model",
                context_tokens=65536,
                max_output_tokens=8192,
            )

    def test_prompt_estimate_is_nonzero(self):
        self.assertEqual(estimate_prompt_tokens(""), 1)
        self.assertGreater(estimate_prompt_tokens("a" * 1000), 300)


if __name__ == "__main__":
    unittest.main()
