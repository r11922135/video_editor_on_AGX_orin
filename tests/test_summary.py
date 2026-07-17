import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.summary import (  # noqa: E402
    MAX_OVERVIEW_PARAGRAPHS,
    MIN_ENGLISH_WORDS,
    MIN_OVERVIEW_PARAGRAPHS,
    SummaryError,
    adaptive_output_token_budget,
    adaptive_summary_targets,
    english_summary_schema,
    estimate_prompt_tokens,
    render_summary_markdown,
    summary_from_raw,
    summary_schema,
    summarize_two_stage,
    translation_schema,
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


def generated_english_payload() -> dict:
    combined = generated_payload()
    return {
        "title_en": combined["title_en"],
        "overview_en": combined["overview_en"],
        "completion_marker": "complete",
    }


def generated_translation_payload(paragraph_count: int | None = None) -> dict:
    count = paragraph_count or MIN_OVERVIEW_PARAGRAPHS
    return {
        "title_zh_tw": "本機機器人訓練",
        "overview_zh_tw": [CHINESE_PARAGRAPH] * count,
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
        self.assertEqual(
            schema["properties"]["overview_en"]["maxItems"],
            MAX_OVERVIEW_PARAGRAPHS,
        )
        self.assertEqual(
            list(english_summary_schema()["properties"]),
            ["title_en", "overview_en", "completion_marker"],
        )
        translated = translation_schema(MIN_OVERVIEW_PARAGRAPHS)
        self.assertEqual(
            translated["properties"]["overview_zh_tw"]["minItems"],
            MIN_OVERVIEW_PARAGRAPHS,
        )
        self.assertEqual(
            translated["properties"]["overview_zh_tw"]["maxItems"],
            MIN_OVERVIEW_PARAGRAPHS,
        )

    def test_adaptive_targets_scale_for_the_two_measured_recordings(self):
        motors = adaptive_summary_targets(6457, 11)
        self.assertEqual(
            motors,
            {
                "english_words_min": 1300,
                "english_words_max": 1800,
                "paragraphs_min": 13,
                "paragraphs_max": 18,
            },
        )
        ros2 = adaptive_summary_targets(9424, 12)
        self.assertEqual(
            ros2,
            {
                "english_words_min": 1900,
                "english_words_max": 2600,
                "paragraphs_min": 16,
                "paragraphs_max": 22,
            },
        )

    def test_adaptive_targets_are_bounded(self):
        short = adaptive_summary_targets(10, 1)
        self.assertEqual(short["english_words_min"], 1200)
        self.assertEqual(short["english_words_max"], 1600)
        long = adaptive_summary_targets(100000, 12)
        self.assertEqual(long["english_words_min"], 2200)
        self.assertEqual(long["english_words_max"], 2600)

    def test_output_budget_uses_config_as_a_cap(self):
        self.assertEqual(adaptive_output_token_budget(1600, 16384), 8192)
        self.assertEqual(adaptive_output_token_budget(2400, 16384), 8192)
        self.assertEqual(adaptive_output_token_budget(2600, 16384), 8192)
        self.assertEqual(adaptive_output_token_budget(4000, 16384), 10240)
        self.assertEqual(adaptive_output_token_budget(2600, 8192), 8192)

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

    def test_two_stage_generation_is_local_aligned_and_preserves_both_raw_outputs(self):
        english_generated = generated_english_payload()
        translated_generated = generated_translation_payload()
        english_raw = json.dumps(english_generated, ensure_ascii=False)
        translation_raw = json.dumps(translated_generated, ensure_ascii=False)
        english_response = {
            "model": "test-model",
            "message": {"content": english_raw},
            "done_reason": "stop",
            "eval_count": 100,
            "load_duration": 123,
        }
        translation_response = {
            "model": "test-model",
            "message": {"content": translation_raw},
            "done_reason": "stop",
            "eval_count": 80,
            "load_duration": 0,
        }
        progress = []
        with tempfile.TemporaryDirectory() as raw:
            english_path = Path(raw) / "summary.en.raw.txt"
            translation_path = Path(raw) / "summary.zh-TW.raw.txt"
            with patch(
                "local_video_editor.summary.ollama_models",
                return_value=["test-model"],
            ), patch(
                "local_video_editor.summary._post_json",
                side_effect=[english_response, translation_response],
            ) as post:
                summary, metrics = summarize_two_stage(
                    [{"start": 0.0, "end": 1.0, "text": "unique transcript marker"}],
                    source_title="Training",
                    ollama_url="http://127.0.0.1:11434",
                    model="test-model",
                    context_tokens=65536,
                    max_output_tokens=8192,
                    english_raw_response_path=english_path,
                    translation_raw_response_path=translation_path,
                    progress=progress.append,
                )
            saved_english = english_path.read_text(encoding="utf-8")
            saved_translation = translation_path.read_text(encoding="utf-8")
        self.assertEqual(summary["overview"]["en"], english_generated["overview_en"])
        self.assertEqual(
            summary["overview"]["zh_tw"],
            translated_generated["overview_zh_tw"],
        )
        self.assertEqual(saved_english, english_raw)
        self.assertEqual(saved_translation, translation_raw)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(progress), 2)
        self.assertEqual(metrics["endpoint_scope"], "loopback")
        self.assertFalse(metrics["generation_reused"])
        self.assertFalse(metrics["post_generation_content_modified"])
        self.assertTrue(metrics["composition_only"])
        self.assertFalse(metrics["format_repair_used"])
        self.assertEqual(metrics["mode"], "two_stage")
        self.assertEqual(metrics["summary_format"], "detailed_overview_v3")
        self.assertEqual(metrics["translation_source"], "generated_english_only")
        translation_source = json.dumps(
            {
                "title_en": english_generated["title_en"],
                "overview_en": english_generated["overview_en"],
            },
            ensure_ascii=False,
            indent=2,
        )
        self.assertEqual(
            metrics["translation_source_sha256"],
            hashlib.sha256(translation_source.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(metrics["summary_target_english_words_min"], 1200)
        self.assertEqual(metrics["summary_target_english_words_max"], 1600)
        self.assertEqual(metrics["summary_target_paragraphs_min"], 10)
        self.assertIn("summary_length_target_met", metrics)
        self.assertEqual(metrics["configured_max_output_tokens"], 8192)
        self.assertEqual(metrics["requested_num_predict_per_stage"], 8192)
        english_request = post.call_args_list[0].args[1]
        translation_request = post.call_args_list[1].args[1]
        self.assertEqual(english_request["format"], english_summary_schema())
        self.assertEqual(
            translation_request["format"],
            translation_schema(MIN_OVERVIEW_PARAGRAPHS),
        )
        self.assertEqual(english_request["keep_alive"], "2m")
        self.assertEqual(translation_request["keep_alive"], 0)
        self.assertEqual(
            english_request["options"]["num_ctx"],
            translation_request["options"]["num_ctx"],
        )
        self.assertEqual(
            metrics["shared_runner_context_tokens"],
            english_request["options"]["num_ctx"],
        )
        self.assertTrue(metrics["same_runner_context_requested"])
        self.assertEqual(metrics["english_stage"]["load_duration_ns"], 123)
        self.assertEqual(metrics["translation_stage"]["load_duration_ns"], 0)
        self.assertIn('<window id="1"', str(english_request["messages"]))
        self.assertIn("Every non-trivial window", str(english_request["messages"]))
        self.assertIn("launch or data-recording", str(english_request["messages"]))
        self.assertNotIn("<transcript>", str(translation_request["messages"]))
        self.assertIn("Unique transcript marker", str(translation_request["messages"]))
        self.assertEqual(english_request["options"]["num_predict"], 8192)
        self.assertEqual(translation_request["options"]["num_predict"], 8192)

    def test_bad_english_stage_is_rejected_before_translation(self):
        missing = generated_english_payload()
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
            ) as post, patch(
                "local_video_editor.summary._best_effort_unload"
            ) as unload:
                with self.assertRaises(SummaryError):
                    summarize_two_stage(
                        [{"start": 0, "end": 1, "text": "Training"}],
                        source_title="Training",
                        ollama_url="http://localhost:11434",
                        model="test-model",
                        context_tokens=65536,
                        max_output_tokens=8192,
                    )
            self.assertEqual(post.call_count, 1)
            unload.assert_called_once_with(
                "http://localhost:11434", "test-model"
            )

    def test_translation_must_match_the_generated_english_paragraph_count(self):
        bad_translation = generated_translation_payload(
            MIN_OVERVIEW_PARAGRAPHS - 1
        )
        responses = [
            {
                "model": "test-model",
                "message": {
                    "content": json.dumps(
                        generated_english_payload(), ensure_ascii=False
                    )
                },
                "done_reason": "stop",
            },
            {
                "model": "test-model",
                "message": {
                    "content": json.dumps(bad_translation, ensure_ascii=False)
                },
                "done_reason": "stop",
            },
        ]
        with patch(
            "local_video_editor.summary.ollama_models", return_value=["test-model"]
        ), patch(
            "local_video_editor.summary._post_json", side_effect=responses
        ) as post:
            with self.assertRaisesRegex(SummaryError, "exactly 8 aligned"):
                summarize_two_stage(
                    [{"start": 0, "end": 1, "text": "Training"}],
                    source_title="Training",
                    ollama_url="http://localhost:11434",
                    model="test-model",
                    context_tokens=65536,
                    max_output_tokens=8192,
                )
        self.assertEqual(post.call_count, 2)

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
            with self.assertRaisesRegex(SummaryError, "English summary reached"):
                summarize_two_stage(
                    [{"start": 0, "end": 1, "text": "Training"}],
                    source_title="Training",
                    ollama_url="http://localhost:11434",
                    model="test-model",
                    context_tokens=65536,
                    max_output_tokens=8192,
                )
        with self.assertRaisesRegex(SummaryError, "Local-only"):
            summarize_two_stage(
                [{"start": 0, "end": 1, "text": "Training"}],
                source_title="Training",
                ollama_url="https://example.com",
                model="test-model",
                context_tokens=65536,
                max_output_tokens=8192,
            )
        with patch(
            "local_video_editor.summary.ollama_models",
            return_value=["test-model:cloud"],
        ):
            with self.assertRaisesRegex(SummaryError, "Cloud model"):
                summarize_two_stage(
                    [{"start": 0, "end": 1, "text": "Training"}],
                    source_title="Training",
                    ollama_url="http://localhost:11434",
                    model="test-model:cloud",
                    context_tokens=65536,
                    max_output_tokens=8192,
                )

    def test_prompt_estimate_is_nonzero(self):
        self.assertEqual(estimate_prompt_tokens(""), 1)
        self.assertGreater(estimate_prompt_tokens("a" * 1000), 300)


if __name__ == "__main__":
    unittest.main()
