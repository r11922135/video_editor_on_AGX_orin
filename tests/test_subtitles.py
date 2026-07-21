from pathlib import Path
import copy
import os
import sys
import tempfile
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.subtitles import (  # noqa: E402
    apply_correction_rules,
    attach_display_tokens,
    build_alignment_chunks,
    build_cues,
    correction_schema,
    _local_model_snapshot,
    render_ass,
    render_subtitle_srt,
    validate_correction_rules,
)


class SubtitleTests(unittest.TestCase):
    def test_local_model_snapshot_resolves_mounted_cache_without_user_home(self):
        with tempfile.TemporaryDirectory() as raw:
            cache = Path(raw) / "hub"
            repository = cache / "models--Qwen--Aligner"
            snapshot = repository / "snapshots" / "abc123"
            snapshot.mkdir(parents=True)
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text("abc123\n")
            with patch.dict(
                os.environ,
                {"HUGGINGFACE_HUB_CACHE": str(cache)},
                clear=False,
            ):
                self.assertEqual(
                    _local_model_snapshot("Qwen/Aligner"), str(snapshot.resolve())
                )

    def test_alignment_chunks_are_ordered_bounded_and_lossless(self):
        segments = [
            {"id": 2, "start": 42.0, "end": 55.0, "text": "  Third   part. "},
            {"id": 0, "start": 0.0, "end": 12.0, "text": "First part."},
            {"id": 1, "start": 15.0, "end": 28.0, "text": "Second part."},
        ]

        chunks = build_alignment_chunks(segments, max_seconds=30)

        self.assertEqual([chunk["id"] for chunk in chunks], ["c001", "c002"])
        self.assertEqual(chunks[0]["start"], 0.0)
        self.assertEqual(chunks[0]["end"], 28.0)
        self.assertEqual(chunks[0]["text"], "First part. Second part.")
        self.assertEqual(chunks[1]["text"], "Third part.")
        self.assertTrue(
            all(chunk["end"] - chunk["start"] <= 30 for chunk in chunks)
        )
        self.assertEqual(
            [segment["id"] for chunk in chunks for segment in chunk["segments"]],
            [0, 1, 2],
        )
        self.assertEqual(segments[0]["text"], "  Third   part. ")

    def test_rule_validation_accepts_conservative_near_matches_only(self):
        chunks = [
            {
                "id": "c001",
                "text": "The CubeBars motor connects over LOS using shark pro files.",
            }
        ]
        response = {
            "rules": [
                {
                    "scope_id": "c001",
                    "original": "CubeBars",
                    "replacement": "CubeMars",
                    "evidence": "technical_context",
                },
                {
                    "scope_id": "c001",
                    "original": "LOS",
                    "replacement": "ROS",
                    "evidence": "standard_term",
                },
                {
                    "scope_id": "c001",
                    "original": "shark pro",
                    "replacement": "SDF",
                    "evidence": "technical_context",
                },
                {
                    "scope_id": "c999",
                    "original": "motor",
                    "replacement": "rotor",
                    "evidence": "technical_context",
                },
                {
                    "scope_id": "c001",
                    "original": "CubeBars",
                    "replacement": "CubeMars",
                    "evidence": "technical_context",
                },
            ]
        }

        rules = validate_correction_rules(response, chunks, max_rules=10)

        self.assertEqual(
            [(rule["original"], rule["replacement"]) for rule in rules],
            [("CubeBars", "CubeMars"), ("LOS", "ROS")],
        )
        self.assertTrue(all(rule["matched_occurrences"] == 1 for rule in rules))

    def test_rule_validation_does_not_treat_unrelated_symbols_or_cjk_as_equal(self):
        chunks = [{"id": "c001", "text": "Use C++, 中文, and VLA here."}]
        response = {
            "rules": [
                {
                    "scope_id": "c001",
                    "original": "C++",
                    "replacement": "C#",
                    "evidence": "standard_term",
                },
                {
                    "scope_id": "c001",
                    "original": "中文",
                    "replacement": "日文",
                    "evidence": "technical_context",
                },
                {
                    "scope_id": "c001",
                    "original": "VLA",
                    "replacement": "VR",
                    "evidence": "technical_context",
                },
            ]
        }
        self.assertEqual(
            validate_correction_rules(response, chunks, max_rules=10), []
        )

    def test_apply_rules_does_not_mutate_canonical_chunks_or_partial_words(self):
        chunks = [
            {
                "id": "c001",
                "start": 0.0,
                "end": 2.0,
                "text": "LOS LOS2 LOS.",
                "segments": [{"text": "LOS LOS2 LOS."}],
            }
        ]
        original = copy.deepcopy(chunks)
        rules = [
            {
                "scope_id": "c001",
                "original": "LOS",
                "replacement": "ROS",
                "evidence": "standard_term",
            }
        ]

        corrected = apply_correction_rules(chunks, rules)

        self.assertEqual(corrected[0]["text"], "ROS LOS2 ROS.")
        self.assertEqual(corrected[0]["applied_rules"][0]["applied_occurrences"], 2)
        self.assertEqual(chunks, original)
        self.assertIsNot(corrected[0], chunks[0])

    def test_correction_schema_limits_scopes_and_rule_count(self):
        schema = correction_schema(["c001", "c002"], 7)
        rules = schema["properties"]["rules"]
        self.assertEqual(rules["maxItems"], 7)
        self.assertEqual(
            rules["items"]["properties"]["scope_id"]["enum"],
            ["c001", "c002"],
        )
        self.assertFalse(schema["additionalProperties"])

    def test_display_tokens_preserve_punctuation_on_aligned_words(self):
        aligned = [
            {"start": 0.0, "end": 0.4, "text": "hello"},
            {"start": 0.5, "end": 0.9, "text": "world"},
        ]
        result = attach_display_tokens("Hello, world!", aligned)
        self.assertEqual([item["text"] for item in result], ["Hello,", "world!"])
        self.assertEqual(result[1]["end"], 0.9)

    def test_cues_obey_readability_bounds_without_overlaps(self):
        words = []
        for index in range(30):
            start = index * 0.32
            words.append(
                {
                    "start": start,
                    "end": start + 0.25,
                    "text": f"word{index}",
                }
            )
        words[5]["text"] = "sentence."

        cues = build_cues(words)

        self.assertGreater(len(cues), 1)
        self.assertTrue(all(cue["end"] > cue["start"] for cue in cues))
        self.assertTrue(all(cue["end"] - cue["start"] <= 6.0 for cue in cues))
        self.assertTrue(all(len(cue["text"]) <= 78 for cue in cues))
        self.assertTrue(
            all(left["end"] <= right["start"] for left, right in zip(cues, cues[1:]))
        )
        self.assertIn("word0 word1", cues[0]["text"])
        self.assertTrue(cues[0]["text"].endswith("sentence."))

        srt = render_subtitle_srt(cues)
        self.assertIn("00:00:00,000 -->", srt)
        text_lines = [
            line
            for line in srt.splitlines()
            if line and not line.isdigit() and " --> " not in line
        ]
        self.assertTrue(all(len(line) <= 42 for line in text_lines))

    def test_cues_normalize_overlapping_and_abnormally_long_fallback_words(self):
        cues = build_cues(
            [
                {"start": 1.0, "end": 15.0, "text": "so"},
                {"start": 2.0, "end": 2.4, "text": "the"},
                {"start": 2.4, "end": 11.0, "text": "motor."},
                {"start": 3.0, "end": 3.5, "text": "Next."},
            ]
        )
        self.assertTrue(all(cue["end"] > cue["start"] for cue in cues))
        self.assertTrue(all(cue["end"] - cue["start"] <= 6.0 for cue in cues))
        self.assertTrue(
            all(left["end"] <= right["start"] for left, right in zip(cues, cues[1:]))
        )

    def test_ass_uses_readable_style_and_escapes_override_characters(self):
        ass = render_ass(
            [
                {
                    "start": 1.25,
                    "end": 3.5,
                    "text": (
                        r"Path C:\robot\config {draft} contains a very long "
                        "technical phrase for line balancing"
                    ),
                }
            ]
        )

        self.assertIn("PlayResX: 1920", ass)
        self.assertIn("PlayResY: 1080", ass)
        self.assertIn("Style: Default,DejaVu Sans,52", ass)
        self.assertIn("Dialogue: 0,0:00:01.25,0:00:03.50", ass)
        self.assertIn("C:\\\u2060robot\\\u2060config", ass)
        self.assertIn(r"\{draft\}", ass)
        self.assertIn(r"\N", ass)


if __name__ == "__main__":
    unittest.main()
