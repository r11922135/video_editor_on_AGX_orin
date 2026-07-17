from pathlib import Path
import math
import re
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.transcript import (  # noqa: E402
    normalize_transcript_text,
    normalized_segment_text,
    readable_transcript_blocks,
    render_srt,
    render_transcript_markdown,
    timestamp,
    transcript_for_prompt,
    transcript_windows_for_prompt,
)


class TranscriptTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": 0.25, "end": 1.75, "text": " Hello world. "},
            {"start": 3661.1, "end": 3662.2, "text": "Second line."},
        ]

    def test_timestamps(self):
        self.assertEqual(timestamp(3661.1), "01:01:01.100")
        self.assertEqual(timestamp(1.25, srt=True), "00:00:01,250")

    def test_renders_srt(self):
        rendered = render_srt(self.segments)
        self.assertIn("00:00:00,250 --> 00:00:01,750", rendered)
        self.assertIn("Hello world.", rendered)

    def test_prompt_has_compact_timestamps(self):
        prompt = transcript_for_prompt(self.segments)
        self.assertEqual(
            prompt.splitlines()[0], "[00:00:00.250] Hello world."
        )

    def test_readable_markdown_uses_five_minute_sections_and_second_timestamps(self):
        rendered = render_transcript_markdown(
            [
                {"start": 0.25, "end": 1.0, "text": "Opening"},
                {"start": 1.0, "end": 2.0, "text": "continues."},
                {"start": 301.75, "end": 302.0, "text": "Later."},
            ],
            title="Training",
        )
        self.assertIn("## 00:00:00–00:05:00", rendered)
        self.assertIn("## 00:05:00–00:10:00", rendered)
        self.assertIn(
            "Automatically transcribed locally; wording may contain recognition errors.",
            rendered,
        )
        self.assertIn("**[00:00:00]** Opening continues.", rendered)
        self.assertIn("**[00:05:01]** Later.", rendered)
        self.assertNotIn("- **[", rendered)
        self.assertIsNone(re.search(r"\*\*\[\d\d:\d\d:\d\d\.\d+\]", rendered))

    def test_readable_prose_is_exactly_the_normalized_raw_text(self):
        segments = [
            {"start": 5.0, "end": 6.0, "text": "  LOS2   uses CubeBars  , "},
            {"start": 6.0, "end": 7.0, "text": "at 120 Nm ."},
            {"start": 10.0, "end": 11.0, "text": "Thank you."},
            {"start": 11.0, "end": 12.0, "text": "Thank you."},
        ]
        rendered = render_transcript_markdown(segments, title="Training")
        prose = []
        for line in rendered.splitlines():
            match = re.match(r"^\*\*\[\d\d:\d\d:\d\d\]\*\* (.*)$", line)
            if match:
                prose.append(match.group(1))
        self.assertEqual(
            normalize_transcript_text(" ".join(prose)),
            normalized_segment_text(segments),
        )
        self.assertEqual(normalized_segment_text(segments).count("Thank you."), 2)
        self.assertIn("LOS2 uses CubeBars, at 120 Nm.", normalized_segment_text(segments))

    def test_readable_blocks_break_on_gap_section_and_length(self):
        gap_blocks = readable_transcript_blocks(
            [
                {"start": 0.0, "end": 1.0, "text": "First fragment"},
                {"start": 1.0, "end": 2.0, "text": "continues."},
                {"start": 4.5, "end": 5.0, "text": "After a gap."},
            ]
        )
        self.assertEqual(len(gap_blocks), 2)
        self.assertEqual(gap_blocks[0]["text"], "First fragment continues.")

        section_blocks = readable_transcript_blocks(
            [
                {"start": 299.0, "end": 299.5, "text": "Before boundary."},
                {"start": 300.0, "end": 300.5, "text": "After boundary."},
            ]
        )
        self.assertEqual([item["section_start"] for item in section_blocks], [0, 300])

        long_piece = "x" * 240
        length_blocks = readable_transcript_blocks(
            [
                {"start": 0.0, "end": 1.0, "text": long_piece},
                {"start": 1.0, "end": 2.0, "text": long_piece},
                {"start": 2.0, "end": 3.0, "text": long_piece},
            ]
        )
        self.assertEqual(len(length_blocks), 2)

    def test_readable_blocks_use_natural_sentence_boundaries(self):
        first = "A" * 160
        second = "B" * 160 + "."
        blocks = readable_transcript_blocks(
            [
                {"start": 0.0, "end": 1.0, "text": first},
                {"start": 1.0, "end": 2.0, "text": second},
                {"start": 2.0, "end": 3.0, "text": "A new thought starts."},
            ]
        )
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0]["text"].endswith("."))

    def test_readable_blocks_sort_timestamps_monotonically_without_losing_text(self):
        segments = [
            {"start": 12.0, "end": 13.0, "text": "Third."},
            {"start": 1.0, "end": 2.0, "text": "First."},
            {"start": 6.0, "end": 7.0, "text": "Second."},
        ]
        blocks = readable_transcript_blocks(segments)
        starts = [float(item["start"]) for item in blocks]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(
            normalized_segment_text(segments), "First. Second. Third."
        )

    def test_summary_windows_merge_single_word_segments(self):
        segments = [
            {
                "start": float(index),
                "end": float(index) + 0.5,
                "text": word,
            }
            for index, word in enumerate(["A", "robot", "uses", "low", "inertia."])
        ]
        prompt, windows = transcript_windows_for_prompt(segments)
        self.assertEqual(len(windows), 1)
        self.assertIn("A robot uses low inertia.", prompt)
        self.assertEqual(prompt.count("<window "), 1)
        self.assertNotIn("00:00:01", prompt)

    def test_summary_windows_adapt_to_at_most_twelve_time_buckets(self):
        segments = [
            {
                "start": float(minute * 60),
                "end": float(minute * 60 + 1),
                "text": f"topic{minute}",
            }
            for minute in range(240)
        ]
        prompt, windows = transcript_windows_for_prompt(segments)
        self.assertLessEqual(len(windows), 12)
        self.assertEqual(windows[0]["segment_count"], 20)
        self.assertIn("topic239", prompt)

    def test_summary_windows_omit_empty_time_buckets(self):
        prompt, windows = transcript_windows_for_prompt(
            [
                {"start": 0.0, "end": 1.0, "text": "Opening."},
                {"start": 900.0, "end": 901.0, "text": "Closing."},
            ]
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual([item["id"] for item in windows], [1, 2])
        self.assertEqual(prompt.count("<window "), 2)

    def test_summary_windows_hold_boundary_at_twelve(self):
        segments = [
            {
                "start": float(index * 300),
                "end": float(index * 300),
                "text": f"topic{index}",
            }
            for index in range(13)
        ]
        _prompt, windows = transcript_windows_for_prompt(segments)
        self.assertEqual(len(windows), 12)
        self.assertIn("topic12", windows[-1]["text"])

    def test_summary_windows_reject_nonfinite_timestamps(self):
        for invalid in (math.inf, math.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite"):
                    transcript_windows_for_prompt(
                        [{"start": invalid, "end": invalid, "text": "Bad"}]
                    )

    def test_summary_windows_preserve_large_noisy_transcript_without_timestamp_bloat(self):
        segments = [
            {
                "start": index * 0.9,
                "end": index * 0.9 + 0.4,
                "text": f"token{index}",
            }
            for index in range(3249)
        ]
        legacy = transcript_for_prompt(segments)
        compact, windows = transcript_windows_for_prompt(segments)
        self.assertEqual(sum(item["segment_count"] for item in windows), 3249)
        self.assertLessEqual(len(windows), 12)
        self.assertEqual(compact.count("token0"), 1)
        self.assertEqual(compact.count("token3248"), 1)
        self.assertLess(compact.index("token1000"), compact.index("token2000"))
        self.assertLess(len(compact), len(legacy) * 0.6)


if __name__ == "__main__":
    unittest.main()
