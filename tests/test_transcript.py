from pathlib import Path
import math
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.transcript import (  # noqa: E402
    render_srt,
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
