from pathlib import Path
import math
import sys
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.silence import (  # noqa: E402
    Interval,
    SilenceConfig,
    build_edit_plan,
    parse_silencedetect,
)


class IntervalAssertions:
    def assertIntervals(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for interval, (start, end) in zip(actual, expected):
            self.assertAlmostEqual(interval.start, start)
            self.assertAlmostEqual(interval.end, end)


class ParseSilencedetectTests(IntervalAssertions, unittest.TestCase):
    def test_parses_regular_and_eof_silence(self):
        stderr = b"""
        [silencedetect @ 0x1] silence_start: 1.25
        [silencedetect @ 0x1] silence_end: 3.5 | silence_duration: 2.25
        [silencedetect @ 0x1] silence_start: 8.0
        """

        intervals = parse_silencedetect(stderr, duration=10.0)

        self.assertIntervals(intervals, [(1.25, 3.5), (8.0, 10.0)])

    def test_recovers_orphan_end_from_reported_duration(self):
        intervals = parse_silencedetect(
            "silence_end: 5.0 | silence_duration: 1.75"
        )

        self.assertIntervals(intervals, [(3.25, 5.0)])

    def test_rejects_unsafe_event_sequences(self):
        with self.assertRaisesRegex(ValueError, "duration is required"):
            parse_silencedetect("silence_start: 2.0")
        with self.assertRaisesRegex(ValueError, "before silence_start"):
            parse_silencedetect("silence_end: 2.0")
        with self.assertRaisesRegex(ValueError, "consecutive"):
            parse_silencedetect("silence_start: 1\nsilence_start: 2", duration=3)
        with self.assertRaisesRegex(ValueError, "before the final"):
            parse_silencedetect("silence_start: 4", duration=3)


class BuildEditPlanTests(IntervalAssertions, unittest.TestCase):
    def setUp(self):
        self.config = SilenceConfig(
            min_silence=1.0,
            target_silence=2.0,
            edge_padding=0.25,
            min_keep=0.5,
            merge_gap=0.05,
        )

    def test_no_silence_keeps_whole_media(self):
        plan = build_edit_plan(12.0, [], self.config)

        self.assertIntervals(plan.keep_intervals, [(0.0, 12.0)])
        self.assertEqual(plan.remove_intervals, ())
        self.assertEqual(plan.edits[0].reason, "content")
        self.assertAlmostEqual(plan.output_duration, 12.0)

    def test_interior_long_silence_is_compressed_not_deleted(self):
        plan = build_edit_plan(20.0, [(5.0, 15.0)], self.config)

        # Two seconds total are retained: one second on each spoken side.
        self.assertIntervals(plan.remove_intervals, [(6.0, 14.0)])
        self.assertIntervals(plan.keep_intervals, [(0.0, 6.0), (14.0, 20.0)])
        self.assertAlmostEqual(plan.removed_duration, 8.0)
        self.assertAlmostEqual(plan.output_duration, 12.0)
        self.assertEqual(
            [edit.reason for edit in plan.edits],
            [
                "content",
                "retained_silence",
                "compressed_silence",
                "retained_silence",
                "content",
            ],
        )

    def test_start_and_end_silence_retain_padding_next_to_content(self):
        config = SilenceConfig(
            min_silence=1.0,
            target_silence=1.0,
            edge_padding=0.25,
            min_keep=0.5,
            merge_gap=0.0,
        )

        plan = build_edit_plan(10.0, [(0.0, 4.0), (7.0, 10.0)], config)

        self.assertIntervals(plan.remove_intervals, [(0.0, 3.0), (8.0, 10.0)])
        self.assertIntervals(plan.keep_intervals, [(3.0, 8.0)])
        self.assertAlmostEqual(plan.output_duration, 5.0)

    def test_all_silent_media_still_retains_target_duration(self):
        config = SilenceConfig(
            min_silence=1.0,
            target_silence=1.5,
            edge_padding=0.25,
            min_keep=0.5,
            merge_gap=0.0,
        )

        plan = build_edit_plan(10.0, [(0.0, 10.0)], config)

        self.assertIntervals(plan.keep_intervals, [(0.0, 1.5)])
        self.assertIntervals(plan.remove_intervals, [(1.5, 10.0)])
        self.assertAlmostEqual(plan.output_duration, 1.5)

    def test_merges_detector_fragments_before_applying_threshold(self):
        config = SilenceConfig(
            min_silence=1.5,
            target_silence=0.4,
            edge_padding=0.1,
            min_keep=0.2,
            merge_gap=0.05,
        )

        plan = build_edit_plan(12.0, [(6.03, 7.0), (5.0, 6.0)], config)

        self.assertIntervals(plan.detected_silences, [(5.0, 7.0)])
        self.assertIntervals(plan.eligible_silences, [(5.0, 7.0)])
        self.assertIntervals(plan.remove_intervals, [(5.2, 6.8)])

    def test_short_silence_is_kept_and_audited(self):
        plan = build_edit_plan(10.0, [(4.0, 4.5)], self.config)

        self.assertEqual(plan.eligible_silences, ())
        self.assertIntervals(plan.keep_intervals, [(0.0, 10.0)])
        self.assertIn("short_silence", [edit.reason for edit in plan.edits])
        self.assertEqual(plan.remove_intervals, ())

    def test_edge_padding_and_min_keep_can_override_target(self):
        config = SilenceConfig(
            min_silence=0.5,
            target_silence=0.1,
            edge_padding=0.2,
            min_keep=1.0,
            merge_gap=0.0,
        )

        plan = build_edit_plan(12.0, [(4.0, 8.0)], config)

        self.assertIntervals(plan.remove_intervals, [(4.5, 7.5)])
        retained = sum(
            edit.duration
            for edit in plan.edits
            if edit.reason == "retained_silence"
        )
        self.assertAlmostEqual(retained, 1.0)

    def test_clips_intervals_to_media_boundaries(self):
        config = SilenceConfig(
            min_silence=0.5,
            target_silence=0.5,
            edge_padding=0.1,
            min_keep=0.1,
            merge_gap=0.0,
        )

        plan = build_edit_plan(10.0, [(-2.0, 2.0), (9.0, 12.0)], config)

        self.assertIntervals(plan.detected_silences, [(0.0, 2.0), (9.0, 10.0)])
        self.assertIntervals(plan.remove_intervals, [(0.0, 1.5), (9.5, 10.0)])

    def test_edit_list_partitions_entire_timeline(self):
        plan = build_edit_plan(
            20.0,
            [(0.0, 3.0), (7.0, 12.0), (18.0, 20.0)],
            self.config,
        )

        self.assertAlmostEqual(sum(edit.duration for edit in plan.edits), 20.0)
        self.assertAlmostEqual(plan.kept_duration + plan.removed_duration, 20.0)
        for previous, current in zip(plan.edits, plan.edits[1:]):
            self.assertAlmostEqual(previous.end, current.start)
        payload = plan.as_dict()
        self.assertEqual(payload["edits"][0]["start"], 0.0)
        self.assertAlmostEqual(payload["output_duration"], plan.output_duration)

    def test_rejects_invalid_inputs_and_configuration(self):
        with self.assertRaisesRegex(ValueError, "duration must be non-negative"):
            build_edit_plan(-1.0, [])
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            build_edit_plan(10.0, [(4.0, 3.0)])
        with self.assertRaisesRegex(ValueError, "finite"):
            build_edit_plan(10.0, [(1.0, math.nan)])
        with self.assertRaisesRegex(ValueError, "min_silence must be non-negative"):
            SilenceConfig(min_silence=-0.1)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            SilenceConfig(time_epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
