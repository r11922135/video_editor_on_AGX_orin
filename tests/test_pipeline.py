from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.config import DEFAULT_CONFIG  # noqa: E402
from local_video_editor.pipeline import (  # noqa: E402
    PipelineError,
    VideoPipeline,
    _config_fingerprint,
    _mark_manifest_failed,
    _pipeline_lock,
    _user_output_names,
    rerender_transcript_job,
    resummarize_job,
)
from local_video_editor.summary import SummaryError  # noqa: E402


SUMMARY = {
    "title": {"en": "Training", "zh_tw": "訓練"},
    "overview": {"en": ["Overview"], "zh_tw": ["概要"]},
}


class PipelineTests(unittest.TestCase):
    def _run_mocked_full_pipeline(self, root, subtitle_effect):
        source = root / "training.mp4"
        source.write_bytes(b"source-video")
        original_segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "The CubeBars motor uses ROS.",
                "words": [
                    {"start": 0.0, "end": 0.4, "word": "The"},
                    {"start": 0.4, "end": 1.0, "word": "CubeBars"},
                    {"start": 1.0, "end": 1.3, "word": "motor"},
                    {"start": 1.3, "end": 1.6, "word": "uses"},
                    {"start": 1.6, "end": 2.0, "word": "ROS."},
                ],
            }
        ]
        probe = {"format": {"duration": "10.0"}}
        observed = {}

        def fake_render(_source, output, *_args, **_kwargs):
            output.write_bytes(b"edited-video")

        def fake_extract(_video, output, _log):
            output.write_bytes(b"analysis-audio")

        def fake_summarize(segments, **kwargs):
            observed["summary_reference"] = segments
            observed["summary_snapshot"] = copy.deepcopy(segments)
            kwargs["english_raw_response_path"].write_text("english raw")
            kwargs["translation_raw_response_path"].write_text("chinese raw")
            return SUMMARY, {
                "model": "test-model",
                "mode": "two_stage",
                "post_generation_content_modified": False,
            }

        def fake_subtitles(**kwargs):
            observed["subtitle_snapshot"] = copy.deepcopy(kwargs["segments"])
            return subtitle_effect(kwargs)

        transcription = {
            "segments": copy.deepcopy(original_segments),
            "language": "en",
            "elapsed_seconds": 0.1,
        }
        with patch(
            "local_video_editor.pipeline.probe_media", return_value=probe
        ), patch(
            "local_video_editor.pipeline.detect_silences", return_value=[]
        ), patch(
            "local_video_editor.pipeline.render_video", side_effect=fake_render
        ), patch(
            "local_video_editor.pipeline.extract_analysis_audio",
            side_effect=fake_extract,
        ), patch(
            "local_video_editor.pipeline.transcribe_faster_whisper",
            return_value=transcription,
        ), patch(
            "local_video_editor.pipeline.summarize_two_stage",
            side_effect=fake_summarize,
        ), patch(
            "local_video_editor.pipeline.create_subtitled_video",
            side_effect=fake_subtitles,
        ):
            result = VideoPipeline(
                copy.deepcopy(DEFAULT_CONFIG),
                model_cache=root / "models",
                status=lambda *_args: None,
            ).run(source, output_root=root / "output", subtitles=True)
        return result, original_segments, observed

    def test_operation_is_part_of_job_identity(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        self.assertNotEqual(
            _config_fingerprint(config, "full"),
            _config_fingerprint(config, "edit_only"),
        )
        changed = copy.deepcopy(config)
        changed["summary"]["model"] = "another-model"
        self.assertEqual(
            _config_fingerprint(config, "edit_only"),
            _config_fingerprint(changed, "edit_only"),
        )
        self.assertNotEqual(
            _config_fingerprint(config, "full"),
            _config_fingerprint(changed, "full"),
        )

    def test_subtitles_are_part_of_full_job_identity(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        self.assertNotEqual(
            _config_fingerprint(config, "full", subtitles=False),
            _config_fingerprint(config, "full", subtitles=True),
        )
        changed = copy.deepcopy(config)
        changed["subtitles"]["alignment_chunk_seconds"] = 90
        self.assertEqual(
            _config_fingerprint(config, "full", subtitles=False),
            _config_fingerprint(changed, "full", subtitles=False),
        )
        self.assertNotEqual(
            _config_fingerprint(config, "full", subtitles=True),
            _config_fingerprint(changed, "full", subtitles=True),
        )

    def test_edit_only_and_subtitles_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as raw:
            pipeline = VideoPipeline(
                copy.deepcopy(DEFAULT_CONFIG), model_cache=Path(raw) / "models"
            )
            with self.assertRaisesRegex(
                PipelineError, "--edit-only and --subtitles"
            ):
                pipeline.run(
                    Path(raw) / "does-not-need-to-exist.mp4",
                    output_root=Path(raw) / "output",
                    edit_only=True,
                    subtitles=True,
                )

    def test_full_subtitle_branch_cannot_mutate_canonical_or_summary_input(self):
        def mutate_subtitle_input(kwargs):
            kwargs["segments"][0]["text"] = "MUTATED BY OPTIONAL SUBTITLE BRANCH"
            kwargs["segments"][0]["words"][0]["word"] = "MUTATED"
            output_dir = kwargs["output_dir"]
            prefix = kwargs["filename_prefix"]
            video_name = f"{prefix}_subtitled.mp4"
            srt_name = f"{prefix}_subtitle.srt"
            ass_name = f"{prefix}_subtitle.ass"
            (output_dir / video_name).write_bytes(b"subtitled-video")
            for name in (
                srt_name,
                ass_name,
                "subtitle.rules.json",
                "subtitle.corrected.json",
                "subtitle.correction.raw.txt",
            ):
                (output_dir / name).write_text("test output")
            return {
                "output": video_name,
                "subtitle_srt": srt_name,
                "subtitle_ass": ass_name,
                "cue_count": 1,
                "fallback_used": False,
            }

        with tempfile.TemporaryDirectory() as raw:
            result, original_segments, observed = self._run_mocked_full_pipeline(
                Path(raw), mutate_subtitle_input
            )
            transcript = json.loads(
                (Path(result["job_dir"]) / "transcript.json").read_text()
            )
            self.assertEqual(observed["summary_snapshot"], original_segments)
            self.assertEqual(observed["subtitle_snapshot"], original_segments)
            self.assertEqual(transcript["segments"], original_segments)
            self.assertEqual(observed["summary_reference"], original_segments)

    def test_optional_subtitle_failure_keeps_summary_and_completes_job(self):
        def fail_subtitles(_kwargs):
            raise RuntimeError("subtitle renderer unavailable")

        with tempfile.TemporaryDirectory() as raw:
            result, original_segments, observed = self._run_mocked_full_pipeline(
                Path(raw), fail_subtitles
            )
            job = Path(result["job_dir"])
            manifest = json.loads((job / "manifest.json").read_text())

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["stages"]["summary"]["state"], "complete")
            self.assertEqual(manifest["stages"]["subtitles"]["state"], "failed")
            self.assertTrue(manifest["stages"]["subtitles"]["optional"])
            self.assertEqual(manifest["warnings"][0]["stage"], "subtitles")
            self.assertEqual(
                manifest["warnings"][0]["message"], "subtitle renderer unavailable"
            )
            self.assertEqual(observed["summary_snapshot"], original_segments)
            for key in (
                "summary_en",
                "summary_zh_tw",
                "summary_json",
                "summary_en_raw",
                "summary_zh_tw_raw",
                "summary_metrics",
            ):
                self.assertIn(key, manifest["outputs"])
                self.assertTrue((job / manifest["outputs"][key]).is_file())
            self.assertNotIn("subtitled_video", manifest["outputs"])
            self.assertTrue((job / "subtitle.error.log").is_file())
            with self.assertRaisesRegex(PipelineError, "without a valid subtitled"):
                VideoPipeline(
                    copy.deepcopy(DEFAULT_CONFIG), model_cache=Path(raw) / "models"
                ).run(
                    Path(raw) / "training.mp4",
                    output_root=Path(raw) / "output",
                    subtitles=True,
                )

    def test_pipeline_lock_rejects_a_second_job(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with _pipeline_lock(root):
                with self.assertRaisesRegex(PipelineError, "already using"):
                    with _pipeline_lock(root):
                        pass

    def test_user_output_names_use_robotics_seminar_date(self):
        names = _user_output_names(
            "Robotics Seminar-20260715_103245-Meeting Recording.mp4"
        )
        self.assertEqual(names["video"], "Robotics_Seminar_20260715_edited.mp4")
        self.assertEqual(
            names["summary_zh_tw"],
            "Robotics_Seminar_20260715_summary.zh-TW.md",
        )
        self.assertEqual(
            names["transcript_md"],
            "Robotics_Seminar_20260715_transcript.md",
        )

    def test_edit_only_stops_before_audio_asr_and_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "training.mp4"
            source.write_bytes(b"source-video")
            probe = {"format": {"duration": "10.0"}}

            def fake_render(_source, output, *_args, **_kwargs):
                output.write_bytes(b"edited-video")

            with patch(
                "local_video_editor.pipeline.probe_media", side_effect=[probe, probe]
            ), patch(
                "local_video_editor.pipeline.detect_silences", return_value=[]
            ), patch(
                "local_video_editor.pipeline.render_video", side_effect=fake_render
            ), patch(
                "local_video_editor.pipeline.extract_analysis_audio"
            ) as extract, patch(
                "local_video_editor.pipeline.transcribe_faster_whisper"
            ) as transcribe, patch(
                "local_video_editor.pipeline.summarize_two_stage"
            ) as summarize:
                result = VideoPipeline(
                    copy.deepcopy(DEFAULT_CONFIG), model_cache=root / "models"
                ).run(source, output_root=root / "output", edit_only=True)

            extract.assert_not_called()
            transcribe.assert_not_called()
            summarize.assert_not_called()
            job = Path(result["job_dir"])
            self.assertTrue((job / "training_edited.mp4").is_file())
            self.assertFalse((job / "analysis.wav").exists())
            self.assertFalse((job / "transcript.json").exists())
            self.assertFalse((job / "summary.json").exists())
            self.assertEqual(result["manifest"]["operation"], "edit_only")
            self.assertEqual(list(result["manifest"]["outputs"]), ["video"])

    def test_resummarize_generates_fresh_overview(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw) / "job"
            job.mkdir()
            (job / "edited.mp4").write_bytes(b"video")
            (job / "transcript.json").write_text(
                json.dumps({"segments": [{"start": 0, "end": 1, "text": "Training"}]})
            )
            (job / "manifest.json").write_text(
                json.dumps({
                    "source": {"name": "training.mp4"},
                    "status": "failed",
                    "stages": {"summary": {"state": "failed"}},
                })
            )
            metrics = {
                "model": "test-model",
                "mode": "two_stage",
                "post_generation_content_modified": False,
            }

            def fake_summarize(*_args, **kwargs):
                kwargs["english_raw_response_path"].write_text("english raw")
                kwargs["translation_raw_response_path"].write_text("chinese raw")
                return SUMMARY, metrics

            with patch(
                "local_video_editor.pipeline.summarize_two_stage",
                side_effect=fake_summarize,
            ) as summarize:
                result = resummarize_job(job, copy.deepcopy(DEFAULT_CONFIG))
            self.assertEqual(result["model"], "test-model")
            self.assertNotIn("detail_level", summarize.call_args.kwargs)
            self.assertNotIn("fallback_model", summarize.call_args.kwargs)
            self.assertEqual(
                summarize.call_args.kwargs["max_output_tokens"], 16384
            )
            self.assertEqual(
                summarize.call_args.kwargs["english_raw_response_path"],
                job / ".summary.pending" / "summary.en.raw.txt",
            )
            self.assertEqual(
                summarize.call_args.kwargs["translation_raw_response_path"],
                job / ".summary.pending" / "summary.zh-TW.raw.txt",
            )
            updated = json.loads((job / "manifest.json").read_text())
            self.assertEqual(updated["status"], "complete")
            self.assertTrue((job / "training_summary.en.md").is_file())
            self.assertTrue((job / "training_summary.zh-TW.md").is_file())
            self.assertTrue((job / "summary.en.raw.txt").is_file())
            self.assertTrue((job / "summary.zh-TW.raw.txt").is_file())
            self.assertFalse((job / ".summary.pending").exists())
            self.assertEqual(updated["outputs"]["summary_json"], "summary.json")

    def test_resummarize_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw) / "job"
            job.mkdir()
            (job / "edited.mp4").write_bytes(b"video")
            (job / "transcript.json").write_text(
                json.dumps({"segments": [{"start": 0, "end": 1, "text": "Training"}]})
            )
            (job / "summary.en.raw.txt").write_text("previous english")
            (job / "summary.zh-TW.raw.txt").write_text("previous chinese")
            (job / "manifest.json").write_text(
                json.dumps({
                    "source": {"name": "training.mp4"},
                    "status": "failed",
                    "stages": {"summary": {"state": "failed"}},
                })
            )
            with patch(
                "local_video_editor.pipeline.summarize_two_stage",
                side_effect=SummaryError("bad model output"),
            ):
                with self.assertRaisesRegex(SummaryError, "bad model output"):
                    resummarize_job(job, copy.deepcopy(DEFAULT_CONFIG))
            updated = json.loads((job / "manifest.json").read_text())
            self.assertEqual(updated["status"], "failed")
            self.assertTrue((job / "failure.traceback.log").is_file())
            self.assertFalse((job / ".summary.pending").exists())
            self.assertEqual(
                (job / "summary.en.raw.txt").read_text(), "previous english"
            )
            self.assertEqual(
                (job / "summary.zh-TW.raw.txt").read_text(), "previous chinese"
            )

    def test_marks_running_stage_failed(self):
        manifest = {
            "status": "running",
            "stages": {"summary": {"state": "running"}},
        }
        _mark_manifest_failed(manifest, SummaryError("invalid summary"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"]["summary"]["state"], "failed")

    def test_rerender_transcript_uses_existing_segments_only(self):
        with tempfile.TemporaryDirectory() as raw:
            job = Path(raw) / "job"
            job.mkdir()
            original = {
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": "First fragment."},
                    {"start": 2.1, "end": 3.0, "text": "Second fragment."},
                ]
            }
            transcript_json = json.dumps(original)
            (job / "transcript.json").write_text(transcript_json)
            (job / "manifest.json").write_text(
                json.dumps(
                    {
                        "source": {"name": "training.mp4"},
                        "status": "failed",
                        "outputs": {"summary_en": "/existing/summary.en.md"},
                    }
                )
            )

            with patch(
                "local_video_editor.pipeline.transcribe_faster_whisper"
            ) as transcribe, patch(
                "local_video_editor.pipeline.summarize_two_stage"
            ) as summarize:
                result = rerender_transcript_job(job)

            transcribe.assert_not_called()
            summarize.assert_not_called()

            self.assertEqual(
                (job / "transcript.json").read_text(), transcript_json
            )
            rendered = (job / "training_transcript.md").read_text()
            self.assertIn("First fragment.", rendered)
            self.assertIn("Second fragment.", rendered)
            self.assertEqual(result["segment_count"], 2)
            manifest = json.loads((job / "manifest.json").read_text())
            self.assertEqual(
                manifest["outputs"]["transcript_md"], "training_transcript.md"
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(
                manifest["outputs"]["summary_en"], "/existing/summary.en.md"
            )


if __name__ == "__main__":
    unittest.main()
