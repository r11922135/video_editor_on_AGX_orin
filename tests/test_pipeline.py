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
    resummarize_job,
)
from local_video_editor.summary import SummaryError  # noqa: E402


SUMMARY = {
    "title": {"en": "Training", "zh_tw": "訓練"},
    "overview": {"en": ["Overview"], "zh_tw": ["概要"]},
}


class PipelineTests(unittest.TestCase):
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

    def test_pipeline_lock_rejects_a_second_job(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with _pipeline_lock(root):
                with self.assertRaisesRegex(PipelineError, "already using"):
                    with _pipeline_lock(root):
                        pass

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
                "local_video_editor.pipeline.summarize_oneshot"
            ) as summarize:
                result = VideoPipeline(
                    copy.deepcopy(DEFAULT_CONFIG), model_cache=root / "models"
                ).run(source, output_root=root / "output", edit_only=True)

            extract.assert_not_called()
            transcribe.assert_not_called()
            summarize.assert_not_called()
            job = Path(result["job_dir"])
            self.assertTrue((job / "edited.mp4").is_file())
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
                "mode": "oneshot",
                "post_generation_content_modified": False,
            }
            with patch(
                "local_video_editor.pipeline.summarize_oneshot",
                return_value=(SUMMARY, metrics),
            ) as summarize:
                result = resummarize_job(job, copy.deepcopy(DEFAULT_CONFIG))
            self.assertEqual(result["model"], "test-model")
            self.assertNotIn("detail_level", summarize.call_args.kwargs)
            self.assertNotIn("fallback_model", summarize.call_args.kwargs)
            updated = json.loads((job / "manifest.json").read_text())
            self.assertEqual(updated["status"], "complete")
            self.assertTrue((job / "summary.en.md").is_file())
            self.assertTrue((job / "summary.zh-TW.md").is_file())

    def test_resummarize_failure_is_recorded(self):
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
            with patch(
                "local_video_editor.pipeline.summarize_oneshot",
                side_effect=SummaryError("bad model output"),
            ):
                with self.assertRaisesRegex(SummaryError, "bad model output"):
                    resummarize_job(job, copy.deepcopy(DEFAULT_CONFIG))
            updated = json.loads((job / "manifest.json").read_text())
            self.assertEqual(updated["status"], "failed")
            self.assertTrue((job / "failure.traceback.log").is_file())

    def test_marks_running_stage_failed(self):
        manifest = {
            "status": "running",
            "stages": {"summary": {"state": "running"}},
        }
        _mark_manifest_failed(manifest, SummaryError("invalid summary"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"]["summary"]["state"], "failed")


if __name__ == "__main__":
    unittest.main()
