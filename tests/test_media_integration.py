from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_video_editor.media import (  # noqa: E402
    burn_ass_subtitles,
    detect_silences,
    media_duration,
    probe_media,
    render_video,
)
from local_video_editor.silence import Interval, SilenceConfig, build_edit_plan  # noqa: E402
from local_video_editor.subtitles import render_ass  # noqa: E402


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MediaIntegrationTests(unittest.TestCase):
    def test_burned_ass_keeps_audio_and_duration(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "edited.mp4"
            subprocess.run(
                [
                    os.environ.get("FFMPEG_BIN", "ffmpeg"),
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=320x180:r=10:d=2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=48000:duration=2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
                capture_output=True,
            )
            ass = root / "subtitle.ass"
            ass.write_text(
                render_ass([{"start": 0.2, "end": 1.8, "text": "Local subtitle"}]),
                encoding="utf-8",
            )
            output = root / "subtitled.mp4"
            burn_ass_subtitles(
                source,
                ass,
                output,
                root / "subtitle.render.log",
                preset="ultrafast",
            )

            source_probe = probe_media(source)
            output_probe = probe_media(output)
            source_audio = next(
                stream for stream in source_probe["streams"] if stream["codec_type"] == "audio"
            )
            output_audio = next(
                stream for stream in output_probe["streams"] if stream["codec_type"] == "audio"
            )
            self.assertEqual(output_audio["codec_name"], source_audio["codec_name"])
            self.assertAlmostEqual(
                media_duration(output_probe), media_duration(source_probe), delta=0.1
            )
            self.assertTrue((root / "subtitle.render.log").is_file())

    def test_detect_plan_and_render_synthetic_video(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.mp4"
            command = [
                os.environ.get("FFMPEG_BIN", "ffmpeg"), "-v", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=10:d=6",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=6",
                "-filter:a", "volume=0:enable='between(t,1,3)'",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(source),
            ]
            subprocess.run(command, check=True, capture_output=True)
            probe = probe_media(source)
            duration = media_duration(probe)
            silences = detect_silences(
                source,
                duration=duration,
                noise_db=-40,
                min_silence=0.8,
                log_path=root / "silence.log",
            )
            self.assertTrue(any(item.duration > 1.5 for item in silences))
            plan = build_edit_plan(
                duration,
                silences,
                SilenceConfig(min_silence=0.8, target_silence=0.3),
            )
            output = root / "edited.mp4"
            render_video(
                source,
                output,
                plan.keep_intervals,
                source_duration=duration,
                filter_script_path=root / "filter.txt",
                log_path=root / "render.log",
                preset="ultrafast",
            )
            edited_duration = media_duration(probe_media(output))
            self.assertLess(edited_duration, duration - 1.0)
            self.assertAlmostEqual(edited_duration, plan.output_duration, delta=0.35)

            many_intervals = [
                Interval(index * 0.2, index * 0.2 + 0.1) for index in range(30)
            ]
            many_output = root / "many-cuts.mp4"
            many_filter = root / "many-filter.txt"
            render_video(
                source,
                many_output,
                many_intervals,
                source_duration=duration,
                filter_script_path=many_filter,
                log_path=root / "many-render.log",
                preset="ultrafast",
            )
            filter_text = many_filter.read_text(encoding="utf-8")
            self.assertIn("select=", filter_text)
            self.assertNotIn("concat=", filter_text)
            self.assertAlmostEqual(
                media_duration(probe_media(many_output)), 3.0, delta=0.25
            )


if __name__ == "__main__":
    unittest.main()
