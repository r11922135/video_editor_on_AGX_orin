from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .io_utils import atomic_write_text
from .silence import Interval, parse_silencedetect


class MediaError(RuntimeError):
    pass


def _media_binary(name: str) -> str:
    variable = "FFMPEG_BIN" if name == "ffmpeg" else "FFPROBE_BIN"
    return os.environ.get(variable, name)


def _command_text(command: list[str]) -> str:
    return shlex.join(command)


def require_media_tools() -> None:
    missing = [
        name
        for name in ("ffmpeg", "ffprobe")
        if not shutil.which(_media_binary(name))
    ]
    if missing:
        raise MediaError(f"Missing required command(s): {', '.join(missing)}")


def probe_media(source: Path) -> dict[str, Any]:
    require_media_tools()
    command = [
        _media_binary("ffprobe"),
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams", [])
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise MediaError("Input has no video stream")
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise MediaError("Input has no audio stream; silence editing requires audio")
    duration = media_duration(payload)
    if duration <= 0:
        raise MediaError("Input duration is missing or zero")
    return payload


def media_duration(probe: dict[str, Any]) -> float:
    candidates: list[Any] = [probe.get("format", {}).get("duration")]
    candidates.extend(stream.get("duration") for stream in probe.get("streams", []))
    for value in candidates:
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            return duration
    return 0.0


def detect_silences(
    source: Path,
    *,
    duration: float,
    noise_db: float,
    min_silence: float,
    log_path: Path,
) -> tuple[Interval, ...]:
    command = [
        _media_binary("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"silencedetect=n={float(noise_db):g}dB:d={float(min_silence):g}",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    atomic_write_text(
        log_path,
        f"$ {_command_text(command)}\n\n{result.stderr}",
    )
    if result.returncode != 0:
        raise MediaError(f"FFmpeg silence detection failed; see {log_path}")
    return parse_silencedetect(result.stderr, duration=duration)


def _normalize_intervals(intervals: Iterable[Interval | dict[str, Any]]) -> list[Interval]:
    normalized: list[Interval] = []
    for value in intervals:
        if isinstance(value, Interval):
            normalized.append(value)
        else:
            normalized.append(Interval(float(value["start"]), float(value["end"])))
    return [item for item in normalized if item.duration > 1e-6]


def _partial_media_path(output: Path) -> Path:
    suffix = output.suffix or ".mp4"
    return output.with_name(f".{output.stem}.partial{suffix}")


def render_video(
    source: Path,
    output: Path,
    keep_intervals: Iterable[Interval | dict[str, Any]],
    *,
    source_duration: float,
    filter_script_path: Path,
    log_path: Path,
    codec: str = "libx264",
    preset: str = "veryfast",
    crf: int = 20,
    audio_bitrate: str = "160k",
) -> None:
    """Render to a partial file and atomically publish the completed MP4."""
    intervals = _normalize_intervals(keep_intervals)
    if not intervals:
        raise MediaError("Edit plan contains no video to keep")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = _partial_media_path(output)
    partial.unlink(missing_ok=True)

    no_cut = (
        len(intervals) == 1
        and intervals[0].start <= 1e-4
        and intervals[0].end >= source_duration - 1e-3
    )
    if no_cut:
        atomic_write_text(filter_script_path, "# No cuts; streams copied without re-encoding.\n")
        command = [
            _media_binary("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0",
            "-map_metadata",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(partial),
        ]
    else:
        # A trim/atrim branch per cut makes long lectures fan every decoded
        # frame into hundreds of filters.  Keep a single video/audio path and
        # close removed timestamp gaps explicitly instead.
        select_expression = "+".join(
            f"gte(t,{item.start:.6f})*lt(t,{item.end:.6f})"
            for item in intervals
        )
        timestamp_shifts = [
            (current.start, current.start - previous.end)
            for previous, current in zip(intervals, intervals[1:])
            if current.start - previous.end > 1e-6
        ]
        if timestamp_shifts:
            shift_expression = "+".join(
                f"gte(T,{start:.6f})*{duration:.6f}"
                for start, duration in timestamp_shifts
            )
            setpts_expression = f"PTS-STARTPTS-({shift_expression})/TB"
        else:
            setpts_expression = "PTS-STARTPTS"
        graph = (
            f"[0:v:0]select='{select_expression}',"
            f"setpts='{setpts_expression}'[vout];\n"
            f"[0:a:0]aselect='{select_expression}',"
            f"asetpts='{setpts_expression}'[aout]\n"
        )
        atomic_write_text(filter_script_path, graph)
        command = [
            _media_binary("ffmpeg"),
            "-hide_banner",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-filter_complex_script",
            str(filter_script_path),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-map_metadata",
            "0",
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            str(partial),
        ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    atomic_write_text(
        log_path,
        f"$ {_command_text(command)}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}",
    )
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise MediaError(f"FFmpeg render failed; see {log_path}")
    partial.chmod(0o600)
    output.unlink(missing_ok=True)
    partial.replace(output)


def extract_analysis_audio(source: Path, output_wav: Path, log_path: Path) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    partial = output_wav.with_name(f".{output_wav.stem}.partial.wav")
    partial.unlink(missing_ok=True)
    command = [
        _media_binary("ffmpeg"),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(partial),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    atomic_write_text(log_path, f"$ {_command_text(command)}\n\n{result.stderr}")
    if result.returncode != 0 or not partial.is_file() or partial.stat().st_size == 0:
        partial.unlink(missing_ok=True)
        raise MediaError(f"Audio extraction failed; see {log_path}")
    partial.chmod(0o600)
    output_wav.unlink(missing_ok=True)
    partial.replace(output_wav)
