from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

from .io_utils import atomic_write_text


def timestamp(seconds: float, *, srt: bool = False) -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def render_srt(segments: Iterable[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, 1):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        blocks.append(
            f"{index}\n{timestamp(segment['start'], srt=True)} --> "
            f"{timestamp(segment['end'], srt=True)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_transcript_markdown(
    segments: Iterable[dict[str, Any]], *, title: str
) -> str:
    lines = [f"# Transcript — {title}", ""]
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"- **[{timestamp(segment['start'])}]** {text}")
    return "\n".join(lines).rstrip() + "\n"


def transcript_for_prompt(segments: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if text:
            lines.append(f"[{timestamp(segment['start'])}] {text}")
    return "\n".join(lines)


def transcript_windows_for_prompt(
    segments: Iterable[dict[str, Any]],
    *,
    min_window_seconds: int = 300,
    max_windows: int = 12,
) -> tuple[str, list[dict[str, Any]]]:
    """Coalesce noisy ASR segments into a bounded set of coverage windows.

    Long recordings can contain thousands of one-word ASR segments.  Repeating a
    timestamp for every segment wastes context and biases a long-context model
    toward the tail.  This representation keeps one time anchor per window while
    preserving every spoken-text segment in chronological order.
    """
    if min_window_seconds <= 0:
        raise ValueError("min_window_seconds must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")

    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = " ".join(str(segment.get("text", "")).split())
        if not text:
            continue
        raw_start = float(segment.get("start", 0.0))
        raw_end = float(segment.get("end", raw_start))
        if not math.isfinite(raw_start) or not math.isfinite(raw_end):
            raise ValueError("Transcript timestamps must be finite")
        start = max(0.0, raw_start)
        end = max(start, raw_end)
        cleaned.append({"start": start, "end": end, "text": text})
    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    if not cleaned:
        return "", []

    duration = max(item["end"] for item in cleaned)
    adaptive_seconds = math.ceil(duration / max_windows / 60.0) * 60
    window_seconds = max(int(min_window_seconds), int(adaptive_seconds))

    buckets: dict[int, list[dict[str, Any]]] = {}
    for item in cleaned:
        bucket = min(int(item["start"] // window_seconds), max_windows - 1)
        buckets.setdefault(bucket, []).append(item)

    windows: list[dict[str, Any]] = []
    blocks: list[str] = []
    for window_id, bucket in enumerate(sorted(buckets), 1):
        items = buckets[bucket]
        start = float(bucket * window_seconds)
        end = min(float((bucket + 1) * window_seconds), duration)
        text = " ".join(item["text"] for item in items)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text).strip()
        metadata = {
            "id": window_id,
            "start": start,
            "end": end,
            "text": text,
            "char_count": len(text),
            "segment_count": len(items),
        }
        windows.append(metadata)
        start_label = timestamp(start).split(".", 1)[0]
        end_label = timestamp(end).split(".", 1)[0]
        blocks.append(
            f'<window id="{window_id}" time="{start_label}-{end_label}">\n'
            f"{text}\n</window>"
        )
    return "\n\n".join(blocks), windows


def write_transcript_files(
    segments: list[dict[str, Any]], *, title: str, output_dir: Path
) -> None:
    atomic_write_text(output_dir / "transcript.srt", render_srt(segments))
    atomic_write_text(
        output_dir / "transcript.md",
        render_transcript_markdown(segments, title=title),
    )
