from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable

from .io_utils import atomic_write_text


READABLE_SECTION_SECONDS = 300
READABLE_PARAGRAPH_GAP_SECONDS = 2.5
READABLE_PARAGRAPH_MIN_CHARS = 300
READABLE_PARAGRAPH_MAX_CHARS = 700
READABLE_PARAGRAPH_MAX_SECONDS = 60.0


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


def normalize_transcript_text(text: str) -> str:
    """Normalize layout artifacts without changing any spoken word."""
    normalized = " ".join(str(text).split())
    return re.sub(r"\s+([,.;:!?])", r"\1", normalized).strip()


def _readable_segments(
    segments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for segment in segments:
        text = normalize_transcript_text(str(segment.get("text", "")))
        if not text:
            continue
        raw_start = float(segment.get("start", 0.0))
        raw_end = float(segment.get("end", raw_start))
        if not math.isfinite(raw_start) or not math.isfinite(raw_end):
            raise ValueError("Transcript timestamps must be finite")
        start = max(0.0, raw_start)
        cleaned.append(
            {
                "start": start,
                "end": max(start, raw_end),
                "text": text,
            }
        )
    return sorted(cleaned, key=lambda item: (item["start"], item["end"]))


def readable_transcript_blocks(
    segments: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group ASR fragments into readable, lossless chronological paragraphs."""
    cleaned = _readable_segments(segments)
    blocks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        text = normalize_transcript_text(" ".join(item["text"] for item in current))
        blocks.append(
            {
                "section_start": int(
                    current[0]["start"] // READABLE_SECTION_SECONDS
                )
                * READABLE_SECTION_SECONDS,
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": text,
            }
        )
        current.clear()

    for segment in cleaned:
        if current:
            current_text = normalize_transcript_text(
                " ".join(item["text"] for item in current)
            )
            current_section = int(
                current[0]["start"] // READABLE_SECTION_SECONDS
            )
            next_section = int(segment["start"] // READABLE_SECTION_SECONDS)
            gap = segment["start"] - current[-1]["end"]
            projected_chars = len(current_text) + 1 + len(segment["text"])
            projected_seconds = segment["end"] - current[0]["start"]
            natural_boundary = (
                len(current_text) >= READABLE_PARAGRAPH_MIN_CHARS
                and re.search(r"[.!?][\"')\]]?$", current[-1]["text"])
                is not None
            )
            if (
                next_section != current_section
                or gap >= READABLE_PARAGRAPH_GAP_SECONDS
                or projected_chars > READABLE_PARAGRAPH_MAX_CHARS
                or projected_seconds > READABLE_PARAGRAPH_MAX_SECONDS
                or natural_boundary
            ):
                flush()
        current.append(segment)
    flush()
    return blocks


def normalized_segment_text(segments: Iterable[dict[str, Any]]) -> str:
    """Return the exact normalized prose represented by readable blocks."""
    return normalize_transcript_text(
        " ".join(item["text"] for item in _readable_segments(segments))
    )


def _second_timestamp(seconds: float) -> str:
    return timestamp(seconds).split(".", 1)[0]


def render_transcript_markdown(
    segments: Iterable[dict[str, Any]], *, title: str
) -> str:
    lines = [
        f"# Transcript — {title}",
        "",
        "> Automatically transcribed locally; wording may contain recognition errors.",
        "",
    ]
    active_section: int | None = None
    for block in readable_transcript_blocks(segments):
        section_start = int(block["section_start"])
        if section_start != active_section:
            if active_section is not None:
                lines.append("")
            section_end = section_start + READABLE_SECTION_SECONDS
            lines.extend(
                [
                    f"## {_second_timestamp(section_start)}–"
                    f"{_second_timestamp(section_end)}",
                    "",
                ]
            )
            active_section = section_start
        lines.extend(
            [
                f"**[{_second_timestamp(float(block['start']))}]** "
                f"{block['text']}",
                "",
            ]
        )
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
    segments: list[dict[str, Any]],
    *,
    title: str,
    output_dir: Path,
    filename_prefix: str | None = None,
) -> None:
    srt_name = (
        f"{filename_prefix}_transcript.srt" if filename_prefix else "transcript.srt"
    )
    markdown_name = (
        f"{filename_prefix}_transcript.md" if filename_prefix else "transcript.md"
    )
    atomic_write_text(output_dir / srt_name, render_srt(segments))
    atomic_write_text(
        output_dir / markdown_name,
        render_transcript_markdown(segments, title=title),
    )
