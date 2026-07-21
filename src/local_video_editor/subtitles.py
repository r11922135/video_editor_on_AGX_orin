from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

from .io_utils import atomic_write_json, atomic_write_text
from .media import burn_ass_subtitles
from .summary import _post_json, ollama_models, require_loopback_ollama
from .transcript import normalize_transcript_text, timestamp


class SubtitleError(RuntimeError):
    pass


def _rule_pattern(original: str) -> re.Pattern[str]:
    prefix = r"(?<!\w)" if original[:1].isalnum() else ""
    suffix = r"(?!\w)" if original[-1:].isalnum() else ""
    return re.compile(prefix + re.escape(original) + suffix)


def _phonetic_similarity(left: str, right: str) -> float:
    def clean(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in normalized if character.isalnum())

    left_clean = clean(left)
    right_clean = clean(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean == right_clean:
        left_symbols = "".join(
            character
            for character in unicodedata.normalize("NFKC", left)
            if not character.isalnum() and not character.isspace()
        )
        right_symbols = "".join(
            character
            for character in unicodedata.normalize("NFKC", right)
            if not character.isalnum() and not character.isspace()
        )
        if left_symbols != right_symbols:
            return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def build_alignment_chunks(
    segments: Iterable[dict[str, Any]], *, max_seconds: int
) -> list[dict[str, Any]]:
    """Group raw ASR segments into bounded correction/alignment scopes."""
    if not 30 <= int(max_seconds) <= 150:
        raise ValueError("max_seconds must be between 30 and 150")
    cleaned: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = normalize_transcript_text(str(segment.get("text", "")))
        if not text:
            continue
        start = max(0.0, float(segment.get("start", 0.0)))
        end = max(start, float(segment.get("end", start)))
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Subtitle segment timestamps must be finite")
        cleaned.append(
            {
                "id": int(segment.get("id", index)),
                "start": start,
                "end": end,
                "text": text,
                "words": list(segment.get("words") or []),
            }
        )
    cleaned.sort(key=lambda item: (item["start"], item["end"]))

    grouped: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for segment in cleaned:
        if current and segment["end"] - current[0]["start"] > max_seconds:
            grouped.append(current)
            current = []
        current.append(segment)
    if current:
        grouped.append(current)

    chunks: list[dict[str, Any]] = []
    for index, items in enumerate(grouped, 1):
        chunks.append(
            {
                "id": f"c{index:03d}",
                "start": float(items[0]["start"]),
                "end": float(items[-1]["end"]),
                "text": normalize_transcript_text(
                    " ".join(str(item["text"]) for item in items)
                ),
                "segments": items,
            }
        )
    return chunks


def correction_schema(scope_ids: list[str], max_rules: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "rules": {
                "type": "array",
                "maxItems": int(max_rules),
                "items": {
                    "type": "object",
                    "properties": {
                        "scope_id": {"type": "string", "enum": scope_ids},
                        "original": {"type": "string"},
                        "replacement": {"type": "string"},
                        "evidence": {
                            "type": "string",
                            "enum": [
                                "same_transcript",
                                "technical_context",
                                "standard_term",
                            ],
                        },
                    },
                    "required": [
                        "scope_id",
                        "original",
                        "replacement",
                        "evidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["rules"],
        "additionalProperties": False,
    }


def validate_correction_rules(
    value: Any,
    chunks: list[dict[str, Any]],
    *,
    max_rules: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("rules"), list):
        raise SubtitleError("Subtitle correction response must contain a rules array")
    scopes = {str(chunk["id"]): str(chunk["text"]) for chunk in chunks}
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in value["rules"]:
        if len(accepted) >= int(max_rules):
            break
        if not isinstance(candidate, dict):
            continue
        scope_id = str(candidate.get("scope_id", ""))
        original = str(candidate.get("original", "")).strip()
        replacement = str(candidate.get("replacement", "")).strip()
        evidence = str(candidate.get("evidence", ""))
        source = scopes.get(scope_id)
        if (
            source is None
            or not original
            or not replacement
            or original == replacement
            or "\n" in original
            or "\n" in replacement
            or len(original) > 120
            or len(replacement) > 120
            or not re.search(r"[A-Za-z0-9]", original + replacement)
            or evidence
            not in {"same_transcript", "technical_context", "standard_term"}
        ):
            continue
        pattern = _rule_pattern(original)
        matches = pattern.findall(source)
        if not matches:
            continue
        if abs(len(original.split()) - len(replacement.split())) > 2:
            continue
        similarity = _phonetic_similarity(original, replacement)
        ascii_length = max(
            len(re.sub(r"[^A-Za-z0-9]", "", original)),
            len(re.sub(r"[^A-Za-z0-9]", "", replacement)),
        )
        minimum_similarity = 0.60 if ascii_length <= 4 else 0.35
        if similarity < minimum_similarity:
            continue
        key = (scope_id, original)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            {
                "scope_id": scope_id,
                "original": original,
                "replacement": replacement,
                "evidence": evidence,
                "matched_occurrences": len(matches),
                "similarity": round(similarity, 4),
            }
        )
    return accepted


def apply_correction_rules(
    chunks: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return corrected chunk copies; the canonical ASR input is never mutated."""
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for rule in rules:
        by_scope.setdefault(str(rule["scope_id"]), []).append(rule)
    corrected: list[dict[str, Any]] = []
    for chunk in chunks:
        source = str(chunk["text"])
        applied: list[dict[str, Any]] = []
        scope_rules = sorted(
            by_scope.get(str(chunk["id"]), []),
            key=lambda item: len(str(item["original"])),
            reverse=True,
        )
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for rule in scope_rules:
            for match in _rule_pattern(str(rule["original"])).finditer(source):
                candidates.append((match.start(), match.end(), rule))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))

        selected: list[tuple[int, int, dict[str, Any]]] = []
        occupied_until = -1
        for start, end, rule in candidates:
            if start < occupied_until:
                continue
            selected.append((start, end, rule))
            occupied_until = end

        pieces: list[str] = []
        cursor = 0
        applied_counts: dict[tuple[str, str], int] = {}
        for start, end, rule in selected:
            pieces.append(source[cursor:start])
            pieces.append(str(rule["replacement"]))
            cursor = end
            key = (str(rule["original"]), str(rule["replacement"]))
            applied_counts[key] = applied_counts.get(key, 0) + 1
        pieces.append(source[cursor:])
        text = "".join(pieces)
        for rule in scope_rules:
            key = (str(rule["original"]), str(rule["replacement"]))
            if applied_counts.get(key):
                applied.append(
                    {**rule, "applied_occurrences": applied_counts[key]}
                )
        corrected.append({**chunk, "text": text, "applied_rules": applied})
    return corrected


def _correction_prompt(chunks: list[dict[str, Any]], max_rules: int) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f'<scope id="{chunk["id"]}" '
            f'time="{timestamp(chunk["start"])}-{timestamp(chunk["end"])}">\n'
            f'{chunk["text"]}\n</scope>'
        )
    return (
        "Inspect the ASR transcript scopes below. Return at most "
        f"{int(max_rules)} conservative correction rules. Correct only highly "
        "certain technical product names, acronyms, commands, capitalization, or "
        "word boundaries. `original` must be the smallest exact substring inside "
        "the same scope. Do not rewrite grammar, filler words, ordinary wording, "
        "or technical meaning. Do not map one valid technical concept to another. "
        "If uncertain, omit the rule. Return each identical rule once per scope; "
        "the program finds all exact occurrences.\n\n" + "\n\n".join(blocks)
    )


def propose_correction_rules(
    chunks: list[dict[str, Any]],
    *,
    ollama_url: str,
    model: str,
    context_tokens: int,
    output_tokens: int,
    max_rules: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    require_loopback_ollama(ollama_url)
    installed = ollama_models(ollama_url)
    if not any(name == model or name.removesuffix(":latest") == model for name in installed):
        raise SubtitleError(f"Local Ollama model is not installed: {model}")
    prompt = _correction_prompt(chunks, max_rules)
    estimated_tokens = max(1, int(len(prompt) / 2.8))
    if estimated_tokens + int(output_tokens) + 1024 > int(context_tokens):
        raise SubtitleError(
            f"Subtitle correction prompt needs about {estimated_tokens:,} tokens; "
            f"configured context is {int(context_tokens):,}"
        )
    request = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "format": correction_schema(
            [str(chunk["id"]) for chunk in chunks], int(max_rules)
        ),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative local ASR correction engine. Output only "
                    "the requested JSON. Prefer leaving an error unchanged over making "
                    "an uncertain correction."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_ctx": int(context_tokens),
            "num_predict": int(output_tokens),
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 20,
            "seed": 42,
        },
    }
    started = time.monotonic()
    response = _post_json(f"{ollama_url.rstrip('/')}/api/chat", request, timeout=1200)
    elapsed = time.monotonic() - started
    raw = str(response.get("message", {}).get("content", ""))
    if response.get("done_reason") == "length":
        raise SubtitleError("Subtitle correction JSON reached its output token limit")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SubtitleError("Subtitle correction model returned invalid JSON") from exc
    rules = validate_correction_rules(decoded, chunks, max_rules=int(max_rules))
    metrics = {
        "status": "complete",
        "model": response.get("model", model),
        "elapsed_seconds": elapsed,
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "load_duration_ns": response.get("load_duration"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_duration_ns": response.get("eval_duration"),
        "total_duration_ns": response.get("total_duration"),
        "done_reason": response.get("done_reason"),
        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "proposed_rule_count": len(decoded.get("rules", [])),
        "accepted_rule_count": len(rules),
    }
    return rules, metrics, raw


def _raw_words_for_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in chunk.get("segments", []):
        valid = []
        for word in segment.get("words", []):
            start = word.get("start")
            end = word.get("end")
            text = str(word.get("word", "")).strip()
            if start is None or end is None or not text:
                continue
            valid.append(
                {"start": float(start), "end": float(end), "text": text}
            )
        if valid:
            words.extend(valid)
            continue
        tokens = re.findall(r"\S+", str(segment.get("text", "")))
        if not tokens:
            continue
        start = float(segment["start"])
        duration = max(0.01, float(segment["end"]) - start)
        for index, token in enumerate(tokens):
            words.append(
                {
                    "start": start + duration * index / len(tokens),
                    "end": start + duration * (index + 1) / len(tokens),
                    "text": token,
                }
            )
    return words


def _display_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"\S+", text):
        if re.search(r"\w", token):
            tokens.append(token)
        elif tokens:
            tokens[-1] += token
    return tokens


def _local_model_snapshot(model_name: str) -> str:
    configured = Path(model_name).expanduser()
    if configured.is_dir():
        return str(configured.resolve())

    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not cache_root and os.environ.get("HF_HOME"):
        cache_root = str(Path(os.environ["HF_HOME"]) / "hub")
    if not cache_root or "/" not in model_name:
        raise SubtitleError(
            f"Forced Aligner is not a local directory and no mounted HF cache "
            f"contains it: {model_name}"
        )
    repository = Path(cache_root) / f"models--{model_name.replace('/', '--')}"
    reference = repository / "refs" / "main"
    if not reference.is_file():
        raise SubtitleError(f"Forced Aligner is not cached locally: {model_name}")
    revision = reference.read_text(encoding="utf-8").strip()
    snapshot_root = (repository / "snapshots").resolve()
    snapshot = (snapshot_root / revision).resolve()
    if snapshot.parent != snapshot_root or not snapshot.is_dir():
        raise SubtitleError(f"Forced Aligner cache snapshot is invalid: {model_name}")
    return str(snapshot)


def attach_display_tokens(
    corrected_text: str, aligned: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    display = _display_tokens(corrected_text)
    if len(display) != len(aligned):
        raise SubtitleError(
            f"Aligner returned {len(aligned)} words for {len(display)} display tokens"
        )
    mismatches = 0
    result: list[dict[str, Any]] = []
    for token, item in zip(display, aligned):
        if _phonetic_similarity(token, str(item["text"])) < 0.3:
            mismatches += 1
        result.append({**item, "text": token})
    if mismatches > max(2, math.ceil(len(result) * 0.08)):
        raise SubtitleError(f"Aligner/display token mismatch count is {mismatches}")
    return result


def _validated_alignment(
    items: list[dict[str, Any]], *, audio_seconds: float
) -> list[dict[str, Any]]:
    if not items:
        raise SubtitleError("Aligner returned no words")
    validated: list[dict[str, Any]] = []
    previous_start = -0.05
    for item in items:
        start = float(item["start"])
        end = float(item["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise SubtitleError("Aligner returned a non-finite timestamp")
        if start < -0.05 or end < start or end > audio_seconds + 0.05:
            raise SubtitleError(
                f"Aligner timestamp {start:.3f}-{end:.3f}s is outside its audio chunk"
            )
        if start + 0.05 < previous_start:
            raise SubtitleError("Aligner timestamps are not monotonic")
        start = min(audio_seconds, max(0.0, start))
        end = min(audio_seconds, max(start, end))
        validated.append({**item, "start": start, "end": end})
        previous_start = start
    return validated


def align_chunks(
    audio_path: Path,
    chunks: list[dict[str, Any]],
    *,
    model_name: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    errors: list[str] = []
    aligned_words: list[dict[str, Any]] = []
    try:
        import soundfile as sf
        import torch
        from qwen_asr import Qwen3ForcedAligner

        if not torch.cuda.is_available():
            raise SubtitleError("CUDA is unavailable for Forced Aligner")
        audio, sample_rate = sf.read(
            str(audio_path), dtype="float32", always_2d=False
        )
        if getattr(audio, "ndim", 1) != 1:
            raise SubtitleError("Forced Aligner expects mono analysis audio")
        torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        local_model = _local_model_snapshot(model_name)
        model = Qwen3ForcedAligner.from_pretrained(
            local_model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            local_files_only=True,
        )
        torch.cuda.synchronize()
        load_seconds = time.monotonic() - load_started
    except Exception as exc:
        fallback = [word for chunk in chunks for word in _raw_words_for_chunk(chunk)]
        return fallback, {
            "status": "fallback",
            "model": model_name,
            "error": str(exc),
            "aligned_chunk_count": 0,
            "fallback_chunk_count": len(chunks),
            "word_count": len(fallback),
            "elapsed_seconds": time.monotonic() - started,
        }

    aligned_count = 0
    fallback_count = 0
    try:
        for index, chunk in enumerate(chunks, 1):
            if progress is not None:
                progress(f"Forced-aligning subtitle chunk {index}/{len(chunks)}")
            start_sample = max(0, int((float(chunk["start"]) - 0.25) * sample_rate))
            end_sample = min(
                len(audio), int((float(chunk["end"]) + 0.25) * sample_rate)
            )
            offset = start_sample / float(sample_rate)
            chunk_audio_seconds = (end_sample - start_sample) / float(sample_rate)
            try:
                result = model.align(
                    audio=(audio[start_sample:end_sample], int(sample_rate)),
                    text=str(chunk["text"]),
                    language="English",
                )[0]
                relative = [
                    {
                        "start": float(item.start_time),
                        "end": float(item.end_time),
                        "text": str(item.text),
                    }
                    for item in result
                ]
                relative = _validated_alignment(
                    relative, audio_seconds=chunk_audio_seconds
                )
                display = attach_display_tokens(str(chunk["text"]), relative)
                for item in display:
                    item["start"] += offset
                    item["end"] += offset
                aligned_words.extend(display)
                aligned_count += 1
            except Exception as exc:
                fallback_count += 1
                errors.append(f"{chunk['id']}: {exc}")
                aligned_words.extend(_raw_words_for_chunk(chunk))
        torch.cuda.synchronize()
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
    finally:
        del model
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    aligned_words.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return aligned_words, {
        "status": "complete" if not errors else "partial_fallback",
        "model": model_name,
        "model_load_seconds": load_seconds,
        "aligned_chunk_count": aligned_count,
        "fallback_chunk_count": fallback_count,
        "word_count": len(aligned_words),
        "peak_cuda_bytes": peak_cuda_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "errors": errors,
    }


def _join_tokens(tokens: Iterable[str]) -> str:
    text = " ".join(str(token).strip() for token in tokens if str(token).strip())
    text = re.sub(r"\s+([,.;:!?%\]\)])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    return text.strip()


def build_cues(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        start = float(word.get("start", 0.0))
        end = max(start, float(word.get("end", start)))
        if math.isfinite(start) and math.isfinite(end):
            cleaned.append({"start": start, "end": end, "text": text})
    cleaned.sort(key=lambda item: (item["start"], item["end"]))

    cues: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        cues.append(
            {
                "start": float(current[0]["start"]),
                "end": float(current[-1]["end"]),
                "text": _join_tokens(item["text"] for item in current),
            }
        )
        current.clear()

    for word in cleaned:
        if current:
            gap = word["start"] - current[-1]["end"]
            projected = _join_tokens(
                [item["text"] for item in current] + [word["text"]]
            )
            duration = word["end"] - current[0]["start"]
            if gap >= 0.65 or duration > 6.0 or len(projected) > 78 or len(current) >= 16:
                flush()
        current.append(word)
        current_text = _join_tokens(item["text"] for item in current)
        current_duration = current[-1]["end"] - current[0]["start"]
        if re.search(r"[.!?][\"')\]]?$", current_text) and current_duration >= 1.0:
            flush()
    flush()

    normalized: list[dict[str, Any]] = []
    for cue in cues:
        current = {
            **cue,
            "start": max(0.0, float(cue["start"])),
            "end": min(
                float(cue["start"]) + 6.0,
                max(float(cue["start"]) + 0.05, float(cue["end"])),
            ),
        }
        if normalized and current["start"] < normalized[-1]["end"]:
            previous = normalized[-1]
            if current["start"] > previous["start"] + 0.08:
                previous["end"] = max(
                    previous["start"] + 0.05, current["start"] - 0.02
                )
            else:
                merged_text = _join_tokens([previous["text"], current["text"]])
                merged_end = min(
                    previous["start"] + 6.0,
                    max(previous["end"], current["end"]),
                )
                if len(merged_text) <= 78:
                    previous["text"] = merged_text
                    previous["end"] = merged_end
                    continue
                current["start"] = previous["end"] + 0.02
                current["end"] = max(current["start"] + 0.05, current["end"])
                current["end"] = min(current["start"] + 6.0, current["end"])
        normalized.append(current)

    for index, cue in enumerate(normalized):
        latest_end = cue["start"] + 6.0
        if index + 1 < len(normalized):
            latest_end = min(latest_end, normalized[index + 1]["start"] - 0.02)
        cue["end"] = min(cue["end"], latest_end)
        cue["end"] = max(cue["end"], min(cue["start"] + 1.0, latest_end))
    return normalized


def _balanced_lines(text: str, width: int = 42) -> str:
    if len(text) <= width:
        return text
    words = text.split()
    candidates: list[tuple[int, int, str, str]] = []
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        overflow = max(0, len(left) - width) + max(0, len(right) - width)
        balance = abs(len(left) - len(right))
        candidates.append((overflow, balance, left, right))
    if not candidates:
        return text
    _overflow, _balance, left, right = min(candidates)
    return f"{left}\n{right}"


def render_subtitle_srt(cues: Iterable[dict[str, Any]]) -> str:
    blocks = []
    for index, cue in enumerate(cues, 1):
        blocks.append(
            f"{index}\n{timestamp(cue['start'], srt=True)} --> "
            f"{timestamp(cue['end'], srt=True)}\n{_balanced_lines(str(cue['text']))}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def _ass_text(text: str) -> str:
    escaped = (
        text.replace("\\", "\\\u2060")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )
    return _balanced_lines(escaped).replace("\n", r"\N")


def render_ass(cues: Iterable[dict[str, Any]]) -> str:
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        "Style: Default,DejaVu Sans,52,&H00FFFFFF,&H000000FF,&H00000000,"
        "&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,48,48,54,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for cue in cues:
        lines.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},"
            f"Default,,0,0,0,,{_ass_text(str(cue['text']))}"
        )
    return "\n".join(lines) + "\n"


def create_subtitled_video(
    *,
    analysis_audio: Path,
    edited_video: Path,
    segments: list[dict[str, Any]],
    output_dir: Path,
    summary_config: dict[str, Any],
    subtitle_config: dict[str, Any],
    video_config: dict[str, Any],
    filename_prefix: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    chunks = build_alignment_chunks(
        segments, max_seconds=int(subtitle_config["alignment_chunk_seconds"])
    )
    if not chunks:
        raise SubtitleError("Transcript has no text to subtitle")

    errors: list[str] = []
    raw_response = ""
    try:
        if progress is not None:
            progress("Generating compact local subtitle correction rules")
        rules, correction, raw_response = propose_correction_rules(
            chunks,
            ollama_url=str(summary_config["ollama_url"]),
            model=str(summary_config["model"]),
            context_tokens=int(subtitle_config["correction_context_tokens"]),
            output_tokens=int(subtitle_config["correction_output_tokens"]),
            max_rules=int(subtitle_config["correction_max_rules"]),
        )
    except Exception as exc:
        rules = []
        correction = {
            "status": "fallback",
            "model": str(summary_config["model"]),
            "accepted_rule_count": 0,
            "error": str(exc),
        }
        errors.append(f"correction: {exc}")
    atomic_write_text(output_dir / "subtitle.correction.raw.txt", raw_response)
    atomic_write_json(
        output_dir / "subtitle.rules.json",
        {
            "canonical_transcript_modified": False,
            "summary_input_modified": False,
            "correction": correction,
            "rules": rules,
        },
    )

    corrected = apply_correction_rules(chunks, rules)
    atomic_write_json(
        output_dir / "subtitle.corrected.json",
        {
            "source": "transcript.json",
            "canonical_transcript_modified": False,
            "chunks": [
                {
                    "id": chunk["id"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "text": chunk["text"],
                    "applied_rules": chunk["applied_rules"],
                }
                for chunk in corrected
            ],
        },
    )

    words, alignment = align_chunks(
        analysis_audio,
        corrected,
        model_name=str(subtitle_config["aligner_model"]),
        progress=progress,
    )
    errors.extend(f"alignment: {item}" for item in alignment.get("errors", []))
    if alignment.get("error"):
        errors.append(f"alignment: {alignment['error']}")
    cues = build_cues(words)
    if not cues:
        raise SubtitleError("No usable subtitle cues were produced")
    subtitle_srt_name = (
        f"{filename_prefix}_subtitle.srt" if filename_prefix else "subtitle.srt"
    )
    subtitle_ass_name = (
        f"{filename_prefix}_subtitle.ass" if filename_prefix else "subtitle.ass"
    )
    output_video_name = (
        f"{filename_prefix}_subtitled.mp4" if filename_prefix else "subtitled.mp4"
    )
    atomic_write_text(output_dir / subtitle_srt_name, render_subtitle_srt(cues))
    atomic_write_text(output_dir / subtitle_ass_name, render_ass(cues))
    if errors:
        atomic_write_text(output_dir / "subtitle.error.log", "\n".join(errors) + "\n")
    else:
        (output_dir / "subtitle.error.log").unlink(missing_ok=True)

    if progress is not None:
        progress("Burning subtitles into a separate video (audio stream copied)")
    output_video = output_dir / output_video_name
    burn_ass_subtitles(
        edited_video,
        output_dir / subtitle_ass_name,
        output_video,
        output_dir / "subtitle.render.log",
        codec=str(video_config["codec"]),
        preset=str(video_config["preset"]),
        crf=int(video_config["crf"]),
    )
    return {
        "output": output_video.name,
        "subtitle_srt": subtitle_srt_name,
        "subtitle_ass": subtitle_ass_name,
        "cue_count": len(cues),
        "correction": correction,
        "accepted_rule_count": len(rules),
        "alignment": alignment,
        "fallback_used": bool(errors),
        "errors": errors,
    }
