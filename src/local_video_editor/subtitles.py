from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from collections import Counter
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
    prefix = r"(?<![\w+/#.-])" if original[:1].isalnum() else ""
    suffix = (
        r"(?![\w+/#_-]|\.(?=\w))" if original[-1:].isalnum() else ""
    )
    return re.compile(prefix + re.escape(original) + suffix)


def _join_tokens(tokens: Iterable[str]) -> str:
    text = " ".join(str(token).strip() for token in tokens if str(token).strip())
    text = re.sub(r"\s+([,.;:!?%\]\)])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    return text.strip()


def _source_token_spans(text: str) -> list[dict[str, Any]]:
    """Split normalized display text into stable source tokens."""
    tokens: list[dict[str, Any]] = []
    for match in re.finditer(r"\S+", text):
        token = match.group(0)
        if re.search(r"\w", token):
            tokens.append(
                {
                    "text": token,
                    "char_start": match.start(),
                    "char_end": match.end(),
                }
            )
        elif tokens:
            tokens[-1]["text"] += token
            tokens[-1]["char_end"] = match.end()
    return tokens


def _token_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in {"'", "’"}
    )


def _timed_source_tokens(
    segment: dict[str, Any], *, segment_id: int, source_ordinal: int
) -> list[dict[str, Any]]:
    text = normalize_transcript_text(str(segment.get("text", "")))
    source = _source_token_spans(text)
    if not source:
        return []
    start = max(0.0, float(segment.get("start", 0.0)))
    end = max(start, float(segment.get("end", start)))
    duration = max(0.01, end - start)
    valid_words: list[dict[str, Any]] = []
    for word in segment.get("words") or []:
        word_start = word.get("start")
        word_end = word.get("end")
        word_text = str(word.get("word", "")).strip()
        if word_start is None or word_end is None or not word_text:
            continue
        word_start = float(word_start)
        word_end = float(word_end)
        if (
            not math.isfinite(word_start)
            or not math.isfinite(word_end)
            or word_end < word_start
        ):
            continue
        key = _token_key(word_text)
        if not key:
            continue
        word_start = min(end, max(start, word_start))
        word_end = min(end, max(word_start, word_end))
        valid_words.append(
            {
                "text": word_text,
                "key": key,
                "start": word_start,
                "end": word_end,
                "probability": word.get("probability"),
            }
        )
    if any(
        current["start"] + 0.05 < previous["start"]
        for previous, current in zip(valid_words, valid_words[1:])
    ):
        valid_words = []

    source_keys = [_token_key(item["text"]) for item in source]
    word_keys = [str(item["key"]) for item in valid_words]
    timings: list[tuple[float, float, str, float | None]] = []
    if (
        valid_words
        and len(source_keys) == len(word_keys)
        and source_keys == word_keys
    ):
        timings = [
            (
                float(item["start"]),
                float(item["end"]),
                "whisper_word",
                None
                if item.get("probability") is None
                else float(item["probability"]),
            )
            for item in valid_words
        ]
    elif valid_words and "".join(source_keys) == "".join(word_keys):
        source_ranges: list[tuple[int, int]] = []
        cursor = 0
        for key in source_keys:
            source_ranges.append((cursor, cursor + len(key)))
            cursor += len(key)
        word_ranges: list[tuple[int, int]] = []
        cursor = 0
        for key in word_keys:
            word_ranges.append((cursor, cursor + len(key)))
            cursor += len(key)
        for source_start, source_end in source_ranges:
            overlaps = [
                item
                for item, (word_start, word_end) in zip(valid_words, word_ranges)
                if word_end > source_start and word_start < source_end
            ]
            if not overlaps:
                timings = []
                break
            probabilities = [
                float(item["probability"])
                for item in overlaps
                if item.get("probability") is not None
            ]
            timings.append(
                (
                    float(overlaps[0]["start"]),
                    float(overlaps[-1]["end"]),
                    "whisper_reconciled",
                    sum(probabilities) / len(probabilities) if probabilities else None,
                )
            )

    if len(timings) != len(source):
        timings = [
            (
                start + duration * index / len(source),
                start + duration * (index + 1) / len(source),
                "segment_interpolation",
                None,
            )
            for index in range(len(source))
        ]

    return [
        {
            **item,
            "source_id": f"s{source_ordinal:06d}:t{index:04d}",
            "segment_id": segment_id,
            "start": timing[0],
            "end": timing[1],
            "timing_source": timing[2],
            "probability": timing[3],
        }
        for index, (item, timing) in enumerate(zip(source, timings))
    ]


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


def build_correction_chunks(
    segments: Iterable[dict[str, Any]], *, max_seconds: int
) -> list[dict[str, Any]]:
    """Group raw ASR segments into bounded local-correction scopes."""
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
        segment_id = int(segment.get("id", index))
        source_tokens = _timed_source_tokens(
            segment, segment_id=segment_id, source_ordinal=index
        )
        if not source_tokens:
            continue
        cleaned.append(
            {
                "id": segment_id,
                "start": start,
                "end": end,
                "text": text,
                "words": list(segment.get("words") or []),
                "source_tokens": source_tokens,
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
        chunk_id = f"c{index:03d}"
        source_tokens = [
            dict(token)
            for item in items
            for token in item.get("source_tokens", [])
        ]
        chunk_text = _join_tokens(token["text"] for token in source_tokens)
        spans = _source_token_spans(chunk_text)
        if len(spans) != len(source_tokens):
            raise SubtitleError(
                "Source token rendering changed the subtitle token count"
            )
        for token_index, (token, span) in enumerate(zip(source_tokens, spans)):
            token.update(
                {
                    "chunk_id": chunk_id,
                    "token_index": token_index,
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "text": span["text"],
                }
            )
        chunks.append(
            {
                "id": chunk_id,
                "start": float(items[0]["start"]),
                "end": float(items[-1]["end"]),
                "text": chunk_text,
                "segments": items,
                "source_tokens": source_tokens,
            }
        )
    return chunks


def _candidate_form(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    while value and unicodedata.category(value[0]).startswith("P"):
        value = value[1:]
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1]
    return value


def _technical_shape(value: str) -> bool:
    letters = [character for character in value if character.isalpha()]
    return bool(
        letters
        and (
            (len(letters) >= 2 and all(character.isupper() for character in letters))
            or any(character.isdigit() for character in value)
            or any(character.isupper() for character in value[1:])
            or bool(re.search(r"[+/#_.-]", value))
        )
    )


def mine_correction_candidates(
    chunks: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    occurrences: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        tokens = list(chunk.get("source_tokens", []))
        for start_index in range(len(tokens)):
            for width in range(1, min(3, len(tokens) - start_index) + 1):
                selected = tokens[start_index : start_index + width]
                if any(
                    re.search(r"[.!?,;:]$", str(token.get("text", "")))
                    for token in selected[:-1]
                ):
                    break
                char_start = int(selected[0]["char_start"])
                char_end = int(selected[-1]["char_end"])
                form = _candidate_form(str(chunk["text"])[char_start:char_end])
                if not form or len(form) > 40 or not _technical_shape(form):
                    continue
                parts = [
                    _candidate_form(str(token.get("text", "")))
                    for token in selected
                ]
                if width > 1 and not (
                    all(re.fullmatch(r"[A-Z0-9]", part or "") for part in parts)
                    or (
                        all(len(part) > 1 for part in parts)
                        and all(part[:1].isupper() for part in parts)
                    )
                ):
                    continue
                probabilities = [
                    float(token["probability"])
                    for token in selected
                    if token.get("probability") is not None
                ]
                context_start = max(0, char_start - 70)
                context_end = min(len(chunk["text"]), char_end + 70)
                occurrences.setdefault(form, []).append(
                    {
                        "scope_id": str(chunk["id"]),
                        "context": str(chunk["text"])[context_start:context_end],
                        "probability": (
                            sum(probabilities) / len(probabilities)
                            if probabilities
                            else None
                        ),
                    }
                )

    forms = list(occurrences)
    near: dict[str, list[tuple[str, float]]] = {form: [] for form in forms}
    ranked_forms = sorted(
        forms, key=lambda form: len(occurrences[form]), reverse=True
    )[:200]
    for index, left in enumerate(ranked_forms):
        for right in ranked_forms[index + 1 :]:
            similarity = _phonetic_similarity(left, right)
            if 0.58 <= similarity < 1.0:
                near[left].append((right, similarity))
                near[right].append((left, similarity))

    ranked = sorted(
        forms,
        key=lambda form: (
            len({item["scope_id"] for item in occurrences[form]}),
            len(occurrences[form]),
            bool(near.get(form)),
            -len(form.split()),
            len(form),
        ),
        reverse=True,
    )[: int(limit)]
    candidates: list[dict[str, Any]] = []
    for index, form in enumerate(ranked, 1):
        items = occurrences[form]
        context_indexes = sorted({0, len(items) // 2, len(items) - 1})
        probabilities = [
            float(item["probability"])
            for item in items
            if item.get("probability") is not None
        ]
        candidates.append(
            {
                "candidate_id": f"t{index:03d}",
                "original": form,
                "occurrence_count": len(items),
                "scope_count": len({item["scope_id"] for item in items}),
                "average_asr_probability": (
                    round(sum(probabilities) / len(probabilities), 4)
                    if probabilities
                    else None
                ),
                "near_variants": [
                    {"text": other, "similarity": round(similarity, 4)}
                    for other, similarity in sorted(
                        near.get(form, []), key=lambda item: item[1], reverse=True
                    )[:5]
                ],
                "contexts": [
                    {
                        "scope_id": items[item_index]["scope_id"],
                        "text": items[item_index]["context"],
                    }
                    for item_index in context_indexes
                ],
            }
        )
    return candidates


def correction_schema(scope_ids: list[str], max_rules: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "document_rules": {
                "type": "array",
                "maxItems": int(max_rules),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                    "required": ["candidate_id", "replacement"],
                    "additionalProperties": False,
                },
            },
            "local_rules": {
                "type": "array",
                "maxItems": int(max_rules),
                "items": {
                    "type": "object",
                    "properties": {
                        "scope_id": {"type": "string", "enum": scope_ids},
                        "original": {"type": "string"},
                        "replacement": {"type": "string"},
                    },
                    "required": ["scope_id", "original", "replacement"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["document_rules", "local_rules"],
        "additionalProperties": False,
    }


def validate_correction_rules(
    value: Any,
    chunks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    max_rules: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("document_rules"), list)
        or not isinstance(value.get("local_rules"), list)
    ):
        raise SubtitleError(
            "Subtitle correction response must contain document_rules and local_rules"
        )
    scopes = {str(chunk["id"]): str(chunk["text"]) for chunk in chunks}
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate for candidate in candidates
    }
    all_text = "\n".join(scopes.values())
    accepted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    rejected: Counter[str] = Counter()

    proposed: list[dict[str, Any]] = []
    for item in value["document_rules"]:
        if not isinstance(item, dict):
            rejected["invalid_object"] += 1
            continue
        candidate = candidate_by_id.get(str(item.get("candidate_id", "")))
        if candidate is None:
            rejected["unknown_candidate"] += 1
            continue
        proposed.append(
            {
                "scope_id": "document",
                "candidate_id": str(candidate["candidate_id"]),
                "original": str(candidate["original"]),
                "replacement": str(item.get("replacement", "")).strip(),
                "candidate_occurrence_count": int(candidate["occurrence_count"]),
                "candidate_scope_count": int(candidate["scope_count"]),
            }
        )
    for item in value["local_rules"]:
        if not isinstance(item, dict):
            rejected["invalid_object"] += 1
            continue
        proposed.append(
            {
                "scope_id": str(item.get("scope_id", "")),
                "original": str(item.get("original", "")).strip(),
                "replacement": str(item.get("replacement", "")).strip(),
            }
        )

    for item in proposed:
        if len(accepted) >= int(max_rules):
            rejected["safety_cap"] += 1
            continue
        scope_id = str(item["scope_id"])
        original = str(item["original"])
        replacement = str(item["replacement"])
        source_texts = (
            list(scopes.values())
            if scope_id == "document"
            else [scopes.get(scope_id)]
        )
        if any(source is None for source in source_texts):
            rejected["unknown_scope"] += 1
            continue
        if (
            not original
            or not replacement
            or original == replacement
            or "\n" in original
            or "\n" in replacement
            or len(original) > 120
            or len(replacement) > 120
            or not re.search(r"[A-Za-z0-9]", original + replacement)
        ):
            rejected["invalid_text"] += 1
            continue
        if scope_id == "document" and (
            int(item.get("candidate_occurrence_count", 0)) < 2
            or not _technical_shape(original)
        ):
            rejected["unsafe_document_scope"] += 1
            continue
        if scope_id != "document" and not (
            _technical_shape(original) or _technical_shape(replacement)
        ):
            rejected["unsafe_local_scope"] += 1
            continue
        matches = sum(
            len(_rule_pattern(original).findall(str(source))) for source in source_texts
        )
        if not matches:
            rejected["no_exact_match"] += 1
            continue
        collapsed_acronym = bool(
            re.fullmatch(r"(?:[A-Za-z0-9]\s+){1,7}[A-Za-z0-9]", original)
            and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", replacement)
        )
        if (
            abs(len(original.split()) - len(replacement.split())) > 1
            and not collapsed_acronym
        ):
            rejected["word_count_delta"] += 1
            continue
        similarity = _phonetic_similarity(original, replacement)
        ascii_length = max(
            len(re.sub(r"[^A-Za-z0-9]", "", original)),
            len(re.sub(r"[^A-Za-z0-9]", "", replacement)),
        )
        replacement_observed = len(_rule_pattern(replacement).findall(all_text))
        minimum_similarity = 0.60 if ascii_length <= 4 else 0.35
        observed_acronym_variant = bool(
            replacement_observed
            and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", original)
            and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", replacement)
        )
        if observed_acronym_variant:
            minimum_similarity = min(minimum_similarity, 0.55)
        acronym_substitution = bool(
            re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", original)
            and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", replacement)
            and len(original) == len(replacement)
            and sum(left != right for left, right in zip(original, replacement)) <= 1
        )
        if (
            scope_id == "document"
            and replacement_observed == 0
            and not acronym_substitution
        ):
            minimum_similarity = max(minimum_similarity, 0.55)
            if original[:2].casefold() == replacement[:2].casefold():
                minimum_similarity = 0.55
        if similarity < minimum_similarity:
            rejected["low_similarity"] += 1
            continue
        key = (scope_id, original)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        accepted.append(
            {
                "rule_id": f"r{len(accepted) + 1:03d}",
                **item,
                "matched_occurrences": matches,
                "replacement_observed_count": replacement_observed,
                "similarity": round(similarity, 4),
            }
        )
    return accepted, {
        "proposed_document_rule_count": len(value["document_rules"]),
        "proposed_local_rule_count": len(value["local_rules"]),
        "accepted_document_rule_count": sum(
            rule["scope_id"] == "document" for rule in accepted
        ),
        "accepted_local_rule_count": sum(
            rule["scope_id"] != "document" for rule in accepted
        ),
        "rejected_rule_count": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "safety_cap_reached": bool(rejected.get("safety_cap")),
    }


def _applicable_rules(
    rules: list[dict[str, Any]], chunk_id: str
) -> list[dict[str, Any]]:
    return [
        rule
        for rule in rules
        if str(rule.get("scope_id")) in {"document", str(chunk_id)}
    ]


def apply_correction_rules(
    chunks: list[dict[str, Any]], rules: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select immutable-source replacements without changing canonical ASR data."""
    corrected: list[dict[str, Any]] = []
    for chunk in chunks:
        source = str(chunk["text"])
        raw_candidates: list[dict[str, Any]] = []
        for rule in _applicable_rules(rules, str(chunk["id"])):
            for match in _rule_pattern(str(rule["original"])).finditer(source):
                raw_candidates.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "priority": 1 if rule["scope_id"] != "document" else 0,
                        "rule": rule,
                    }
                )

        conflicts: list[dict[str, Any]] = []
        blocked: set[int] = set()
        exact_groups: dict[tuple[int, int, int], list[int]] = {}
        for index, candidate in enumerate(raw_candidates):
            exact_groups.setdefault(
                (candidate["start"], candidate["end"], candidate["priority"]), []
            ).append(index)
        for indexes in exact_groups.values():
            replacements = {
                str(raw_candidates[index]["rule"]["replacement"]) for index in indexes
            }
            if len(replacements) > 1:
                blocked.update(indexes)
                conflicts.append(
                    {
                        "reason": "same_span_different_replacement",
                        "rule_ids": [
                            raw_candidates[index]["rule"].get("rule_id")
                            for index in indexes
                        ],
                    }
                )

        ordered = sorted(
            (
                candidate
                for index, candidate in enumerate(raw_candidates)
                if index not in blocked
            ),
            key=lambda item: (
                -item["priority"],
                -(item["end"] - item["start"]),
                item["start"],
                str(item["rule"].get("rule_id", "")),
            ),
        )
        selected: list[dict[str, Any]] = []
        for candidate in ordered:
            if any(
                candidate["start"] < chosen["end"]
                and chosen["start"] < candidate["end"]
                for chosen in selected
            ):
                continue
            selected.append(candidate)
        selected.sort(key=lambda item: item["start"])

        pieces: list[str] = []
        cursor = 0
        occurrences: list[dict[str, Any]] = []
        applied_counts: Counter[str] = Counter()
        for index, candidate in enumerate(selected, 1):
            rule = candidate["rule"]
            pieces.append(source[cursor : candidate["start"]])
            pieces.append(str(rule["replacement"]))
            cursor = candidate["end"]
            rule_id = str(rule.get("rule_id", f"manual-{index:03d}"))
            occurrence_id = f"{chunk['id']}:o{index:04d}"
            occurrences.append(
                {
                    "occurrence_id": occurrence_id,
                    "rule_id": rule_id,
                    "scope_id": str(rule["scope_id"]),
                    "char_start": candidate["start"],
                    "char_end": candidate["end"],
                    "original": str(rule["original"]),
                    "replacement": str(rule["replacement"]),
                }
            )
            applied_counts[rule_id] += 1
        pieces.append(source[cursor:])
        applied = [
            {**rule, "applied_occurrences": applied_counts[str(rule.get("rule_id"))]}
            for rule in _applicable_rules(rules, str(chunk["id"]))
            if applied_counts[str(rule.get("rule_id"))]
        ]
        corrected.append(
            {
                **chunk,
                "source_text": source,
                "text": "".join(pieces),
                "applied_rules": applied,
                "occurrences": occurrences,
                "conflicts": conflicts,
            }
        )
    return corrected


def _correction_prompt(
    chunks: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    source_title: str,
    max_rules: int,
) -> str:
    blocks = [
        f'<scope id="{chunk["id"]}" '
        f'time="{timestamp(chunk["start"])}-{timestamp(chunk["end"])}">\n'
        f'{chunk["text"]}\n</scope>'
        for chunk in chunks
    ]
    candidate_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    return (
        f"Source title: {source_title}\n\n"
        "The program mined the exact technical-looking forms below from the whole "
        "ASR transcript. Candidate mining never decides a replacement. First identify "
        "repeated systematic ASR errors and return them as document_rules by "
        "candidate_id; one document rule is applied to every exact case-sensitive "
        "whole-form occurrence. Then return only highly certain one-off corrections "
        "as local_rules. Prioritize repeated acronym variants such as one-letter "
        "confusions. A replacement may change the word count by at most one, except "
        "when collapsing a spaced acronym such as R O S. Do not "
        "rewrite grammar, filler, ordinary wording, or technical meaning. Do not "
        "expand an abbreviation into an explanation. If uncertain, omit it. The total "
        f"number of rules must remain below the safety ceiling of {int(max_rules)}.\n\n"
        f"<candidates>{candidate_json}</candidates>\n\n"
        + "\n\n".join(blocks)
    )


def propose_correction_rules(
    chunks: list[dict[str, Any]],
    *,
    source_title: str,
    ollama_url: str,
    model: str,
    context_tokens: int,
    output_tokens: int,
    candidate_limit: int,
    max_rules: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, list[dict[str, Any]]]:
    require_loopback_ollama(ollama_url)
    installed = ollama_models(ollama_url)
    if not any(
        name == model or name.removesuffix(":latest") == model
        for name in installed
    ):
        raise SubtitleError(f"Local Ollama model is not installed: {model}")
    candidates = mine_correction_candidates(chunks, limit=int(candidate_limit))
    prompt = _correction_prompt(
        chunks, candidates, source_title=source_title, max_rules=int(max_rules)
    )
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
                    "You are a conservative local ASR terminology correction engine. "
                    "Output only the requested JSON. Prefer leaving an error unchanged "
                    "over making an uncertain correction."
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
    rules, validation = validate_correction_rules(
        decoded, chunks, candidates, max_rules=int(max_rules)
    )
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
        "candidate_count": len(candidates),
        "accepted_rule_count": len(rules),
        **validation,
    }
    return rules, metrics, raw, candidates


def _whisper_source_timeline(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **token,
            "source_text": str(token["text"]),
            "timing_source": str(token.get("timing_source", "segment_interpolation")),
        }
        for token in chunk.get("source_tokens", [])
    ]


def build_whisper_timeline(
    chunks: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use only Faster Whisper word or segment timestamps for subtitle timing."""
    words = [
        word
        for chunk in chunks
        for word in _whisper_source_timeline(chunk)
    ]
    timing_sources = Counter(str(word["timing_source"]) for word in words)
    return words, {
        "status": "complete",
        "source": "faster_whisper",
        "additional_timing_model_used": False,
        "word_count": len(words),
        "timing_source_counts": dict(sorted(timing_sources.items())),
    }


def project_corrections(
    timed_words: list[dict[str, Any]], corrected_chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Overlay selected display replacements onto an immutable source timeline."""
    timed_by_id = {str(word["source_id"]): word for word in timed_words}
    display: list[dict[str, Any]] = []
    sequence = 0
    for chunk in corrected_chunks:
        source_tokens = list(chunk.get("source_tokens", []))
        timeline: list[dict[str, Any]] = []
        for token in source_tokens:
            timed = timed_by_id.get(str(token["source_id"]))
            if timed is None:
                raise SubtitleError(
                    f"Timeline is missing source token {token['source_id']}"
                )
            timeline.append(timed)

        located: list[dict[str, Any]] = []
        for occurrence in chunk.get("occurrences", []):
            indexes = [
                index
                for index, token in enumerate(source_tokens)
                if int(token["char_start"]) < int(occurrence["char_end"])
                and int(token["char_end"]) > int(occurrence["char_start"])
            ]
            if not indexes:
                raise SubtitleError(
                    f"Correction {occurrence['occurrence_id']} has no source token span"
                )
            located.append(
                {
                    **occurrence,
                    "token_start": min(indexes),
                    "token_end": max(indexes),
                }
            )

        index = 0
        while index < len(timeline):
            starting = [item for item in located if item["token_start"] == index]
            if not starting:
                word = timeline[index]
                display.append(
                    {
                        **word,
                        "sequence": sequence,
                        "text": str(word["source_text"]),
                        "correction_ids": [],
                    }
                )
                sequence += 1
                index += 1
                continue

            group = list(starting)
            group_end = max(item["token_end"] for item in group)
            changed = True
            while changed:
                changed = False
                for item in located:
                    if item in group:
                        continue
                    if item["token_start"] <= group_end and item["token_end"] >= index:
                        group.append(item)
                        new_end = max(group_end, item["token_end"])
                        changed = changed or new_end != group_end
                        group_end = new_end

            char_start = int(source_tokens[index]["char_start"])
            char_end = int(source_tokens[group_end]["char_end"])
            text = str(chunk["source_text"])[char_start:char_end]
            for occurrence in sorted(
                group, key=lambda item: int(item["char_start"]), reverse=True
            ):
                local_start = int(occurrence["char_start"]) - char_start
                local_end = int(occurrence["char_end"]) - char_start
                text = (
                    text[:local_start]
                    + str(occurrence["replacement"])
                    + text[local_end:]
                )
            timing_sources = {
                str(timeline[item_index].get("timing_source", "unknown"))
                for item_index in range(index, group_end + 1)
            }
            display.append(
                {
                    "sequence": sequence,
                    "chunk_id": str(chunk["id"]),
                    "source_ids": [
                        str(timeline[item_index]["source_id"])
                        for item_index in range(index, group_end + 1)
                    ],
                    "start": min(
                        float(timeline[item_index]["start"])
                        for item_index in range(index, group_end + 1)
                    ),
                    "end": max(
                        float(timeline[item_index]["end"])
                        for item_index in range(index, group_end + 1)
                    ),
                    "text": text,
                    "source_text": str(chunk["source_text"])[char_start:char_end],
                    "timing_source": (
                        next(iter(timing_sources))
                        if len(timing_sources) == 1
                        else "mixed"
                    ),
                    "correction_ids": [
                        str(item["occurrence_id"])
                        for item in sorted(group, key=lambda item: item["char_start"])
                    ],
                }
            )
            sequence += 1
            index = group_end + 1
    return display


def audit_correction_delivery(
    corrected_chunks: list[dict[str, Any]],
    display_words: list[dict[str, Any]],
    cues: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    occurrences = {
        str(item["occurrence_id"]): item
        for chunk in corrected_chunks
        for item in chunk.get("occurrences", [])
    }
    projected = Counter(
        str(correction_id)
        for word in display_words
        for correction_id in word.get("correction_ids", [])
    )
    delivered = Counter(
        str(correction_id)
        for cue in cues
        for correction_id in cue.get("correction_ids", [])
    )
    expected_ids = set(occurrences)
    if (
        set(projected) != expected_ids
        or set(delivered) != expected_ids
        or any(projected[item] != 1 or delivered[item] != 1 for item in expected_ids)
    ):
        raise SubtitleError(
            "Selected subtitle corrections were not delivered exactly once to cues"
        )
    rule_audit = []
    for rule in rules:
        rule_id = str(rule["rule_id"])
        selected_ids = [
            occurrence_id
            for occurrence_id, occurrence in occurrences.items()
            if str(occurrence["rule_id"]) == rule_id
        ]
        rule_audit.append(
            {
                "rule_id": rule_id,
                "matched_occurrences": int(rule["matched_occurrences"]),
                "selected_occurrences": len(selected_ids),
                "projected_occurrences": sum(projected[item] for item in selected_ids),
                "delivered_occurrences": sum(delivered[item] for item in selected_ids),
            }
        )
    return {
        "selected_occurrence_count": len(expected_ids),
        "projected_occurrence_count": sum(projected.values()),
        "delivered_occurrence_count": sum(delivered.values()),
        "all_selected_corrections_delivered": True,
        "rules": rule_audit,
    }


def build_cues(words: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        start = float(word.get("start", 0.0))
        end = max(start, float(word.get("end", start)))
        if math.isfinite(start) and math.isfinite(end):
            cleaned.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "sequence": int(word.get("sequence", len(cleaned))),
                    "correction_ids": list(word.get("correction_ids", [])),
                }
            )
    cleaned.sort(key=lambda item: item["sequence"])

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
                "correction_ids": [
                    correction_id
                    for item in current
                    for correction_id in item.get("correction_ids", [])
                ],
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
            if (
                gap >= 0.65
                or duration > 6.0
                or len(projected) > 78
                or len(current) >= 16
            ):
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
                    previous["correction_ids"].extend(
                        current.get("correction_ids", [])
                    )
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
    edited_video: Path,
    segments: list[dict[str, Any]],
    source_title: str,
    output_dir: Path,
    summary_config: dict[str, Any],
    subtitle_config: dict[str, Any],
    video_config: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    chunks = build_correction_chunks(
        segments, max_seconds=int(subtitle_config["correction_scope_seconds"])
    )
    if not chunks:
        raise SubtitleError("Transcript has no text to subtitle")

    errors: list[str] = []
    raw_response = ""
    candidates: list[dict[str, Any]] = []
    try:
        if progress is not None:
            progress("Generating document-wide and local subtitle correction rules")
        rules, correction, raw_response, candidates = propose_correction_rules(
            chunks,
            source_title=source_title,
            ollama_url=str(summary_config["ollama_url"]),
            model=str(summary_config["model"]),
            context_tokens=int(subtitle_config["correction_context_tokens"]),
            output_tokens=int(subtitle_config["correction_output_tokens"]),
            candidate_limit=int(subtitle_config["correction_candidate_limit"]),
            max_rules=int(subtitle_config["correction_rule_safety_cap"]),
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
                    "source_text": chunk["source_text"],
                    "text": chunk["text"],
                    "applied_rules": chunk["applied_rules"],
                    "occurrences": chunk["occurrences"],
                    "conflicts": chunk["conflicts"],
                }
                for chunk in corrected
            ],
        },
    )

    if progress is not None:
        progress("Projecting corrected text onto Whisper timestamps")
    words, timing = build_whisper_timeline(chunks)
    display_words = project_corrections(words, corrected)
    cues = build_cues(display_words)
    if not cues:
        raise SubtitleError("No usable subtitle cues were produced")
    delivery_audit = audit_correction_delivery(
        corrected, display_words, cues, rules
    )
    atomic_write_text(output_dir / "subtitle.srt", render_subtitle_srt(cues))
    atomic_write_text(output_dir / "subtitle.ass", render_ass(cues))
    audit_by_rule = {
        str(item["rule_id"]): item for item in delivery_audit["rules"]
    }
    atomic_write_json(
        output_dir / "subtitle.rules.json",
        {
            "canonical_transcript_modified": False,
            "summary_input_modified": False,
            "timing_source": "faster_whisper",
            "additional_timing_model_used": False,
            "correction_projection_changed_source_timestamps": False,
            "correction": correction,
            "candidates": candidates,
            "rules": [
                {**rule, "delivery": audit_by_rule.get(str(rule["rule_id"]), {})}
                for rule in rules
            ],
            "delivery_audit": delivery_audit,
        },
    )
    if errors:
        atomic_write_text(output_dir / "subtitle.error.log", "\n".join(errors) + "\n")
    else:
        (output_dir / "subtitle.error.log").unlink(missing_ok=True)

    if progress is not None:
        progress("Burning subtitles into a separate video (audio stream copied)")
    output_video = output_dir / "subtitled.mp4"
    burn_ass_subtitles(
        edited_video,
        output_dir / "subtitle.ass",
        output_video,
        output_dir / "subtitle.render.log",
        codec=str(video_config["codec"]),
        preset=str(video_config["preset"]),
        crf=int(video_config["crf"]),
    )
    return {
        "output": output_video.name,
        "cue_count": len(cues),
        "correction": correction,
        "accepted_rule_count": len(rules),
        "selected_correction_count": delivery_audit[
            "selected_occurrence_count"
        ],
        "all_selected_corrections_delivered": delivery_audit[
            "all_selected_corrections_delivered"
        ],
        "timing": timing,
        "fallback_used": bool(errors),
        "errors": errors,
    }
