from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text
from .transcript import transcript_windows_for_prompt


class SummaryError(RuntimeError):
    pass


TARGET_OVERVIEW_MIN = 10
TARGET_OVERVIEW_MAX = 14
MIN_OVERVIEW_PARAGRAPHS = 8
MIN_ENGLISH_WORDS = 1000


def summary_schema() -> dict[str, Any]:
    paragraphs = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": MIN_OVERVIEW_PARAGRAPHS,
        "maxItems": 16,
    }
    return {
        "type": "object",
        "properties": {
            "title_en": {"type": "string"},
            "title_zh_tw": {"type": "string"},
            "overview_en": paragraphs,
            "overview_zh_tw": dict(paragraphs),
            "completion_marker": {"type": "string", "enum": ["complete"]},
        },
        "required": [
            "title_en",
            "title_zh_tw",
            "overview_en",
            "overview_zh_tw",
            "completion_marker",
        ],
        "additionalProperties": False,
    }


SUMMARY_SCHEMA = summary_schema()


SYSTEM_PROMPT = """Create faithful, detailed notes for internal technical training videos.
Use only facts stated in the transcript. Never invent APIs, decisions, examples, or action items.
Keep product names, API names, code identifiers, numbers, and caveats exact.
Write clear professional English and natural Traditional Chinese used in Taiwan.
The Chinese overview must faithfully cover the same content as the English overview.
Return only the requested JSON object."""


def _user_prompt(title: str, transcript: str, *, window_count: int) -> str:
    return f"""Create one detailed bilingual overview of the complete transcript below.

Requirements:
- Read all {window_count} transcript windows before writing. Cover the beginning, middle, and end.
- Write {TARGET_OVERVIEW_MIN}-{TARGET_OVERVIEW_MAX} cohesive thematic paragraphs per language.
- Across overview_en, target 1,200-1,600 words of information-dense content.
- Each overview_zh_tw paragraph must cover the same material as the English paragraph at the same index.
- Explain supported technical reasoning, dependencies, examples, trade-offs, commands, numbers, and caveats.
- Preserve English product names, APIs, identifiers, and commands where translation would be misleading.
- If audio is ambiguous, state only what is supported; do not invent a resolution.
- Do not add Key Takeaways, Uncertainties, sections, action items, or any fields not shown below.
- Do not mention these instructions or the act of summarizing.

Use exactly this flat JSON shape:
{{"title_en":"...","title_zh_tw":"...","overview_en":["..."],
"overview_zh_tw":["..."],"completion_marker":"complete"}}

Source title: {title}

<transcript>
{transcript}
</transcript>

Return one valid JSON object. After closing overview_zh_tw, write the final scalar field
"completion_marker":"complete" and close the object. Do not use Markdown fences or add
text before or after the JSON object.
"""


def _post_json(url: str, payload: dict[str, Any], timeout: int = 7200) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SummaryError(f"LLM HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SummaryError(f"Cannot reach local LLM at {url}: {exc}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SummaryError("Local LLM server returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise SummaryError("Local LLM server response must be a JSON object")
    return parsed


def ollama_models(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SummaryError(f"Cannot query Ollama at {url}: {exc}") from exc
    return [str(item.get("name", "")) for item in payload.get("models", [])]


def require_loopback_ollama(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    hostname = parsed.hostname
    local = hostname == "localhost"
    if hostname:
        try:
            local = local or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if parsed.scheme not in {"http", "https"} or not local:
        raise SummaryError(
            "Local-only summary generation requires Ollama on localhost or a "
            f"loopback IP; received {base_url!r}"
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise SummaryError("Model returned malformed summary JSON") from exc
    if not isinstance(value, dict):
        raise SummaryError("Model summary must be a JSON object")
    return value


def _required_string(value: dict[str, Any], field: str) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise SummaryError(f"Model summary field {field} must be a non-empty string")
    return item


def _required_string_list(value: dict[str, Any], field: str) -> list[str]:
    items = value.get(field)
    if not isinstance(items, list) or any(
        not isinstance(item, str) or not item.strip() for item in items
    ):
        raise SummaryError(f"Model summary field {field} must be a list of text")
    return list(items)


def summary_from_raw(value: dict[str, Any]) -> dict[str, Any]:
    """Map the exact flat model contract without modifying any prose."""
    expected = set(SUMMARY_SCHEMA["required"])
    if set(value) != expected:
        raise SummaryError(
            "Model summary fields must be exactly: " + ", ".join(SUMMARY_SCHEMA["required"])
        )
    if value.get("completion_marker") != "complete":
        raise SummaryError("Model summary is missing the completion marker")
    return {
        "title": {
            "en": _required_string(value, "title_en"),
            "zh_tw": _required_string(value, "title_zh_tw"),
        },
        "overview": {
            "en": _required_string_list(value, "overview_en"),
            "zh_tw": _required_string_list(value, "overview_zh_tw"),
        },
    }


def _english_word_count(items: list[str]) -> int:
    return len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9_./+#'-]*\b", " ".join(items)))


def validate_summary_quality(value: dict[str, Any]) -> dict[str, int]:
    title = value.get("title", {})
    overview = value.get("overview", {})
    if not isinstance(title, dict) or not isinstance(overview, dict):
        raise SummaryError("Summary must contain bilingual title and overview objects")
    english = overview.get("en")
    chinese = overview.get("zh_tw")
    if not isinstance(english, list) or not isinstance(chinese, list):
        raise SummaryError("Summary overview must contain bilingual paragraph lists")
    if len(english) < MIN_OVERVIEW_PARAGRAPHS or len(english) != len(chinese):
        raise SummaryError(
            f"Detailed overview requires at least {MIN_OVERVIEW_PARAGRAPHS} aligned "
            "paragraphs per language"
        )
    if not re.search(r"[A-Za-z]", str(title.get("en", ""))):
        raise SummaryError("English title does not contain Latin text")
    if not re.search(r"[\u3400-\u9fff]", str(title.get("zh_tw", ""))):
        raise SummaryError("Traditional Chinese title does not contain Chinese text")
    if any(not re.search(r"[A-Za-z]", item) for item in english):
        raise SummaryError("Every English overview paragraph needs Latin text")
    if any(not re.search(r"[\u3400-\u9fff]", item) for item in chinese):
        raise SummaryError(
            "Every Traditional Chinese overview paragraph needs Chinese text"
        )
    word_count = _english_word_count(english)
    if word_count < MIN_ENGLISH_WORDS:
        raise SummaryError(
            f"Detailed overview requires at least {MIN_ENGLISH_WORDS:,} English "
            f"words; received {word_count:,}"
        )
    return {
        "overview_count": len(english),
        "english_visible_word_count": word_count,
        "minimum_english_words": MIN_ENGLISH_WORDS,
    }


def estimate_prompt_tokens(text: str) -> int:
    return max(1, int(len(text) / 2.8))


def _memory_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _select_installed_model(installed: list[str], model: str) -> str:
    if any(name == model or name.removesuffix(":latest") == model for name in installed):
        return model
    raise SummaryError(f"Ollama model '{model}' is not installed. Run: ollama pull {model}")


def summarize_oneshot(
    segments: list[dict[str, Any]],
    *,
    source_title: str,
    ollama_url: str,
    model: str,
    context_tokens: int,
    max_output_tokens: int,
    raw_response_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require_loopback_ollama(ollama_url)
    transcript, windows = transcript_windows_for_prompt(segments)
    if not windows:
        raise SummaryError("Transcript has no non-empty windows")
    prompt = _user_prompt(source_title, transcript, window_count=len(windows))
    estimated = estimate_prompt_tokens(SYSTEM_PROMPT + prompt)
    input_budget = int(context_tokens) - int(max_output_tokens) - 1024
    if estimated > input_budget:
        raise SummaryError(
            f"One-shot prompt is estimated at {estimated:,} tokens but the safe input "
            f"budget is {input_budget:,}; shorten the transcript or raise context"
        )
    required_context = estimated + int(max_output_tokens) + 1024
    effective_context = min(
        int(context_tokens), max(8192, ((required_context + 2047) // 2048) * 2048)
    )
    selected = _select_installed_model(ollama_models(ollama_url), model)

    available_memory = _memory_available_bytes()
    estimated_memory = None
    if "qwen" in selected.lower() and "27b" in selected.lower():
        gib = 1024**3
        estimated_memory = int((18.0 + 4.2 * effective_context / 8192 + 4.0) * gib)
        if available_memory is not None and available_memory < estimated_memory:
            raise SummaryError(
                f"Not enough unified memory for {selected}: estimated "
                f"{estimated_memory / gib:.1f} GiB, available "
                f"{available_memory / gib:.1f} GiB"
            )

    request_payload = {
        "model": selected,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "format": summary_schema(),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "num_ctx": effective_context,
            "num_predict": int(max_output_tokens),
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.0,
        },
    }
    started = time.monotonic()
    response = _post_json(f"{ollama_url.rstrip('/')}/api/chat", request_payload)
    raw_content = str(response.get("message", {}).get("content", ""))
    if raw_response_path is not None:
        atomic_write_text(raw_response_path, raw_content)
    if response.get("done_reason") == "length":
        raise SummaryError(
            "The model reached summary.max_output_tokens before completing JSON"
        )
    try:
        raw_summary = _extract_json_object(raw_content)
        summary = summary_from_raw(raw_summary)
        quality = validate_summary_quality(summary)
    except SummaryError as exc:
        raise type(exc)(
            f"{exc}; done_reason={response.get('done_reason', 'unknown')}, "
            f"response_chars={len(raw_content):,}. The draft was not published."
        ) from exc

    metrics = {
        "provider": "ollama",
        "endpoint_scope": "loopback",
        "requested_model": model,
        "model": str(response.get("model", selected)),
        "mode": "oneshot",
        "summary_format": "detailed_overview_v1",
        "thinking": False,
        "generation_reused": False,
        "post_generation_content_modified": False,
        "raw_response_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        "raw_response_chars": len(raw_content),
        "context_tokens_cap": int(context_tokens),
        "context_tokens": effective_context,
        "estimated_prompt_tokens": estimated,
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_duration_ns": response.get("eval_duration"),
        "total_duration_ns": response.get("total_duration"),
        "elapsed_seconds": time.monotonic() - started,
        "format_repair_used": False,
        "quality_gate": "structural_detailed_overview_v1",
        "summary_overview_count": quality["overview_count"],
        "summary_english_visible_word_count": quality["english_visible_word_count"],
        "summary_minimum_english_words": quality["minimum_english_words"],
        "transcript_window_count": len(windows),
        "compacted_transcript_chars": len(transcript),
        "source_segment_count": len(segments),
        "memory_available_before_bytes": available_memory,
        "estimated_memory_required_bytes": estimated_memory,
    }
    return summary, metrics


def render_summary_markdown(
    summary: dict[str, Any],
    *,
    language: str,
    source_name: str,
    model: str,
) -> str:
    if language not in {"en", "zh_tw"}:
        raise ValueError("language must be en or zh_tw")
    title = str(summary["title"][language])
    heading = "## Overview" if language == "en" else "## 詳細摘要"
    source_label = "Source" if language == "en" else "來源"
    model_label = "Local model" if language == "en" else "本機模型"
    lines = [
        f"# {title}",
        "",
        f"_{source_label}: {source_name} · {model_label}: {model}_",
        "",
        heading,
        "",
    ]
    for paragraph in summary["overview"][language]:
        lines.extend([str(paragraph), ""])
    return "\n".join(lines).rstrip() + "\n"


def write_summary_files(
    summary: dict[str, Any],
    metrics: dict[str, Any],
    *,
    source_name: str,
    output_dir: Path,
) -> None:
    model = str(metrics.get("model", "unknown"))
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(output_dir / "summary.metrics.json", metrics)
    atomic_write_text(
        output_dir / "summary.en.md",
        render_summary_markdown(
            summary, language="en", source_name=source_name, model=model
        ),
    )
    atomic_write_text(
        output_dir / "summary.zh-TW.md",
        render_summary_markdown(
            summary, language="zh_tw", source_name=source_name, model=model
        ),
    )
