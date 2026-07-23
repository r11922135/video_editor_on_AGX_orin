from __future__ import annotations

import ipaddress
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_CONFIG: dict[str, Any] = {
    "silence": {"noise_db": -35.0, "min_silence": 1.2, "target_silence": 0.35},
    "video": {
        "codec": "libx264",
        "preset": "veryfast",
        "crf": 20,
        "audio_bitrate": "160k",
    },
    "asr": {
        "backend": "faster-whisper",
        "model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "device": "cuda",
        "compute_type": "float16",
        "language": "en",
        "beam_size": 5,
    },
    "summary": {
        "ollama_url": "http://127.0.0.1:11435",
        "model": "qwen3.6:27b",
        "context_tokens": 65536,
        "max_output_tokens": 16384,
    },
    "subtitles": {
        "correction_context_tokens": 65536,
        "correction_output_tokens": 2048,
        "correction_candidate_limit": 48,
        "correction_rule_safety_cap": 32,
        "correction_scope_seconds": 120,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("Config root must be a JSON object")
        supplied_subtitles = supplied.get("subtitles")
        if isinstance(supplied_subtitles, dict):
            if (
                "correction_scope_seconds" not in supplied_subtitles
                and "alignment_chunk_seconds" in supplied_subtitles
            ):
                supplied_subtitles["correction_scope_seconds"] = supplied_subtitles[
                    "alignment_chunk_seconds"
                ]
            supplied_subtitles.pop("alignment_chunk_seconds", None)
            supplied_subtitles.pop("aligner_model", None)
        _merge(config, supplied)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    silence = config["silence"]
    minimum = float(silence["min_silence"])
    target = float(silence["target_silence"])
    if minimum <= 0:
        raise ValueError("silence.min_silence must be greater than zero")
    if target < 0 or target >= minimum:
        raise ValueError("silence.target_silence must be >= 0 and < min_silence")
    crf = int(config["video"]["crf"])
    if not 0 <= crf <= 51:
        raise ValueError("video.crf must be between 0 and 51")
    summary = config["summary"]
    parsed_ollama_url = urlparse(str(summary.get("ollama_url", "")))
    hostname = parsed_ollama_url.hostname
    loopback = hostname == "localhost"
    if hostname:
        try:
            loopback = loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if parsed_ollama_url.scheme not in {"http", "https"} or not loopback:
        raise ValueError(
            "summary.ollama_url must use localhost or a loopback IP for local-only "
            "generation"
        )
    if int(summary["context_tokens"]) < 8192:
        raise ValueError("summary.context_tokens must be at least 8192")
    if int(summary["context_tokens"]) > 65536:
        raise ValueError(
            "summary.context_tokens must not exceed the verified Orin limit of 65536"
        )
    if int(summary["max_output_tokens"]) < 8192:
        raise ValueError(
            "summary.max_output_tokens must be at least 8192 for the detailed overview"
        )
    if int(summary["max_output_tokens"]) >= int(summary["context_tokens"]):
        raise ValueError(
            "summary.max_output_tokens must be smaller than summary.context_tokens"
        )
    subtitles = config["subtitles"]
    correction_context = int(subtitles["correction_context_tokens"])
    correction_output = int(subtitles["correction_output_tokens"])
    if not 8192 <= correction_context <= 65536:
        raise ValueError(
            "subtitles.correction_context_tokens must be between 8192 and 65536"
        )
    if not 128 <= correction_output < correction_context:
        raise ValueError(
            "subtitles.correction_output_tokens must be >= 128 and smaller than context"
        )
    if not 8 <= int(subtitles["correction_candidate_limit"]) <= 96:
        raise ValueError(
            "subtitles.correction_candidate_limit must be between 8 and 96"
        )
    if not 1 <= int(subtitles["correction_rule_safety_cap"]) <= 64:
        raise ValueError(
            "subtitles.correction_rule_safety_cap must be between 1 and 64"
        )
    if not 30 <= int(subtitles["correction_scope_seconds"]) <= 150:
        raise ValueError(
            "subtitles.correction_scope_seconds must be between 30 and 150"
        )
