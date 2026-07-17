from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any


class ASRError(RuntimeError):
    pass


def transcribe_faster_whisper(
    audio_path: Path,
    *,
    model_name: str,
    model_cache: Path,
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    beam_size: int = 5,
) -> dict[str, Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRError(
            "faster-whisper is not installed; run this command through the Jetson container"
        ) from exc

    model_cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = None
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(model_cache),
            cpu_threads=8,
            num_workers=1,
        )
        generated, info = model.transcribe(
            str(audio_path),
            language=language or None,
            task="transcribe",
            beam_size=int(beam_size),
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=True,
        )
        segments: list[dict[str, Any]] = []
        for segment in generated:
            text = str(segment.text).strip()
            if not text:
                continue
            words = []
            for word in segment.words or []:
                words.append(
                    {
                        "start": None if word.start is None else float(word.start),
                        "end": None if word.end is None else float(word.end),
                        "word": str(word.word),
                        "probability": float(word.probability),
                    }
                )
            segments.append(
                {
                    "id": len(segments),
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "words": words,
                }
            )
        return {
            "backend": "faster-whisper",
            "model": model_name,
            "device": device,
            "compute_type": compute_type,
            "language": str(info.language),
            "language_probability": float(info.language_probability),
            "duration": float(info.duration),
            "duration_after_vad": float(info.duration_after_vad),
            "elapsed_seconds": time.monotonic() - started,
            "segments": segments,
        }
    except Exception as exc:
        raise ASRError(f"ASR failed using {model_name} on {device}: {exc}") from exc
    finally:
        if model is not None:
            del model
        gc.collect()

