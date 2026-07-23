from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

from .asr import transcribe_faster_whisper
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    read_json,
    safe_stem,
    source_fingerprint,
)
from .media import (
    detect_silences,
    extract_analysis_audio,
    media_duration,
    probe_media,
    render_video,
)
from .silence import SilenceConfig, build_edit_plan
from .summary import summarize_two_stage, write_summary_files
from .subtitles import create_subtitled_video
from .transcript import render_transcript_markdown, write_transcript_files


StatusCallback = Callable[[str, str], None]
FULL_PIPELINE_REVISION = "cli-two-stage-subtitles-v7-whisper-timing"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineError(RuntimeError):
    pass


def _config_fingerprint(
    config: dict[str, Any],
    operation: str = "full",
    *,
    subtitles: bool = False,
) -> str:
    if operation in {"plan", "edit_only"}:
        relevant = {"silence": config["silence"], "video": config["video"]}
    else:
        relevant = dict(config)
        if not subtitles:
            relevant.pop("subtitles", None)
    identity: dict[str, Any] = {
        "operation": operation,
        "config": relevant,
        "subtitles_requested": bool(subtitles),
    }
    if operation == "full":
        identity["pipeline_revision"] = FULL_PIPELINE_REVISION
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _pipeline_lock(output_root: Path) -> Iterator[None]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".pipeline.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        lock_path.chmod(0o600)
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(
                f"Another video/summary job is already using {output_root}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _serialized_pipeline(function):
    @wraps(function)
    def wrapper(self, source: Path, *, output_root: Path, **kwargs: Any):
        resolved_output = output_root.expanduser().resolve()
        with _pipeline_lock(resolved_output):
            return function(
                self, source, output_root=resolved_output, **kwargs
            )

    return wrapper


_GENERATED_JOB_FILES = (
    "analysis.wav",
    "audio_extract.log",
    "edit_plan.json",
    "edited.mp4",
    "edited_probe.json",
    "failure.traceback.log",
    "ffmpeg_filter.txt",
    "manifest.json",
    "probe.json",
    "render.log",
    "silencedetect.log",
    "summary.en.md",
    "summary.json",
    "summary.metrics.json",
    "summary.raw.txt",
    "summary.en.raw.txt",
    "summary.zh-TW.raw.txt",
    "summary.zh-TW.md",
    "subtitle.ass",
    "subtitle.corrected.json",
    "subtitle.correction.raw.txt",
    "subtitle.error.log",
    "subtitle.probe.json",
    "subtitle.render.log",
    "subtitle.rules.json",
    "subtitle.srt",
    "subtitled.mp4",
    "transcript.json",
    "transcript.md",
    "transcript.srt",
)


def _clear_generated_job_files(job_dir: Path) -> None:
    for name in _GENERATED_JOB_FILES:
        (job_dir / name).unlink(missing_ok=True)
    shutil.rmtree(job_dir / ".summary.pending", ignore_errors=True)


def _default_status(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def _mark_manifest_failed(manifest: dict[str, Any], exc: BaseException) -> None:
    now = _utc_now()
    error = {"type": type(exc).__name__, "message": str(exc)}
    stages = manifest.setdefault("stages", {})
    for stage in stages.values():
        if isinstance(stage, dict) and stage.get("state") == "running":
            stage.update(
                {
                    "state": "failed",
                    "updated_at": now,
                    "message": str(exc),
                    "error": error,
                }
            )
    manifest["status"] = "failed"
    manifest["updated_at"] = now
    manifest["failed_at"] = now
    manifest["error"] = error


class VideoPipeline:
    def __init__(
        self,
        config: dict[str, Any],
        *,
        model_cache: Path,
        status: StatusCallback | None = None,
    ) -> None:
        self.config = config
        self.model_cache = model_cache
        self.status = status or _default_status

    def _stage(
        self,
        manifest: dict[str, Any],
        manifest_path: Path,
        stage: str,
        state: str,
        message: str,
        **details: Any,
    ) -> None:
        manifest.setdefault("stages", {})[stage] = {
            "state": state,
            "updated_at": _utc_now(),
            "message": message,
            **details,
        }
        manifest["updated_at"] = _utc_now()
        atomic_write_json(manifest_path, manifest)
        self.status(stage, message)

    @_serialized_pipeline
    def run(
        self,
        source: Path,
        *,
        output_root: Path,
        plan_only: bool = False,
        edit_only: bool = False,
        subtitles: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        if plan_only and edit_only:
            raise PipelineError("plan_only and edit_only cannot be used together")
        if edit_only and subtitles:
            raise PipelineError("--edit-only and --subtitles cannot be used together")
        source = source.expanduser().resolve(strict=True)
        if not source.is_file():
            raise PipelineError(f"Input is not a file: {source}")
        fingerprint = source_fingerprint(source)
        operation = "plan" if plan_only else "edit_only" if edit_only else "full"
        config_fingerprint = _config_fingerprint(
            self.config, operation, subtitles=subtitles
        )
        job_id = (
            f"{safe_stem(source)}-{fingerprint[:10]}-{config_fingerprint[:8]}"
        )
        job_dir = output_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_dir.chmod(0o700)
        manifest_path = job_dir / "manifest.json"
        failure_log = job_dir / "failure.traceback.log"

        if manifest_path.exists() and not force:
            old = read_json(manifest_path)
            if old.get("source", {}).get("fingerprint") != fingerprint:
                raise PipelineError("Existing job fingerprint does not match the source")
            old_status = str(old.get("status", "unknown"))
            terminal_status = "planned" if plan_only else "complete"
            if old_status == terminal_status:
                if subtitles and (
                    old.get("stages", {}).get("subtitles", {}).get("state")
                    != "complete"
                    or not (job_dir / "subtitled.mp4").is_file()
                ):
                    raise PipelineError(
                        "Existing job completed without a valid subtitled.mp4; "
                        f"use --force to retry it: {job_dir}"
                    )
                self.status("cached", f"Using existing job: {job_dir}")
                return {"job_id": job_id, "job_dir": str(job_dir), "manifest": old}
            raise PipelineError(
                f"Existing job is {old_status}; use --force to replace it: {job_dir}"
            )
        if force:
            _clear_generated_job_files(job_dir)
        failure_log.unlink(missing_ok=True)
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "job_id": job_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "source": {
                "path": str(source),
                "name": source.name,
                "size": source.stat().st_size,
                "mtime_ns": source.stat().st_mtime_ns,
                "fingerprint": fingerprint,
                "config_fingerprint": config_fingerprint,
            },
            "operation": operation,
            "subtitles_requested": bool(subtitles),
            "config": self.config,
            "stages": {},
            "status": "running",
        }
        atomic_write_json(manifest_path, manifest)

        try:
            probe_path = job_dir / "probe.json"
            self._stage(manifest, manifest_path, "probe", "running", "Inspecting media")
            probe = probe_media(source)
            atomic_write_json(probe_path, probe)
            duration = media_duration(probe)
            self._stage(
                manifest,
                manifest_path,
                "probe",
                "complete",
                f"Media duration: {duration:.2f}s",
                duration=duration,
            )

            if not plan_only:
                free_bytes = shutil.disk_usage(job_dir).free
                analysis_wav_bytes = 0 if edit_only else int(duration * 32_000)
                output_multiplier = 3 if subtitles else 2
                estimated_required = max(
                    5 * 1024**3,
                    source.stat().st_size * output_multiplier + analysis_wav_bytes,
                )
                if free_bytes < estimated_required:
                    raise PipelineError(
                        f"Insufficient free disk: need an estimated "
                        f"{estimated_required / 1024**3:.1f} GiB but only "
                        f"{free_bytes / 1024**3:.1f} GiB is available"
                    )
                self._stage(
                    manifest,
                    manifest_path,
                    "storage",
                    "complete",
                    f"Disk preflight passed ({free_bytes / 1024**3:.1f} GiB free)",
                    free_bytes=free_bytes,
                    estimated_required_bytes=estimated_required,
                )

            silence_cfg_raw = self.config["silence"]
            silence_cfg = SilenceConfig(
                min_silence=float(silence_cfg_raw["min_silence"]),
                target_silence=float(silence_cfg_raw["target_silence"]),
            )
            self._stage(
                manifest,
                manifest_path,
                "silence",
                "running",
                "Detecting and planning long-silence compression",
            )
            detected = detect_silences(
                source,
                duration=duration,
                noise_db=float(silence_cfg_raw["noise_db"]),
                min_silence=silence_cfg.min_silence,
                log_path=job_dir / "silencedetect.log",
            )
            plan = build_edit_plan(duration, detected, silence_cfg)
            plan_payload = plan.as_dict()
            atomic_write_json(job_dir / "edit_plan.json", plan_payload)
            self._stage(
                manifest,
                manifest_path,
                "silence",
                "complete",
                f"Will remove {plan.removed_duration:.2f}s across "
                f"{len(plan.remove_intervals)} long pauses",
                detected_count=len(detected),
                cut_count=len(plan.remove_intervals),
                removed_duration=plan.removed_duration,
                output_duration=plan.output_duration,
            )

            if plan_only:
                manifest["status"] = "planned"
                manifest["completed_at"] = _utc_now()
                atomic_write_json(manifest_path, manifest)
                return {"job_id": job_id, "job_dir": str(job_dir), "manifest": manifest}

            edited_video = job_dir / "edited.mp4"
            self._stage(
                manifest,
                manifest_path,
                "render",
                "running",
                "Rendering edited video (the source file is never modified)",
            )
            render_video(
                source,
                edited_video,
                plan.keep_intervals,
                source_duration=duration,
                filter_script_path=job_dir / "ffmpeg_filter.txt",
                log_path=job_dir / "render.log",
                codec=str(self.config["video"]["codec"]),
                preset=str(self.config["video"]["preset"]),
                crf=int(self.config["video"]["crf"]),
                audio_bitrate=str(self.config["video"]["audio_bitrate"]),
            )
            edited_probe = probe_media(edited_video)
            atomic_write_json(job_dir / "edited_probe.json", edited_probe)
            self._stage(
                manifest,
                manifest_path,
                "render",
                "complete",
                f"Edited video ready: {edited_video.name}",
                output=str(edited_video),
                duration=media_duration(edited_probe),
                size=edited_video.stat().st_size,
            )

            if edit_only:
                manifest["status"] = "complete"
                manifest["completed_at"] = _utc_now()
                manifest["outputs"] = {"video": edited_video.name}
                atomic_write_json(manifest_path, manifest)
                failure_log.unlink(missing_ok=True)
                self.status("complete", f"Edit-only job completed: {job_dir}")
                return {"job_id": job_id, "job_dir": str(job_dir), "manifest": manifest}

            analysis_audio = job_dir / "analysis.wav"
            self._stage(
                manifest,
                manifest_path,
                "audio",
                "running",
                "Extracting 16kHz mono audio for local ASR",
            )
            extract_analysis_audio(
                edited_video, analysis_audio, job_dir / "audio_extract.log"
            )
            self._stage(
                manifest,
                manifest_path,
                "audio",
                "complete",
                "Analysis audio ready",
                size=analysis_audio.stat().st_size,
            )

            asr_cfg = self.config["asr"]
            if str(asr_cfg.get("backend")) != "faster-whisper":
                raise PipelineError(
                    f"Unsupported ASR backend: {asr_cfg.get('backend')}"
                )
            self._stage(
                manifest,
                manifest_path,
                "asr",
                "running",
                f"Transcribing locally with {asr_cfg['model']}",
            )
            transcription = transcribe_faster_whisper(
                analysis_audio,
                model_name=str(asr_cfg["model"]),
                model_cache=self.model_cache,
                device=str(asr_cfg["device"]),
                compute_type=str(asr_cfg["compute_type"]),
                language=str(asr_cfg["language"]),
                beam_size=int(asr_cfg["beam_size"]),
            )
            atomic_write_json(job_dir / "transcript.json", transcription)
            segments = transcription["segments"]
            write_transcript_files(segments, title=source.stem, output_dir=job_dir)
            self._stage(
                manifest,
                manifest_path,
                "asr",
                "complete",
                f"Transcribed {len(segments)} timestamped segments",
                segment_count=len(segments),
                language=transcription.get("language"),
                elapsed_seconds=transcription.get("elapsed_seconds"),
            )

            if not segments:
                raise PipelineError(
                    "ASR detected no speech; refusing to generate a summary from only the filename"
                )

            summary_cfg = self.config["summary"]
            self._stage(
                manifest,
                manifest_path,
                "summary",
                "running",
                f"Two-stage English overview and zh-TW translation with "
                f"{summary_cfg['model']}",
            )
            summary, metrics = summarize_two_stage(
                segments,
                source_title=source.stem,
                ollama_url=str(summary_cfg["ollama_url"]),
                model=str(summary_cfg["model"]),
                context_tokens=int(summary_cfg["context_tokens"]),
                max_output_tokens=int(summary_cfg["max_output_tokens"]),
                english_raw_response_path=job_dir / "summary.en.raw.txt",
                translation_raw_response_path=job_dir / "summary.zh-TW.raw.txt",
                progress=lambda message: self._stage(
                    manifest,
                    manifest_path,
                    "summary",
                    "running",
                    message,
                ),
            )
            write_summary_files(
                summary,
                metrics,
                source_name=source.name,
                output_dir=job_dir,
            )
            self._stage(
                manifest,
                manifest_path,
                "summary",
                "complete",
                f"Two-stage bilingual Markdown overview ready ({metrics['model']})",
                **metrics,
            )

            subtitle_outputs: dict[str, str] = {}
            if subtitles:
                self._stage(
                    manifest,
                    manifest_path,
                    "subtitles",
                    "running",
                    "Preparing best-effort corrected and burned subtitles",
                    optional=True,
                )
                try:
                    subtitle_metrics = create_subtitled_video(
                        edited_video=edited_video,
                        segments=deepcopy(segments),
                        source_title=source.stem,
                        output_dir=job_dir,
                        summary_config=summary_cfg,
                        subtitle_config=self.config["subtitles"],
                        video_config=self.config["video"],
                        progress=lambda message: self._stage(
                            manifest,
                            manifest_path,
                            "subtitles",
                            "running",
                            message,
                            optional=True,
                        ),
                    )
                    subtitled_video = job_dir / "subtitled.mp4"
                    subtitle_probe = probe_media(subtitled_video)
                    subtitle_duration = media_duration(subtitle_probe)
                    edited_duration = media_duration(edited_probe)
                    if abs(subtitle_duration - edited_duration) > 1.0:
                        raise PipelineError(
                            "Subtitled video duration differs from edited.mp4 by more "
                            "than one second"
                        )
                    atomic_write_json(job_dir / "subtitle.probe.json", subtitle_probe)
                    subtitle_outputs = {
                        "subtitled_video": "subtitled.mp4",
                        "subtitle_srt": "subtitle.srt",
                        "subtitle_ass": "subtitle.ass",
                        "subtitle_rules": "subtitle.rules.json",
                        "subtitle_corrected": "subtitle.corrected.json",
                        "subtitle_correction_raw": "subtitle.correction.raw.txt",
                    }
                    if (job_dir / "subtitle.error.log").is_file():
                        subtitle_outputs["subtitle_error"] = "subtitle.error.log"
                    fallback_note = (
                        " (subtitle correction fallback used)"
                        if subtitle_metrics["fallback_used"]
                        else ""
                    )
                    self._stage(
                        manifest,
                        manifest_path,
                        "subtitles",
                        "complete",
                        f"Burned {subtitle_metrics['cue_count']} subtitle cues into "
                        f"subtitled.mp4{fallback_note}",
                        optional=True,
                        duration=subtitle_duration,
                        size=subtitled_video.stat().st_size,
                        **subtitle_metrics,
                    )
                except Exception as exc:
                    (job_dir / "subtitled.mp4").unlink(missing_ok=True)
                    (job_dir / "subtitle.probe.json").unlink(missing_ok=True)
                    error_path = job_dir / "subtitle.error.log"
                    previous = (
                        error_path.read_text(encoding="utf-8")
                        if error_path.is_file()
                        else ""
                    )
                    atomic_write_text(
                        error_path,
                        previous + traceback.format_exc(),
                    )
                    warning = {
                        "stage": "subtitles",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    manifest.setdefault("warnings", []).append(warning)
                    self._stage(
                        manifest,
                        manifest_path,
                        "subtitles",
                        "failed",
                        f"Optional subtitles failed; edit, transcript, and summary "
                        f"remain valid: {exc}",
                        optional=True,
                        error=warning,
                    )

            analysis_audio.unlink(missing_ok=True)
            manifest["status"] = "complete"
            manifest["completed_at"] = _utc_now()
            manifest["outputs"] = {
                "video": edited_video.name,
                "transcript_json": "transcript.json",
                "transcript_srt": "transcript.srt",
                "transcript_md": "transcript.md",
                "summary_en": "summary.en.md",
                "summary_zh_tw": "summary.zh-TW.md",
                "summary_json": "summary.json",
                "summary_en_raw": "summary.en.raw.txt",
                "summary_zh_tw_raw": "summary.zh-TW.raw.txt",
                "summary_metrics": "summary.metrics.json",
                **subtitle_outputs,
            }
            atomic_write_json(manifest_path, manifest)
            failure_log.unlink(missing_ok=True)
            self.status("complete", f"Job completed: {job_dir}")
            return {"job_id": job_id, "job_dir": str(job_dir), "manifest": manifest}
        except BaseException as exc:
            _mark_manifest_failed(manifest, exc)
            atomic_write_json(manifest_path, manifest)
            atomic_write_text(failure_log, traceback.format_exc())
            self.status("failed", str(exc))
            raise


def _resummarize_job_unlocked(
    job_dir: Path,
    config: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve(strict=True)
    transcript_path = job_dir / "transcript.json"
    manifest_path = job_dir / "manifest.json"
    if not transcript_path.is_file() or not manifest_path.is_file():
        raise PipelineError("Job must contain transcript.json and manifest.json")
    transcript = read_json(transcript_path)
    manifest = read_json(manifest_path)
    segments = transcript.get("segments", [])
    if not segments:
        raise PipelineError("Transcript has no speech segments to summarize")
    pending_dir = job_dir / ".summary.pending"
    shutil.rmtree(pending_dir, ignore_errors=True)
    pending_dir.mkdir(mode=0o700)
    english_raw_response_path = pending_dir / "summary.en.raw.txt"
    translation_raw_response_path = pending_dir / "summary.zh-TW.raw.txt"
    summary_cfg = dict(config["summary"])
    if model:
        summary_cfg["model"] = model
    failure_log = job_dir / "failure.traceback.log"
    now = _utc_now()
    manifest["status"] = "running"
    manifest["updated_at"] = now
    summary_message = (
        f"Two-stage English overview and zh-TW translation with "
        f"{summary_cfg['model']}"
    )
    manifest.setdefault("stages", {})["summary"] = {
        "state": "running",
        "updated_at": now,
        "message": summary_message,
    }
    atomic_write_json(manifest_path, manifest)
    try:
        source_title = Path(manifest["source"]["name"]).stem

        def progress(message: str) -> None:
            now = _utc_now()
            manifest.setdefault("stages", {})["summary"] = {
                "state": "running",
                "updated_at": now,
                "message": message,
            }
            manifest["updated_at"] = now
            atomic_write_json(manifest_path, manifest)
            print(f"[summary] {message}", flush=True)

        summary, metrics = summarize_two_stage(
            segments,
            source_title=source_title,
            ollama_url=str(summary_cfg["ollama_url"]),
            model=str(summary_cfg["model"]),
            context_tokens=int(summary_cfg["context_tokens"]),
            max_output_tokens=int(summary_cfg["max_output_tokens"]),
            english_raw_response_path=english_raw_response_path,
            translation_raw_response_path=translation_raw_response_path,
            progress=progress,
        )
        write_summary_files(
            summary,
            metrics,
            source_name=str(manifest["source"]["name"]),
            output_dir=pending_dir,
        )
        for name in (
            "summary.en.raw.txt",
            "summary.zh-TW.raw.txt",
            "summary.json",
            "summary.metrics.json",
            "summary.en.md",
            "summary.zh-TW.md",
        ):
            (pending_dir / name).replace(job_dir / name)
        pending_dir.rmdir()
        (job_dir / "summary.raw.txt").unlink(missing_ok=True)
        now = _utc_now()
        manifest.setdefault("stages", {})["summary"] = {
            "state": "complete",
            "updated_at": now,
            "message": (
                f"Two-stage bilingual Markdown overview ready ({metrics['model']})"
            ),
            **metrics,
        }
        outputs = manifest.setdefault("outputs", {})
        for key, path in {
            "video": job_dir / "edited.mp4",
            "transcript_json": job_dir / "transcript.json",
            "transcript_srt": job_dir / "transcript.srt",
            "transcript_md": job_dir / "transcript.md",
        }.items():
            if path.is_file():
                outputs[key] = path.name
        outputs.update(
            {
                "summary_en": "summary.en.md",
                "summary_zh_tw": "summary.zh-TW.md",
                "summary_json": "summary.json",
                "summary_en_raw": "summary.en.raw.txt",
                "summary_zh_tw_raw": "summary.zh-TW.raw.txt",
                "summary_metrics": "summary.metrics.json",
            }
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = now
        manifest.pop("failed_at", None)
        manifest.pop("error", None)
        manifest["updated_at"] = now
        atomic_write_json(manifest_path, manifest)
        failure_log.unlink(missing_ok=True)
        return metrics
    except BaseException as exc:
        shutil.rmtree(pending_dir, ignore_errors=True)
        _mark_manifest_failed(manifest, exc)
        atomic_write_json(manifest_path, manifest)
        atomic_write_text(failure_log, traceback.format_exc())
        raise


def resummarize_job(
    job_dir: Path,
    config: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    resolved = job_dir.expanduser().resolve(strict=True)
    with _pipeline_lock(resolved.parent):
        return _resummarize_job_unlocked(
            resolved,
            config,
            model=model,
        )


def _rerender_transcript_job_unlocked(job_dir: Path) -> dict[str, Any]:
    job_dir = job_dir.expanduser().resolve(strict=True)
    transcript_path = job_dir / "transcript.json"
    manifest_path = job_dir / "manifest.json"
    if not transcript_path.is_file() or not manifest_path.is_file():
        raise PipelineError("Job must contain transcript.json and manifest.json")

    transcript = read_json(transcript_path)
    manifest = read_json(manifest_path)
    segments = transcript.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PipelineError("Transcript has no speech segments to format")

    source_name = str(manifest.get("source", {}).get("name", "Transcript"))
    output_path = job_dir / "transcript.md"
    atomic_write_text(
        output_path,
        render_transcript_markdown(segments, title=Path(source_name).stem),
    )

    now = _utc_now()
    manifest.setdefault("outputs", {})["transcript_md"] = output_path.name
    manifest.setdefault("stages", {})["transcript_format"] = {
        "state": "complete",
        "updated_at": now,
        "message": "Readable Markdown transcript ready",
        "segment_count": len(segments),
    }
    manifest["updated_at"] = now
    atomic_write_json(manifest_path, manifest)
    return {
        "job_dir": str(job_dir),
        "transcript_md": str(output_path),
        "segment_count": len(segments),
    }


def rerender_transcript_job(job_dir: Path) -> dict[str, Any]:
    resolved = job_dir.expanduser().resolve(strict=True)
    with _pipeline_lock(resolved.parent):
        return _rerender_transcript_job_unlocked(resolved)
