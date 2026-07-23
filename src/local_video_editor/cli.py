from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config
from .pipeline import VideoPipeline, rerender_transcript_job, resummarize_job


def _common_process_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Source video (never overwritten)")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--model-cache", type=Path, default=Path("models/asr"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orin-video-editor",
        description=(
            "Local silence editing, GPU transcription, bilingual overview, and "
            "optional burned subtitles"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Detect silence and write an edit plan only")
    _common_process_args(plan)

    process = sub.add_parser("process", help="Run the complete local pipeline")
    _common_process_args(process)
    process_mode = process.add_mutually_exclusive_group()
    process_mode.add_argument(
        "--edit-only",
        action="store_true",
        help="Only create the date-prefixed edited video; skip ASR and summary",
    )
    process_mode.add_argument(
        "--subtitles",
        action="store_true",
        help="Also correct and burn English subtitles using Whisper timestamps",
    )

    summarize = sub.add_parser(
        "summarize", help="Generate a fresh overview from an existing transcript"
    )
    summarize.add_argument("job_dir", type=Path)
    summarize.add_argument("--config", type=Path)
    summarize.add_argument("--model")

    transcript = sub.add_parser(
        "transcript",
        help="Rebuild the readable date-prefixed Markdown transcript",
    )
    transcript.add_argument("job_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "transcript":
            result = rerender_transcript_job(args.job_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        config = load_config(getattr(args, "config", None))
        if args.command in {"plan", "process"}:
            pipeline = VideoPipeline(config, model_cache=args.model_cache)
            result = pipeline.run(
                args.input,
                output_root=args.output_root,
                plan_only=args.command == "plan",
                edit_only=getattr(args, "edit_only", False),
                subtitles=getattr(args, "subtitles", False),
                force=args.force,
            )
            print(
                json.dumps(
                    {"job_id": result["job_id"], "job_dir": result["job_dir"]},
                    indent=2,
                )
            )
            return 0

        if args.command == "summarize":
            metrics = resummarize_job(
                args.job_dir,
                config,
                model=args.model,
            )
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

        raise AssertionError(f"Unhandled command: {args.command}")
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
