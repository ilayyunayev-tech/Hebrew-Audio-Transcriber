#!/usr/bin/env python3
"""Local Hebrew audio transcription CLI using faster-whisper."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
TURBO_MODEL_DIR = SCRIPT_DIR / "models" / "ivrit-turbo"
DEFAULT_MODEL_NAME = "turbo"
LEGACY_MODEL = "ivrit-ai/faster-whisper-v2-d4"
MODEL_ALIASES = {
    "turbo": TURBO_MODEL_DIR,
    "small": "small",
    "medium": "medium",
    "large": LEGACY_MODEL,
}
SUPPORTED_FORMATS = {".mp3", ".wav", ".mp4", ".ogg", ".m4a"}
PROGRESS_BAR_WIDTH = 30


class TranscriptionError(RuntimeError):
    """An expected user-facing transcription failure."""


def load_model(model_name: str = DEFAULT_MODEL_NAME, verbose: bool = False) -> Any:
    """Load a faster-whisper model configured for CPU inference."""
    model_source = MODEL_ALIASES[model_name]
    if model_name == "turbo" and not (TURBO_MODEL_DIR / "model.bin").is_file():
        raise TranscriptionError(
            f"Error: local Turbo model not found in {TURBO_MODEL_DIR}"
        )

    status = (
        f"Loading transcription model ({model_source})..."
        if verbose
        else
        "Loading transcription model..."
    )
    print(status, file=sys.stderr, flush=True)

    try:
        from faster_whisper import WhisperModel

        return WhisperModel(
            str(model_source),
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
            local_files_only=model_name == "turbo",
        )
    except Exception as exc:
        if verbose:
            logging.exception("Model loading failed")
        raise TranscriptionError(
            "Error: failed to load transcription model"
        ) from exc


def _show_progress(current_seconds: float, total_seconds: float) -> None:
    """Render an in-place transcription progress bar on stderr."""
    if total_seconds <= 0:
        return

    ratio = min(max(current_seconds / total_seconds, 0.0), 1.0)
    completed = round(ratio * PROGRESS_BAR_WIDTH)
    bar = "#" * completed + "-" * (PROGRESS_BAR_WIDTH - completed)
    percent = round(ratio * 100)
    sys.stderr.write(f"\rTranscribing [{bar}] {percent:3d}%")
    sys.stderr.flush()


def transcribe_audio(
    model: Any,
    audio_path: Path,
    verbose: bool = False,
    accurate: bool = False,
) -> tuple[str, list[Any]]:
    """Transcribe Hebrew speech and return the full text and timed segments."""
    if verbose:
        print(f"Audio file: {audio_path}", file=sys.stderr)

    try:
        segments_iterator, info = model.transcribe(
            str(audio_path),
            language="he",
            beam_size=5 if accurate else 1,
            best_of=5 if accurate else 1,
            vad_filter=True,
        )
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        segments = []
        _show_progress(0.0, duration)
        for segment in segments_iterator:
            segments.append(segment)
            _show_progress(float(segment.end), duration)
        _show_progress(duration, duration)
        if duration > 0:
            print(file=sys.stderr)
    except Exception as exc:
        print(file=sys.stderr)
        if verbose:
            logging.exception("Transcription failed")
        raise TranscriptionError("Error: transcription failed") from exc

    text = " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )
    return text, segments


def save_txt(text: str, output_path: Path) -> None:
    """Save a plain-text UTF-8 transcript."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def _format_timeline_timestamp(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"


def format_timeline(segments: Iterable[Any]) -> str:
    """Format transcription segments as readable timestamped lines."""
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = _format_timeline_timestamp(float(segment.start))
        end = _format_timeline_timestamp(float(segment.end))
        lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


def save_timeline(segments: Iterable[Any], output_path: Path) -> None:
    """Save a readable UTF-8 transcript with a time range per segment."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = format_timeline(segments)
    output_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
    )


def save_srt(segments: Iterable[Any], output_path: Path) -> None:
    """Save timed transcription segments in UTF-8 SRT format."""
    blocks = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        index = len(blocks) + 1
        blocks.append(
            "\n".join(
                (
                    str(index),
                    (
                        f"{_format_srt_timestamp(segment.start)} --> "
                        f"{_format_srt_timestamp(segment.end)}"
                    ),
                    text,
                )
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n".join(blocks)
    output_path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe Hebrew audio locally with faster-whisper."
    )
    parser.add_argument("audio", type=Path, help="Path to an audio/video file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save output to this file instead of printing it",
    )
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument(
        "--srt",
        action="store_true",
        help="Save an SRT subtitle file (defaults to AUDIO.srt)",
    )
    output_format.add_argument(
        "--timeline",
        action="store_true",
        help="Include a time range on every transcript line",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_ALIASES),
        default=DEFAULT_MODEL_NAME,
        help="Model (default: turbo, the fast local ivrit.ai Hebrew model)",
    )
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Use slower beam search for potentially higher accuracy",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed status messages and error diagnostics",
    )
    return parser


def _validate_audio_path(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise TranscriptionError("Error: file not found")
    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise TranscriptionError("Error: unsupported audio format")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_audio_path(args.audio)
        model = load_model(args.model, args.verbose)
        text, segments = transcribe_audio(
            model, args.audio, args.verbose, args.accurate
        )

        if args.srt:
            output_path = args.output or args.audio.with_suffix(".srt")
            save_srt(segments, output_path)
            if args.verbose:
                print(f"Saved SRT: {output_path}", file=sys.stderr)
        elif args.timeline:
            if args.output:
                save_timeline(segments, args.output)
                if args.verbose:
                    print(
                        f"Saved timeline transcript: {args.output}",
                        file=sys.stderr,
                    )
            else:
                print(format_timeline(segments))
        elif args.output:
            save_txt(text, args.output)
            if args.verbose:
                print(f"Saved transcript: {args.output}", file=sys.stderr)
        else:
            print(text)

        return 0
    except KeyboardInterrupt:
        print("\nTranscription cancelled.", file=sys.stderr)
        return 130
    except TranscriptionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except OSError as exc:
        if args.verbose:
            logging.exception("Could not write output")
        print(f"Error: failed to write output file: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
