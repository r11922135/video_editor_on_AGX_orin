"""Parse FFmpeg silence logs and build an auditable edit decision list.

The planner deliberately removes only the middle (or the outer portion at a
media boundary) of a long silence.  It therefore preserves a configurable
amount of breathing room instead of turning every detected pause into a hard
cut.

All times are expressed in seconds on the source-media timeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Literal, Sequence


Action = Literal["keep", "remove"]


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class Interval:
    """A half-open source interval, ``[start, end)``."""

    start: float
    end: float

    def __post_init__(self) -> None:
        start = _finite_float("interval start", self.start)
        end = _finite_float("interval end", self.end)
        if end < start:
            raise ValueError("interval end must be greater than or equal to start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "duration": self.duration}


@dataclass(frozen=True)
class SilenceConfig:
    """Controls which pauses are shortened and how much silence is retained.

    ``target_silence`` is the desired *total* retained duration.  For an
    interior pause it is divided between both sides of the removed region.
    ``edge_padding`` is the minimum retained silence adjacent to each spoken
    side.  ``min_keep`` is a final lower bound on the total retained silence.
    ``merge_gap`` joins detector fragments separated by a very small gap.
    """

    min_silence: float = 0.8
    target_silence: float = 0.35
    edge_padding: float = 0.08
    min_keep: float = 0.10
    merge_gap: float = 0.08
    time_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "min_silence",
            "target_silence",
            "edge_padding",
            "min_keep",
            "merge_gap",
            "time_epsilon",
        ):
            value = _finite_float(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.time_epsilon == 0.0:
            raise ValueError("time_epsilon must be greater than zero")


@dataclass(frozen=True)
class EditInterval:
    """One auditable decision on a source-media interval."""

    start: float
    end: float
    action: Action
    reason: str
    silence_index: int | None = None

    def __post_init__(self) -> None:
        start = _finite_float("edit start", self.start)
        end = _finite_float("edit end", self.end)
        if start < 0.0:
            raise ValueError("edit start must be non-negative")
        if end < start:
            raise ValueError("edit end must be greater than or equal to start")
        if self.action not in ("keep", "remove"):
            raise ValueError("edit action must be 'keep' or 'remove'")
        if not self.reason:
            raise ValueError("edit reason must not be empty")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "action": self.action,
            "reason": self.reason,
            "silence_index": self.silence_index,
        }


@dataclass(frozen=True)
class EditPlan:
    """Complete edit decision list covering the source exactly once."""

    duration: float
    detected_silences: tuple[Interval, ...]
    eligible_silences: tuple[Interval, ...]
    edits: tuple[EditInterval, ...]
    config: SilenceConfig

    @property
    def edit_intervals(self) -> tuple[EditInterval, ...]:
        """Alias useful to callers that prefer an explicit property name."""

        return self.edits

    def _coalesced(self, action: Action) -> tuple[Interval, ...]:
        result: list[Interval] = []
        epsilon = self.config.time_epsilon
        for edit in self.edits:
            if edit.action != action:
                continue
            current = Interval(edit.start, edit.end)
            if result and current.start <= result[-1].end + epsilon:
                result[-1] = Interval(result[-1].start, max(result[-1].end, current.end))
            else:
                result.append(current)
        return tuple(result)

    @property
    def keep_intervals(self) -> tuple[Interval, ...]:
        """Coalesced source spans to retain in the rendered output."""

        return self._coalesced("keep")

    @property
    def remove_intervals(self) -> tuple[Interval, ...]:
        return self._coalesced("remove")

    @property
    def kept_duration(self) -> float:
        return sum(edit.duration for edit in self.edits if edit.action == "keep")

    @property
    def removed_duration(self) -> float:
        return sum(edit.duration for edit in self.edits if edit.action == "remove")

    @property
    def output_duration(self) -> float:
        return self.kept_duration

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "output_duration": self.output_duration,
            "removed_duration": self.removed_duration,
            "config": asdict(self.config),
            "detected_silences": [item.as_dict() for item in self.detected_silences],
            "eligible_silences": [item.as_dict() for item in self.eligible_silences],
            "keep_intervals": [item.as_dict() for item in self.keep_intervals],
            "remove_intervals": [item.as_dict() for item in self.remove_intervals],
            "edits": [item.as_dict() for item in self.edits],
        }


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_EVENT_RE = re.compile(
    rf"silence_(?P<kind>start|end)\s*:\s*(?P<time>{_NUMBER})"
    rf"(?:\s*\|\s*silence_duration\s*:\s*(?P<reported>{_NUMBER}))?"
)


def parse_silencedetect(
    stderr: str | bytes,
    *,
    duration: float | None = None,
) -> tuple[Interval, ...]:
    """Parse ``ffmpeg -af silencedetect`` stderr into source intervals.

    A final ``silence_start`` has no matching end when media finishes during a
    pause; pass ``duration`` to close it at EOF.  If a log starts in the middle
    of a pause, an orphan ``silence_end`` can be recovered from FFmpeg's
    ``silence_duration`` field.  Other malformed event sequences raise
    ``ValueError`` instead of silently producing unsafe cuts.
    """

    if isinstance(stderr, bytes):
        text = stderr.decode("utf-8", errors="replace")
    elif isinstance(stderr, str):
        text = stderr
    else:
        raise TypeError("stderr must be str or bytes")

    media_duration: float | None = None
    if duration is not None:
        media_duration = _finite_float("duration", duration)
        if media_duration < 0.0:
            raise ValueError("duration must be non-negative")

    result: list[Interval] = []
    pending_start: float | None = None
    for match in _EVENT_RE.finditer(text):
        event_time = _finite_float("silence event time", match.group("time"))
        if match.group("kind") == "start":
            if pending_start is not None:
                raise ValueError(
                    "silencedetect log contains consecutive silence_start events"
                )
            pending_start = event_time
            continue

        if pending_start is None:
            reported = match.group("reported")
            if reported is None:
                raise ValueError("silence_end appears before silence_start")
            reported_duration = _finite_float("reported silence duration", reported)
            if reported_duration < 0.0:
                raise ValueError("reported silence duration must be non-negative")
            start = event_time - reported_duration
        else:
            start = pending_start
        if event_time < start:
            raise ValueError("silence_end occurs before silence_start")
        result.append(Interval(start, event_time))
        pending_start = None

    if pending_start is not None:
        if media_duration is None:
            raise ValueError(
                "silencedetect log ends during silence; duration is required"
            )
        if media_duration < pending_start:
            raise ValueError("duration occurs before the final silence_start")
        result.append(Interval(pending_start, media_duration))

    return tuple(result)


def _coerce_interval(value: Interval | Sequence[float], index: int) -> Interval:
    if isinstance(value, Interval):
        return value
    try:
        start, end = value
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"silence interval {index} must contain exactly two numbers"
        ) from exc
    return Interval(start, end)


def _normalize_silences(
    duration: float,
    silences: Iterable[Interval | Sequence[float]],
    config: SilenceConfig,
) -> tuple[Interval, ...]:
    clipped: list[Interval] = []
    for index, raw in enumerate(silences):
        interval = _coerce_interval(raw, index)
        start = min(duration, max(0.0, interval.start))
        end = min(duration, max(0.0, interval.end))
        if end - start <= config.time_epsilon:
            continue
        clipped.append(Interval(start, end))

    clipped.sort(key=lambda item: (item.start, item.end))
    merged: list[Interval] = []
    for interval in clipped:
        if not merged:
            merged.append(interval)
            continue
        previous = merged[-1]
        if interval.start <= previous.end + config.merge_gap + config.time_epsilon:
            merged[-1] = Interval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def _removal_for_silence(
    duration: float,
    silence: Interval,
    config: SilenceConfig,
) -> Interval | None:
    epsilon = config.time_epsilon
    touches_start = silence.start <= epsilon
    touches_end = duration - silence.end <= epsilon

    if touches_start and touches_end:
        retained = max(config.target_silence, config.min_keep)
        remove_start = silence.start + min(retained, silence.duration)
        remove_end = silence.end
    elif touches_start:
        retained = max(
            config.target_silence, config.edge_padding, config.min_keep
        )
        remove_start = silence.start
        remove_end = silence.end - min(retained, silence.duration)
    elif touches_end:
        retained = max(
            config.target_silence, config.edge_padding, config.min_keep
        )
        remove_start = silence.start + min(retained, silence.duration)
        remove_end = silence.end
    else:
        retained = max(
            config.target_silence,
            2.0 * config.edge_padding,
            config.min_keep,
        )
        retained = min(retained, silence.duration)
        left_retained = retained / 2.0
        right_retained = retained - left_retained
        remove_start = silence.start + left_retained
        remove_end = silence.end - right_retained

    if remove_end - remove_start <= epsilon:
        return None
    return Interval(remove_start, remove_end)


def build_edit_plan(
    duration: float,
    silences: Iterable[Interval | Sequence[float]],
    config: SilenceConfig | None = None,
) -> EditPlan:
    """Build a deterministic, full-timeline silence-compression plan.

    Input intervals are validated, clipped to the media bounds, sorted and
    merged before ``min_silence`` is applied.  ``edits`` partitions the whole
    source and records why each segment is kept or removed.  The coalesced
    ``keep_intervals`` are the execution-oriented spans suitable for the
    single-path FFmpeg select/timestamp renderer.
    """

    media_duration = _finite_float("duration", duration)
    if media_duration < 0.0:
        raise ValueError("duration must be non-negative")
    settings = config if config is not None else SilenceConfig()
    if not isinstance(settings, SilenceConfig):
        raise TypeError("config must be a SilenceConfig instance")

    detected = _normalize_silences(media_duration, silences, settings)
    eligible_indices = {
        index
        for index, item in enumerate(detected)
        if item.duration + settings.time_epsilon >= settings.min_silence
    }
    eligible = tuple(detected[index] for index in sorted(eligible_indices))
    removals = {
        index: removal
        for index in eligible_indices
        if (removal := _removal_for_silence(
            media_duration, detected[index], settings
        ))
        is not None
    }

    boundaries = {0.0, media_duration}
    for silence in detected:
        boundaries.update((silence.start, silence.end))
    for removal in removals.values():
        boundaries.update((removal.start, removal.end))
    ordered = sorted(boundaries)

    edits: list[EditInterval] = []
    silence_cursor = 0
    for start, end in zip(ordered, ordered[1:]):
        if end - start <= settings.time_epsilon:
            continue
        midpoint = start + (end - start) / 2.0
        while (
            silence_cursor < len(detected)
            and detected[silence_cursor].end <= midpoint
        ):
            silence_cursor += 1

        silence_index: int | None = None
        if (
            silence_cursor < len(detected)
            and detected[silence_cursor].start <= midpoint
            and midpoint < detected[silence_cursor].end
        ):
            silence_index = silence_cursor

        action: Action = "keep"
        reason = "content"
        if silence_index is not None:
            removal = removals.get(silence_index)
            if (
                removal is not None
                and removal.start <= midpoint
                and midpoint < removal.end
            ):
                action = "remove"
                reason = "compressed_silence"
            elif silence_index not in eligible_indices:
                reason = "short_silence"
            elif removal is None:
                reason = "silence_kept"
            else:
                reason = "retained_silence"

        edit = EditInterval(start, end, action, reason, silence_index)
        if (
            edits
            and edits[-1].action == edit.action
            and edits[-1].reason == edit.reason
            and edits[-1].silence_index == edit.silence_index
            and edit.start <= edits[-1].end + settings.time_epsilon
        ):
            previous = edits[-1]
            edits[-1] = EditInterval(
                previous.start,
                max(previous.end, edit.end),
                previous.action,
                previous.reason,
                previous.silence_index,
            )
        else:
            edits.append(edit)

    return EditPlan(
        duration=media_duration,
        detected_silences=detected,
        eligible_silences=eligible,
        edits=tuple(edits),
        config=settings,
    )


__all__ = [
    "EditInterval",
    "EditPlan",
    "Interval",
    "SilenceConfig",
    "build_edit_plan",
    "parse_silencedetect",
]
