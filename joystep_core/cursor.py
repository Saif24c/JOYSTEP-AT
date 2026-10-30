"""Cursor mapping and opt-in accessibility output controls.

Stage 4 leaves landmark classification, existing EMA filters, and DSM
calibration untouched. The accessibility layer receives only mapped pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import time
from typing import Protocol


class HandSmoother(Protocol):
    def update(self, x: float, y: float) -> tuple[float, float]: ...
    def reset(self) -> None: ...


class FaceMapper(Protocol):
    def update(self, nose_x: float, nose_y: float) -> tuple[int, int] | None: ...


class CursorPipeline(Protocol):
    def hand_target(self, index_norm: tuple[float, float], screen_w: int, screen_h: int,
                    smoother: HandSmoother) -> tuple[int, int]: ...
    def face_target(self, mapper: FaceMapper, nose_x: float, nose_y: float) -> tuple[int, int] | None: ...
    def reset_hand(self, smoother: HandSmoother) -> None: ...


@dataclass(frozen=True)
class PassThroughCursorPipeline:
    """Exact extraction of the legacy hand and face cursor mapping."""
    frame_margin: float

    def hand_target(self, index_norm, screen_w, screen_h, smoother):
        mg = (1.0 - self.frame_margin) / 2.0
        nx = max(0.0, min((index_norm[0] - mg) / self.frame_margin, 1.0))
        ny = max(0.0, min((index_norm[1] - mg) / self.frame_margin, 1.0))
        sx, sy = smoother.update(nx, ny)
        return int(sx * screen_w), int(sy * screen_h)

    def face_target(self, mapper, nose_x, nose_y):
        return mapper.update(nose_x, nose_y)

    def reset_hand(self, smoother):
        smoother.reset()


class CursorOutputState(Enum):
    PASS_THROUGH = auto()
    STABLE_HOLD = auto()
    LOCKED = auto()


@dataclass
class AccessibilityCursorPipeline:
    """Output-stage controls; all are disabled by default.

    Stabilization holds an emitted target inside a hysteresis radius. Precision
    scales from an explicit anchor so it never jumps. Recenter is an explicit
    offset request: this class never silently recalibrates a user.
    """
    stabilization_enabled: bool = False
    precision_enabled: bool = False
    locked: bool = False
    hold_radius: float = 3.0
    release_radius: float = 6.0
    precision_scale: float = 0.35
    _last_base: tuple[int, int] | None = None
    _last_output: tuple[int, int] | None = None
    _precision_base: tuple[int, int] | None = None
    _precision_cursor: tuple[int, int] | None = None
    _recenter_offset: tuple[int, int] = (0, 0)

    @property
    def state(self) -> CursorOutputState:
        if self.locked:
            return CursorOutputState.LOCKED
        if self.stabilization_enabled and self._last_output is not None:
            return CursorOutputState.STABLE_HOLD
        return CursorOutputState.PASS_THROUGH

    def output_target(self, base_target: tuple[int, int] | None) -> tuple[int, int] | None:
        if base_target is None:
            return None
        self._last_base = base_target
        if self.locked:
            return None
        target = self._apply_recenter(base_target)
        if self.precision_enabled and self._precision_base and self._precision_cursor:
            target = (
                round(self._precision_cursor[0] + self.precision_scale * (target[0] - self._precision_base[0])),
                round(self._precision_cursor[1] + self.precision_scale * (target[1] - self._precision_base[1])),
            )
        if self.stabilization_enabled and self._last_output is not None:
            distance = math.dist(target, self._last_output)
            if distance < self.release_radius:
                return self._last_output
        self._last_output = target
        return target

    def set_stabilization(self, enabled: bool) -> None:
        self.stabilization_enabled = enabled

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        if not locked:
            self._last_output = None  # require a fresh post-unlock target

    def set_precision(self, enabled: bool, cursor_position: tuple[int, int]) -> bool:
        if enabled:
            if self._last_base is None:
                return False
            self.precision_enabled = True
            self._precision_base = self._apply_recenter(self._last_base)
            self._precision_cursor = cursor_position
            self._last_output = cursor_position
            return True
        self.precision_enabled = False
        self._precision_base = self._precision_cursor = None
        return True

    def recenter(self, cursor_position: tuple[int, int]) -> bool:
        if self._last_base is None:
            return False
        self._recenter_offset = (
            cursor_position[0] - self._last_base[0], cursor_position[1] - self._last_base[1]
        )
        self._last_output = cursor_position
        return True

    def _apply_recenter(self, target: tuple[int, int]) -> tuple[int, int]:
        return target[0] + self._recenter_offset[0], target[1] + self._recenter_offset[1]


@dataclass(frozen=True)
class ClickEvent:
    source: str
    button: str = "left"
    created_at: float = 0.0


@dataclass
class ClickArbiter:
    """Accept fresh click events once, with physical-input safety gates."""
    freshness_seconds: float = 0.75
    cooldowns: dict[str, float] = field(default_factory=lambda: {"dwell": 0.8})
    physical_locked: bool = False
    physical_armed: bool = True
    _last_accepted: dict[tuple[str, str], float] = field(default_factory=dict)

    def gate_physical(self) -> None:
        self.physical_armed = False

    def arm_physical(self) -> None:
        self.physical_armed = True

    def set_physical_locked(self, locked: bool) -> None:
        self.physical_locked = locked
        if not locked:
            self.gate_physical()

    def admit(self, event: ClickEvent, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if event.created_at and now - event.created_at > self.freshness_seconds:
            return False
        physical = event.source in {"hand", "face", "dwell"}
        if physical and (self.physical_locked or not self.physical_armed):
            return False
        key = (event.source, event.button)
        cooldown = self.cooldowns.get(event.source, 0.0)
        if now - self._last_accepted.get(key, float("-inf")) < cooldown:
            return False
        self._last_accepted[key] = now
        return True


@dataclass
class DwellClickController:
    """Optional one-click-per-stable-entry dwell state machine."""
    enabled: bool = False
    delay_seconds: float = 1.0
    stable_radius: float = 8.0
    _anchor: tuple[int, int] | None = None
    _entered_at: float | None = None
    _fired: bool = False

    def cancel(self) -> None:
        self._anchor = self._entered_at = None
        self._fired = False

    def update(self, target: tuple[int, int] | None, now: float, eligible: bool) -> bool:
        if not self.enabled or not eligible or target is None:
            self.cancel()
            return False
        if self._anchor is None or math.dist(target, self._anchor) > self.stable_radius:
            self._anchor, self._entered_at, self._fired = target, now, False
            return False
        if not self._fired and now - self._entered_at >= self.delay_seconds:
            self._fired = True
            return True
        return False
