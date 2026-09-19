"""Cursor mapping seam introduced in Stage 3.

This module is deliberately pass-through only.  It is a home for later
accessibility layers, but contains no stabilization, precision, dwell, lock,
or adaptive-sensitivity behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HandSmoother(Protocol):
    def update(self, x: float, y: float) -> tuple[float, float]: ...
    def reset(self) -> None: ...


class FaceMapper(Protocol):
    def update(self, nose_x: float, nose_y: float) -> tuple[int, int] | None: ...


class CursorPipeline(Protocol):
    def hand_target(
        self, index_norm: tuple[float, float], screen_w: int, screen_h: int,
        smoother: HandSmoother,
    ) -> tuple[int, int]: ...
    def face_target(self, mapper: FaceMapper, nose_x: float, nose_y: float) -> tuple[int, int] | None: ...
    def reset_hand(self, smoother: HandSmoother) -> None: ...


@dataclass(frozen=True)
class PassThroughCursorPipeline:
    """Exact extraction of the legacy hand and face cursor mapping.

    ``frame_margin`` and the supplied smoother/DSM mapper retain ownership of
    all existing mapping and smoothing semantics.  This class must remain
    behaviorally transparent until a later, separately approved stage.
    """
    frame_margin: float

    def hand_target(
        self, index_norm: tuple[float, float], screen_w: int, screen_h: int,
        smoother: HandSmoother,
    ) -> tuple[int, int]:
        mg = (1.0 - self.frame_margin) / 2.0
        nx = max(0.0, min((index_norm[0] - mg) / self.frame_margin, 1.0))
        ny = max(0.0, min((index_norm[1] - mg) / self.frame_margin, 1.0))
        sx, sy = smoother.update(nx, ny)
        return int(sx * screen_w), int(sy * screen_h)

    def face_target(self, mapper: FaceMapper, nose_x: float, nose_y: float) -> tuple[int, int] | None:
        return mapper.update(nose_x, nose_y)

    def reset_hand(self, smoother: HandSmoother) -> None:
        smoother.reset()
