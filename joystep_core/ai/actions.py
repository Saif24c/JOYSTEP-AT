"""Strict, finite AI action vocabulary; deliberately no code/shell actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ActionKind(Enum):
    MOVE_CURSOR = auto()
    CLICK = auto()
    DOUBLE_CLICK = auto()
    RIGHT_CLICK = auto()
    DRAG = auto()
    SCROLL = auto()
    TYPE_TEXT = auto()
    PRESS_KEY = auto()
    HOTKEY = auto()
    WAIT = auto()
    OBSERVE = auto()


class RiskTier(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass(frozen=True)
class AIAction:
    kind: ActionKind
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


LOW_RISK = {ActionKind.MOVE_CURSOR, ActionKind.CLICK, ActionKind.DOUBLE_CLICK,
            ActionKind.RIGHT_CLICK, ActionKind.DRAG, ActionKind.SCROLL,
            ActionKind.PRESS_KEY, ActionKind.HOTKEY, ActionKind.WAIT, ActionKind.OBSERVE}


def classify_risk(action: AIAction) -> RiskTier:
    """Classify fixed input semantics before considering provider-supplied hints."""
    # These keystrokes have known destructive or persistence effects even when
    # a provider omits a risk hint.  Treat them conservatively rather than
    # trusting a model to classify its own output.
    if action.kind is ActionKind.PRESS_KEY and action.arguments.get("key") in {"delete", "backspace"}:
        return RiskTier.HIGH
    if action.kind is ActionKind.HOTKEY:
        keys = tuple(action.arguments.get("keys", ()))
        if keys in {("ctrl", "s"), ("ctrl", "x"), ("ctrl", "v"), ("ctrl", "z"), ("ctrl", "y")}:
            return RiskTier.MEDIUM
    if action.arguments.get("high_risk") is True:
        return RiskTier.HIGH
    if action.arguments.get("medium_risk") is True:
        return RiskTier.MEDIUM
    if action.kind is ActionKind.TYPE_TEXT:
        return RiskTier.LOW
    return RiskTier.LOW


def validate_action(action: AIAction, width: int = 1920, height: int = 1080) -> tuple[bool, str]:
    """Validate action data before it reaches any executor."""
    args = action.arguments
    if action.kind in {ActionKind.MOVE_CURSOR, ActionKind.CLICK, ActionKind.DOUBLE_CLICK, ActionKind.RIGHT_CLICK}:
        x, y = args.get("x"), args.get("y")
        if not isinstance(x, int) or not isinstance(y, int) or not (0 <= x < width and 0 <= y < height):
            return False, "coordinates must be within the configured screen bounds"
    elif action.kind is ActionKind.DRAG:
        values = (args.get("start_x"), args.get("start_y"), args.get("end_x"), args.get("end_y"))
        if not all(isinstance(v, int) for v in values) or not all(0 <= v < (width if i % 2 == 0 else height) for i, v in enumerate(values)):
            return False, "drag coordinates must be within the configured screen bounds"
    elif action.kind is ActionKind.SCROLL:
        if not isinstance(args.get("amount"), int) or abs(args["amount"]) > 100:
            return False, "scroll amount must be an integer between -100 and 100"
    elif action.kind is ActionKind.TYPE_TEXT:
        if not isinstance(args.get("text"), str) or not args["text"] or len(args["text"]) > 5000:
            return False, "text must be non-empty and at most 5000 characters"
    elif action.kind is ActionKind.PRESS_KEY:
        if args.get("key") not in {"enter", "escape", "space", "tab", "backspace", "delete", "home", "end", "left", "right", "up", "down"}:
            return False, "key is not allowlisted"
    elif action.kind is ActionKind.HOTKEY:
        keys = args.get("keys")
        allowed = {"ctrl", "shift", "alt", "win", "s", "z", "y", "c", "v", "x", "a", "b", "i", "u", "f", "home", "end"}
        if not isinstance(keys, (list, tuple)) or not keys or not all(key in allowed for key in keys):
            return False, "hotkey contains a non-allowlisted key"
    elif action.kind is ActionKind.WAIT:
        seconds = args.get("seconds")
        if not isinstance(seconds, (int, float)) or not 0 <= seconds <= 10:
            return False, "wait must be between zero and ten seconds"
    elif action.kind is ActionKind.OBSERVE:
        if set(args) - {"scope"}:
            return False, "observe accepts only an optional scope"
    return True, ""
