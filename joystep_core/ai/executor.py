"""Validated action executor that respects JOYSTEP cursor and click safety."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .actions import AIAction, ActionKind
from ..cursor import ClickEvent


@dataclass
class RestrictedInputExecutor:
    """A fixed dispatcher, not an evaluator of model-provided code or commands."""
    input: Any
    click_arbiter: Any
    cursor_locked: callable

    def execute(self, action: AIAction) -> bool:
        if action.kind in {ActionKind.MOVE_CURSOR, ActionKind.CLICK, ActionKind.DOUBLE_CLICK,
                           ActionKind.RIGHT_CLICK, ActionKind.DRAG} and self.cursor_locked():
            return False
        args = action.arguments
        if action.kind is ActionKind.MOVE_CURSOR:
            self.input.moveTo(args["x"], args["y"]); return True
        if action.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK, ActionKind.RIGHT_CLICK}:
            source = "ai"
            event = ClickEvent(source, "left", time.monotonic())
            if not self.click_arbiter.admit(event): return False
            if action.kind is ActionKind.CLICK: self.input.click(args["x"], args["y"])
            elif action.kind is ActionKind.DOUBLE_CLICK: self.input.doubleClick(args["x"], args["y"])
            else: self.input.rightClick(args["x"], args["y"])
            return True
        if action.kind is ActionKind.SCROLL:
            self.input.scroll(args["amount"]); return True
        if action.kind is ActionKind.TYPE_TEXT:
            self.input.write(args["text"], interval=0.03); return True
        if action.kind is ActionKind.PRESS_KEY:
            self.input.press(args["key"]); return True
        if action.kind is ActionKind.HOTKEY:
            self.input.hotkey(*args["keys"]); return True
        if action.kind is ActionKind.WAIT:
            time.sleep(args["seconds"]); return True
        if action.kind is ActionKind.OBSERVE:
            return True
        # Drag is intentionally not wired into the live executor until a
        # dedicated safe pointer API is introduced.
        return False
