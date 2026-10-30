"""Typed, allowlisted voice intents for the incremental command architecture."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import re
from pathlib import Path


class IntentKind(Enum):
    UNKNOWN = auto()
    CLICK = auto()
    MOVE_CURSOR = auto()
    SCROLL = auto()
    TYPE_TEXT = auto()
    HOTKEY = auto()
    OPEN_APPLICATION = auto()
    SWITCH_WINDOW = auto()
    SAVE = auto()
    UNDO = auto()
    REDO = auto()
    COPY = auto()
    CUT = auto()
    PASTE = auto()
    SELECT_ALL = auto()
    DELETE = auto()
    BACKSPACE = auto()
    NAVIGATE = auto()
    MODE_CHANGE = auto()
    DICTATION = auto()
    SAFETY = auto()
    SYSTEM = auto()
    ACCESSIBILITY_CURSOR = auto()
    DOCUMENT = auto()
    AI_ASSIST = auto()


@dataclass(frozen=True)
class CommandIntent:
    kind: IntentKind
    raw_text: str
    arguments: dict[str, object] = field(default_factory=dict)


def parse_voice_intent(text: str) -> CommandIntent:
    """Recognize only named commands; unknown text is explicitly non-executable."""
    raw = text.strip()
    command = raw.lower()
    # Speech engines commonly omit punctuation.  Keep this anchored so an
    # ordinary command mentioning the words does not enter autonomous mode.
    match = re.fullmatch(r"ai\s+assist\s*(?::|,)?\s+(.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.AI_ASSIST, command, {"task": match.group(1)})
    if command in {"click", "right click", "double click"}:
        return CommandIntent(IntentKind.CLICK, command, {"button": command})
    if command in {"bold", "italic", "underline", "remove formatting", "select current word",
                   "select current sentence", "select current paragraph", "move to beginning of document",
                   "move to end of document", "move next paragraph", "move previous paragraph",
                   "move next sentence", "move previous sentence", "align left", "align center", "align right",
                   "confirm save", "confirm overwrite", "confirm delete", "confirm replace all"}:
        return CommandIntent(IntentKind.DOCUMENT, command, {"action": command})
    match = re.fullmatch(r"(?:set )?font size (\d{1,3})", command)
    if match:
        return CommandIntent(IntentKind.DOCUMENT, command, {"action": "font_size", "size": int(match.group(1))})
    match = re.fullmatch(r"(?:apply )?heading ([1-3])", command)
    if match:
        return CommandIntent(IntentKind.DOCUMENT, command, {"action": "heading", "level": int(match.group(1))})
    match = re.fullmatch(r"find (.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.DOCUMENT, command, {"action": "find", "text": match.group(1)})
    match = re.fullmatch(r"replace( all)? (.+) with (.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.DOCUMENT, command, {
            "action": "replace_all" if match.group(1) else "replace",
            "find": match.group(2), "replace": match.group(3),
        })
    match = re.fullmatch(r"save as (.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.DOCUMENT, command, {"action": "save_as", "path": match.group(1)})
    match = re.fullmatch(r"move (up|down|left|right|north|south|east|west)(?:\s+(\d+))?", command)
    if match:
        return CommandIntent(IntentKind.MOVE_CURSOR, command,
                             {"direction": match.group(1), "amount": int(match.group(2) or 100)})
    if command in {"scroll up", "scroll down"}:
        return CommandIntent(IntentKind.SCROLL, command, {"direction": command.rsplit(" ", 1)[1]})
    match = re.fullmatch(r"type\s+(.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.TYPE_TEXT, command, {"text": match.group(1)})
    match = re.fullmatch(r"press\s+(enter|escape|space|tab|backspace|delete)", command)
    if match:
        return CommandIntent(IntentKind.HOTKEY, command, {"keys": (match.group(1),)})
    match = re.fullmatch(r"(?:open|switch to)\s+(.+)", raw, flags=re.IGNORECASE)
    if match:
        return CommandIntent(IntentKind.OPEN_APPLICATION, command, {"target": match.group(1)})
    if command in {"switch window", "alt tab", "next window"}:
        return CommandIntent(IntentKind.SWITCH_WINDOW, command)
    shortcuts = {
        "save": IntentKind.SAVE, "undo": IntentKind.UNDO, "redo": IntentKind.REDO,
        "copy": IntentKind.COPY, "cut": IntentKind.CUT, "paste": IntentKind.PASTE, "select all": IntentKind.SELECT_ALL,
        "delete": IntentKind.DELETE, "backspace": IntentKind.BACKSPACE,
    }
    if command in shortcuts:
        return CommandIntent(shortcuts[command], command)
    if command in {"home", "end", "find", "new file", "enter", "escape", "space", "tab"}:
        return CommandIntent(IntentKind.NAVIGATE, command, {"command": command})
    if command in {"hand mode", "switch hand", "face mode", "switch face"}:
        return CommandIntent(IntentKind.MODE_CHANGE, command)
    if command in {"dictation on", "dictation off", "start dictation", "stop dictation", "start typing", "stop typing"}:
        return CommandIntent(IntentKind.DICTATION, command)
    if command in {"pause", "pause tracking", "resume", "wake up", "unpause", "continue", "activate", "emergency stop", "stop joystep"}:
        return CommandIntent(IntentKind.SAFETY, command)
    if command in {"shutdown", "restart", "reboot", "sleep", "lock screen", "lock computer", "cancel shutdown"}:
        return CommandIntent(IntentKind.SYSTEM, command)
    if command in {
        "cursor stabilization on", "cursor stabilization off", "turn on cursor stabilization",
        "turn off cursor stabilization", "precision mode on", "precision mode off",
        "turn on precision mode", "turn off precision mode", "lock cursor", "unlock cursor",
        "cursor lock", "cursor unlock", "recenter cursor", "recenter tracking",
        "dwell clicking on", "dwell clicking off", "turn on dwell clicking", "turn off dwell clicking",
    }:
        return CommandIntent(IntentKind.ACCESSIBILITY_CURSOR, command)
    return CommandIntent(IntentKind.UNKNOWN, command)


def validate_intent(intent: CommandIntent) -> bool:
    """Reject malformed commands before they reach an executor or adapter."""
    if intent.kind is IntentKind.UNKNOWN:
        return False
    if intent.kind is IntentKind.MOVE_CURSOR:
        amount = intent.arguments.get("amount")
        return isinstance(amount, int) and 0 < amount <= 10000
    if intent.kind is IntentKind.TYPE_TEXT:
        text = intent.arguments.get("text")
        return isinstance(text, str) and bool(text.strip()) and len(text) <= 5000
    if intent.kind is IntentKind.OPEN_APPLICATION:
        target = intent.arguments.get("target")
        return isinstance(target, str) and bool(target.strip()) and len(target) <= 200
    if intent.kind is IntentKind.DOCUMENT:
        for key in ("text", "find", "replace", "path"):
            if key in intent.arguments:
                value = intent.arguments[key]
                if not isinstance(value, str) or not value.strip() or len(value) > 5000:
                    return False
        size = intent.arguments.get("size")
        if isinstance(size, int) and not 6 <= size <= 144:
            return False
        path = intent.arguments.get("path")
        if path is not None:
            candidate = Path(str(path))
            return candidate.is_absolute() and candidate.suffix.lower() == ".docx"
        return True
    if intent.kind is IntentKind.AI_ASSIST:
        task = intent.arguments.get("task")
        return isinstance(task, str) and bool(task.strip()) and len(task) <= 2000
    return True
