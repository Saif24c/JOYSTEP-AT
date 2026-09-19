"""Typed state and events shared by the legacy entry point and tests."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class VoiceMode(Enum):
    COMMAND_IDLE = auto()
    COMMAND_LISTENING = auto()
    COMMAND_EXECUTING = auto()
    DICTATION_ACTIVE = auto()


class EventType(Enum):
    VOICE_COMMAND = auto()
    DICTATION_TEXT = auto()
    EMERGENCY_STOP = auto()


@dataclass(frozen=True)
class AppEvent:
    type: EventType
    payload: str = ""
    source: str = "voice"


@dataclass
class ApplicationState:
    """The small state contract introduced in Stage 2.

    The legacy ``state`` dictionary remains the runtime source of truth during
    this stage; this type lets new code and tests stop relying on untyped keys.
    """
    mode: Any = None
    paused: bool = False
    dictation: bool = False
    voice_mode: VoiceMode = VoiceMode.COMMAND_IDLE

    # Temporary compatibility bridge: Stage 2 lets the existing entry point
    # keep its established dictionary-style state access while new code gets a
    # typed, discoverable state contract.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
