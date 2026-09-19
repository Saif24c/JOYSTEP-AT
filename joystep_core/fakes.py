"""Deterministic fakes for JOYSTEP regression tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeClock:
    now: float = 0.0
    def monotonic(self) -> float:
        return self.now
    def sleep(self, seconds: float) -> None:
        self.now += seconds
    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeCamera:
    frames: list[Any] = field(default_factory=list)
    released: bool = False
    def read(self) -> tuple[bool, Any]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)
    def release(self) -> None:
        self.released = True


@dataclass
class FakeTracker:
    results: list[Any] = field(default_factory=list)
    closed: bool = False
    def process(self, rgb_frame: Any) -> Any:
        return self.results.pop(0) if self.results else None
    def close(self) -> None:
        self.closed = True


@dataclass
class FakeScreenCapture:
    image: Any = None
    captures: int = 0
    def capture(self) -> Any:
        self.captures += 1
        return self.image


@dataclass
class FakeInputController:
    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: list[str] = field(default_factory=list)
    scrolls: list[int] = field(default_factory=list)
    typed: list[str] = field(default_factory=list)
    hotkeys: list[tuple[str, ...]] = field(default_factory=list)
    screenshot_value: Any = None
    def move_to(self, x: int, y: int, **kwargs: Any) -> None:
        self.moves.append((x, y))
    def click(self, button: str = "left") -> None:
        self.clicks.append(button)
    def scroll(self, amount: int) -> None:
        self.scrolls.append(amount)
    def type_text(self, text: str, **kwargs: Any) -> None:
        self.typed.append(text)
    def hotkey(self, *keys: str) -> None:
        self.hotkeys.append(keys)
    def screenshot(self) -> Any:
        return self.screenshot_value


@dataclass
class FakeTTS:
    messages: list[tuple[str, bool]] = field(default_factory=list)
    stopped: bool = False
    def say(self, text: str, priority: bool = False) -> None:
        self.messages.append((text, priority))
    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeSystemActions:
    calls: list[str] = field(default_factory=list)
    launchable: set[str] = field(default_factory=set)
    def shutdown(self) -> None: self.calls.append("shutdown")
    def cancel_shutdown(self) -> None: self.calls.append("cancel_shutdown")
    def restart(self) -> None: self.calls.append("restart")
    def sleep(self) -> None: self.calls.append("sleep")
    def lock(self) -> None: self.calls.append("lock")
    def open_application(self, name: str) -> bool:
        self.calls.append(f"open:{name}")
        return name in self.launchable


@dataclass
class FakeRecognizer:
    phrases: list[str] = field(default_factory=list)
    def listen(self, timeout: float, phrase_time_limit: float) -> str:
        if not self.phrases:
            raise TimeoutError("no fake phrase")
        return self.phrases.pop(0)
    def recognize(self, audio: Any) -> str:
        return str(audio)
