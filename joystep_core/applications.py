"""Application context and adapter contracts. No application automation yet."""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import platform
from typing import Protocol

from .intents import CommandIntent


@dataclass(frozen=True)
class ApplicationContext:
    platform_name: str
    app_name: str = ""
    window_title: str = ""


class ApplicationAdapter(Protocol):
    name: str
    def supports(self, intent: CommandIntent, context: ApplicationContext) -> bool: ...
    def execute(self, intent: CommandIntent, context: ApplicationContext) -> bool: ...


class ApplicationContextProvider(Protocol):
    def current(self) -> ApplicationContext: ...


class WindowsApplicationContextProvider:
    """Best-effort foreground-window context using only the Windows stdlib API."""
    def current(self) -> ApplicationContext:
        if platform.system() != "Windows":
            return ApplicationContext(platform.system())
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, title, length + 1)
            return ApplicationContext("Windows", window_title=title.value)
        except (AttributeError, OSError):
            return ApplicationContext("Windows")


@dataclass
class ApplicationRouter:
    adapters: tuple[ApplicationAdapter, ...] = ()

    def route(self, intent: CommandIntent, context: ApplicationContext) -> ApplicationAdapter | None:
        for adapter in self.adapters:
            if adapter.supports(intent, context):
                return adapter
        return None
