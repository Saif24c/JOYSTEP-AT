"""Recognition -> validation -> routing -> execution boundary."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .applications import ApplicationContextProvider, ApplicationRouter
from .intents import CommandIntent, parse_voice_intent, validate_intent


LegacyExecutor = Callable[[str], None]


@dataclass
class CommandExecutor:
    context_provider: ApplicationContextProvider
    router: ApplicationRouter
    last_message: str = ""

    def execute(self, intent: CommandIntent, legacy_execute: LegacyExecutor) -> bool:
        self.last_message = ""
        if not validate_intent(intent):
            # Preserve current user-facing unknown-command feedback and all
            # legacy commands not yet promoted to a typed intent.
            legacy_execute(intent.raw_text)
            return False
        context = self.context_provider.current()
        adapter = self.router.route(intent, context)
        if adapter is not None:
            handled = adapter.execute(intent, context)
            self.last_message = str(getattr(adapter, "last_message", ""))
            return handled
        # The legacy allowlisted implementation stays authoritative until each
        # command obtains a dedicated typed handler in a later migration.
        legacy_execute(intent.raw_text)
        return True


def recognize_and_execute(text: str, executor: CommandExecutor, legacy_execute: LegacyExecutor) -> bool:
    return executor.execute(parse_voice_intent(text), legacy_execute)
