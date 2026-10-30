"""Bounded observe-plan-act-verify state machine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import time
from typing import Callable, Protocol
import os

from .actions import AIAction, RiskTier, classify_risk, validate_action
from .provider import AIProvider, Observation, PerceptionProvider


class AISessionState(Enum):
    IDLE = auto(); OBSERVING = auto(); PLANNING = auto(); CONFIRMATION_REQUIRED = auto()
    EXECUTING = auto(); VERIFYING = auto(); COMPLETED = auto(); NEEDS_HELP = auto()
    CANCELLED = auto(); ERROR = auto()


class SafeActionExecutor(Protocol):
    def execute(self, action: AIAction) -> bool: ...


StatusCallback = Callable[[AISessionState, str], None]


@dataclass
class AIAssistSession:
    provider: AIProvider
    perception: PerceptionProvider
    executor: SafeActionExecutor
    status: StatusCallback | None = None
    # A task is deliberately a short, bounded loop.  One fresh observation
    # and one provider decision are used for each action.
    max_actions: int = field(default_factory=lambda: int(os.environ.get("JOYSTEP_AI_MAX_ACTIONS", "8")))
    max_duration_seconds: float = 60.0
    state: AISessionState = AISessionState.IDLE
    task: str = ""
    _started_at: float = 0.0
    _executed: int = 0
    _pending: AIAction | None = None

    def start(self, task: str, now: float | None = None) -> None:
        if not task.strip():
            self._set(AISessionState.NEEDS_HELP, "AI needs help: describe a task."); return
        self.task, self._started_at, self._executed, self._pending = task, (time.monotonic() if now is None else now), 0, None
        self._set(AISessionState.OBSERVING, "AI observing")

    def cancel(self) -> None:
        self._pending = None; self._set(AISessionState.CANCELLED, "AI cancelled")

    def confirm(self) -> None:
        if self.state is not AISessionState.CONFIRMATION_REQUIRED or self._pending is None:
            return
        self._set(AISessionState.EXECUTING, "AI executing")

    def step(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state in {AISessionState.IDLE, AISessionState.COMPLETED, AISessionState.NEEDS_HELP, AISessionState.CANCELLED, AISessionState.ERROR, AISessionState.CONFIRMATION_REQUIRED}:
            return
        if now - self._started_at > self.max_duration_seconds or self._executed >= self.max_actions:
            self._set(AISessionState.NEEDS_HELP, "AI needs help: task limit reached"); return
        try:
            if self.state is AISessionState.OBSERVING:
                self._observation = self.perception.observe()
                self._set(AISessionState.PLANNING, "AI planning")
            if self.state is AISessionState.PLANNING:
                plan = self.provider.plan(self.task, self._observation)
                if not plan: self._set(AISessionState.COMPLETED, "AI completed"); return
                # Providers are schema-constrained to one next action.  Keep
                # this check here too, so a custom provider cannot smuggle a
                # pre-authorized plan around later observations.
                if len(plan) != 1:
                    self._set(AISessionState.NEEDS_HELP, "AI needs help: provider returned too many actions"); return
                self._pending = plan[0]
                valid, message = validate_action(self._pending)
                if not valid: self._set(AISessionState.NEEDS_HELP, f"AI needs help: {message}"); return
                if classify_risk(self._pending) is not RiskTier.LOW:
                    self._set(AISessionState.CONFIRMATION_REQUIRED, "AI confirmation required"); return
                self._set(AISessionState.EXECUTING, "AI executing")
                return
            if self.state is AISessionState.EXECUTING:
                if self._pending is None or not self.executor.execute(self._pending):
                    self._set(AISessionState.NEEDS_HELP, "AI needs help: action failed"); return
                self._executed += 1; self._pending = None
                self._set(AISessionState.VERIFYING, "AI verifying")
            if self.state is AISessionState.VERIFYING:
                # Verify with a fresh screen and make the next decision from
                # that exact observation; never reuse an old provider plan.
                self._observation = self.perception.observe()
                plan = self.provider.plan(self.task, self._observation)
                if not plan:
                    self._set(AISessionState.COMPLETED, "AI completed"); return
                if len(plan) != 1:
                    self._set(AISessionState.NEEDS_HELP, "AI needs help: provider returned too many actions"); return
                self._pending = plan[0]
                valid, message = validate_action(self._pending)
                if not valid:
                    self._set(AISessionState.NEEDS_HELP, f"AI needs help: {message}"); return
                if classify_risk(self._pending) is not RiskTier.LOW:
                    self._set(AISessionState.CONFIRMATION_REQUIRED, "AI confirmation required"); return
                # Execute at most one action per call, ensuring cancellation
                # can always interrupt between actions.
                self._set(AISessionState.EXECUTING, "AI executing")
        except Exception:
            self._set(AISessionState.ERROR, "AI needs help: session error")

    def _set(self, state: AISessionState, message: str) -> None:
        self.state = state
        if self.status: self.status(state, message)
