"""Provider protocol plus deterministic test provider; no external SDKs."""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
from io import BytesIO
import json
import os
from typing import Any, Callable, Protocol

from .actions import AIAction


@dataclass(frozen=True)
class Observation:
    application_state: dict[str, object] = field(default_factory=dict)
    description: str = ""
    image_data_url: str | None = None


class PerceptionProvider(Protocol):
    def observe(self) -> Observation: ...


class AIProvider(Protocol):
    def plan(self, task: str, observation: Observation) -> list[AIAction]: ...


@dataclass
class MockAIProvider:
    plans: list[list[AIAction]] = field(default_factory=list)
    calls: list[tuple[str, Observation]] = field(default_factory=list)

    def plan(self, task: str, observation: Observation) -> list[AIAction]:
        self.calls.append((task, observation))
        return self.plans.pop(0) if self.plans else []


class UnavailableAIProvider:
    def plan(self, task: str, observation: Observation) -> list[AIAction]:
        raise RuntimeError("AI Assist needs help: no AI provider is configured.")


@dataclass
class ConfigurationUnavailableAIProvider(UnavailableAIProvider):
    """Unavailable provider with a non-secret, configuration-specific help message."""
    message: str

    def plan(self, task: str, observation: Observation) -> list[AIAction]:
        raise RuntimeError(self.message)


@dataclass
class StaticPerception:
    observation: Observation = field(default_factory=Observation)
    calls: int = 0
    def observe(self) -> Observation:
        self.calls += 1
        return self.observation


@dataclass
class EphemeralScreenPerception:
    """Capture only on ``observe``; JPEG data stays in-memory for that request."""
    capture: Callable[[], Any]
    application_context: Callable[[], object] | None = None
    max_dimension: int = 1280
    jpeg_quality: int = 70

    def observe(self) -> Observation:
        image = self.capture()
        if image is None:
            raise RuntimeError("screen observation unavailable")
        try:
            # PIL Image-compatible screen captures are reduced before encoding.
            image.thumbnail((self.max_dimension, self.max_dimension))
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=self.jpeg_quality, optimize=True)
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            context = self.application_context() if self.application_context else {}
            state = {"application": getattr(context, "app_name", ""), "window_title": getattr(context, "window_title", "")}
            return Observation(application_state=state, image_data_url=f"data:image/jpeg;base64,{encoded}")
        finally:
            close = getattr(image, "close", None)
            if close: close()


ACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "actions": {
            "type": "array", "maxItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": [
                        "MOVE_CURSOR", "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "DRAG", "SCROLL",
                        "TYPE_TEXT", "PRESS_KEY", "HOTKEY", "WAIT", "OBSERVE"]},
                    "arguments": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "arguments", "reason"],
            },
        },
    },
    "required": ["actions"]
}


@dataclass
class OpenAIProvider:
    """OpenAI Responses adapter returning only Stage 7 typed actions."""
    api_key: str | None = None
    model: str | None = None
    client: Any = None
    timeout_seconds: float = 20.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        self.model = self.model or os.environ.get("JOYSTEP_OPENAI_MODEL", "gpt-5")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def plan(self, task: str, observation: Observation) -> list[AIAction]:
        if not self.configured:
            raise RuntimeError("AI Assist needs help: OPENAI_API_KEY is not configured.")
        client = self.client or self._client()
        content: list[dict[str, str]] = [{"type": "input_text", "text": self._instructions(task, observation)}]
        if observation.image_data_url:
            content.append({"type": "input_image", "image_url": observation.image_data_url, "detail": "low"})
        response = client.responses.create(
            model=self.model, input=[{"role": "user", "content": content}],
            text={"format": {"type": "json_schema", "name": "joystep_action_plan", "strict": True, "schema": ACTION_SCHEMA}},
            timeout=self.timeout_seconds,
        )
        return self._parse(getattr(response, "output_text", ""))

    def _client(self) -> Any:
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def _instructions(self, task: str, observation: Observation) -> str:
        return (
            "You are JOYSTEP AI Assist. Return exactly one action in the supplied JSON schema. "
            "Never return code, shell commands, subprocesses, filesystem operations, or unlisted actions. "
            "Use observation only to propose the next small reversible step. "
            "When structured application capabilities are present, prefer them over visual pointer actions. "
            f"Task: {task}\nApplication context: {json.dumps(observation.application_state)}"
        )

    def _parse(self, text: str) -> list[AIAction]:
        try:
            payload = json.loads(text)
            output: list[AIAction] = []
            from .actions import ActionKind
            for item in payload["actions"]:
                output.append(AIAction(ActionKind[item["kind"]], dict(item["arguments"]), str(item["reason"])))
            return output
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("AI Assist needs help: provider returned an invalid action plan.") from error


@dataclass
class GeminiProvider:
    """Google Gen AI SDK adapter that only returns the existing typed action plan."""
    api_key: str | None = None
    model: str | None = None
    client: Any = None

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        # Stable multimodal model with structured output and a documented free tier.
        self.model = self.model or os.environ.get("JOYSTEP_GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def plan(self, task: str, observation: Observation) -> list[AIAction]:
        if not self.configured:
            raise RuntimeError("AI Assist needs help: GEMINI_API_KEY is not configured.")
        client = self.client or self._client()
        contents: list[Any] = [self._instructions(task, observation)]
        if observation.image_data_url:
            contents.append(self._image_part(observation.image_data_url))
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"response_mime_type": "application/json", "response_json_schema": ACTION_SCHEMA},
        )
        return _parse_action_plan(getattr(response, "text", ""))

    def _client(self) -> Any:
        from google import genai
        return genai.Client(api_key=self.api_key)

    @staticmethod
    def _image_part(image_data_url: str) -> Any:
        """Convert the existing in-memory data URL to an SDK image part; never persist it."""
        try:
            header, encoded = image_data_url.split(",", 1)
            if not header.startswith("data:image/") or ";base64" not in header:
                raise ValueError("unsupported image data URL")
            mime_type = header[5:].split(";", 1)[0]
            from google.genai import types
            return types.Part.from_bytes(data=base64.b64decode(encoded, validate=True), mime_type=mime_type)
        except (ValueError, TypeError) as error:
            raise RuntimeError("AI Assist needs help: screen observation is invalid.") from error

    @staticmethod
    def _instructions(task: str, observation: Observation) -> str:
        return (
            "You are JOYSTEP AI Assist. Return exactly one action in the supplied JSON schema. "
            "Never return code, shell commands, subprocesses, filesystem operations, or unlisted actions. "
            "Use observation only to propose the next small reversible step. "
            "When structured application capabilities are present, prefer them over visual pointer actions. "
            f"Task: {task}\nApplication context: {json.dumps(observation.application_state)}"
        )


def select_ai_provider(provider_name: str | None = None) -> AIProvider:
    """Select a provider by environment choice, preserving legacy OpenAI selection."""
    selected = (provider_name if provider_name is not None else os.environ.get("JOYSTEP_AI_PROVIDER", "")).strip().lower()
    if selected == "":
        # Stage 8 selected OpenAI when its credential was present.  Keep that
        # compatibility; without a credential it remains safely unavailable.
        provider = OpenAIProvider()
        return provider if provider.configured else UnavailableAIProvider()
    if selected == "openai":
        provider = OpenAIProvider()
        return provider if provider.configured else ConfigurationUnavailableAIProvider(
            "AI Assist needs help: OpenAI is not configured. Set OPENAI_API_KEY.")
    if selected == "gemini":
        provider = GeminiProvider()
        return provider if provider.configured else ConfigurationUnavailableAIProvider(
            "AI Assist needs help: Gemini is not configured. Set GEMINI_API_KEY.")
    if selected == "mock":
        return MockAIProvider()
    if selected in {"", "unavailable"}:
        return UnavailableAIProvider()
    return ConfigurationUnavailableAIProvider("AI Assist needs help: selected AI provider is unsupported.")


def _parse_action_plan(text: str) -> list[AIAction]:
    """Provider-neutral conversion into the finite Stage 7 action vocabulary."""
    try:
        payload = json.loads(text)
        actions = payload["actions"]
        if not isinstance(actions, list) or len(actions) > 1:
            raise ValueError("expected no more than one action")
        output: list[AIAction] = []
        from .actions import ActionKind
        for item in actions:
            if set(item) != {"kind", "arguments", "reason"}:
                raise ValueError("unexpected action fields")
            output.append(AIAction(ActionKind[item["kind"]], dict(item["arguments"]), str(item["reason"])))
        return output
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("AI Assist needs help: provider returned an invalid action plan.") from error
