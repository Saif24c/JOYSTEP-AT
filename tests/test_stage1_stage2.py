from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import joystep_at as app
from joystep_core.cursor import (
    AccessibilityCursorPipeline, ClickArbiter, ClickEvent,
    DwellClickController, PassThroughCursorPipeline,
)
from joystep_core.applications import ApplicationContext, ApplicationRouter, WindowsApplicationContextProvider
from joystep_core.commands import CommandExecutor, recognize_and_execute
from joystep_core.intents import CommandIntent, IntentKind, parse_voice_intent, validate_intent
from joystep_core.word import WordAdapter
from joystep_core.ai.actions import AIAction, ActionKind, RiskTier, classify_risk, validate_action
from joystep_core.ai.executor import RestrictedInputExecutor
from joystep_core.ai.provider import MockAIProvider, Observation, StaticPerception, UnavailableAIProvider
from joystep_core.ai.provider import EphemeralScreenPerception, GeminiProvider, OpenAIProvider, select_ai_provider
from joystep_core.ai.session import AIAssistSession, AISessionState
from joystep_core.fakes import (
    FakeCamera, FakeClock, FakeInputController, FakeScreenCapture,
    FakeSystemActions, FakeTTS, FakeTracker, FakeRecognizer,
)
from joystep_core.metrics import BaselineMetrics
from joystep_core.state import ApplicationState


def landmarks() -> list[SimpleNamespace]:
    return [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]


def hand_for(thumb=False, index=False, middle=False, ring=False, pinky=False):
    lms = landmarks()
    # Right hand: thumb tip x > IP means extended; other fingers tip y < PIP.
    values = [(thumb, app.THUMB_TIP, app.THUMB_IP), (index, app.INDEX_TIP, app.INDEX_PIP),
              (middle, app.MIDDLE_TIP, app.MIDDLE_PIP), (ring, app.RING_TIP, app.RING_PIP),
              (pinky, app.PINKY_TIP, app.PINKY_PIP)]
    for extended, tip, pip in values:
        if tip == app.THUMB_TIP:
            lms[tip].x, lms[pip].x = (0.8, 0.5) if extended else (0.4, 0.5)
        else:
            lms[tip].y, lms[pip].y = (0.3, 0.5) if extended else (0.7, 0.5)
    lms[app.WRIST].x, lms[app.WRIST].y = 0.4, 0.7
    lms[app.MIDDLE_MCP].x, lms[app.MIDDLE_MCP].y = 0.5, 0.5
    return SimpleNamespace(landmark=lms)


@pytest.mark.parametrize(("hand", "expected"), [
    (hand_for(thumb=True), app.HandGesture.PINCH_L),
    (hand_for(index=True, middle=True), app.HandGesture.SCROLL),
    (hand_for(), app.HandGesture.FIST),
    (hand_for(index=True, middle=True, ring=True), app.HandGesture.TRACKING),
])
def test_existing_hand_gesture_classification(hand, expected):
    assert app.classify_hand(hand, "Right") is expected


def test_existing_ema_smoothing_and_reset():
    ema = app.EMA(0.25)
    assert ema.update(100, 40) == (100.0, 40.0)
    assert ema.update(0, 0) == (75.0, 30.0)
    ema.reset()
    assert ema.update(3, 4) == (3.0, 4.0)


def legacy_hand_target(index_norm, screen_w, screen_h, smoother):
    """The Stage 2 hand mapping, retained here as the regression oracle."""
    mg = (1.0 - app.FRAME_MARGIN) / 2.0
    nx = max(0.0, min((index_norm[0] - mg) / app.FRAME_MARGIN, 1.0))
    ny = max(0.0, min((index_norm[1] - mg) / app.FRAME_MARGIN, 1.0))
    sx, sy = smoother.update(nx, ny)
    return int(sx * screen_w), int(sy * screen_h)


@pytest.mark.parametrize("points", [
    [(0.5, 0.5)],
    [(0.0, 0.0), (1.0, 1.0)],
    [(0.29, 0.71), (0.7, 0.3), (0.5, 0.5)],
])
def test_pass_through_hand_pipeline_matches_legacy_mapping(points):
    old_smoother = app.EMA(app.CURSOR_SMOOTHING)
    new_smoother = app.EMA(app.CURSOR_SMOOTHING)
    pipeline = PassThroughCursorPipeline(app.FRAME_MARGIN)
    for point in points:
        assert pipeline.hand_target(point, 1920, 1080, new_smoother) == (
            legacy_hand_target(point, 1920, 1080, old_smoother)
        )


def test_pass_through_hand_pipeline_preserves_reset_behavior():
    smoother = app.EMA(app.CURSOR_SMOOTHING)
    pipeline = PassThroughCursorPipeline(app.FRAME_MARGIN)
    pipeline.hand_target((0.8, 0.8), 1920, 1080, smoother)
    pipeline.reset_hand(smoother)
    assert pipeline.hand_target((0.4, 0.6), 1920, 1080, smoother) == (
        int(((0.4 - ((1 - app.FRAME_MARGIN) / 2)) / app.FRAME_MARGIN) * 1920),
        int(((0.6 - ((1 - app.FRAME_MARGIN) / 2)) / app.FRAME_MARGIN) * 1080),
    )


def test_pass_through_face_pipeline_delegates_to_existing_dsm(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    old_dsm = app.DSMCalibrator(1920, 1080)
    new_dsm = app.DSMCalibrator(1920, 1080)
    for dsm in (old_dsm, new_dsm):
        dsm.start_calibration(0.5, 0.5)
        dsm.update(0.2, 0.2)
        dsm.update(0.8, 0.8)
    now[0] += app.CAL_DURATION + 0.01
    old_dsm.update(0.8, 0.8)
    new_dsm.update(0.8, 0.8)
    pipeline = PassThroughCursorPipeline(app.FRAME_MARGIN)
    for nose in [(0.5, 0.5), (0.2, 0.8), (0.8, 0.2)]:
        assert pipeline.face_target(new_dsm, *nose) == old_dsm.update(*nose)


def test_accessibility_pipeline_is_exact_pass_through_by_default():
    pipeline = AccessibilityCursorPipeline()
    assert pipeline.output_target((123, 456)) == (123, 456)
    assert pipeline.output_target((126, 455)) == (126, 455)
    assert pipeline.stabilization_enabled is False
    assert pipeline.precision_enabled is False
    assert pipeline.locked is False


def test_stabilization_holds_jitter_then_releases_only_when_enabled():
    pipeline = AccessibilityCursorPipeline()
    assert pipeline.output_target((100, 100)) == (100, 100)
    assert pipeline.output_target((102, 101)) == (102, 101)  # default pass-through
    pipeline.set_stabilization(True)
    assert pipeline.output_target((100, 100)) == (102, 101)  # no enable-time jump
    assert pipeline.output_target((104, 100)) == (102, 101)  # within release radius
    assert pipeline.output_target((108, 100)) == (108, 100)  # release


def test_explicit_recenter_adds_offset_without_changing_base_mapping():
    pipeline = AccessibilityCursorPipeline()
    pipeline.output_target((100, 100))
    assert pipeline.recenter((120, 90))
    assert pipeline.output_target((100, 100)) == (120, 90)
    assert pipeline.output_target((110, 110)) == (130, 100)


def test_precision_anchors_current_cursor_without_jump_then_scales_delta():
    pipeline = AccessibilityCursorPipeline(precision_scale=0.5)
    pipeline.output_target((100, 100))
    assert pipeline.set_precision(True, (400, 300))
    assert pipeline.output_target((100, 100)) == (400, 300)
    assert pipeline.output_target((120, 140)) == (410, 320)
    assert pipeline.set_precision(False, (410, 320))
    assert pipeline.output_target((120, 140)) == (120, 140)


def test_lock_gates_physical_output_and_unlock_requires_fresh_target():
    pipeline = AccessibilityCursorPipeline()
    pipeline.output_target((100, 100))
    pipeline.set_locked(True)
    assert pipeline.output_target((200, 200)) is None
    pipeline.set_locked(False)
    assert pipeline.output_target((210, 210)) == (210, 210)


def test_dwell_click_requires_stability_fires_once_and_cancels():
    dwell = DwellClickController(enabled=True, delay_seconds=1.0, stable_radius=5.0)
    assert not dwell.update((100, 100), 0.0, eligible=True)
    assert not dwell.update((104, 102), 0.8, eligible=True)
    assert dwell.update((104, 102), 1.01, eligible=True)
    assert not dwell.update((104, 102), 2.0, eligible=True)
    assert not dwell.update((120, 100), 2.1, eligible=True)  # new stable entry
    assert not dwell.update((120, 100), 2.5, eligible=False)  # cancellation
    assert not dwell.update((120, 100), 2.6, eligible=True)


def test_click_arbiter_rejects_stale_locked_gated_and_duplicate_dwell_events():
    arbiter = ClickArbiter(freshness_seconds=0.5, cooldowns={"dwell": 1.0})
    assert not arbiter.admit(ClickEvent("hand", created_at=1.0), now=2.0)
    arbiter.gate_physical()
    assert not arbiter.admit(ClickEvent("hand", created_at=2.0), now=2.0)
    arbiter.arm_physical()
    assert arbiter.admit(ClickEvent("dwell", created_at=2.0), now=2.0)
    assert not arbiter.admit(ClickEvent("dwell", created_at=2.1), now=2.1)
    arbiter.set_physical_locked(True)
    assert not arbiter.admit(ClickEvent("face", created_at=2.2), now=2.2)
    assert arbiter.admit(ClickEvent("voice", created_at=2.2), now=2.2)


def test_accessibility_voice_controls_are_opt_in_and_cursor_lock_keeps_voice_click(command_context):
    banner, tts, state, input_fake = command_context
    cursor = AccessibilityCursorPipeline()
    cursor.output_target((100, 100))
    arbiter = ClickArbiter()
    app.process_voice_command("turn on precision mode", banner, tts, state, app.ClickRipple(), cursor, arbiter)
    assert cursor.precision_enabled
    app.process_voice_command("lock cursor", banner, tts, state, app.ClickRipple(), cursor, arbiter)
    assert cursor.locked and arbiter.physical_locked
    app.process_voice_command("click", banner, tts, state, app.ClickRipple(), cursor, arbiter)
    assert input_fake.clicks == ["left"]
    app.process_voice_command("unlock cursor", banner, tts, state, app.ClickRipple(), cursor, arbiter)
    assert not cursor.locked and not arbiter.physical_armed


@pytest.mark.parametrize(("text", "kind"), [
    ("click", IntentKind.CLICK), ("right click", IntentKind.CLICK),
    ("move left 250", IntentKind.MOVE_CURSOR), ("scroll down", IntentKind.SCROLL),
    ("type hello", IntentKind.TYPE_TEXT), ("open notepad", IntentKind.OPEN_APPLICATION),
    ("press enter", IntentKind.HOTKEY),
    ("save", IntentKind.SAVE), ("undo", IntentKind.UNDO), ("copy", IntentKind.COPY),
    ("select all", IntentKind.SELECT_ALL), ("backspace", IntentKind.BACKSPACE),
    ("face mode", IntentKind.MODE_CHANGE), ("emergency stop", IntentKind.SAFETY),
])
def test_typed_intent_parser_recognizes_existing_command_families(text, kind):
    assert parse_voice_intent(text).kind is kind


def test_intent_validation_rejects_malformed_or_unbounded_inputs():
    assert not validate_intent(parse_voice_intent("move left 10001"))
    assert not validate_intent(parse_voice_intent(""))
    assert validate_intent(parse_voice_intent("move left 100"))
    assert validate_intent(parse_voice_intent("type hello"))


def test_document_intent_preserves_text_and_path_payload_casing():
    assert parse_voice_intent("type Hello JOYSTEP").arguments["text"] == "Hello JOYSTEP"
    assert parse_voice_intent(r"save as C:\Users\Admin\Report.docx").arguments["path"] == r"C:\Users\Admin\Report.docx"


class StaticContext:
    def current(self):
        return ApplicationContext("Windows", app_name="test", window_title="Test window")


class RecordingAdapter:
    name = "recording"
    def __init__(self): self.calls = []
    def supports(self, intent, context):
        return intent.kind is IntentKind.SAVE and context.app_name == "test"
    def execute(self, intent, context):
        self.calls.append((intent, context))
        return True


def test_application_router_uses_adapter_before_legacy_executor():
    adapter = RecordingAdapter()
    executor = CommandExecutor(StaticContext(), ApplicationRouter((adapter,)))
    legacy = []
    assert recognize_and_execute("save", executor, legacy.append)
    assert len(adapter.calls) == 1
    assert legacy == []


def test_unhandled_and_unknown_intents_fall_back_to_legacy_behavior():
    executor = CommandExecutor(StaticContext(), ApplicationRouter())
    legacy = []
    assert recognize_and_execute("copy", executor, legacy.append)
    assert not recognize_and_execute("say something unusual", executor, legacy.append)
    assert legacy == ["copy", "say something unusual"]


def test_windows_context_provider_requires_no_external_dependency():
    context = WindowsApplicationContextProvider().current()
    assert context.platform_name


def test_process_voice_command_remains_backward_compatible(command_context):
    banner, tts, state, input_fake = command_context
    app.process_voice_command("copy", banner, tts, state, app.ClickRipple())
    assert input_fake.hotkeys == [("ctrl", "c")]


class FakeFind:
    def __init__(self): self.calls = []; self.result = True
    def Execute(self, **kwargs): self.calls.append(kwargs); return self.result


class FakeFont:
    Bold = Italic = Underline = Size = None


class FakeParagraphFormat:
    Alignment = None


class FakeSelection:
    def __init__(self):
        self.Font = FakeFont(); self.ParagraphFormat = FakeParagraphFormat(); self.Find = FakeFind()
        self.Start = 0; self.End = 0; self.calls = []; self.Style = None
    def TypeText(self, text): self.calls.append(("type", text))
    def Copy(self): self.calls.append(("copy",))
    def Cut(self): self.calls.append(("cut",))
    def Paste(self): self.calls.append(("paste",))
    def Delete(self): self.calls.append(("delete",))
    def WholeStory(self): self.calls.append(("all",))
    def Expand(self, unit): self.calls.append(("expand", unit))
    def HomeKey(self, **kwargs): self.calls.append(("home", kwargs))
    def EndKey(self, **kwargs): self.calls.append(("end", kwargs))
    def MoveDown(self, **kwargs): self.calls.append(("down", kwargs))
    def MoveUp(self, **kwargs): self.calls.append(("up", kwargs))
    def ClearFormatting(self): self.calls.append(("clear",))


class FakeDocument:
    def __init__(self, readonly=False): self.ReadOnly = readonly; self.saved = 0; self.saved_as = []
    def Save(self): self.saved += 1
    def SaveAs2(self, path): self.saved_as.append(path)


class FakeDocuments:
    def __init__(self, document): self.Count = 1 if document else 0


class FakeWordApp:
    def __init__(self, document=None):
        self.ActiveDocument = document; self.Documents = FakeDocuments(document); self.Selection = FakeSelection()
        self.Visible = False; self.undo = self.redo = 0
    def Undo(self): self.undo += 1
    def Redo(self): self.redo += 1


class FakeWordApi:
    def __init__(self, app=None): self.app = app; self.starts = []
    def application(self, start=False): self.starts.append(start); return self.app


def word_intent(text): return parse_voice_intent(text)


def test_word_adapter_routes_and_opens_word_structurally():
    app_obj = FakeWordApp(FakeDocument())
    adapter = WordAdapter(api=FakeWordApi(app_obj))
    intent = word_intent("open word")
    assert adapter.supports(intent, ApplicationContext("Windows"))
    assert adapter.execute(intent, ApplicationContext("Windows"))
    assert app_obj.Visible and adapter.last_message == "Word is ready."


def test_word_adapter_formatting_navigation_and_text_are_structured():
    app_obj = FakeWordApp(FakeDocument()); adapter = WordAdapter(api=FakeWordApi(app_obj)); context = ApplicationContext("Windows")
    for command in ("bold", "italic", "underline", "set font size 14", "select current paragraph", "move next paragraph", "type hello"):
        assert adapter.execute(word_intent(command), context)
    assert app_obj.Selection.Font.Bold == -1 and app_obj.Selection.Font.Italic == -1
    assert app_obj.Selection.Font.Underline == 1 and app_obj.Selection.Font.Size == 14
    assert ("expand", 4) in app_obj.Selection.calls
    assert ("down", {"Unit": 4, "Count": 1}) in app_obj.Selection.calls
    assert ("type", "hello") in app_obj.Selection.calls


def test_word_find_replace_all_requires_explicit_confirmation():
    app_obj = FakeWordApp(FakeDocument()); adapter = WordAdapter(api=FakeWordApi(app_obj)); context = ApplicationContext("Windows")
    assert adapter.execute(word_intent("find accessibility"), context)
    assert app_obj.Selection.Find.calls[-1]["FindText"] == "accessibility"
    assert adapter.execute(word_intent("replace all old with new"), context)
    assert "Confirmation required" in adapter.last_message
    assert adapter.execute(word_intent("confirm replace all"), context)
    assert app_obj.Selection.Find.calls[-1]["Replace"] == 2


def test_word_save_and_large_delete_require_confirmation():
    document = FakeDocument(); app_obj = FakeWordApp(document); adapter = WordAdapter(api=FakeWordApi(app_obj)); context = ApplicationContext("Windows")
    assert adapter.execute(word_intent("save"), context)
    assert document.saved == 0
    assert adapter.execute(word_intent("confirm save"), context)
    assert document.saved == 1
    app_obj.Selection.End = 100
    assert adapter.execute(word_intent("delete"), context)
    assert not any(call[0] == "delete" for call in app_obj.Selection.calls)
    assert adapter.execute(word_intent("confirm delete"), context)
    assert ("delete",) in app_obj.Selection.calls


def test_word_adapter_reports_no_word_no_document_and_read_only_failures():
    context = ApplicationContext("Windows")
    no_word = WordAdapter(api=FakeWordApi(None))
    assert no_word.execute(word_intent("bold"), context)
    assert "open Microsoft Word" in no_word.last_message
    no_document = WordAdapter(api=FakeWordApi(FakeWordApp(None)))
    assert no_document.execute(word_intent("bold"), context)
    assert "no active document" in no_document.last_message
    readonly_doc = FakeDocument(readonly=True)
    readonly = WordAdapter(api=FakeWordApi(FakeWordApp(readonly_doc)))
    readonly.execute(word_intent("save"), context)
    readonly.execute(word_intent("confirm save"), context)
    assert "read-only" in readonly.last_message


def test_word_adapter_does_not_claim_unsafe_save_as_paths_or_unknown_commands(tmp_path):
    adapter = WordAdapter(api=FakeWordApi(FakeWordApp(FakeDocument())))
    context = ApplicationContext("Windows")
    assert not validate_intent(word_intent("save as relative.docx"))
    unsupported = CommandIntent(IntentKind.DOCUMENT, "unsupported", {"action": "unsupported"})
    assert adapter.execute(unsupported, context)
    assert "unsupported" in adapter.last_message

    target = tmp_path / "saved.docx"
    assert adapter.execute(word_intent(f"save as {target}"), context)
    assert "Confirmation required" in adapter.last_message
    assert adapter.execute(word_intent("confirm save"), context)
    assert adapter.api.app.ActiveDocument.saved_as == [str(target)]
    target.write_text("existing")
    assert adapter.execute(word_intent(f"save as {target}"), context)
    assert adapter.execute(word_intent("confirm save"), context)
    assert "confirm overwrite" in adapter.last_message
    assert adapter.execute(word_intent("confirm overwrite"), context)


def test_dsm_calibration_and_deadzone(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    dsm = app.DSMCalibrator(1000, 500)
    dsm.start_calibration(0.5, 0.5)
    dsm.update(0.2, 0.2)
    dsm.update(0.8, 0.8)
    now[0] = app.CAL_DURATION + 0.01
    dsm.update(0.8, 0.8)
    assert dsm.state == dsm.ACTIVE
    assert dsm.update(0.5, 0.5) is not None
    assert dsm.in_deadzone is True
    assert dsm.update(0.8, 0.8)[0] > 500


def test_mouth_hold_and_cooldown(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    lms = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    lms[10].y, lms[152].y = 0.2, 0.8
    lms[app.MOUTH_TOP].y, lms[app.MOUTH_BOT].y = 0.4, 0.5
    detector = app.MouthClickDetector()
    assert detector.update(lms, 480) == "none"
    now[0] += app.MOUTH_HOLD_DUR + 0.01
    assert detector.update(lms, 480) == "left_click"
    assert detector.update(lms, 480) == "none"


def test_wink_click_cooldown(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    detector = app.WinkDetector()
    assert detector.update(0.1, 0.3) == "left_click"
    assert detector.update(0.3, 0.1) == "none"
    now[0] += app.BLINK_COOLDOWN + 0.01
    assert detector.update(0.3, 0.3) == "none"
    assert detector.update(0.3, 0.1) == "left_click"


def test_long_blink_click_cooldown(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(app.time, "monotonic", lambda: now[0])
    detector = app.BlinkDetector()
    assert detector.update(0.1) == "none"
    now[0] = app.LONG_BLINK_DUR + 0.01
    assert detector.update(0.1) == "right_click"
    assert detector.update(0.1) == "none"


class Banner:
    def __init__(self): self.messages = []
    def show(self, *args, **kwargs): self.messages.append((args, kwargs))


@pytest.fixture
def command_context(monkeypatch):
    input_fake = FakeInputController()
    # process_voice_command needs only this small PyAutoGUI surface in tests.
    monkeypatch.setattr(app, "pyautogui", SimpleNamespace(
        click=lambda *a, **k: input_fake.click(),
        rightClick=lambda *a, **k: input_fake.click("right"),
        doubleClick=lambda *a, **k: input_fake.click("double"),
        scroll=input_fake.scroll,
        write=lambda text, **k: input_fake.type_text(text),
        hotkey=input_fake.hotkey,
        press=lambda key: input_fake.hotkey(key),
        position=lambda: (0, 0),
        moveTo=lambda *a, **k: None,
        easeInOutQuad=None,
    ))
    return Banner(), FakeTTS(), ApplicationState(mode=app.InputMode.HAND), input_fake


def test_voice_mode_pause_resume_and_typing(command_context):
    banner, tts, state, input_fake = command_context
    app.process_voice_command("switch face", banner, tts, state, app.ClickRipple())
    assert state.mode is app.InputMode.FACE
    app.process_voice_command("pause", banner, tts, state, app.ClickRipple())
    assert state.paused is True
    app.process_voice_command("resume", banner, tts, state, app.ClickRipple())
    assert state.paused is False
    app.process_voice_command("type hello", banner, tts, state, app.ClickRipple())
    assert input_fake.typed == ["hello"]


def test_shutdown_cancel_precedes_shutdown(command_context, monkeypatch):
    banner, tts, state, _ = command_context
    actions = FakeSystemActions()
    monkeypatch.setattr(app, "sys_shutdown", actions.shutdown)
    monkeypatch.setattr(app, "sys_cancel_shutdown", actions.cancel_shutdown)
    app.process_voice_command("cancel shutdown", banner, tts, state, app.ClickRipple())
    assert actions.calls == ["cancel_shutdown"]


def test_shutdown_command_keeps_existing_schedule_behavior(command_context, monkeypatch):
    banner, tts, state, _ = command_context
    actions = FakeSystemActions()
    monkeypatch.setattr(app, "sys_shutdown", actions.shutdown)
    app.process_voice_command("shutdown", banner, tts, state, app.ClickRipple())
    assert actions.calls == ["shutdown"]


@pytest.mark.parametrize("text", ["dictation off", "stop dictation", "pause", "resume", "emergency stop"])
def test_local_dictation_controls_are_recognized(text):
    assert app.is_local_dictation_command(text)


def test_dictation_does_not_treat_normal_prose_as_command():
    assert not app.is_local_dictation_command("today I am writing my report")


def test_dictation_router_queues_only_local_controls():
    commands: queue.Queue = queue.Queue()
    assert not app.route_dictation_text("today I am writing my report", commands)
    assert commands.empty()
    assert app.route_dictation_text("dictation off", commands)
    event = commands.get_nowait()
    assert event.type.name == "VOICE_COMMAND"
    assert event.payload == "dictation off"


@pytest.mark.parametrize("voice_state", [
    app.VoiceState.COMMAND_LISTENING,
    app.VoiceState.COMMAND_EXECUTING,
    app.VoiceState.DICTATION_ACTIVE,
])
def test_all_active_voice_states_suppress_face_clicks(voice_state):
    assert app.is_voice_capture_active(voice_state)


def test_idle_voice_does_not_suppress_face_clicks():
    assert not app.is_voice_capture_active(app.VoiceState.COMMAND_IDLE)


def test_voice_state_contract_exposes_the_four_stage_two_states():
    assert {state.name for state in app.VoiceState} == {
        "COMMAND_IDLE", "COMMAND_LISTENING", "COMMAND_EXECUTING", "DICTATION_ACTIVE",
    }


def test_stage_one_fakes_and_metrics_need_no_devices():
    clock = FakeClock()
    clock.advance(0.5)
    assert clock.monotonic() == 0.5
    metrics = BaselineMetrics(enabled=True)
    metrics.frame()
    metrics.cursor_update()
    metrics.sample("hand_inference", 0.010)
    metrics.sample("voice_latency", 0.050)
    metrics.process_sample({"cpu_percent": 4.0, "rss_bytes": 1024.0})
    assert metrics.summary()["hand_inference_mean_ms"] == pytest.approx(10.0)
    assert metrics.summary()["voice_latency_mean_ms"] == pytest.approx(50.0)
    assert metrics.summary()["rss_bytes_mean"] == pytest.approx(1024.0)


def test_camera_tracking_and_screen_fakes_are_deterministic():
    camera = FakeCamera(frames=["frame"])
    assert camera.read() == (True, "frame")
    assert camera.read() == (False, None)
    camera.release()
    assert camera.released
    tracker = FakeTracker(results=["landmarks"])
    assert tracker.process("rgb") == "landmarks"
    tracker.close()
    assert tracker.closed
    screen = FakeScreenCapture(image="screen")
    assert screen.capture() == "screen"
    assert screen.captures == 1


@pytest.mark.parametrize("action, valid", [
    (AIAction(ActionKind.MOVE_CURSOR, {"x": 10, "y": 20}), True),
    (AIAction(ActionKind.CLICK, {"x": -1, "y": 20}), False),
    (AIAction(ActionKind.TYPE_TEXT, {"text": "hello"}), True),
    (AIAction(ActionKind.HOTKEY, {"keys": ["ctrl", "s"]}), True),
    (AIAction(ActionKind.HOTKEY, {"keys": ["powershell"]}), False),
    (AIAction(ActionKind.WAIT, {"seconds": 99}), False),
])
def test_ai_action_allowlist_validation(action, valid):
    assert validate_action(action)[0] is valid


def test_ai_risk_confirmation_and_observe_plan_execute_verify_loop():
    high = AIAction(ActionKind.TYPE_TEXT, {"text": "delete", "high_risk": True})
    assert classify_risk(AIAction(ActionKind.SCROLL, {"amount": 1})) is RiskTier.LOW
    assert classify_risk(high) is RiskTier.HIGH
    executed, perception = [], StaticPerception(Observation({"app": "fake"}))
    provider = MockAIProvider([[high]])
    executor = type("Exec", (), {"execute": lambda _, action: executed.append(action) or True})()
    session = AIAssistSession(provider, perception, executor)
    session.start("task", now=0); session.step(now=0)
    assert session.state is AISessionState.CONFIRMATION_REQUIRED and not executed
    session.confirm(); session.step(now=1)
    assert session.state is AISessionState.COMPLETED and executed == [high] and perception.calls == 2

    medium = AIAction(ActionKind.TYPE_TEXT, {"text": "save", "medium_risk": True})
    medium_session = AIAssistSession(MockAIProvider([[medium]]), StaticPerception(), executor)
    medium_session.start("save", now=0); medium_session.step(now=0)
    assert medium_session.state is AISessionState.CONFIRMATION_REQUIRED


def test_ai_risk_classification_does_not_trust_provider_for_known_editing_actions():
    assert classify_risk(AIAction(ActionKind.PRESS_KEY, {"key": "delete"})) is RiskTier.HIGH
    assert classify_risk(AIAction(ActionKind.PRESS_KEY, {"key": "backspace"})) is RiskTier.HIGH
    assert classify_risk(AIAction(ActionKind.HOTKEY, {"keys": ["ctrl", "s"]})) is RiskTier.MEDIUM
    assert classify_risk(AIAction(ActionKind.HOTKEY, {"keys": ["ctrl", "x"]})) is RiskTier.MEDIUM
    assert classify_risk(AIAction(ActionKind.HOTKEY, {"keys": ["ctrl", "a"]})) is RiskTier.LOW


def test_ai_mock_provider_cancel_timeout_errors_and_invalid_plan_are_safe():
    executor = type("Exec", (), {"execute": lambda *_: True})()
    session = AIAssistSession(MockAIProvider(), StaticPerception(), executor, max_duration_seconds=1)
    session.start("task", now=0); session.cancel(); assert session.state is AISessionState.CANCELLED
    session.start("task", now=0); session.step(now=2); assert session.state is AISessionState.NEEDS_HELP
    unavailable = AIAssistSession(UnavailableAIProvider(), StaticPerception(), executor)
    unavailable.start("task", now=0); unavailable.step(now=0); assert unavailable.state is AISessionState.ERROR
    invalid = AIAssistSession(MockAIProvider([[AIAction(ActionKind.CLICK, {"x": -1, "y": 0})]]), StaticPerception(), executor)
    invalid.start("task", now=0); invalid.step(now=0); assert invalid.state is AISessionState.NEEDS_HELP


class SafeInput:
    def __init__(self): self.calls = []
    def __getattr__(self, name): return lambda *args, **kwargs: self.calls.append((name, args))


def test_ai_executor_respects_cursor_lock_and_click_arbiter():
    input_fake, arbiter, locked = SafeInput(), ClickArbiter(), [True]
    executor = RestrictedInputExecutor(input_fake, arbiter, lambda: locked[0])
    assert not executor.execute(AIAction(ActionKind.CLICK, {"x": 1, "y": 1}))
    locked[0] = False
    assert executor.execute(AIAction(ActionKind.CLICK, {"x": 1, "y": 1}))
    assert input_fake.calls[0][0] == "click"


def test_ai_assist_intent_and_voice_invocation_needs_provider_without_legacy_execution(command_context):
    banner, tts, state, _ = command_context
    assert parse_voice_intent("AI Assist: make heading bold").kind is IntentKind.AI_ASSIST
    session = AIAssistSession(UnavailableAIProvider(), StaticPerception(), type("Exec", (), {"execute": lambda *_: True})())
    app.process_voice_command("AI Assist: make heading bold", banner, tts, state, app.ClickRipple(), ai_session=session)
    assert session.state is AISessionState.ERROR


class FakeResponse:
    def __init__(self, output_text): self.output_text = output_text


class FakeResponses:
    def __init__(self, response): self.response = response; self.calls = []
    def create(self, **kwargs): self.calls.append(kwargs); return self.response


class FakeOpenAIClient:
    def __init__(self, response): self.responses = FakeResponses(response)


class FakeGeminiModels:
    def __init__(self, response): self.response = response; self.calls = []
    def generate_content(self, **kwargs): self.calls.append(kwargs); return self.response


class FakeGeminiClient:
    def __init__(self, response): self.models = FakeGeminiModels(response)


class FakeGeminiResponse:
    def __init__(self, text): self.text = text


def test_openai_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider(api_key=None, client=FakeOpenAIClient(FakeResponse('{"actions":[]}')))
    assert not provider.configured
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        provider.plan("task", Observation())


def test_openai_provider_sends_bounded_structured_vision_request_and_parses_plan():
    response = FakeResponse('{"actions":[{"kind":"SCROLL","arguments":{"amount":2},"reason":"continue"}]}')
    client = FakeOpenAIClient(response)
    provider = OpenAIProvider(api_key="test", model="test-model", client=client)
    observation = Observation({"application": "Word", "structured_capabilities": ["formatting"]}, image_data_url="data:image/jpeg;base64,abc")
    assert provider.plan("format title", observation) == [AIAction(ActionKind.SCROLL, {"amount": 2}, "continue")]
    request = client.responses.calls[0]
    assert request["model"] == "test-model" and request["text"]["format"]["type"] == "json_schema"
    assert request["input"][0]["content"][1]["type"] == "input_image"
    assert request["text"]["format"]["schema"]["properties"]["actions"]["maxItems"] == 1
    assert "prefer them over visual pointer actions" in request["input"][0]["content"][0]["text"]


@pytest.mark.parametrize("payload", ["not json", '{"actions":[{"kind":"SHELL","arguments":{},"reason":"bad"}]}'])
def test_openai_provider_rejects_malformed_or_nonallowlisted_output(payload):
    provider = OpenAIProvider(api_key="test", client=FakeOpenAIClient(FakeResponse(payload)))
    with pytest.raises(RuntimeError, match="invalid action plan"):
        provider.plan("task", Observation())


def test_gemini_provider_requires_key_and_uses_configurable_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiProvider(api_key=None, client=FakeGeminiClient(FakeResponse('{"actions":[]}')))
    assert not provider.configured
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        provider.plan("task", Observation())
    monkeypatch.setenv("JOYSTEP_GEMINI_MODEL", "test-model")
    configured = GeminiProvider(api_key="test", client=FakeGeminiClient(FakeResponse('{"actions":[]}')))
    assert configured.configured and configured.model == "test-model"


def test_gemini_provider_sends_one_action_schema_and_ephemeral_image_to_fake_client():
    response = FakeGeminiResponse('{"actions":[{"kind":"SCROLL","arguments":{"amount":2},"reason":"continue"}]}')
    client = FakeGeminiClient(response)
    provider = GeminiProvider(api_key="test", model="test-model", client=client)
    observation = Observation({"application": "Word"}, image_data_url="data:image/jpeg;base64,YWJj")
    assert provider.plan("scroll", observation) == [AIAction(ActionKind.SCROLL, {"amount": 2}, "continue")]
    request = client.models.calls[0]
    assert request["model"] == "test-model"
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["response_json_schema"]["properties"]["actions"]["maxItems"] == 1
    assert len(request["contents"]) == 2


@pytest.mark.parametrize("payload", ["not json", '{"actions":[{"kind":"SHELL","arguments":{},"reason":"bad"}]}'])
def test_gemini_provider_rejects_malformed_or_nonallowlisted_output(payload):
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(FakeGeminiResponse(payload)))
    with pytest.raises(RuntimeError, match="invalid action plan"):
        provider.plan("task", Observation())


def test_gemini_invalid_action_is_rejected_by_the_existing_session_validator():
    provider = GeminiProvider(api_key="test", client=FakeGeminiClient(
        FakeGeminiResponse('{"actions":[{"kind":"CLICK","arguments":{"x":-1,"y":0},"reason":"bad"}]}')))
    session = AIAssistSession(provider, StaticPerception(), type("Exec", (), {"execute": lambda *_: True})())
    session.start("task", now=0); session.step(now=0)
    assert session.state is AISessionState.NEEDS_HELP


def test_awake_session_accepts_multiple_commands_and_manual_v_fallback_starts_it():
    session = app.AwakeSession(timeout_seconds=30)
    assert not session.active
    # The V-key and wake phrase both call wake(), so no command-specific path
    # can regress the manual fallback.
    session.wake(now=10)
    assert session.active and not session.expired(now=39)
    session.touch(now=39)
    assert not session.expired(now=60)


@pytest.mark.parametrize("command", ["go to sleep", "stop listening", "sleep JARVIS", "stop listening JARVIS"])
def test_sleep_commands_exit_awake_session(command):
    session = app.AwakeSession(timeout_seconds=30)
    session.wake(now=0)
    assert app.is_sleep_session_command(command)
    session.sleep()
    assert not session.active


def test_awake_session_timeout_returns_to_wake_word_state():
    session = app.AwakeSession(timeout_seconds=10)
    session.wake(now=2)
    assert session.expired(now=12)
    session.sleep()
    assert not session.active


def test_voice_recognition_retries_short_error_without_retaining_audio():
    recognizer = FakeRecognizer(phrases=["open youtube"], failures=[TimeoutError("brief miss")])
    assert app.recognize_phrase(recognizer, object(), timeout=1, phrase_time_limit=4, retries=1) == "open youtube"
    assert recognizer.phrases == []


def test_tts_cleanup_unloads_pygame_audio_before_deleting_temp_file():
    calls = []
    music = type("Music", (), {
        "stop": lambda self: calls.append("stop"),
        "unload": lambda self: calls.append("unload"),
    })()
    assert app.release_tts_temp_file(music, "voice.mp3", remove=lambda path: calls.append(("delete", path)))
    assert calls == ["stop", "unload", ("delete", "voice.mp3")]


@pytest.mark.parametrize("text", [
    "AI Assist move the cursor to the centre",
    "AI Assist: move the cursor to the centre",
    "AI Assist, move the cursor to the centre",
])
def test_ai_assist_accepts_spoken_punctuation_variants(text):
    intent = parse_voice_intent(text)
    assert intent.kind is IntentKind.AI_ASSIST
    assert intent.arguments["task"] == "move the cursor to the centre"


def test_ordinary_command_does_not_accidentally_become_ai_assist():
    assert parse_voice_intent("open AI Assist settings").kind is IntentKind.OPEN_APPLICATION


def test_ai_assist_executes_multiple_freshly_observed_steps():
    first = AIAction(ActionKind.SCROLL, {"amount": 2}, "move toward video")
    second = AIAction(ActionKind.PRESS_KEY, {"key": "space"}, "play")
    executed = []
    perception = StaticPerception()
    session = AIAssistSession(MockAIProvider([[first], [second], []]), perception,
                              type("Exec", (), {"execute": lambda _, action: executed.append(action) or True})())
    session.start("play and forward", now=0)
    session.step(now=0)  # observe, decide first action
    session.step(now=1)  # execute/verify first, decide second
    session.step(now=2)  # execute/verify second, provider declares complete
    assert session.state is AISessionState.COMPLETED
    assert executed == [first, second]
    assert perception.calls == 3


def test_ai_assist_maximum_action_limit_and_cancellation_are_enforced():
    action = AIAction(ActionKind.SCROLL, {"amount": 1}, "continue")
    executed = []
    executor = type("Exec", (), {"execute": lambda _, value: executed.append(value) or True})()
    capped = AIAssistSession(MockAIProvider([[action], [action], [action]]), StaticPerception(), executor, max_actions=1)
    capped.start("bounded", now=0); capped.step(now=0); capped.step(now=1); capped.step(now=2)
    assert capped.state is AISessionState.NEEDS_HELP and executed == [action]

    cancellable = AIAssistSession(MockAIProvider([[action]]), StaticPerception(), executor)
    cancellable.start("cancel me", now=0); cancellable.step(now=0); cancellable.cancel(); cancellable.step(now=1)
    assert cancellable.state is AISessionState.CANCELLED


def test_ai_each_step_remains_validated_and_completion_does_not_sleep_voice():
    unsafe = AIAction(ActionKind.CLICK, {"x": -1, "y": 0}, "unsafe")
    executed = []
    ai = AIAssistSession(MockAIProvider([[unsafe]]), StaticPerception(),
                         type("Exec", (), {"execute": lambda _, value: executed.append(value) or True})())
    awake = app.AwakeSession(); awake.wake(now=0)
    ai.start("unsafe", now=0); ai.step(now=0)
    assert ai.state is AISessionState.NEEDS_HELP and not executed
    assert awake.active  # AI terminal state never owns the voice session.


def test_provider_selection_is_explicit_and_fails_closed_for_missing_gemini_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("JOYSTEP_AI_PROVIDER", raising=False)
    assert isinstance(select_ai_provider(), UnavailableAIProvider)
    monkeypatch.setenv("JOYSTEP_AI_PROVIDER", "gemini")
    missing = select_ai_provider()
    assert isinstance(missing, UnavailableAIProvider)
    with pytest.raises(RuntimeError, match="Gemini is not configured"):
        missing.plan("task", Observation())
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    assert isinstance(select_ai_provider(), GeminiProvider)
    monkeypatch.setenv("JOYSTEP_AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert isinstance(select_ai_provider(), OpenAIProvider)
    monkeypatch.delenv("JOYSTEP_AI_PROVIDER", raising=False)
    assert isinstance(select_ai_provider(), OpenAIProvider)  # Stage 8 compatibility
    monkeypatch.setenv("JOYSTEP_AI_PROVIDER", "mock")
    assert isinstance(select_ai_provider(), MockAIProvider)
    monkeypatch.setenv("JOYSTEP_AI_PROVIDER", "unavailable")
    assert isinstance(select_ai_provider(), UnavailableAIProvider)


def test_gemini_provider_failure_is_contained_by_the_existing_session():
    class FailingModels:
        def generate_content(self, **kwargs): raise TimeoutError("provider timeout")
    provider = GeminiProvider(api_key="test", client=type("Client", (), {"models": FailingModels()})())
    session = AIAssistSession(provider, StaticPerception(), type("Exec", (), {"execute": lambda *_: True})())
    session.start("task", now=0); session.step(now=0)
    assert session.state is AISessionState.ERROR


def test_ephemeral_screen_observation_is_resized_in_memory_and_not_saved(tmp_path):
    from PIL import Image
    image = Image.new("RGB", (2400, 1200), "white")
    perception = EphemeralScreenPerception(lambda: image, max_dimension=600, jpeg_quality=50)
    observation = perception.observe()
    assert observation.image_data_url.startswith("data:image/jpeg;base64,")
    assert not list(tmp_path.iterdir())


def test_provider_timeout_or_observation_failure_stops_session_safely():
    class BrokenPerception:
        def observe(self): raise TimeoutError("screen timed out")
    session = AIAssistSession(MockAIProvider(), BrokenPerception(), type("Exec", (), {"execute": lambda *_: True})())
    session.start("task", now=0); session.step(now=0)
    assert session.state is AISessionState.ERROR
