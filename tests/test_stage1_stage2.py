from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import joystep_at as app
from joystep_core.cursor import PassThroughCursorPipeline
from joystep_core.fakes import (
    FakeCamera, FakeClock, FakeInputController, FakeScreenCapture,
    FakeSystemActions, FakeTTS, FakeTracker,
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
