"""
╔══════════════════════════════════════════════════════════════════════════╗
║                  JOYSTEP AT — ULTIMATE EDITION                           ║
║              Full Assistive Technology Suite  v4.0                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  THREE INPUT MODES  (switch anytime — even mid-session)                  ║
║  ────────────────────────────────────────────────────────────────────    ║
║  HAND MODE    (default)  — for users with hand mobility                  ║
║    🖐 Open palm      → cursor follows INDEX fingertip                    ║
║    ✊ Fist           → cursor FREEZES  (rest your hand)                  ║
║    👍 Thumb open     → LEFT CLICK  (smooth + visual ripple)              ║
║    🤙 Pinch mid      → RIGHT CLICK                                       ║
║    ✌ Peace sign     → SCROLL MODE  (move hand up/down)                  ║
║                                                                          ║
║  FACE MODE  — for users with NO hands / limited hand mobility            ║
║    Press C to run 5-sec DSM calibration (nose-tip absolute mapping)      ║
║    After calibration: tiny head moves map to full screen corners          ║
║    MOUTH HOLD (jaw drop held 0.4s)        → LEFT CLICK  ← UPDATED       ║
║    Wink (one eye closed, other open)      → LEFT CLICK  ← NEW           ║
║    Long blink (hold > 1.5s)               → RIGHT CLICK                 ║
║                                                                          ║
║  VOICE-ONLY MODE  — camera not required                                  ║
║    100% voice-driven, no camera needed                                   ║
║                                                                          ║
║  VOICE COMMANDS  (all modes — say "Hey JARVIS" first)                   ║
║  ────────────────────────────────────────────────────────────────────    ║
║  CURSOR & CLICKS                                                         ║
║    "click" / "right click" / "double click"                              ║
║    "move up/down/left/right [amount]"                                    ║
║                                                                          ║
║  TYPING & DICTATION                                                      ║
║    "type [text]"          → types instantly                              ║
║    "dictation on/off"     → continuous speech-to-type mode               ║
║    "press enter/space/tab/escape/backspace/delete"                       ║
║    "copy / paste / cut / undo / redo / save / select all"                ║
║                                                                          ║
║  SYSTEM CONTROL                                                          ║
║    "shutdown" / "turn off"  → shuts down the computer                   ║
║    "restart" / "reboot"     → restarts the computer                     ║
║    "sleep"                  → puts computer to sleep                    ║
║    "lock screen"            → locks the PC                              ║
║    "screenshot"             → takes and saves a screenshot              ║
║    "volume up/down/mute"    → media keys                                ║
║    "brightness up/down"     → screen brightness (Windows)               ║
║                                                                          ║
║  WINDOW MANAGEMENT                                                       ║
║    "minimize" / "maximize" / "close window"                              ║
║    "switch window"  → Alt+Tab                                            ║
║    "show desktop"   → Win+D                                              ║
║    "task manager"   → Ctrl+Shift+Esc                                    ║
║    "full screen"    → F11                                               ║
║                                                                          ║
║  APP LAUNCH                                                              ║
║    "open notepad / calculator / paint / settings / file explorer"        ║
║    "open [website]"  → browser                                          ║
║                                                                          ║
║  MODE SWITCHING                                                          ║
║    "switch to hand mode"                                                 ║
║    "switch to face mode"                                                 ║
║    "switch to voice mode"                                                ║
║    "pause" / "resume"  → freeze/unfreeze all input                      ║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  INSTALL (run once):                                                     ║
║    py -3.11 -m pip install opencv-python mediapipe==0.10.9               ║
║                           pyautogui numpy SpeechRecognition pyttsx3      ║
║    py -3.11 -m pip install pipwin                                        ║
║    py -3.11 -m pipwin install pyaudio                                    ║
║                                                                          ║
║  RUN:   py -3.11 joystep_ultimate.py                                     ║
║  KEYS:  Q=quit  M=cycle mode  V=voice  P=pause  H=hand  F=face  C=calibrate ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════
import os
import sys
import time
import platform
import subprocess
import threading
import webbrowser
import queue
import re
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0.0   # we handle timing ourselves

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ← tune everything here
# ═══════════════════════════════════════════════════════════════════════

# ── Camera ───────────────────────────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_W, FRAME_H  = 640, 480

# ── Hand mode ────────────────────────────────────────────────────────
CURSOR_SMOOTHING  = 0.13   # EMA alpha: lower=smoother, higher=more responsive
FRAME_MARGIN      = 0.55   # fraction of camera frame that maps to full screen
                            # lower = less hand movement needed
PINCH_L_THRESH    = 0.075  # (kept for reference, not used for left click anymore)
PINCH_R_THRESH    = 0.090  # thumb+middle distance for right click
PINCH_COOLDOWN    = 0.85   # min seconds between clicks
FIST_MAX_FINGERS  = 1      # fingers up before considered a fist (forgiving)
SCROLL_SENS       = 2800   # scroll multiplier for peace-sign scroll
HAND_FPS          = 20     # hand thread fps cap

# ── Face mode ────────────────────────────────────────────────────────
FACE_SPEED        = 22     # pixels/frame at max tilt (relative mode)
FACE_SMOOTHING    = 0.18   # EMA for nose position
FACE_DZ_W         = 0.08   # deadzone half-width  (fraction of frame)
FACE_DZ_H         = 0.08   # deadzone half-height
EAR_CLOSE         = 0.20   # EAR below this = eye closed
EAR_OPEN          = 0.26   # EAR above this = eye open
BLINK_MAX_DUR     = 0.40   # max duration of one blink phase
DOUBLE_BLINK_WIN  = 1.5    # window for double-blink (kept for reference)
LONG_BLINK_DUR    = 1.5    # hold blink this long for right click
BLINK_COOLDOWN    = 1.1    # min seconds between click events

# ── Voice ────────────────────────────────────────────────────────────
WAKE_WORD         = "jarvis"
LISTEN_TIMEOUT    = 6      # seconds to wait for command after wake
SPEECH_RATE       = 160    # TTS words per minute
DICTATION_PAUSE   = 1.5    # pause in speech before committing dictated text

# ── Click animation ──────────────────────────────────────────────────
CLICK_RIPPLE_MS   = 350    # duration of click ripple animation in ms

# ── Mouth click (left click in face mode) ────────────────────────────
MOUTH_OPEN_THRESH = 0.06   # open-ratio (vertical gap / face height)
MOUTH_COOLDOWN    = 0.90   # min seconds between mouth-click events
MOUTH_HOLD_DUR    = 0.40   # mouth must stay open this long to fire click
                            # prevents "Hey JARVIS" speech from triggering

# ── Screenshot folder ────────────────────────────────────────────────
SCREENSHOT_DIR    = os.path.join(os.path.expanduser("~"), "Pictures", "JoystepAT")

# ── Colours (BGR) ────────────────────────────────────────────────────
C_TRACK  = ( 50, 230, 120)   # green   — tracking active
C_FREEZE = ( 80,  80, 220)   # blue    — frozen
C_LCLICK = (  0, 230,  80)   # lime    — left click
C_RCLICK = ( 50, 160, 255)   # amber   — right click
C_SCROLL = (255, 200,   0)   # yellow  — scroll
C_VOICE  = (255, 140,  40)   # orange  — voice
C_PAUSE  = ( 30,  30, 200)   # red     — paused
C_JARVIS = ( 50, 210, 255)   # cyan    — JARVIS brand
C_FACE   = (200,  80, 255)   # purple  — face mode
C_DZ     = (  0, 200, 100)   # deadzone idle
C_DZ_ACT = (  0,  80, 255)   # deadzone active
C_CAL    = (  0, 200, 255)   # calibration mode
C_HEARD  = ( 80, 255, 200)   # voice heard confirmation
BANNER_DUR = 1.3


# ═══════════════════════════════════════════════════════════════════════
#  LANDMARK INDICES
# ═══════════════════════════════════════════════════════════════════════

# Hand
WRIST        = 0
THUMB_TIP    = 4;  THUMB_IP    = 3;   THUMB_MCP   = 2
INDEX_TIP    = 8;  INDEX_PIP   = 6
MIDDLE_TIP   = 12; MIDDLE_PIP  = 10;  MIDDLE_MCP  = 9
RING_TIP     = 16; RING_PIP    = 14
PINKY_TIP    = 20; PINKY_PIP   = 18
TIPS         = [4, 8, 12, 16, 20]
PIPS         = [3, 6, 10, 14, 18]

HAND_BONES   = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# Face
NOSE_TIP         = 4
L_EYE_IDX        = [362, 385, 387, 263, 373, 380]
R_EYE_IDX        = [33,  160, 158, 133, 153, 144]
L_EYE_OUTER      = 263
R_EYE_OUTER      = 33

# Mouth landmarks for open-mouth left-click
MOUTH_TOP        = 13   # upper inner lip
MOUTH_BOT        = 14   # lower inner lip
MOUTH_L          = 78   # left mouth corner
MOUTH_R          = 308  # right mouth corner

# ── DSM Calibration (Face mode — nose-tip absolute mapping) ──────────────
CAL_DURATION     = 5.0    # seconds
CAL_SAFETY_GAP   = 0.01   # prevents zero-division
NOSE_EMA_ALPHA   = 0.10   # smoothing for absolute nose cursor (anti-jitter)
# Fraction of calibrated range at centre that freezes cursor (helps clicking)
NOSE_DEADZONE    = 0.10   # 10% of each axis around the neutral point = cursor stays still


# ═══════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════
class InputMode(Enum):
    HAND  = "HAND"
    FACE  = "FACE"
    # Voice is always active in both modes — no separate voice-only mode

class HandGesture(Enum):
    NONE     = auto()
    TRACKING = auto()
    FIST     = auto()
    PINCH_L  = auto()
    PINCH_R  = auto()
    SCROLL   = auto()

class VoiceState(Enum):
    IDLE      = auto()
    LISTENING = auto()
    THINKING  = auto()
    DICTATING = auto()
    HEARD     = auto()   # briefly shows what was recognised


# ═══════════════════════════════════════════════════════════════════════
#  EMA FILTER
# ═══════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, a: float):
        self.a = a
        self.x = self.y = None

    def update(self, x, y):
        if self.x is None:
            self.x, self.y = float(x), float(y)
        else:
            self.x = self.a * x + (1 - self.a) * self.x
            self.y = self.a * y + (1 - self.a) * self.y
        return self.x, self.y
    def reset(self):
        self.x = self.y = None


# ═══════════════════════════════════════════════════════════════════════
#  BANNER
# ═══════════════════════════════════════════════════════════════════════
class Banner:
    def __init__(self):
        self.text  = ""
        self.col   = C_JARVIS
        self.until = 0.0
        self.sub   = ""

    def show(self, text, col=C_JARVIS, dur=BANNER_DUR, sub=""):
        self.text, self.col = text, col
        self.until = time.monotonic() + dur
        self.sub   = sub

    def draw(self, frame):
        if not self.text or time.monotonic() > self.until:
            return
        h, w = frame.shape[:2]
        sz, _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_DUPLEX, 1.05, 2)
        tw = sz[0]
        x0 = max((w - tw)//2 - 22, 0)
        x1 = min((w + tw)//2 + 22, w)
        ov = frame.copy()
        cv2.rectangle(ov, (x0, h//2 - 54), (x1, h//2 + 26), (8, 8, 8), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, self.text, ((w - tw)//2, h//2 - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 1.05, self.col, 2)
        if self.sub:
            ss, _ = cv2.getTextSize(self.sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.putText(frame, self.sub, ((w - ss[0])//2, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1)


# ═══════════════════════════════════════════════════════════════════════
#  CLICK RIPPLE  (smooth visual feedback on clicks)
# ═══════════════════════════════════════════════════════════════════════
class ClickRipple:
    """Draws an expanding+fading circle at the click position in camera space."""
    def __init__(self):
        self._events: list = []   # list of (start_t, cx, cy, col)

    def add(self, cx: int, cy: int, col: tuple):
        self._events.append((time.monotonic(), cx, cy, col))

    def draw(self, frame):
        now  = time.monotonic()
        dur  = CLICK_RIPPLE_MS / 1000.0
        keep = []
        for (t0, cx, cy, col) in self._events:
            frac = (now - t0) / dur
            if frac >= 1.0:
                continue
            keep.append((t0, cx, cy, col))
            r     = int(8 + frac * 30)
            alpha = 1.0 - frac
            ov    = frame.copy()
            cv2.circle(ov, (cx, cy), r, col, 2)
            cv2.circle(ov, (cx, cy), r // 2, col, -1)
            cv2.addWeighted(ov, alpha * 0.7, frame, 1 - alpha * 0.7, 0, frame)
        self._events = keep


# ═══════════════════════════════════════════════════════════════════════
#  JARVIS VOICE  (TTS — non-blocking, guaranteed audio output)
#
#  Strategy:
#    1. Try pyttsx3 (offline, instant)
#    2. If pyttsx3 silent/broken → fall back to gTTS (online) + pygame mixer
#       Install fallback:  pip install gTTS pygame
# ═══════════════════════════════════════════════════════════════════════
class JarvisVoice:
    def __init__(self):
        self._q      = queue.Queue()
        self._engine = None   # "pyttsx3" | "gtts" | None
        self._t      = threading.Thread(target=self._run, daemon=True, name="TTS")
        self._t.start()

    # ── TTS worker thread ─────────────────────────────────────────────
    def _run(self):
        # ── Try pyttsx3 first ────────────────────────────────────────
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty('rate', SPEECH_RATE)
            # Pick a male voice if available
            voices = e.getProperty('voices')
            for v in voices:
                if any(k in v.name.lower() for k in ('male', 'david', 'mark', 'zira')):
                    e.setProperty('voice', v.id)
                    break
            # Quick smoke-test — if pyttsx3 is broken it will raise here
            e.say(" ")
            e.runAndWait()
            self._engine = "pyttsx3"
            print("[TTS] Engine: pyttsx3 ✓")
            while True:
                text = self._q.get()
                if text is None:
                    break
                print(f"[JARVIS] {text}")
                e.say(text)
                e.runAndWait()
            return
        except Exception as ex:
            print(f"[TTS] pyttsx3 failed ({ex}), trying gTTS fallback…")

        # ── gTTS + pygame fallback ────────────────────────────────────
        try:
            from gtts import gTTS
            import pygame
            import tempfile, os as _os
            pygame.mixer.init()
            self._engine = "gtts"
            print("[TTS] Engine: gTTS + pygame ✓")
            while True:
                text = self._q.get()
                if text is None:
                    break
                print(f"[JARVIS] {text}")
                try:
                    with tempfile.NamedTemporaryFile(suffix=".mp3",
                                                    delete=False) as f:
                        fpath = f.name
                    gTTS(text=text, lang='en', slow=False).save(fpath)
                    pygame.mixer.music.load(fpath)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
                    _os.unlink(fpath)
                except Exception as e2:
                    print(f"[TTS] gTTS error: {e2}")
            return
        except Exception as ex2:
            print(f"[TTS] gTTS fallback also failed ({ex2}).")
            print("[TTS] Install with:  pip install gTTS pygame")

        # ── Silent fallback — at least print ─────────────────────────
        self._engine = None
        while True:
            text = self._q.get()
            if text is None:
                break
            print(f"[JARVIS] {text}")

    def say(self, text: str, priority: bool = False):
        print(f"[JARVIS] {text}")
        if priority:
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except Exception:
                    pass
        self._q.put(text)

    def stop(self):
        self._q.put(None)


# ═══════════════════════════════════════════════════════════════════════
#  HAND GESTURE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════
def _hand_scale(lms) -> float:
    w = np.array([lms[WRIST].x, lms[WRIST].y])
    m = np.array([lms[MIDDLE_MCP].x, lms[MIDDLE_MCP].y])
    return max(float(np.linalg.norm(m - w)), 1e-4)

def _pdist(lms, a, b) -> float:
    pa = np.array([lms[a].x, lms[a].y])
    pb = np.array([lms[b].x, lms[b].y])
    return float(np.linalg.norm(pa - pb)) / _hand_scale(lms)

def _up(lms, tip, pip, thumb=False, hand="Right") -> bool:
    t, p = lms[tip], lms[pip]
    if thumb:
        return t.x > p.x if hand == "Right" else t.x < p.x
    return t.y < p.y

def classify_hand(hl, handedness: str) -> HandGesture:
    lms = hl.landmark
    ext = [_up(lms, TIPS[i], PIPS[i], i == 0, handedness) for i in range(5)]
    n   = sum(ext)

    # ── CHANGE 1: Thumb open (thumbs-up) → LEFT CLICK ────────────────
    # Thumb extended, all other fingers curled (none of index/mid/ring/pinky up)
    if ext[0] and sum(ext[1:]) == 0:
        return HandGesture.PINCH_L

    # Pinch right (thumb+middle distance) → RIGHT CLICK
    if _pdist(lms, THUMB_TIP, MIDDLE_TIP) < PINCH_R_THRESH:
        return HandGesture.PINCH_R

    if ext[1] and ext[2] and not ext[3] and not ext[4]:
        return HandGesture.SCROLL
    if n <= FIST_MAX_FINGERS:
        return HandGesture.FIST
    if n >= 3:
        return HandGesture.TRACKING
    return HandGesture.NONE


# ═══════════════════════════════════════════════════════════════════════
#  HAND THREAD STATE
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class HandResult:
    gesture:     HandGesture    = HandGesture.NONE
    lm_px:       list           = field(default_factory=list)
    index_norm:  tuple          = (0.5, 0.5)
    lock:        threading.Lock = field(default_factory=threading.Lock)

def hand_thread_fn(shared_frame, frame_lock, hr: HandResult,
                   stop: threading.Event, enabled: list):
    mp_h = mp.solutions.hands
    iv   = 1.0 / HAND_FPS
    with mp_h.Hands(static_image_mode=False, max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.75,
                    min_tracking_confidence=0.65) as hands:
        while not stop.is_set():
            t0 = time.monotonic()
            if not enabled[0]:
                with hr.lock:
                    hr.gesture, hr.lm_px = HandGesture.NONE, []
                time.sleep(iv); continue
            with frame_lock:
                if not shared_frame or shared_frame[0] is None:
                    time.sleep(0.01); continue
                fc = shared_frame[0].copy()
            fh, fw = fc.shape[:2]
            res = hands.process(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))
            g, px, inorm = HandGesture.NONE, [], (0.5, 0.5)
            if res.multi_hand_landmarks and res.multi_handedness:
                hl  = res.multi_hand_landmarks[0]
                hnd = res.multi_handedness[0].classification[0].label
                g   = classify_hand(hl, hnd)
                px  = [(int(lm.x * fw), int(lm.y * fh)) for lm in hl.landmark]
                inorm = (hl.landmark[INDEX_TIP].x, hl.landmark[INDEX_TIP].y)
            with hr.lock:
                hr.gesture, hr.lm_px, hr.index_norm = g, px, inorm
            dt = time.monotonic() - t0
            if iv - dt > 0: time.sleep(iv - dt)


# ═══════════════════════════════════════════════════════════════════════
#  FACE / EAR HELPERS
# ═══════════════════════════════════════════════════════════════════════
def calc_ear(lms, idxs, w, h) -> float:
    p = [np.array([lms[i].x * w, lms[i].y * h]) for i in idxs]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    hd = np.linalg.norm(p[0] - p[3])
    return 0.0 if hd < 1e-6 else (v1 + v2) / (2 * hd)


# ═══════════════════════════════════════════════════════════════════════
#  DOUBLE-BLINK / LONG-BLINK DETECTOR
# ═══════════════════════════════════════════════════════════════════════
class BlinkDetector:
    """
    Tracks blink patterns and fires:
      - right_click   on one long held blink (>= LONG_BLINK_DUR)

    Note: double-blink left-click has been replaced by WinkDetector.
    The state machine still tracks the sequence but double_click is
    no longer acted upon — only right_click (long blink) is used.
    """
    def __init__(self):
        self._s        = "OPEN"
        self._t        = 0.0        # time eye closed
        self._open1_t  = 0.0        # time eye re-opened after blink 1
        self.last_fire = 0.0

    def update(self, ear: float) -> str:
        now    = time.monotonic()
        closed = ear < EAR_CLOSE
        opened = ear > EAR_OPEN

        # ── OPEN: waiting for first close ────────────────────────────
        if self._s == "OPEN":
            if closed:
                self._s = "C1"
                self._t = now

        # ── C1: first eye-close in progress ──────────────────────────
        elif self._s == "C1":
            dur = now - self._t
            if dur >= LONG_BLINK_DUR:
                # Long blink → right click (check cooldown at fire)
                if (now - self.last_fire) >= BLINK_COOLDOWN:
                    self._s = "OPEN"
                    self.last_fire = now
                    return "right_click"
                else:
                    self._s = "OPEN"   # cooldown active, discard
            elif opened:
                if dur <= BLINK_MAX_DUR:
                    # Clean fast blink — go wait for second
                    self._s = "O1"
                    self._open1_t = now
                else:
                    self._s = "OPEN"   # blink was too slow, ignore

        # ── O1: between blink 1 and blink 2 ──────────────────────────
        elif self._s == "O1":
            if (now - self._open1_t) > DOUBLE_BLINK_WIN:
                self._s = "OPEN"   # window expired
            elif closed:
                self._s = "C2"
                self._t = now

        # ── C2: second eye-close in progress ─────────────────────────
        elif self._s == "C2":
            dur = now - self._t
            if opened:
                if dur <= BLINK_MAX_DUR:
                    # Valid second blink — double_click no longer used,
                    # just reset state cleanly
                    self._s = "OPEN"
                else:
                    self._s = "OPEN"   # second blink too slow
            elif dur > BLINK_MAX_DUR:
                self._s = "OPEN"       # held too long, abort

        return "none"


# ═══════════════════════════════════════════════════════════════════════
#  WINK DETECTOR  — CHANGE 2: replaces double-blink for left click
# ═══════════════════════════════════════════════════════════════════════
class WinkDetector:
    """
    Fires left_click on a wink: one eye closes while the other stays
    clearly open.  Uses individual left/right EAR values so it can
    distinguish a wink from a regular two-eye blink.

    Trigger: closed→open transition on the winking eye while the other
    eye remains above EAR_OPEN throughout.
    """
    def __init__(self):
        self._winking  = False   # True while a single-eye wink is in progress
        self.last_fire = 0.0

    def update(self, ear_left: float, ear_right: float) -> str:
        now = time.monotonic()
        if (now - self.last_fire) < BLINK_COOLDOWN:
            return "none"

        l_closed = ear_left  < EAR_CLOSE
        r_closed = ear_right < EAR_CLOSE
        l_open   = ear_left  > EAR_OPEN
        r_open   = ear_right > EAR_OPEN

        # Wink = exactly one eye closed, the other clearly open
        is_winking = (l_closed and r_open) or (r_closed and l_open)

        if is_winking and not self._winking:
            # Rising edge — fire left click
            self._winking  = True
            self.last_fire = now
            return "left_click"

        if not is_winking:
            self._winking = False

        return "none"


# ═══════════════════════════════════════════════════════════════════════
#  MOUTH CLICK DETECTOR  — CHANGE 3: hold-duration prevents JARVIS clash
# ═══════════════════════════════════════════════════════════════════════
class MouthClickDetector:
    """
    Fires a left_click when the mouth is held open beyond MOUTH_OPEN_THRESH
    for at least MOUTH_HOLD_DUR seconds.

    The hold-duration requirement means that brief mouth movements while
    speaking "Hey JARVIS" do NOT trigger a click — only a deliberate,
    sustained jaw-drop fires it.
    """
    def __init__(self):
        self._open_since = None   # monotonic time mouth first opened
        self._fired      = False  # already fired for this open phase
        self.last_fire   = 0.0

    def _ratio(self, lms, frame_h: int) -> float:
        """Vertical gap between inner lips, normalised by face height."""
        top    = lms[MOUTH_TOP].y * frame_h
        bot    = lms[MOUTH_BOT].y * frame_h
        face_h = max(abs(lms[152].y - lms[10].y) * frame_h, 1.0)
        return (bot - top) / face_h

    def update(self, lms, frame_h: int) -> str:
        now     = time.monotonic()
        ratio   = self._ratio(lms, frame_h)
        is_open = ratio > MOUTH_OPEN_THRESH

        if is_open:
            if self._open_since is None:
                self._open_since = now   # start timing the hold
                self._fired      = False
            elif not self._fired:
                held = now - self._open_since
                if (held >= MOUTH_HOLD_DUR
                        and (now - self.last_fire) >= MOUTH_COOLDOWN):
                    self._fired    = True
                    self.last_fire = now
                    return "left_click"
        else:
            # Mouth closed — reset for next open
            self._open_since = None
            self._fired      = False

        return "none"


# ═══════════════════════════════════════════════════════════════════════
#  DSM CALIBRATOR  (nose-tip absolute mapping with dynamic sensitivity)
# ═══════════════════════════════════════════════════════════════════════
class DSMCalibrator:
    """
    State machine:
      WAITING   → press C → CALIBRATING (5s) → ACTIVE

    Calibration records min/max AND the nose position at the START of
    calibration as the natural 'neutral' resting point.  The deadzone is
    then centred on that neutral — not on 0.5 — so the cursor freezes
    exactly where the user's head naturally sits.
    """
    WAITING     = "WAITING"
    CALIBRATING = "CALIBRATING"
    ACTIVE      = "ACTIVE"

    def __init__(self, sw: int, sh: int):
        self.sw, self.sh    = sw, sh
        self.state          = self.WAITING
        self._cal_end       = 0.0
        self._min_x = self._min_y =  9999.0
        self._max_x = self._max_y = -9999.0
        self._neutral_x     = 0.5   # learned resting nose X (normalised)
        self._neutral_y     = 0.5   # learned resting nose Y (normalised)
        self._ema_cx        = float(sw // 2)
        self._ema_cy        = float(sh // 2)
        # Published info
        self.mult_x         = 1.0
        self.mult_y         = 1.0
        self.countdown      = 0.0
        self.in_deadzone    = False

    def start_calibration(self, first_nose_x: float = 0.5,
                          first_nose_y: float = 0.5):
        """
        Call with the CURRENT nose position so we record the natural
        resting point before the user starts moving their head.
        """
        self._min_x = self._min_y =  9999.0
        self._max_x = self._max_y = -9999.0
        self._neutral_x = first_nose_x
        self._neutral_y = first_nose_y
        self._cal_end   = time.monotonic() + CAL_DURATION
        self.state      = self.CALIBRATING
        print(f"  [DSM] Neutral resting point: ({first_nose_x:.3f}, {first_nose_y:.3f})")

    def reset(self):
        self.__init__(self.sw, self.sh)

    def update(self, nose_x: float, nose_y: float):
        """
        Call every frame.  Returns (screen_cx, screen_cy) when ACTIVE,
        else None.
        """
        now = time.monotonic()

        if self.state == self.CALIBRATING:
            self.countdown = max(0.0, self._cal_end - now)
            self._min_x = min(self._min_x, nose_x)
            self._max_x = max(self._max_x, nose_x)
            self._min_y = min(self._min_y, nose_y)
            self._max_y = max(self._max_y, nose_y)
            if now >= self._cal_end:
                self._finish_calibration(nose_x, nose_y)
            return None

        if self.state == self.ACTIVE:
            safe_max_x = max(self._max_x, self._min_x + CAL_SAFETY_GAP)
            safe_max_y = max(self._max_y, self._min_y + CAL_SAFETY_GAP)

            raw_cx = float(np.clip(
                np.interp(nose_x, [self._min_x, safe_max_x], [0.0, 1.0]),
                0.0, 1.0))
            raw_cy = float(np.clip(
                np.interp(nose_y, [self._min_y, safe_max_y], [0.0, 1.0]),
                0.0, 1.0))

            # ── Deadzone centred on the LEARNED neutral position ──────
            # Convert neutral raw coords to normalised [0,1] space
            nx_norm = float(np.clip(
                np.interp(self._neutral_x, [self._min_x, safe_max_x], [0.0, 1.0]),
                0.0, 1.0))
            ny_norm = float(np.clip(
                np.interp(self._neutral_y, [self._min_y, safe_max_y], [0.0, 1.0]),
                0.0, 1.0))

            dx = abs(raw_cx - nx_norm)
            dy = abs(raw_cy - ny_norm)
            self.in_deadzone = (dx < NOSE_DEADZONE and dy < NOSE_DEADZONE)

            if not self.in_deadzone:
                target_cx = raw_cx * self.sw
                target_cy = raw_cy * self.sh
                self._ema_cx = (NOSE_EMA_ALPHA * target_cx
                                + (1.0 - NOSE_EMA_ALPHA) * self._ema_cx)
                self._ema_cy = (NOSE_EMA_ALPHA * target_cy
                                + (1.0 - NOSE_EMA_ALPHA) * self._ema_cy)
            return int(self._ema_cx), int(self._ema_cy)

        return None   # WAITING

    def _finish_calibration(self, nose_x: float, nose_y: float):
        spread_x = max(self._max_x - self._min_x, CAL_SAFETY_GAP)
        spread_y = max(self._max_y - self._min_y, CAL_SAFETY_GAP)
        self.mult_x = round(1.0 / spread_x, 1)
        self.mult_y = round(1.0 / spread_y, 1)
        safe_max_x  = max(self._max_x, self._min_x + CAL_SAFETY_GAP)
        safe_max_y  = max(self._max_y, self._min_y + CAL_SAFETY_GAP)
        self._ema_cx = float(np.interp(nose_x, [self._min_x, safe_max_x], [0, self.sw]))
        self._ema_cy = float(np.interp(nose_y, [self._min_y, safe_max_y], [0, self.sh]))
        self.state   = self.ACTIVE
        print(f"  [DSM] X spread={self._max_x-self._min_x:.4f}  mult={self.mult_x}x")
        print(f"  [DSM] Y spread={self._max_y-self._min_y:.4f}  mult={self.mult_y}x")
        print(f"  [DSM] Neutral mapped to: ({self._neutral_x:.3f}, {self._neutral_y:.3f})")
#  SYSTEM COMMANDS
# ═══════════════════════════════════════════════════════════════════════
def _is_windows() -> bool:
    return platform.system() == "Windows"

def _is_mac() -> bool:
    return platform.system() == "Darwin"

def sys_shutdown():
    if _is_windows():
        subprocess.run(["shutdown", "/s", "/t", "5"], check=False)
    elif _is_mac():
        subprocess.run(["osascript", "-e", 'tell app "System Events" to shut down'], check=False)
    else:
        subprocess.run(["shutdown", "-h", "+0"], check=False)

def sys_restart():
    if _is_windows():
        subprocess.run(["shutdown", "/r", "/t", "5"], check=False)
    elif _is_mac():
        subprocess.run(["osascript", "-e", 'tell app "System Events" to restart'], check=False)
    else:
        subprocess.run(["shutdown", "-r", "+0"], check=False)

def sys_sleep():
    if _is_windows():
        subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=False)
    elif _is_mac():
        subprocess.run(["pmset", "sleepnow"], check=False)
    else:
        subprocess.run(["systemctl", "suspend"], check=False)

def sys_lock():
    if _is_windows():
        subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=False)
    elif _is_mac():
        subprocess.run(["pmset", "displaysleepnow"], check=False)
    else:
        subprocess.run(["xdg-screensaver", "lock"], check=False)

def sys_screenshot(jarvis: "JarvisVoice", banner: "Banner") -> str:
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    fname = os.path.join(SCREENSHOT_DIR,
                         f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    img = pyautogui.screenshot()
    img.save(fname)
    return fname

def sys_brightness(direction: str):
    """Windows brightness via PowerShell WMI."""
    if not _is_windows():
        return
    cur = subprocess.run(
        ["powershell", "-Command",
         "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"],
        capture_output=True, text=True
    )
    try:
        level = int(cur.stdout.strip())
    except:
        level = 50
    new_level = min(100, level + 20) if direction == "up" else max(0, level - 20)
    subprocess.run(
        ["powershell", "-Command",
         f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
         f".WmiSetBrightness(1,{new_level})"],
        check=False
    )

def open_app(app_name: str):
    """Launch common desktop applications by name."""
    win_apps = {
        "notepad":        "notepad.exe",
        "calculator":     "calc.exe",
        "paint":          "mspaint.exe",
        "settings":       "ms-settings:",
        "file explorer":  "explorer.exe",
        "explorer":       "explorer.exe",
        "control panel":  "control.exe",
        "task manager":   "taskmgr.exe",
        "wordpad":        "wordpad.exe",
        "command prompt": "cmd.exe",
        "powershell":     "powershell.exe",
        "snipping tool":  "SnippingTool.exe",
        "camera":         "microsoft.windows.camera:",
        "calendar":       "outlookcal:",
        "mail":           "ms-mail:",
        "photos":         "ms-photos:",
        "store":          "ms-windows-store:",
        "clock":          "ms-clock:",
        "maps":           "bingmaps:",
        "weather":        "bingweather:",
    }
    mac_apps = {
        "safari": "Safari", "finder": "Finder", "notes": "Notes",
        "calculator": "Calculator", "terminal": "Terminal",
        "system preferences": "System Preferences",
    }
    if _is_windows():
        for key, cmd in win_apps.items():
            if key in app_name:
                if cmd.endswith(":"):
                    subprocess.Popen(["start", cmd], shell=True)
                else:
                    subprocess.Popen(cmd, shell=True)
                return True
    elif _is_mac():
        for key, app in mac_apps.items():
            if key in app_name:
                subprocess.Popen(["open", "-a", app])
                return True
    return False


# ═══════════════════════════════════════════════════════════════════════
#  VOICE COMMAND PROCESSOR
# ═══════════════════════════════════════════════════════════════════════
SITES = {
    "youtube":   "https://www.youtube.com",
    "google":    "https://www.google.com",
    "gmail":     "https://mail.google.com",
    "maps":      "https://maps.google.com",
    "facebook":  "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter":   "https://www.twitter.com",
    "netflix":   "https://www.netflix.com",
    "amazon":    "https://www.amazon.com",
    "wikipedia": "https://www.wikipedia.org",
    "github":    "https://www.github.com",
    "whatsapp":  "https://web.whatsapp.com",
    "spotify":   "https://open.spotify.com",
    "linkedin":  "https://www.linkedin.com",
    "reddit":    "https://www.reddit.com",
    "chatgpt":   "https://chat.openai.com",
    "claude":    "https://claude.ai",
}

def smooth_move(x: int, y: int, duration: float = 0.15):
    """Move cursor to (x,y) with smooth tweening."""
    pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)

def smooth_click(x: int = None, y: int = None):
    if x is not None:
        smooth_move(x, y, 0.10)
    pyautogui.click()

def process_voice_command(
    cmd: str,
    banner: Banner,
    jarvis: JarvisVoice,
    state: dict,          # mutable app state dict
    ripple: ClickRipple,
) -> None:
    """
    Parse and execute any voice command.
    state keys used: 'paused', 'mode', 'dictation'
    """
    cmd = cmd.strip().lower()
    ts  = time.strftime('%H:%M:%S')
    print(f"[{ts}] CMD: '{cmd}'")

    # ── Mode switching ────────────────────────────────────────
    if "hand mode" in cmd or "switch hand" in cmd:
        state['mode'] = InputMode.HAND
        banner.show("HAND MODE", C_TRACK, sub="Palm open to move cursor")
        jarvis.say("Switching to hand mode.")
        return
    if "face mode" in cmd or "switch face" in cmd:
        state['mode'] = InputMode.FACE
        banner.show("FACE MODE", C_FACE, sub="Press C to calibrate nose tracking")
        jarvis.say("Switching to face mode. Press C to calibrate.")
        return

    # ── Kill switch ───────────────────────────────────────────
    if any(w in cmd for w in ("pause tracking", "pause", "freeze", "sleep mode")):
        state['paused'] = True
        banner.show("JARVIS PAUSED", C_PAUSE, dur=2.0,
                    sub="Say 'Hey JARVIS resume' to wake")
        jarvis.say("Tracking paused. Say Hey JARVIS resume to continue.", priority=True)
        return
    if any(w in cmd for w in ("resume", "wake up", "unpause", "continue", "activate")):
        state['paused'] = False
        banner.show("JARVIS ONLINE", C_TRACK, dur=2.0)
        jarvis.say("I am back online.", priority=True)
        return

    # ── Dictation mode ────────────────────────────────────────
    if "dictation on" in cmd or "start dictation" in cmd or "start typing" in cmd:
        state['dictation'] = True
        banner.show("DICTATION ON", C_VOICE, sub="Everything you say will be typed")
        jarvis.say("Dictation mode on. Start speaking and I will type everything.")
        return
    if "dictation off" in cmd or "stop dictation" in cmd or "stop typing" in cmd:
        state['dictation'] = False
        banner.show("DICTATION OFF", C_JARVIS)
        jarvis.say("Dictation mode off.")
        return

    # ── SYSTEM: shutdown ──────────────────────────────────────
    if any(w in cmd for w in ("shutdown", "shut down", "turn off", "power off")):
        banner.show("SHUTTING DOWN in 5s", C_PAUSE, dur=5.0,
                    sub="Say 'cancel' quickly to abort")
        jarvis.say("Shutting down the computer in 5 seconds.")
        sys_shutdown()
        return
    if "cancel shutdown" in cmd or ("cancel" in cmd and "shutdown" in cmd):
        if _is_windows():
            subprocess.run(["shutdown", "/a"], check=False)
        banner.show("Shutdown cancelled", C_JARVIS)
        jarvis.say("Shutdown cancelled.")
        return

    # ── SYSTEM: restart ───────────────────────────────────────
    if any(w in cmd for w in ("restart", "reboot", "restart computer")):
        banner.show("RESTARTING in 5s", C_PAUSE, dur=5.0)
        jarvis.say("Restarting in 5 seconds.")
        sys_restart()
        return

    # ── SYSTEM: sleep ─────────────────────────────────────────
    if "sleep" in cmd and "mode" not in cmd:
        banner.show("Going to sleep", C_PAUSE, dur=2.0)
        jarvis.say("Putting the computer to sleep.")
        time.sleep(1.5)
        sys_sleep()
        return

    # ── SYSTEM: lock ─────────────────────────────────────────
    if "lock screen" in cmd or "lock computer" in cmd or "lock" in cmd:
        banner.show("Screen Locked", C_PAUSE, dur=1.5)
        jarvis.say("Locking screen.")
        sys_lock()
        return

    # ── SYSTEM: screenshot ────────────────────────────────────
    if "screenshot" in cmd or "take screenshot" in cmd or "capture screen" in cmd:
        fname = sys_screenshot(jarvis, banner)
        short = os.path.basename(fname)
        banner.show("Screenshot saved", C_JARVIS, sub=short)
        jarvis.say(f"Screenshot saved to your Pictures folder.")
        return

    # ── SYSTEM: brightness ────────────────────────────────────
    if "brightness up" in cmd or "brighter" in cmd:
        sys_brightness("up")
        banner.show("Brightness Up", C_JARVIS)
        jarvis.say("Brightness increased.")
        return
    if "brightness down" in cmd or "dimmer" in cmd or "dim screen" in cmd:
        sys_brightness("down")
        banner.show("Brightness Down", C_JARVIS)
        jarvis.say("Brightness decreased.")
        return

    # ── WINDOW: management ───────────────────────────────────
    if "minimize" in cmd or "minimise" in cmd:
        pyautogui.hotkey("win", "down")
        banner.show("Minimized", C_JARVIS)
        jarvis.say("Window minimized.")
        return
    if "maximize" in cmd or "maximise" in cmd or "full screen" in cmd:
        pyautogui.hotkey("win", "up")
        banner.show("Maximized", C_JARVIS)
        jarvis.say("Window maximized.")
        return
    if "close window" in cmd or "close app" in cmd:
        pyautogui.hotkey("alt", "f4")
        banner.show("Window Closed", C_JARVIS)
        jarvis.say("Window closed.")
        return
    if "switch window" in cmd or "alt tab" in cmd or "next window" in cmd:
        pyautogui.hotkey("alt", "tab")
        banner.show("Switch Window", C_JARVIS)
        jarvis.say("Switching window.")
        return
    if "show desktop" in cmd or "hide windows" in cmd:
        pyautogui.hotkey("win", "d")
        banner.show("Show Desktop", C_JARVIS)
        jarvis.say("Showing desktop.")
        return
    if "task manager" in cmd:
        pyautogui.hotkey("ctrl", "shift", "esc")
        banner.show("Task Manager", C_JARVIS)
        jarvis.say("Opening task manager.")
        return
    if "snap left" in cmd or "window left" in cmd:
        pyautogui.hotkey("win", "left")
        return
    if "snap right" in cmd or "window right" in cmd:
        pyautogui.hotkey("win", "right")
        return

    # ── OPEN: apps ────────────────────────────────────────────
    m = re.search(r'open\s+(.+)', cmd)
    if m:
        target = m.group(1).strip()
        # Try desktop apps first
        if open_app(target):
            banner.show(f"Opening {target.title()}", C_JARVIS)
            jarvis.say(f"Opening {target}.")
            return
        # Try websites
        for kw, url in SITES.items():
            if kw in target:
                webbrowser.open_new_tab(url)
                banner.show(f"Opening {kw.title()}", C_VOICE, sub="Browser")
                jarvis.say(f"Opening {kw}.")
                return
        # Google search fallback
        q = target.replace(" ", "+")
        webbrowser.open_new_tab(f"https://www.google.com/search?q={q}")
        banner.show(f"Searching: {target[:24]}", C_VOICE)
        jarvis.say(f"Searching for {target}.")
        return

    # ── TYPING ────────────────────────────────────────────────
    m = re.search(r'type\s+(.+)', cmd)
    if m:
        text = m.group(1)
        pyautogui.write(text, interval=0.03)
        banner.show(f"Typed: {text[:24]}", C_VOICE)
        jarvis.say("Typed.")
        return

    # ── SCROLL ────────────────────────────────────────────────
    if "scroll down" in cmd:
        pyautogui.scroll(-5)
        banner.show("Scroll Down", C_VOICE)
        jarvis.say("Scrolling down.")
        return
    if "scroll up" in cmd:
        pyautogui.scroll(5)
        banner.show("Scroll Up", C_VOICE)
        jarvis.say("Scrolling up.")
        return

    # ── CURSOR MOVE by voice ──────────────────────────────────
    move_map = {
        r'move (up|north)(?:\s+(\d+))?':    (0, -1),
        r'move (down|south)(?:\s+(\d+))?':  (0,  1),
        r'move (left|west)(?:\s+(\d+))?':   (-1, 0),
        r'move (right|east)(?:\s+(\d+))?':  ( 1, 0),
    }
    for pattern, (dx, dy) in move_map.items():
        mt = re.search(pattern, cmd)
        if mt:
            amt = int(mt.group(2)) if mt.lastindex >= 2 and mt.group(2) else 100
            cx, cy = pyautogui.position()
            smooth_move(cx + dx * amt, cy + dy * amt)
            jarvis.say(f"Moving {mt.group(1)}.")
            return

    # ── CLICKS ────────────────────────────────────────────────
    if "double click" in cmd:
        pyautogui.doubleClick()
        banner.show("Double Click", C_LCLICK)
        jarvis.say("Double click.")
        return
    if "right click" in cmd:
        pyautogui.rightClick()
        banner.show("Right Click", C_RCLICK)
        jarvis.say("Right click.")
        return
    if "click" in cmd:
        pyautogui.click()
        banner.show("Click", C_LCLICK)
        jarvis.say("Click.")
        return

    # ── KEYBOARD SHORTCUTS ────────────────────────────────────
    key_cmds = {
        "enter":      ("enter",),
        "escape":     ("escape",),
        "space":      ("space",),
        "tab":        ("tab",),
        "backspace":  ("backspace",),
        "delete":     ("delete",),
        "copy":       ("ctrl", "c"),
        "paste":      ("ctrl", "v"),
        "cut":        ("ctrl", "x"),
        "undo":       ("ctrl", "z"),
        "redo":       ("ctrl", "y"),
        "save":       ("ctrl", "s"),
        "select all": ("ctrl", "a"),
        "find":       ("ctrl", "f"),
        "new file":   ("ctrl", "n"),
        "print":      ("ctrl", "p"),
        "zoom in":    ("ctrl", "+"),
        "zoom out":   ("ctrl", "-"),
        "home":       ("ctrl", "home"),
        "end":        ("ctrl", "end"),
    }
    for phrase, keys in key_cmds.items():
        if phrase in cmd:
            pyautogui.hotkey(*keys) if len(keys) > 1 else pyautogui.press(keys[0])
            banner.show(phrase.title(), C_JARVIS)
            jarvis.say(phrase)
            return

    # ── BROWSER SHORTCUTS ─────────────────────────────────────
    browser_cmds = {
        "new tab":     ("ctrl", "t"),
        "close tab":   ("ctrl", "w"),
        "next tab":    ("ctrl", "tab"),
        "previous tab":("ctrl", "shift", "tab"),
        "back":        ("alt",  "left"),
        "forward":     ("alt",  "right"),
        "refresh":     ("f5",),
        "reload":      ("f5",),
        "address bar": ("ctrl", "l"),
        "bookmark":    ("ctrl", "d"),
        "incognito":   ("ctrl", "shift", "n"),
    }
    for phrase, keys in browser_cmds.items():
        if phrase in cmd:
            pyautogui.hotkey(*keys) if len(keys) > 1 else pyautogui.press(keys[0])
            banner.show(phrase.title(), C_JARVIS)
            jarvis.say(phrase)
            return

    # ── VOLUME / MEDIA ────────────────────────────────────────
    if "volume up" in cmd or "louder" in cmd:
        [pyautogui.press("volumeup") for _ in range(5)]
        banner.show("Volume Up", C_JARVIS); jarvis.say("Volume up."); return
    if "volume down" in cmd or "quieter" in cmd:
        [pyautogui.press("volumedown") for _ in range(5)]
        banner.show("Volume Down", C_JARVIS); jarvis.say("Volume down."); return
    if "mute" in cmd:
        pyautogui.press("volumemute")
        banner.show("Muted", C_JARVIS); jarvis.say("Muted."); return
    if "play" in cmd or "pause music" in cmd:
        pyautogui.press("playpause")
        banner.show("Play/Pause", C_JARVIS); jarvis.say("Play pause."); return
    if "next song" in cmd or "next track" in cmd:
        pyautogui.press("nexttrack")
        banner.show("Next Track", C_JARVIS); jarvis.say("Next track."); return

    # ── WHAT TIME / DATE ──────────────────────────────────────
    if "what time" in cmd or "current time" in cmd:
        t = datetime.now().strftime("%I:%M %p")
        banner.show(f"Time: {t}", C_JARVIS)
        jarvis.say(f"The time is {t}.")
        return
    if "what date" in cmd or "today's date" in cmd:
        d = datetime.now().strftime("%A, %B %d %Y")
        banner.show(d, C_JARVIS)
        jarvis.say(f"Today is {d}.")
        return

    # ── UNRECOGNISED ──────────────────────────────────────────
    banner.show(f"Unknown: {cmd[:28]}", (160, 160, 160),
                sub="Try 'Hey JARVIS help' for commands")
    jarvis.say("Sorry, I don't know that command yet.")


# ═══════════════════════════════════════════════════════════════════════
#  VOICE THREAD
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class VoiceStatus:
    state:    VoiceState    = VoiceState.IDLE
    last_cmd: str           = ""
    lock:     threading.Lock = field(default_factory=threading.Lock)

def voice_thread_fn(cmd_q: queue.Queue, vs: VoiceStatus,
                    stop: threading.Event,
                    jarvis: JarvisVoice, state: dict,
                    trigger_evt: threading.Event = None):
    try:
        import speech_recognition as sr
    except ImportError:
        print("[VOICE] Missing. Run: py -3.11 -m pip install SpeechRecognition pyaudio")
        return

    rec = sr.Recognizer()
    rec.energy_threshold         = 300
    rec.dynamic_energy_threshold = True
    rec.dynamic_energy_adjustment_damping = 0.15  # adapts faster to outdoor noise
    rec.pause_threshold        = 0.6

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[VOICE] No mic: {e}"); return

    print("[VOICE] Calibrating mic (2s)…")
    with mic as src:
        rec.adjust_for_ambient_noise(src, duration=1.0)
    print(f"[VOICE] Ready — say '{WAKE_WORD.upper()}' to activate.")
    jarvis.say("JARVIS online. Say Hey JARVIS for a command.")

    while not stop.is_set():
        # Voice is always active — works in HAND and FACE mode

        # ── Dictation mode: continuous speech-to-type ──────────
        if state.get('dictation'):
            with vs.lock:
                vs.state = VoiceState.DICTATING
            try:
                with mic as src:
                    audio = rec.listen(src, timeout=3,
                                       phrase_time_limit=10)
                text = rec.recognize_google(audio)
                pyautogui.write(text + " ", interval=0.02)
                print(f"[DICTATION] '{text}'")
                with vs.lock:
                    vs.last_cmd = f"[dictation] {text[:40]}"
            except Exception:
                pass
            continue

        # ── Normal: wait for wake word ─────────────────────────
        # ── Normal: wait for wake word ─────────────────────────
        with vs.lock:
            vs.state = VoiceState.IDLE

        # V-key already pressed — skip listen entirely
        if trigger_evt is not None and trigger_evt.is_set():
            trigger_evt.clear()
            print("[VOICE] V-key instant trigger ✓")
            with vs.lock:
                vs.state = VoiceState.LISTENING
            jarvis.say("Yes?", priority=True)
            try:
                with mic as src:
                    audio = rec.listen(src, timeout=LISTEN_TIMEOUT,
                                       phrase_time_limit=8)
                with vs.lock:
                    vs.state = VoiceState.THINKING
                cmd = rec.recognize_google(audio).lower().strip()
                with vs.lock:
                    vs.last_cmd = cmd
                    vs.state    = VoiceState.HEARD
                cmd_q.put(cmd)
                time.sleep(0.3)
                with vs.lock:
                    vs.state = VoiceState.IDLE
            except sr.WaitTimeoutError:
                jarvis.say("No speech detected. Try again.")
            except sr.UnknownValueError:
                jarvis.say("Couldn't understand. Speak louder or closer.")
            except sr.RequestError:
                jarvis.say("Network error. Check internet.")
            except Exception:
                jarvis.say("Something went wrong.")
                with vs.lock:
                    vs.state = VoiceState.IDLE
            continue

        try:
            with mic as src:
                audio = rec.listen(src, timeout=0.5, phrase_time_limit=4)
            heard = rec.recognize_google(audio).lower()
        except Exception:
            continue

        if WAKE_WORD not in heard:
            continue

        # Wake word hit
        with vs.lock:
            vs.state = VoiceState.LISTENING
        jarvis.say("Yes?", priority=True)
        print("[VOICE] Wake word ✓")

        try:
            with mic as src:
                audio = rec.listen(src, timeout=LISTEN_TIMEOUT,
                                   phrase_time_limit=10)
            with vs.lock:
                vs.state = VoiceState.THINKING
            cmd = rec.recognize_google(audio).lower().strip()
            print(f"[VOICE] '{cmd}'")
            with vs.lock:
                vs.last_cmd = cmd
                vs.state    = VoiceState.HEARD
            cmd_q.put(cmd)
            time.sleep(0.3)
            with vs.lock:
                vs.state = VoiceState.IDLE
        except Exception:
            jarvis.say("I didn't catch that.")
            with vs.lock:
                vs.state = VoiceState.IDLE


# ═══════════════════════════════════════════════════════════════════════
#  DRAWING
# ═══════════════════════════════════════════════════════════════════════
def draw_hand_skeleton(frame, hr: HandResult, gesture: HandGesture):
    with hr.lock:
        lm = list(hr.lm_px)
    if not lm: return

    col = {
        HandGesture.TRACKING: C_TRACK,
        HandGesture.FIST:     C_FREEZE,
        HandGesture.PINCH_L:  C_LCLICK,
        HandGesture.PINCH_R:  C_RCLICK,
        HandGesture.SCROLL:   C_SCROLL,
    }.get(gesture, (150, 150, 150))

    for a, b in HAND_BONES:
        if a < len(lm) and b < len(lm):
            cv2.line(frame, lm[a], lm[b], col, 2)
    for i, pt in enumerate(lm):
        r = 7 if i in TIPS else 3
        cv2.circle(frame, pt, r, col, -1)
        cv2.circle(frame, pt, r + 1, (0, 0, 0), 1)

    if gesture == HandGesture.TRACKING and len(lm) > INDEX_TIP:
        ix, iy = lm[INDEX_TIP]
        cv2.circle(frame, (ix, iy), 16, C_TRACK, 2)
        cv2.line(frame, (ix-20, iy), (ix+20, iy), C_TRACK, 1)
        cv2.line(frame, (ix, iy-20), (ix, iy+20), C_TRACK, 1)

    labels = {HandGesture.TRACKING: "TRACKING",
              HandGesture.FIST: "FROZEN",
              HandGesture.PINCH_L: "L-CLICK",
              HandGesture.PINCH_R: "R-CLICK",
              HandGesture.SCROLL: "SCROLL"}
    if gesture in labels and lm:
        cv2.putText(frame, labels[gesture], (lm[0][0] - 50, lm[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)


def draw_face_overlay(frame, nose_px, smooth_px, dz_rect,
                      is_active, trail, face_lms, ear_val, blink_st, w, h):
    x1, y1, x2, y2 = dz_rect
    col = C_DZ_ACT if is_active else C_DZ
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), col, -1)
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)
    mx, my = (x1 + x2)//2, (y1 + y2)//2
    cv2.line(frame, (mx-12, my), (mx+12, my), col, 1)
    cv2.line(frame, (mx, my-12), (mx, my+12), col, 1)

    for i in range(1, len(trail)):
        a = i / len(trail)
        cv2.line(frame, trail[i-1], trail[i],
                 (int(C_FACE[2]*a), int(C_FACE[1]*a), int(C_FACE[0]*a)), 1)

    cv2.circle(frame, nose_px,   5, (255,255,0), -1)
    cv2.circle(frame, nose_px,   9, (255,255,0),  1)
    cv2.circle(frame, smooth_px, 4, (255,255,255), -1)

    if face_lms:
        ecol = (255, 200, 0) if blink_st else (120, 220, 120)
        for idx in L_EYE_IDX + R_EYE_IDX:
            cv2.circle(frame,
                       (int(face_lms[idx].x * w), int(face_lms[idx].y * h)),
                       2, ecol, -1)
    cv2.putText(frame, f"EAR:{ear_val:.2f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1)


def draw_face_overlay_dsm(frame, nose_px, trail, face_lms,
                          ear_val, blink_st, mouth_st,
                          dsm: "DSMCalibrator", w, h):
    """
    Overlay for the new DSM nose-tip face mode.
    Shows: nose dot + trail, eye dots, calibration state / multipliers.
    """
    # ── Trail ─────────────────────────────────────────────────────────
    for i in range(1, len(trail)):
        a = i / max(len(trail), 1)
        cv2.line(frame, trail[i-1], trail[i],
                 (int(C_FACE[2]*a), int(C_FACE[1]*a), int(C_FACE[0]*a)), 1)

    # ── Nose dot — colour shows deadzone state ────────────────────────
    if dsm.state == DSMCalibrator.ACTIVE:
        if dsm.in_deadzone:
            dot_col = (0, 255, 255)   # yellow-cyan = FROZEN / safe to click
            cv2.circle(frame, nose_px, 14, dot_col, 2)   # outer freeze ring
            cv2.putText(frame, "STEADY", (nose_px[0] + 16, nose_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, dot_col, 1)
        else:
            dot_col = (0, 255, 140)   # green = moving
    else:
        dot_col = (255, 255, 0)       # yellow = not yet calibrated
    cv2.circle(frame, nose_px, 6,  dot_col, -1)
    cv2.circle(frame, nose_px, 10, dot_col,  1)

    # ── Eye landmarks ─────────────────────────────────────────────────
    if face_lms:
        ecol = (255, 200, 0) if blink_st else (120, 220, 120)
        for idx in L_EYE_IDX + R_EYE_IDX:
            cv2.circle(frame,
                       (int(face_lms[idx].x * w), int(face_lms[idx].y * h)),
                       2, ecol, -1)
        # Mouth corners highlight when open
        mcol = C_LCLICK if mouth_st else (100, 100, 100)
        for idx in [MOUTH_TOP, MOUTH_BOT, MOUTH_L, MOUTH_R]:
            cv2.circle(frame,
                       (int(face_lms[idx].x * w), int(face_lms[idx].y * h)),
                       3, mcol, -1)

    # ── DSM state panel (top-left info) ───────────────────────────────
    if dsm.state == DSMCalibrator.WAITING:
        cv2.putText(frame, "Press C to calibrate nose tracking",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_CAL, 1)

    elif dsm.state == DSMCalibrator.CALIBRATING:
        # Dark box
        ov = frame.copy()
        cv2.rectangle(ov, (0, 38), (w, 90), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
        # Progress bar
        frac = max(0.0, 1.0 - dsm.countdown / CAL_DURATION)
        cv2.rectangle(frame, (10, 80), (w - 10, 88), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 80), (10 + int((w - 20) * frac), 88), C_CAL, -1)
        cv2.putText(frame,
                    f"CALIBRATING — move nose to corners!  ({dsm.countdown:.1f}s)",
                    (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, C_CAL, 2)

    elif dsm.state == DSMCalibrator.ACTIVE:
        # Compact multiplier badge
        cv2.putText(frame, f"DSM  X:{dsm.mult_x}x  Y:{dsm.mult_y}x",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_CAL, 1)

    # ── Diagnostics ───────────────────────────────────────────────────
    cv2.putText(frame, f"EAR:{ear_val:.2f}", (10, 80 if dsm.state != DSMCalibrator.CALIBRATING else 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # ── Voice-busy suppression badge ─────────────────────────────────
    # (drawn by caller when needed — placeholder for alignment)


def draw_status_panel(frame, mode: InputMode, gesture,
                      vs: VoiceStatus,
                      paused: bool, dictation: bool):
    fh, fw = frame.shape[:2]

    # ── Mode badge (top left) ─────────────────────────────────
    mode_col = {
        InputMode.HAND:  C_TRACK,
        InputMode.FACE:  C_FACE,
    }.get(mode, C_JARVIS)
    if paused:
        mode_col = C_PAUSE

    mode_label = "PAUSED" if paused else mode.value
    if dictation:
        mode_label = "DICTATING"
        mode_col   = C_VOICE

    cv2.putText(frame, f"[{mode_label}]", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mode_col, 2)

    # ── Voice status (top right) — always shown ───────────────
    with vs.lock:
        vstate = vs.state
        vlast  = vs.last_cmd

    vstxt = {
        VoiceState.IDLE:      ("Say 'Hey JARVIS'", (110,110,110)),
        VoiceState.LISTENING: ("LISTENING...",      C_VOICE),
        VoiceState.THINKING:  ("Thinking...",       C_JARVIS),
        VoiceState.DICTATING: ("DICTATING...",      C_VOICE),
        VoiceState.HEARD:     ("✓ HEARD",           C_HEARD),
    }[vstate]
    cv2.putText(frame, vstxt[0], (fw - 210, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, vstxt[1], 1)
    if vstate in (VoiceState.LISTENING, VoiceState.DICTATING, VoiceState.HEARD):
        pulse = int(abs(np.sin(time.monotonic() * 6)) * 8) + 5
        col_p = C_HEARD if vstate == VoiceState.HEARD else C_VOICE
        cv2.circle(frame, (fw - 15, 18), pulse, col_p, -1)

    # ── Last command (bottom) ─────────────────────────────────────────
    if vlast:
        label_col = C_HEARD if vstate == VoiceState.HEARD else (150, 150, 150)
        prefix    = "✓ Heard: " if vstate == VoiceState.HEARD else '"'
        suffix    = "" if vstate == VoiceState.HEARD else '"'
        cv2.putText(frame, f'{prefix}{vlast[:36]}{suffix}', (10, fh - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_col, 1)

    # ── Legend (bottom right compact) ────────────────────────
    if mode == InputMode.HAND:
        items = [("Palm=move",    C_TRACK),
                 ("Fist=freeze",  C_FREEZE),
                 ("ThumbUp=click",C_LCLICK),   # updated from Pinch
                 ("Peace=scroll", C_SCROLL)]
    else:  # FACE
        items = [("MouthHold=lclick", C_LCLICK),   # updated: hold not instant
                 ("Wink=lclick",      C_LCLICK),   # updated: wink not dbl-blink
                 ("LongBlink=rclick", C_RCLICK),
                 ("C=calibrate",      C_CAL)]

    for i, (txt, col) in enumerate(items):
        cv2.putText(frame, txt, (fw - 200, fh - 12 - i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)

    # ── Key hints (bottom left) ───────────────────────────────
    cv2.putText(frame, "Q:quit  M:mode  P:pause  H:hand  F:face  C:calibrate  [VOICE ALWAYS ON]",
                (10, fh - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120,120,120), 1)


def draw_paused_overlay(frame):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (fw, fh), (0,0,80), -1)
    cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)
    msg = "JARVIS PAUSED"
    sz, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
    cv2.putText(frame, msg, ((fw-sz[0])//2, fh//2-10),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, C_PAUSE, 2)
    cv2.putText(frame, "Say 'Hey JARVIS resume'  or press P",
                ((fw-sz[0])//2-10, fh//2+34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200,200,220), 1)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    SW, SH = pyautogui.size()
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # ── Camera ───────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    camera_ok = cap.isOpened()
    if not camera_ok:
        print("[WARN] No camera found — starting in VOICE ONLY mode.")

    # ── App state (shared with voice processor) ───────────────
    state = {
        'mode':      InputMode.HAND,   # voice always active in all modes
        'paused':    False,
        'dictation': False,
    }

    # ── Objects ───────────────────────────────────────────────
    cursor_ema  = EMA(CURSOR_SMOOTHING)
    face_ema    = EMA(FACE_SMOOTHING)
    banner      = Banner()
    ripple      = ClickRipple()
    hr          = HandResult()
    vs          = VoiceStatus()
    jarvis      = JarvisVoice()
    blink_det   = BlinkDetector()
    wink_det    = WinkDetector()      # CHANGE 2: wink detector added
    mouth_det   = MouthClickDetector()
    dsm         = DSMCalibrator(SW, SH)
    cmd_q: queue.Queue = queue.Queue()
    voice_trigger_evt = threading.Event()

    last_click_t  = [0.0]
    prev_gesture  = [HandGesture.NONE]
    scroll_ref_y  = [None]
    face_trail: list = []
    TRAIL_MAX     = 22
    dsm_prev_state = [DSMCalibrator.WAITING]   # track state transitions

    # ── Thread infrastructure ─────────────────────────────────
    shared_frame: list = [None]
    frame_lock         = threading.Lock()
    stop_evt           = threading.Event()
    hand_enabled       = [True]

    ht = threading.Thread(target=hand_thread_fn,
                          args=(shared_frame, frame_lock, hr,
                                stop_evt, hand_enabled),
                          daemon=True, name="HandThread")
    if camera_ok:
        ht.start()

    vt = threading.Thread(target=voice_thread_fn,
                      args=(cmd_q, vs, stop_evt,
                            jarvis, state, voice_trigger_evt),
                      daemon=True, name="VoiceThread")
    vt.start()

    # ── Face Mesh (used in FACE mode, on main thread) ─────────
    mp_fm = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) if camera_ok else None

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          JOYSTEP AT — ULTIMATE EDITION  v5.0             ║")
    print("║  2 modes: HAND  |  FACE  — voice active in both          ║")
    print("║  HAND: ThumbUp=LeftClick  Fist=Freeze  Peace=Scroll      ║")
    print("║  FACE: Wink=LeftClick  LongBlink=RightClick  MouthHold=Click ║")
    print("║  Say 'Hey JARVIS' anytime for voice commands             ║")
    print("║  Keys: Q=quit  M=mode  P=pause  H=hand  F=face  C=cal   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    while True:
        # ── Get frame ────────────────────────────────────────
        if camera_ok:
            ok, frame = cap.read()
            if not ok: continue
            frame = cv2.flip(frame, 1)
        else:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera — voice commands active",
                        (20, FRAME_H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, C_VOICE, 1)

        fh, fw = frame.shape[:2]

        if camera_ok:
            with frame_lock:
                shared_frame[0] = frame.copy()

        # ── Drain voice commands ──────────────────────────────
        while not cmd_q.empty():
            try:
                cmd = cmd_q.get_nowait()
                process_voice_command(cmd, banner, jarvis, state, ripple)
            except queue.Empty:
                break

        now = time.monotonic()

        # ══════════════════════════════════════════════════════
        #  HAND MODE
        # ══════════════════════════════════════════════════════
        if state['mode'] == InputMode.HAND and camera_ok and not state['paused']:
            with hr.lock:
                gesture  = hr.gesture
                idx_norm = hr.index_norm
                lm_px    = list(hr.lm_px)

            if gesture == HandGesture.TRACKING:
                scroll_ref_y[0] = None
                mg  = (1.0 - FRAME_MARGIN) / 2.0
                nx  = max(0.0, min((idx_norm[0] - mg) / FRAME_MARGIN, 1.0))
                ny  = max(0.0, min((idx_norm[1] - mg) / FRAME_MARGIN, 1.0))
                sx, sy = cursor_ema.update(nx, ny)
                pyautogui.moveTo(int(sx * SW), int(sy * SH),
                                 duration=0.05, tween=pyautogui.easeOutQuad)

            elif gesture == HandGesture.FIST:
                scroll_ref_y[0] = None
                cursor_ema.reset()

            elif gesture == HandGesture.PINCH_L:
                # ── CHANGE 1: now triggered by thumb-open gesture ─────
                scroll_ref_y[0] = None
                cursor_ema.reset()
                if (prev_gesture[0] != HandGesture.PINCH_L and
                        (now - last_click_t[0]) >= PINCH_COOLDOWN):
                    last_click_t[0] = now
                    cx, cy = pyautogui.position()
                    pyautogui.click(cx, cy)
                    banner.show("Thumb Up — Left Click", C_LCLICK)
                    jarvis.say("Click")
                    if lm_px and len(lm_px) > INDEX_TIP:
                        ripple.add(lm_px[INDEX_TIP][0], lm_px[INDEX_TIP][1], C_LCLICK)

            elif gesture == HandGesture.PINCH_R:
                scroll_ref_y[0] = None
                cursor_ema.reset()
                if (prev_gesture[0] != HandGesture.PINCH_R and
                        (now - last_click_t[0]) >= PINCH_COOLDOWN):
                    last_click_t[0] = now
                    pyautogui.rightClick()
                    banner.show("Right Click", C_RCLICK)
                    jarvis.say("Right click")
                    if lm_px and len(lm_px) > INDEX_TIP:
                        ripple.add(lm_px[INDEX_TIP][0], lm_px[INDEX_TIP][1], C_RCLICK)

            elif gesture == HandGesture.SCROLL:
                cur_y = idx_norm[1]
                if scroll_ref_y[0] is None:
                    scroll_ref_y[0] = cur_y
                else:
                    delta = (cur_y - scroll_ref_y[0]) * SCROLL_SENS
                    if abs(delta) > 50:
                        pyautogui.scroll(-int(delta))
                        scroll_ref_y[0] = cur_y
            else:
                scroll_ref_y[0] = None

            prev_gesture[0] = gesture
            draw_hand_skeleton(frame, hr, gesture)

        # ══════════════════════════════════════════════════════
        #  FACE MODE  — DSM nose-tip absolute mapping
        # ══════════════════════════════════════════════════════
        elif state['mode'] == InputMode.FACE and camera_ok and not state['paused']:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            ear_v    = 0.30
            blink_st = False
            mouth_st = False
            face_lms = None
            nose_px  = (fw//2, fh//2)

            # Check voice state ONCE per frame outside landmark block
            with vs.lock:
                _vs_now = vs.state
            _voice_busy = _vs_now in (VoiceState.LISTENING, VoiceState.THINKING,
                                      VoiceState.HEARD)

            if result.multi_face_landmarks:
                lms      = result.multi_face_landmarks[0].landmark
                face_lms = lms

                raw_nx = lms[NOSE_TIP].x
                raw_ny = lms[NOSE_TIP].y
                nose_px = (int(raw_nx * fw), int(raw_ny * fh))

                # ── DSM: update calibrator & move cursor ──────
                pos = dsm.update(raw_nx, raw_ny)
                if pos is not None:
                    pyautogui.moveTo(pos[0], pos[1])

                face_trail.append(nose_px)
                if len(face_trail) > TRAIL_MAX: face_trail.pop(0)

                # ── Blink / wink detection ────────────────────
                el  = calc_ear(lms, L_EYE_IDX, fw, fh)
                er  = calc_ear(lms, R_EYE_IDX, fw, fh)
                ear_v = (el + er) / 2.0

                blink_event = blink_det.update(ear_v)         # long blink → right click
                wink_event  = wink_det.update(el, er)         # CHANGE 2: wink → left click
                blink_st    = blink_event != "none" or wink_event != "none"

                # _voice_busy already set above — suppress clicks during voice
                if not _voice_busy:
                    # ── CHANGE 2: wink fires left click ──────
                    if wink_event == "left_click":
                        pyautogui.click()
                        banner.show("Wink — Left Click", C_LCLICK)
                        jarvis.say("Click")
                        ripple.add(nose_px[0], nose_px[1], C_LCLICK)
                    # Long blink still fires right click
                    elif blink_event == "right_click":
                        pyautogui.rightClick()
                        banner.show("Long Blink — Right Click", C_RCLICK)
                        jarvis.say("Right click")
                        ripple.add(nose_px[0], nose_px[1], C_RCLICK)

                # ── CHANGE 3: Mouth-hold → LEFT CLICK ────────
                # MouthClickDetector now requires MOUTH_HOLD_DUR (0.4s) of
                # sustained open before firing — brief speech won't trigger it
                mouth_event = mouth_det.update(lms, fh)
                mouth_st    = mouth_event == "left_click"
                if mouth_st and not _voice_busy:
                    pyautogui.click()
                    banner.show("Mouth Hold — Left Click", C_LCLICK)
                    jarvis.say("Click")
                    ripple.add(nose_px[0], nose_px[1], C_LCLICK)

            else:
                face_trail.clear()

            # ── Draw face overlay ─────────────────────────────
            draw_face_overlay_dsm(frame, nose_px, face_trail, face_lms,
                                  ear_v, blink_st, mouth_st, dsm, fw, fh)

            # ── Voice-busy: show suppression notice ───────────
            if _voice_busy:
                ov = frame.copy()
                cv2.rectangle(ov, (0, fh - 32), (fw, fh), (0, 0, 80), -1)
                cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
                cv2.putText(frame,
                            "MIC ACTIVE — mouth & blink clicks paused",
                            (10, fh - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_VOICE, 1)

            # ── Announce when calibration just completed ───────
            if (dsm_prev_state[0] == DSMCalibrator.CALIBRATING
                    and dsm.state == DSMCalibrator.ACTIVE):
                banner.show(
                    f"DSM ACTIVE  X:{dsm.mult_x}x  Y:{dsm.mult_y}x",
                    C_CAL, dur=3.5,
                    sub="Wink=lclick  |  MouthHold=lclick  |  LongBlink=rclick"
                )
                jarvis.say(
                    f"Calibration complete. X multiplier {dsm.mult_x}. "
                    f"Y multiplier {dsm.mult_y}. You are now in control."
                )
            dsm_prev_state[0] = dsm.state

        # ── Ripple + banner + status ──────────────────────────
        ripple.draw(frame)

        if state['paused']:
            draw_paused_overlay(frame)

        draw_status_panel(frame,
                          state['mode'],
                          prev_gesture[0] if state['mode'] == InputMode.HAND
                          else None,
                          vs,
                          state['paused'], state['dictation'])
        banner.draw(frame)

        cv2.imshow("JOYSTEP AT — Ultimate Edition", frame)

        # ── Keys ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle between HAND and FACE
            state['mode'] = (InputMode.FACE
                             if state['mode'] == InputMode.HAND
                             else InputMode.HAND)
            jarvis.say(f"{state['mode'].value} mode.")
        elif key == ord('v') or key == ord('V'):
            voice_trigger_evt.set()
            banner.show("JARVIS LISTENING", C_VOICE, dur=1.0,
                        sub="Speak your command now")
        elif key == ord('p'):
            state['paused'] = not state['paused']
            jarvis.say("Paused." if state['paused'] else "Resumed.")
        elif key == ord('h'):
            state['mode'] = InputMode.HAND
            jarvis.say("Hand mode.")
        elif key == ord('f'):
            state['mode'] = InputMode.FACE
            jarvis.say("Face mode. Press C to calibrate.")
        elif key == ord('c') or key == ord('C'):
            if state['mode'] == InputMode.FACE and not state['paused']:
                # Snapshot current nose as the neutral resting position
                _nx, _ny = 0.5, 0.5
                if face_mesh:
                    _tmp = face_mesh.process(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if _tmp.multi_face_landmarks:
                        _lm = _tmp.multi_face_landmarks[0].landmark
                        _nx, _ny = _lm[NOSE_TIP].x, _lm[NOSE_TIP].y
                dsm.start_calibration(_nx, _ny)
                blink_det   = BlinkDetector()
                wink_det    = WinkDetector()    # CHANGE 2: reset wink on recalibrate
                mouth_det   = MouthClickDetector()
                face_trail.clear()
                banner.show("CALIBRATING — move nose to all corners!", C_CAL, dur=5.2)
                jarvis.say("Calibrating. Move your nose to all four corners now.")
                print("  [DSM] Calibration started.")

    # ── Shutdown ──────────────────────────────────────────────
    stop_evt.set()
    if face_mesh: face_mesh.close()
    jarvis.say("Shutting down. Goodbye.")
    time.sleep(1.2)
    jarvis.stop()
    if camera_ok: cap.release()
    cv2.destroyAllWindows()
    print("JARVIS offline.")


if __name__ == "__main__":
    main()