"""LEAP Hand Interactive Booth Kiosk Application.

A touchscreen & keyboard-friendly exhibition kiosk UI supporting:
1. Real-time Teleoperation (with C / F calibration and A / D arming).
2. Interactive Rock-Paper-Scissors Game (with 3-2-1 countdown, robot random move, gesture detection, and live scoreboard).
3. Gesture & Showcase Demo.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from cube_reorient_controller import (
    DEFAULT_POLICY_PATH as DEFAULT_REORIENT_POLICY_PATH,
    CubeReorientController,
)
from hand_angles import (
    ANGLE_NAMES,
    calculate_leap_control_angles,
)
from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController
from neutral_calibration import NeutralCalibration
from one_euro_filter import OneEuroFilter
from rps.gesture import classify_rps_gesture
from rps.moves import MOVE_NAMES
from rps.postures import get_posture
from rps.rounds import human_result
from webcam_hand_tracking import (
    HAND_CONNECTIONS,
    draw_hand,
)

DEFAULT_CALIB_PATH = Path("calibration/neutral_calibration.json")
DEFAULT_MOTOR_CALIB_PATH = Path("calibration/hardware_motors.yaml")
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")
DEFAULT_LOGO_PATH = Path("assets/shape_mark.png")

# Colours in BGR, taken from the SHAPE site (snu-shape.com): a light ground,
# one blue that carries the brand, and a crimson for the things that need to
# stand apart from it. Contrast is against the white card, measured, so the
# muted greys stay readable rather than merely looking calm.
COLOR_BG = (251, 248, 247)        # #f7f8fb  page ground
COLOR_CARD_BG = (255, 255, 255)   # #ffffff  cards sit above the ground
COLOR_CARD_BORDER = (234, 228, 224)  # #e0e4ea  hairline, not a frame
COLOR_TEXT_MAIN = (26, 36, 52)    # #34241a  ink, 14.9:1
COLOR_TEXT_MUTED = (127, 113, 105)   # #69717f  secondary, 4.9:1
COLOR_PRIMARY = (243, 85, 40)     # #2855f3  brand blue, 5.7:1
COLOR_SECONDARY = (70, 36, 182)   # #b62446  crimson, 6.3:1
COLOR_SUCCESS = (75, 127, 26)     # #1a7f4b  green, 5.0:1
COLOR_DANGER = (70, 36, 182)      # #b62446  the crimson again -- failure is
                                  # the loudest thing on a light page
COLOR_WARNING = (0, 106, 178)     # #b26a00  amber, 4.2:1 -- large text only
COLOR_ROBOT = (124, 56, 20)       # #14387c  deep blue, the gradient's dark end
COLOR_HUMAN = (70, 36, 182)       # #b62446  crimson, opposite the robot

# Surfaces derived from the palette, so a screen never invents its own grey.
COLOR_CARD_HOVER = (250, 245, 240)   # a card under the cursor
COLOR_SUNKEN = (247, 242, 238)       # wells: viewports, inset panels
COLOR_DISABLED_BG = (245, 243, 242)
COLOR_DISABLED_TEXT = (188, 184, 180)
COLOR_SHADOW = (52, 24, 12)          # #0c1834, the site's blue-cast shadow


def get_hand_posture(name: str) -> np.ndarray:
    """Return 16-element joint posture degrees."""
    n = name.lower()
    if n in ("neutral", "paper"):
        return get_posture("paper").copy()
    if n == "rock":
        return get_posture("rock").copy()
    if n == "scissors":
        return get_posture("scissors").copy()
    if n in ("thumbs_up", "thumb_up"):
        # Fist with thumb pointing up
        deg = get_posture("rock").copy()
        deg[12] = 0.0  # thumb_cmc_side
        deg[13] = 0.0  # thumb_cmc_flex
        deg[14] = 0.0  # thumb_mcp_flex
        deg[15] = 0.0  # thumb_ip_flex
        return deg
    if n in ("ok_sign", "ok"):
        # Index & Thumb pinch, Middle/Ring extended
        deg = np.zeros(16, dtype=np.float64)
        deg[0] = 0.0   # index_mcp_side
        deg[1] = 45.0  # index_mcp_flex
        deg[2] = 65.0  # index_pip_flex
        deg[3] = 45.0  # index_dip_flex
        deg[12] = 20.0 # thumb_cmc_side
        deg[13] = 45.0 # thumb_cmc_flex
        deg[14] = 40.0 # thumb_mcp_flex
        deg[15] = 20.0 # thumb_ip_flex
        return deg
    if n == "pointing":
        # Index extended, others closed
        deg = get_posture("rock").copy()
        deg[0] = 0.0
        deg[1] = 0.0
        deg[2] = 0.0
        deg[3] = 0.0
        return deg
    if n in ("rock_on", "rockon", "love"):
        # Index & Ring & Thumb extended, Middle closed
        deg = np.zeros(16, dtype=np.float64)
        deg[4] = 0.0   # middle_mcp_side
        deg[5] = 75.0  # middle_mcp_flex
        deg[6] = 85.0  # middle_pip_flex
        deg[7] = 65.0  # middle_dip_flex
        return deg
    return get_posture(name).copy()


class MediaPipeTracker:
    """Wrapper around MediaPipe HandLandmarker for video frame inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.landmarker = None
        self.last_timestamp_ms = 0
        if model_path.is_file():
            try:
                import mediapipe as mp

                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(model_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                    options
                )
            except Exception as e:
                print(f"[BOOTH WARN] Failed to load MediaPipe HandLandmarker: {e}")

    def process_frame(self, bgr_frame: np.ndarray | None) -> list[Any]:
        if self.landmarker is None or bgr_frame is None:
            return []
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        now_ms = time.perf_counter_ns() // 1_000_000
        now_ms = max(now_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = now_ms
        res = self.landmarker.detect_for_video(mp_image, now_ms)
        return res.hand_landmarks or []


class AppScreen(Enum):
    HOME = auto()
    TELEOP = auto()
    RPS = auto()
    SHOWCASE = auto()
    REORIENT = auto()


class RpsState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    SHOOT = auto()
    RESULT = auto()


# --- drawing primitives -------------------------------------------------------
#
# OpenCV draws squared-off rectangles and nothing else, so the rounded corners
# and soft shadows the design leans on have to be built here. Every screen goes
# through these rather than calling cv2.rectangle directly, which is also what
# keeps the corner radius and the shadow the same everywhere.

RADIUS_CARD = 12
SHADOW_DROP = 3   # a shadow sits slightly below what casts it
RADIUS_WELL = 10

# Type scale. Thirteen different sizes had accumulated across the screens; five
# is enough to say what a piece of text is, and a fixed set is what makes two
# screens look like the same application.
TYPE_DISPLAY = 1.5    # a single number the room should read
TYPE_TITLE = 0.75     # screen and card titles
TYPE_SUBTITLE = 0.6   # section labels
TYPE_BODY = 0.5       # ordinary text
TYPE_CAPTION = 0.42   # keys, units, footnotes


def rounded_rect(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    radius: int = RADIUS_CARD,
    thickness: int = -1,
) -> None:
    """A filled or stroked rectangle with rounded corners.

    Composed from two overlapping rectangles and four corner arcs, which is the
    only way to get one out of OpenCV. A radius past half the shorter side would
    make the corners overlap, so it is clamped rather than left to fold in.
    """
    x, y, w, h = rect
    radius = max(0, min(radius, w // 2, h // 2))
    if radius == 0:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)
        return

    x2, y2 = x + w, y + h
    centres = (
        ((x + radius, y + radius), 180),
        ((x2 - radius, y + radius), 270),
        ((x2 - radius, y2 - radius), 0),
        ((x + radius, y2 - radius), 90),
    )
    if thickness < 0:
        cv2.rectangle(canvas, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(canvas, (x, y + radius), (x2, y2 - radius), color, -1)
        for centre, start in centres:
            cv2.ellipse(canvas, centre, (radius, radius), start, 0, 90, color, -1, cv2.LINE_AA)
        return

    cv2.line(canvas, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    for centre, start in centres:
        cv2.ellipse(canvas, centre, (radius, radius), start, 0, 90, color, thickness, cv2.LINE_AA)


def pill_rect(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    thickness: int = -1,
) -> None:
    """A fully rounded rectangle -- the site's 999px radius, for buttons."""
    rounded_rect(canvas, rect, color, rect[3] // 2, thickness)


@lru_cache(maxsize=32)
def _shadow_on_ground(
    width: int, height: int, radius: int, spread: int, strength: float,
    ground: tuple[int, int, int],
) -> np.ndarray:
    """The finished pixels of a shadow cast onto a known flat colour.

    Most shadows in the booth fall on the page and nothing else -- the viewport
    is the same rectangle in the same place on every screen, over the same
    ground, every frame. Compositing it once and blitting the result turns the
    largest of them, 800x530, from about 3.5 ms a frame into a memory copy.
    """
    mask = _shadow_mask(width, height, radius, spread, strength)
    base = np.empty((mask.shape[0], mask.shape[1], 3), np.float32)
    base[:] = ground
    return (base * (1.0 - mask) + np.asarray(COLOR_SHADOW, np.float32) * mask).astype(
        np.uint8
    )


@lru_cache(maxsize=64)
def _shadow_mask(
    width: int, height: int, radius: int, spread: int, strength: float
) -> np.ndarray:
    """The alpha of one shadow shape, built once and reused.

    Concentric rounded rectangles at a low alpha stand in for a blur. Only a
    handful of distinct button and card shapes exist, so building the falloff
    per shape and caching it turns a per-frame cost into a startup one.
    """
    pad = max(2, spread)
    mask = np.zeros((height + 2 * pad + SHADOW_DROP, width + 2 * pad), np.float32)
    layers = max(1, spread // 2)
    for index in range(layers, 0, -1):
        grow = index * 2
        band = (pad - grow, pad - grow + SHADOW_DROP, width + 2 * grow, height + 2 * grow)
        layer = np.zeros_like(mask)
        rounded_rect(layer, band, 1.0, radius + grow, -1)
        mask += layer * (strength / layers)
    return np.clip(mask, 0.0, 1.0)[:, :, None]


def soft_shadow(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    radius: int = RADIUS_CARD,
    spread: int = 10,
    strength: float = 0.10,
    ground: tuple[int, int, int] | None = None,
) -> None:
    """Lay a soft blue-cast shadow under a rectangle.

    Pass ``ground`` when the shadow falls on a known flat colour and nothing
    else; the composited result is then cached rather than recomputed.

    On a light ground a card needs somewhere to sit, and a hard border makes it
    look boxed rather than raised.

    Only the pixels the shadow actually covers are touched. Blending the whole
    canvas once per shadow -- which is what this did first -- cost around 20 ms
    a frame on the screens carrying a dozen buttons, and the kiosk visibly
    stuttered; the region is a few hundred pixels across.
    """
    x, y, w, h = rect
    if w <= 0 or h <= 0 or not SHADOWS_ENABLED:
        return
    pad = max(2, spread)
    mask = _shadow_mask(w, h, radius, spread, strength)

    # Where the shadow lands, clipped to the canvas, and the matching window
    # into the mask so the two stay aligned.
    top, left = y - pad, x - pad
    canvas_h, canvas_w = canvas.shape[:2]
    y0, x0 = max(0, top), max(0, left)
    y1 = min(canvas_h, top + mask.shape[0])
    x1 = min(canvas_w, left + mask.shape[1])
    if y0 >= y1 or x0 >= x1:
        return

    if ground is not None:
        # Nothing under it but the page, so the answer is already known.
        ready = _shadow_on_ground(w, h, radius, spread, strength, ground)
        np.copyto(canvas[y0:y1, x0:x1],
                  ready[y0 - top:y1 - top, x0 - left:x1 - left])
        return

    window = mask[y0 - top:y1 - top, x0 - left:x1 - left]
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    region *= 1.0 - window
    region += np.asarray(COLOR_SHADOW, np.float32) * window
    canvas[y0:y1, x0:x1] = region.astype(np.uint8)


def card(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    title: str | None = None,
    title_color: tuple[int, int, int] = COLOR_TEXT_MAIN,
    fill: tuple[int, int, int] = COLOR_CARD_BG,
    shadow: bool = True,
) -> int:
    """Draw a card, optionally with a title and its rule.

    Returns the y a caller should start its own content at, so a card's body
    does not have to know whether it was given a title.
    """
    x, y, w, h = rect
    if shadow:
        soft_shadow(canvas, rect)
    rounded_rect(canvas, rect, fill, RADIUS_CARD, -1)
    rounded_rect(canvas, rect, COLOR_CARD_BORDER, RADIUS_CARD, 1)
    if title is None:
        return y + 20
    cv2.putText(canvas, title, (x + 20, y + 32), cv2.FONT_HERSHEY_DUPLEX,
                TYPE_SUBTITLE, title_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (x + 20, y + 46), (x + w - 20, y + 46), COLOR_CARD_BORDER, 1, cv2.LINE_AA)
    return y + 46


def badge(
    canvas: np.ndarray,
    origin: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    filled: bool = False,
    scale: float = TYPE_CAPTION,
) -> int:
    """A pill carrying one short label. Returns the width it used."""
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    w, h = text_w + 26, 28
    if filled:
        pill_rect(canvas, (x, y, w, h), color, -1)
        text_color = (255, 255, 255)
    else:
        pill_rect(canvas, (x, y, w, h), COLOR_CARD_BG, -1)
        pill_rect(canvas, (x, y, w, h), color, 1)
        text_color = color
    cv2.putText(canvas, text, (x + 13, y + 19), cv2.FONT_HERSHEY_SIMPLEX,
                scale, text_color, 1, cv2.LINE_AA)
    return w


def load_logo(path: Path, height: int) -> np.ndarray | None:
    """Read the brand mark and scale it to a header height.

    The file is a dark mark on a white artboard rather than a cut-out, so it is
    composited by multiplication below rather than masked: white goes to
    nothing, the mark keeps its gradient. Returns None if the file is missing,
    since a booth with no logo should still open.
    """
    if not path.is_file():
        return None
    image = cv2.imread(str(path))
    if image is None:
        return None
    width = max(1, round(image.shape[1] * height / image.shape[0]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def draw_logo(canvas: np.ndarray, logo: np.ndarray, origin: tuple[int, int]) -> int:
    """Multiply the mark onto the canvas. Returns the width it used.

    Multiplication is what lets a mark on a white artboard sit on a white
    header without a rectangle around it: white (1.0) leaves the ground alone
    and the mark darkens it exactly as much as it is dark.
    """
    x, y = origin
    h, w = logo.shape[:2]
    if y + h > canvas.shape[0] or x + w > canvas.shape[1]:
        return 0
    region = canvas[y:y + h, x:x + w].astype(np.float32)
    blended = region * (logo.astype(np.float32) / 255.0)
    canvas[y:y + h, x:x + w] = blended.astype(np.uint8)
    return w


CAMERA_SCAN_LIMIT = 6

# Shadows are the most expensive thing the theme asks for. On a machine that
# cannot spare them, this turns them off and leaves the hairlines behind, which
# is a flatter page rather than a broken one.
SHADOWS_ENABLED = True


def set_shadows_enabled(enabled: bool) -> None:
    global SHADOWS_ENABLED
    SHADOWS_ENABLED = enabled


def open_camera(preferred: int, scan_limit: int = CAMERA_SCAN_LIMIT):
    """Open the first camera that actually delivers a frame.

    Opening an index is not enough: a machine can expose several /dev/video
    nodes per physical camera, where some open and then read nothing. This
    machine is one of them -- index 0 opens as a device and hands back no
    frames, while 1 works -- so the booth would sit in GUI-only mode on the
    default settings. The preferred index is tried first and the rest scanned
    only if it fails.

    Returns an opened VideoCapture, or None if nothing on the machine works.
    """
    for index in [preferred] + [i for i in range(scan_limit) if i != preferred]:
        capture = cv2.VideoCapture(index)
        if capture.isOpened():
            frame_ok, _ = capture.read()
            if frame_ok:
                if index != preferred:
                    print(f"[BOOTH] Camera {preferred} gave no frames; using {index}.")
                return capture
        capture.release()
    return None


VIEWPORT = (40, 85, 800, 530)   # where a screen's main image goes


@lru_cache(maxsize=16)
def _corner_mask(
    width: int, height: int, radius: int
) -> tuple[tuple[int, int, np.ndarray], ...]:
    """The four corner patches of a rounded rectangle, as alpha.

    Returned as (y, x, alpha) so a caller can blend just those squares rather
    than testing every pixel of an image against a full-size mask.
    """
    full = np.zeros((height, width), np.float32)
    rounded_rect(full, (0, 0, width, height), 1.0, radius, -1)
    size = radius + 1
    return (
        (0, 0, full[:size, :size, None].copy()),
        (0, width - size, full[:size, width - size:, None].copy()),
        (height - size, 0, full[height - size:, :size, None].copy()),
        (height - size, width - size, full[height - size:, width - size:, None].copy()),
    )


def viewport(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int] = VIEWPORT,
    frame: np.ndarray | None = None,
    placeholder: str = "No stream available",
) -> None:
    """A well holding a camera or simulation image, or saying it has none.

    The image is written inside the rounded frame rather than over it, so the
    corners stay rounded -- a straight blit would square them off again.
    """
    x, y, w, h = rect
    soft_shadow(canvas, rect, RADIUS_CARD, ground=COLOR_BG)
    rounded_rect(canvas, rect, COLOR_SUNKEN, RADIUS_CARD, -1)

    if frame is not None:
        inner_x, inner_y, inner_w, inner_h = x + 5, y + 5, w - 10, h - 10
        resized = cv2.resize(frame, (inner_w, inner_h), interpolation=cv2.INTER_AREA)
        corners = _corner_mask(inner_w, inner_h, RADIUS_WELL)
        region = canvas[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
        # Only the corners are blended; the rest is a straight copy. Masking the
        # whole image cost about 12 ms a frame, which is most of a 60 Hz budget
        # spent rounding four corners.
        np.copyto(region, resized)
        for cy, cx, patch in corners:
            block = region[cy:cy + patch.shape[0], cx:cx + patch.shape[1]]
            np.copyto(block, (block * patch + COLOR_SUNKEN * (1.0 - patch)).astype(np.uint8))
    else:
        (text_w, _), _ = cv2.getTextSize(placeholder, cv2.FONT_HERSHEY_SIMPLEX,
                                         TYPE_SUBTITLE, 1)
        cv2.putText(canvas, placeholder, (x + (w - text_w) // 2, y + h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_SUBTITLE, COLOR_TEXT_MUTED,
                    1, cv2.LINE_AA)

    rounded_rect(canvas, rect, COLOR_CARD_BORDER, RADIUS_CARD, 1)


def overlay_label(
    canvas: np.ndarray,
    origin: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
) -> None:
    """A label that has to stay readable over an unknown image beneath it."""
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, 1)
    x, y = origin
    w, h = text_w + 30, 32
    overlay = canvas.copy()
    rounded_rect(overlay, (x, y, w, h), COLOR_CARD_BG, h // 2, -1)
    cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0.0, canvas)
    rounded_rect(canvas, (x, y, w, h), COLOR_CARD_BORDER, h // 2, 1)
    cv2.putText(canvas, text, (x + 15, y + 21), cv2.FONT_HERSHEY_SIMPLEX,
                TYPE_BODY, color, 1, cv2.LINE_AA)


@dataclass
class Button:
    id: str
    label: str
    rect: tuple[int, int, int, int]  # (x, y, w, h)
    shortcut: str = ""
    icon: str = ""
    bg_color: tuple[int, int, int] = COLOR_CARD_BG
    hover_color: tuple[int, int, int] = (60, 48, 40)
    text_color: tuple[int, int, int] = COLOR_TEXT_MAIN
    border_color: tuple[int, int, int] = COLOR_CARD_BORDER
    enabled: bool = True
    active: bool = False

    def is_inside(self, px: int, py: int) -> bool:
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def draw(self, canvas: np.ndarray, mouse_pos: tuple[int, int]) -> None:
        """Draw the button in whichever of its four states it is in.

        The home screen's mode cards come through here too -- they are buttons
        filling a whole card -- so the shape is chosen from the rectangle: a
        tall one is a card, a short one is a pill.
        """
        x, y, w, h = self.rect
        hovered = self.is_inside(*mouse_pos) and self.enabled
        is_card = h > 120

        radius = RADIUS_CARD if is_card else h // 2
        accent = self.text_color if self.text_color != COLOR_TEXT_MAIN else COLOR_PRIMARY

        if not self.enabled:
            fill, border, label_color = COLOR_DISABLED_BG, COLOR_CARD_BORDER, COLOR_DISABLED_TEXT
        elif self.active:
            # The one state that inverts: an active toggle is the only thing on
            # the screen carrying a solid brand colour, so it cannot be missed.
            fill, border, label_color = accent, accent, (255, 255, 255)
        elif hovered:
            fill, border, label_color = COLOR_CARD_HOVER, accent, accent
        else:
            fill, border, label_color = COLOR_CARD_BG, COLOR_CARD_BORDER, self.text_color

        # A resting pill gets its hairline and nothing else -- the site puts
        # shadows under cards, not under every control, and a shadow per button
        # per frame was about 4.7 ms of a redraw on the busier screens. Cards
        # keep theirs, and any button lifts when it is pointed at.
        if self.enabled and (hovered or self.active or is_card):
            soft_shadow(canvas, self.rect, radius, spread=12, strength=0.13)

        rounded_rect(canvas, self.rect, fill, radius, -1)
        rounded_rect(canvas, self.rect, border, radius, 1)

        # A mode card carries its own title and copy, written over it after the
        # buttons are drawn; a label here would be a second heading on top of
        # that one. So a card contributes its shape and its key cap only.
        if not is_card:
            (text_w, text_h), _ = cv2.getTextSize(
                self.label, cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, 1
            )
            cap_room = 34 if self.shortcut else 0
            text_x = x + max(14, (w - cap_room - text_w) // 2)
            cv2.putText(canvas, self.label, (text_x, y + (h + text_h) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, label_color, 1, cv2.LINE_AA)

        if self.shortcut:
            self._draw_shortcut(canvas, is_card, label_color, border)

    def _draw_shortcut(
        self,
        canvas: np.ndarray,
        is_card: bool,
        label_color: tuple[int, int, int],
        border: tuple[int, int, int],
    ) -> None:
        """Set the shortcut as a key cap rather than "[K] " before the label.

        Written into the caption it competed with the label for the same line
        and pushed every label off centre; as a cap it reads as the key it is.
        """
        x, y, w, h = self.rect
        key = self.shortcut
        (key_w, _), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, 1)
        cap_w, cap_h = max(22, key_w + 14), 20

        if is_card:
            cap_x, cap_y = x + 22, y + 26
        else:
            cap_x, cap_y = x + w - cap_w - 14, y + (h - cap_h) // 2

        if not self.enabled:
            cap_bg, cap_fg = COLOR_DISABLED_BG, COLOR_DISABLED_TEXT
        elif self.active:
            cap_bg, cap_fg = (255, 255, 255), label_color
        else:
            cap_bg, cap_fg = COLOR_SUNKEN, COLOR_TEXT_MUTED

        rounded_rect(canvas, (cap_x, cap_y, cap_w, cap_h), cap_bg, 6, -1)
        if not self.active:
            rounded_rect(canvas, (cap_x, cap_y, cap_w, cap_h), border, 6, 1)
        cv2.putText(canvas, key, (cap_x + (cap_w - key_w) // 2, cap_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, cap_fg, 1, cv2.LINE_AA)


@dataclass
class RpsScoreboard:
    total_rounds: int = 0
    human_wins: int = 0
    robot_wins: int = 0
    ties: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=8))

    def record_round(self, human_move: str, robot_move: str) -> str:
        verdict = human_result(human_move, robot_move)
        self.total_rounds += 1
        if verdict == "win":
            self.human_wins += 1
        elif verdict == "loss":
            self.robot_wins += 1
        else:
            self.ties += 1

        self.history.append((self.total_rounds, human_move, robot_move, verdict))
        return verdict

    def reset(self) -> None:
        self.total_rounds = 0
        self.human_wins = 0
        self.robot_wins = 0
        self.ties = 0
        self.history.clear()


class BoothKioskApp:
    """Main interactive booth application controller and renderer."""

    def __init__(
        self,
        *,
        mode: str = "hardware",
        port: str = "/dev/ttyUSB0",
        profile: str = "jiwoo",
        calib_path: Path = DEFAULT_CALIB_PATH,
        motor_calib_path: Path = DEFAULT_MOTOR_CALIB_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        logo_path: Path = DEFAULT_LOGO_PATH,
        camera_id: int = 0,
        current_limit: int = 350,
        max_joint_speed: float = 350.0,
        max_tracking_error: float = 50.0,
        enable_mujoco: bool = True,
        enable_hardware: bool = False,
        enable_reorient: bool = True,
        reorient_policy_path: Path = DEFAULT_REORIENT_POLICY_PATH,
        playground_root: Path | None = None,
        reorient_tilt_degrees: float = 0.0,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.mode = mode
        self.port = port
        self.profile = profile
        self.motor_calib_path = motor_calib_path
        self.camera_id = camera_id
        self.current_limit = current_limit
        self.max_joint_speed = max_joint_speed
        self.max_tracking_error = max_tracking_error
        self.enable_mujoco = enable_mujoco
        self.enable_hardware = enable_hardware
        self.enable_reorient = enable_reorient
        self.reorient_policy_path = reorient_policy_path
        self.playground_root = playground_root
        self.reorient_tilt_degrees = reorient_tilt_degrees
        self.width = width
        self.height = height

        self.current_screen = AppScreen.HOME
        self.logo = load_logo(logo_path, 38)
        self.mouse_pos = (0, 0)
        self.window_name = "LEAP Hand Interactive Booth Kiosk"

        # Teleop Controllers & Calibration
        self.tracker = MediaPipeTracker(model_path=model_path)
        self.calibrator = NeutralCalibration(path=calib_path, profile=profile)
        self.filter = OneEuroFilter()
        self.hardware_controller: LeapHandHardwareController | None = None
        self.mujoco_controller: MujocoHandController | None = None

        # Cube Reorientation (RL policy, simulation only -- never the hand)
        self.reorient_controller: CubeReorientController | None = None

        # Teleop State
        self.armed = False
        self.calib_open_in_progress = False
        self.calib_fist_in_progress = False

        # RPS Game State
        self.rps_state = RpsState.IDLE
        self.rps_scoreboard = RpsScoreboard()
        self.countdown_start_time = 0.0
        self.countdown_duration = 3.0  # 3 seconds countdown
        self.shoot_time = 0.0
        self.shoot_duration = 2.5      # Hold verdict for 2.5 seconds
        self.auto_play = False
        self.robot_chosen_move: str | None = None
        self.human_detected_move: str | None = None
        self.last_round_verdict: str | None = None
        self.consecutive_detected_gesture: str | None = None
        self.gesture_streak_count = 0

        # Showcase Dynamic Animation State
        self.active_animation: str | None = None
        self.animation_start_time: float = 0.0

        # Status & Telemetry
        self.fps = 0.0
        self.last_frame_time = time.monotonic()
        self.current_joint_angles = np.zeros(16, dtype=np.float64)
        self.target_joint_angles = np.zeros(16, dtype=np.float64)
        self.status_message = "Ready. Welcome to the LEAP Hand Booth!"
        self.status_color = COLOR_SUCCESS

        # Initialize buttons
        self.buttons: dict[AppScreen, list[Button]] = {
            AppScreen.HOME: [],
            AppScreen.TELEOP: [],
            AppScreen.RPS: [],
            AppScreen.SHOWCASE: [],
            AppScreen.REORIENT: [],
        }
        self._init_buttons()

        # Connect hardware / MuJoCo if configured
        self._init_controllers()

    def _home_card_layout(self) -> tuple[int, int, int, int, int]:
        """Geometry of the home screen's mode cards: (w, h, gap, x0, y).

        Both the buttons and the text drawn inside them come from here, so a
        card cannot end up with its label in one place and its hit box in
        another. Four cards is what makes this worth sharing -- the labels are
        short because at this width the longer ones no longer fit.
        """
        card_w, card_h = 280, 292
        gap = 30
        columns = 4
        start_x = (self.width - (columns * card_w + (columns - 1) * gap)) // 2
        return card_w, card_h, gap, start_x, 210

    def _init_buttons(self) -> None:
        # HOME SCREEN BUTTONS
        card_w, card_h, gap, start_x, y_pos = self._home_card_layout()

        self.buttons[AppScreen.HOME] = [
            Button(
                id="goto_teleop",
                label="Teleoperation",
                shortcut="1",
                rect=(start_x, y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="goto_rps",
                label="RPS Game",
                shortcut="2",
                rect=(start_x + card_w + gap, y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="goto_showcase",
                label="Gesture Showcase",
                shortcut="3",
                rect=(start_x + 2 * (card_w + gap), y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="goto_reorient",
                label="Cube Reorient",
                shortcut="4",
                rect=(start_x + 3 * (card_w + gap), y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="quit_app",
                label="Exit (Quit)",
                shortcut="Q",
                rect=(self.width - 160, 20, 130, 35),
                bg_color=(50, 20, 20),
                hover_color=(80, 30, 30),
                text_color=COLOR_DANGER,
            ),
        ]

        # TELEOP SCREEN BUTTONS
        bx, by, bw, bh = 880, 100, 370, 48
        b_gap = 12
        self.buttons[AppScreen.TELEOP] = [
            Button(
                id="calib_open",
                label="Open Hand Calibration",
                shortcut="C",
                rect=(bx, by, bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="calib_fist",
                label="Closed Fist Calibration",
                shortcut="F",
                rect=(bx, by + (bh + b_gap), bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="toggle_arm",
                label="ARM / Follow Hand",
                shortcut="A",
                rect=(bx, by + 2 * (bh + b_gap), bw, bh),
                bg_color=(20, 60, 20),
                hover_color=(30, 90, 30),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="disarm",
                label="DISARM / Pause Torque",
                shortcut="D",
                rect=(bx, by + 3 * (bh + b_gap), bw, bh),
                bg_color=(60, 20, 20),
                hover_color=(90, 30, 30),
                text_color=COLOR_DANGER,
            ),
            Button(
                id="reset_calib",
                label="Reset Calibration Profile",
                shortcut="R",
                rect=(bx, by + 4 * (bh + b_gap), bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="back_home_teleop",
                label="Return to Main Menu",
                shortcut="H",
                rect=(bx, by + 5 * (bh + b_gap) + 15, bw, bh),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

        # RPS SCREEN BUTTONS
        rx, ry, rw, rh = 880, 100, 370, 52
        self.buttons[AppScreen.RPS] = [
            Button(
                id="rps_start",
                label="START RPS MATCH",
                shortcut="Space",
                rect=(rx, ry, rw, rh),
                bg_color=(20, 65, 30),
                hover_color=(30, 95, 45),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="rps_auto_toggle",
                label="Auto-Play Mode: OFF",
                shortcut="P",
                rect=(rx, ry + rh + 12, rw, rh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="rps_reset_score",
                label="Reset Scoreboard",
                shortcut="R",
                rect=(rx, ry + 2 * (rh + 12), rw, rh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="back_home_rps",
                label="Return to Main Menu",
                shortcut="H",
                rect=(rx, ry + 3 * (rh + 12) + 50, rw, rh),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

        # SHOWCASE SCREEN BUTTONS (2-Column Grid + Dynamic Animations)
        # The column starts where every other screen's does. It began at 815,
        # which put the first 25 pixels of every button over the viewport --
        # invisible while both were flat, obvious once the buttons cast a shadow.
        c1_x, c2_x = 880, 1070
        bw, bh = 180, 42
        wide_w = 370
        gap_y = 8
        sy = 90

        self.buttons[AppScreen.SHOWCASE] = [
            # Column 1: Classic & Basic
            Button(
                id="showcase_rock",
                label="Rock (바위)",
                shortcut="1",
                rect=(c1_x, sy, bw, bh),
            ),
            Button(
                id="showcase_paper",
                label="Paper (보)",
                shortcut="2",
                rect=(c1_x, sy + (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_scissors",
                label="Scissors (가위)",
                shortcut="3",
                rect=(c1_x, sy + 2 * (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_neutral",
                label="Neutral (편 손)",
                shortcut="4",
                rect=(c1_x, sy + 3 * (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_middle",
                label="Middle (중지)",
                shortcut="5",
                rect=(c1_x, sy + 4 * (bh + gap_y), bw, bh),
                text_color=COLOR_WARNING,
            ),
            # Column 2: Expressive Gestures & Wave
            Button(
                id="showcase_thumbs_up",
                label="Thumbs Up (엄지척)",
                shortcut="6",
                rect=(c2_x, sy, bw, bh),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="showcase_ok",
                label="OK Sign (OK사인)",
                shortcut="7",
                rect=(c2_x, sy + (bh + gap_y), bw, bh),
                text_color=COLOR_PRIMARY,
            ),
            Button(
                id="showcase_pointing",
                label="Pointing (가리키기)",
                shortcut="8",
                rect=(c2_x, sy + 2 * (bh + gap_y), bw, bh),
                text_color=COLOR_SECONDARY,
            ),
            Button(
                id="showcase_rock_on",
                label="Rock On (락앤롤)",
                shortcut="9",
                rect=(c2_x, sy + 3 * (bh + gap_y), bw, bh),
                text_color=COLOR_ROBOT,
            ),
            Button(
                id="showcase_finger_wave",
                label="Wave (웨이브)",
                shortcut="W",
                rect=(c2_x, sy + 4 * (bh + gap_y), bw, bh),
                bg_color=(35, 45, 55),
                hover_color=(50, 65, 80),
                text_color=COLOR_PRIMARY,
            ),
            # Dynamic Animation: Wave Hello
            Button(
                id="showcase_wave_hello",
                label="Wave Hello (손 인사 애니메이션)",
                shortcut="V",
                rect=(c1_x, sy + 5 * (bh + gap_y) + 4, wide_w, bh),
                bg_color=(30, 50, 35),
                hover_color=(45, 75, 50),
                text_color=COLOR_SUCCESS,
            ),
            # Return Home
            Button(
                id="back_home_showcase",
                label="Return to Main Menu (메인 메뉴)",
                shortcut="H",
                rect=(c1_x, sy + 6 * (bh + gap_y) + 20, wide_w, 46),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

        # CUBE REORIENTATION SCREEN BUTTONS
        # The goal is turned a quarter turn at a time about a world axis, so
        # the six axis buttons sit in a 3x2 grid and everything else is a
        # full-width row beneath them.
        rx, ry, rw, rh = 880, 100, 370, 44
        small_w = (rw - 2 * 8) // 3
        row_gap = 54

        def axis_button(column: int, row: int, name: str, shortcut: str) -> Button:
            return Button(
                id=f"reorient_goal_{name.lower()}",
                label=name,
                shortcut=shortcut,
                rect=(rx + column * (small_w + 8), ry + row * (rh + 10), small_w, rh),
                bg_color=(45, 35, 45),
                hover_color=(70, 55, 70),
                text_color=COLOR_ROBOT,
            )

        stack_y = ry + 2 * (rh + 10) + 16

        def stacked(index: int, height: int = 44) -> tuple[int, int, int, int]:
            return (rx, stack_y + index * row_gap, rw, height)

        self.buttons[AppScreen.REORIENT] = [
            axis_button(0, 0, "X+", "1"),
            axis_button(1, 0, "Y+", "2"),
            axis_button(2, 0, "Z+", "3"),
            # 4/5/6 rather than Q/W/E: Q is the kiosk-wide quit key.
            axis_button(0, 1, "X-", "4"),
            axis_button(1, 1, "Y-", "5"),
            axis_button(2, 1, "Z-", "6"),
            Button(
                id="reorient_random",
                label="Random Goal",
                shortcut="R",
                rect=stacked(0),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="reorient_reset_goal",
                label="Reset Goal (Upright)",
                shortcut="0",
                rect=stacked(1),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="reorient_adopt",
                label="Goal = Cube Now",
                shortcut="G",
                rect=stacked(2),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="reorient_auto",
                label="Auto Goal: OFF",
                shortcut="A",
                rect=stacked(3),
                bg_color=(30, 45, 55),
                hover_color=(45, 65, 80),
                text_color=COLOR_PRIMARY,
            ),
            Button(
                id="reorient_pause",
                label="Pause Policy",
                shortcut="Space",
                rect=stacked(4),
                bg_color=(50, 45, 20),
                hover_color=(75, 65, 30),
                text_color=COLOR_WARNING,
            ),
            Button(
                id="reorient_reset",
                label="Reset Cube & Hand",
                shortcut="X",
                rect=stacked(5),
                bg_color=(50, 25, 25),
                hover_color=(80, 40, 40),
                text_color=COLOR_DANGER,
            ),
            Button(
                id="back_home_reorient",
                label="Return to Main Menu (메인 메뉴)",
                shortcut="H",
                rect=stacked(6, 46),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

    def _init_controllers(self) -> None:
        """Initialize Hardware and MuJoCo simulation backend controllers."""
        if self.enable_mujoco:
            try:
                self.mujoco_controller = MujocoHandController()
                if self.mode in ("mujoco", "both"):
                    self.mujoco_controller.launch_viewer()
                self.status_message = "MuJoCo 3D simulation connected."
            except Exception as e:
                print(f"[BOOTH WARN] MuJoCo simulation launch skipped: {e}")
                self.mujoco_controller = None

        if self.enable_hardware:
            try:
                motor_calib = (
                    self.motor_calib_path
                    if self.motor_calib_path.is_file()
                    else None
                )
                self.hardware_controller = LeapHandHardwareController(
                    port=self.port,
                    current_limit_milliamps=self.current_limit,
                    motor_calibration=motor_calib,
                )
                self.hardware_controller.connect()
                self.hardware_controller.configure()
                self.status_message = f"LEAP Hand connected on {self.port}."
            except Exception as e:
                print(f"[BOOTH WARN] Hardware connection skipped: {e}")
                self.hardware_controller = None

        if self.enable_reorient:
            try:
                self.reorient_controller = CubeReorientController(
                    self.reorient_policy_path,
                    playground_root=self.playground_root,
                    tilt_degrees=self.reorient_tilt_degrees,
                )
                self.status_message = "Cube reorientation policy loaded."
            except Exception as e:
                # Missing policy or Playground checkout must not stop the booth:
                # the other three modes have nothing to do with either.
                print(f"[BOOTH WARN] Cube reorientation unavailable: {e}")
                self.reorient_controller = None

    def send_robot_posture_degrees(self, posture_degrees: Sequence[float]) -> None:
        """Set target angles for smooth interpolation towards the target posture."""
        angles = np.asarray(posture_degrees, dtype=np.float64)
        self.target_joint_angles = angles.copy()

    def step_smooth_control(self, dt: float = 0.02) -> None:
        """Smoothly interpolate current joint angles for Showcase and RPS modes only."""
        if self.current_screen == AppScreen.TELEOP:
            # Teleoperation mode uses direct 1:1 instantaneous commands without showcase rate-limiting
            return

        if self.current_screen == AppScreen.REORIENT:
            # The reorientation demo runs entirely in its own simulation. This
            # function is the booth's only path to the hand, and it would keep
            # commanding whichever posture the showcase left behind, so it stops
            # here: nothing on that screen may reach a motor.
            return

        # Dynamic Animation Update for Showcase
        if self.current_screen == AppScreen.SHOWCASE and self.active_animation:
            t = time.monotonic() - self.animation_start_time
            if self.active_animation == "finger_wave":
                wave_pose = np.zeros(16, dtype=np.float64)
                # Index wave
                wave_pose[1] = 40.0 + 35.0 * math.sin(4.0 * t)
                wave_pose[2] = 45.0 + 40.0 * math.sin(4.0 * t)
                wave_pose[3] = 35.0 + 30.0 * math.sin(4.0 * t)
                # Middle wave (phase offset -1.0)
                wave_pose[5] = 40.0 + 35.0 * math.sin(4.0 * t - 1.0)
                wave_pose[6] = 45.0 + 40.0 * math.sin(4.0 * t - 1.0)
                wave_pose[7] = 35.0 + 30.0 * math.sin(4.0 * t - 1.0)
                # Ring wave (phase offset -2.0)
                wave_pose[9] = 40.0 + 35.0 * math.sin(4.0 * t - 2.0)
                wave_pose[10] = 45.0 + 40.0 * math.sin(4.0 * t - 2.0)
                wave_pose[11] = 35.0 + 30.0 * math.sin(4.0 * t - 2.0)
                # Thumb wave (phase offset -3.0)
                wave_pose[13] = 30.0 + 25.0 * math.sin(4.0 * t - 3.0)
                wave_pose[14] = 35.0 + 30.0 * math.sin(4.0 * t - 3.0)
                wave_pose[15] = 25.0 + 20.0 * math.sin(4.0 * t - 3.0)
                self.target_joint_angles = wave_pose
            elif self.active_animation == "wave_hello":
                hello_pose = np.zeros(16, dtype=np.float64)
                side_val = 18.0 * math.sin(5.0 * t)
                hello_pose[0] = side_val
                hello_pose[4] = side_val
                hello_pose[8] = side_val
                hello_pose[12] = 10.0 * math.sin(5.0 * t)
                flex_val = 20.0 + 15.0 * math.sin(3.0 * t)
                hello_pose[1] = flex_val
                hello_pose[2] = flex_val
                hello_pose[5] = flex_val
                hello_pose[6] = flex_val
                hello_pose[9] = flex_val
                hello_pose[10] = flex_val
                self.target_joint_angles = hello_pose

        # Smooth trajectory interpolation for Showcases and RPS transitions (~350 deg/s)
        max_step = self.max_joint_speed * max(0.001, min(dt, 0.05))
        diff = self.target_joint_angles - self.current_joint_angles
        step = np.clip(diff, -max_step, max_step)
        self.current_joint_angles += step

        if self.mujoco_controller is not None:
            self.mujoco_controller.set_target_degrees(self.current_joint_angles)
            self.mujoco_controller.step_for(dt)
            self.mujoco_controller.sync_viewer()

        if self.hardware_controller is not None and self.hardware_controller.torque_enabled:
            try:
                self.hardware_controller.command_degrees(self.current_joint_angles)
            except Exception as e:
                print(f"[HARDWARE ERROR] Command failed: {e}")

    @staticmethod
    def _key_codes(*characters: str) -> tuple[int, ...]:
        """Both cases of each character, so shortcuts work with caps lock on."""
        codes = []
        for character in characters:
            codes.extend((ord(character.lower()), ord(character.upper())))
        return tuple(dict.fromkeys(codes))

    def _screen_key_bindings(self) -> list[tuple[tuple[int, ...], str]]:
        """The keys the current screen answers to, paired with their action.

        One table rather than a chain of comparisons in the run loop, so a
        button's shortcut and the key that fires it stay together.
        """
        if self.current_screen == AppScreen.TELEOP:
            return [
                (self._key_codes("c"), "calib_open"),
                (self._key_codes("f"), "calib_fist"),
                (self._key_codes("a"), "toggle_arm"),
                (self._key_codes("d", " "), "disarm"),
                (self._key_codes("r"), "reset_calib"),
            ]
        if self.current_screen == AppScreen.RPS:
            return [
                (self._key_codes(" "), "rps_start"),
                (self._key_codes("p"), "rps_auto_toggle"),
                (self._key_codes("r"), "rps_reset_score"),
            ]
        if self.current_screen == AppScreen.SHOWCASE:
            return [
                (self._key_codes("1"), "showcase_rock"),
                (self._key_codes("2"), "showcase_paper"),
                (self._key_codes("3"), "showcase_scissors"),
                (self._key_codes("4"), "showcase_neutral"),
                (self._key_codes("5"), "showcase_middle"),
                (self._key_codes("6"), "showcase_thumbs_up"),
                (self._key_codes("7"), "showcase_ok"),
                (self._key_codes("8"), "showcase_pointing"),
                (self._key_codes("9"), "showcase_rock_on"),
                (self._key_codes("w"), "showcase_finger_wave"),
                (self._key_codes("v"), "showcase_wave_hello"),
            ]
        if self.current_screen == AppScreen.REORIENT:
            return [
                (self._key_codes("1"), "reorient_goal_x+"),
                (self._key_codes("2"), "reorient_goal_y+"),
                (self._key_codes("3"), "reorient_goal_z+"),
                (self._key_codes("4"), "reorient_goal_x-"),
                (self._key_codes("5"), "reorient_goal_y-"),
                (self._key_codes("6"), "reorient_goal_z-"),
                (self._key_codes("r"), "reorient_random"),
                (self._key_codes("0"), "reorient_reset_goal"),
                (self._key_codes("g"), "reorient_adopt"),
                (self._key_codes("a"), "reorient_auto"),
                (self._key_codes(" "), "reorient_pause"),
                (self._key_codes("x"), "reorient_reset"),
            ]
        return []

    def _handle_screen_key(self, key: int) -> bool:
        """Fire the current screen's action for this key. True if it took it."""
        for codes, action_id in self._screen_key_bindings():
            if key in codes:
                self.handle_action(action_id)
                return True
        return False

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse movement and clicks."""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)

    def _handle_click(self, x: int, y: int) -> None:
        for btn in self.buttons.get(self.current_screen, []):
            if btn.is_inside(x, y) and btn.enabled:
                self.handle_action(btn.id)
                break

    def handle_action(self, action_id: str) -> None:
        """Execute logic triggered by button click or keyboard shortcut."""
        # Navigation
        if action_id == "goto_teleop":
            self.current_screen = AppScreen.TELEOP
            self.status_message = "Teleoperation: [C] Open Calib, [F] Fist Calib, [A] Arm"
            self.status_color = COLOR_PRIMARY
        elif action_id == "goto_rps":
            self.current_screen = AppScreen.RPS
            self.rps_state = RpsState.IDLE
            self.status_message = "RPS Match: Press [Space] to start countdown!"
            self.status_color = COLOR_SECONDARY
        elif action_id == "goto_showcase":
            self.current_screen = AppScreen.SHOWCASE
            self.status_message = "Showcase: Click buttons to test postures."
            self.status_color = COLOR_SUCCESS
        elif action_id == "goto_reorient":
            self.current_screen = AppScreen.REORIENT
            # Nothing on this screen commands the hand, so leaving it armed
            # would hold torque on a hand nobody is watching -- [4] reaches
            # here straight from an armed teleop session.
            self.disarm_robot()
            if self.reorient_controller is None:
                self.status_message = (
                    "Cube reorientation is unavailable -- see the console for why."
                )
                self.status_color = COLOR_DANGER
            else:
                self.status_message = (
                    "Cube Reorient: set a goal and watch the policy match it. "
                    "Simulation only."
                )
                self.status_color = COLOR_ROBOT
        elif action_id in (
            "back_home_teleop",
            "back_home_rps",
            "back_home_showcase",
            "back_home_reorient",
        ):
            self.current_screen = AppScreen.HOME
            self.disarm_robot()
            self.status_message = "Main Menu. Select a mode to begin."
            self.status_color = COLOR_TEXT_MAIN

        # Teleop Actions
        elif action_id == "calib_open":
            self.calib_open_in_progress = True
            self.calib_fist_in_progress = False
            self.calibrator.start("Right", time.perf_counter(), "neutral")
            self.status_message = "Capturing Open Hand pose for 3 seconds..."
            self.status_color = COLOR_WARNING
        elif action_id == "calib_fist":
            self.calib_fist_in_progress = True
            self.calib_open_in_progress = False
            try:
                self.calibrator.start("Right", time.perf_counter(), "closed")
                self.status_message = "Capturing Closed Fist pose for 3 seconds..."
                self.status_color = COLOR_WARNING
            except ValueError as e:
                self.status_message = f"Error: {e}"
                self.status_color = COLOR_DANGER
                self.calib_fist_in_progress = False
        elif action_id == "toggle_arm":
            self.arm_robot()
        elif action_id == "disarm":
            self.disarm_robot()
        elif action_id == "reset_calib":
            self.calibrator.reset("Right")
            self.status_message = "Calibration profile reset to default."
            self.status_color = COLOR_SUCCESS

        # RPS Game Actions
        elif action_id == "rps_start":
            self.start_rps_countdown()
        elif action_id == "rps_auto_toggle":
            self.auto_play = not self.auto_play
            for btn in self.buttons[AppScreen.RPS]:
                if btn.id == "rps_auto_toggle":
                    btn.label = f"Auto-Play Mode: {'ON' if self.auto_play else 'OFF'}"
                    btn.active = self.auto_play
            self.status_message = f"Auto-play mode {'enabled' if self.auto_play else 'disabled'}."
        elif action_id == "rps_reset_score":
            self.rps_scoreboard.reset()
            self.status_message = "Scoreboard reset to 0."

        # Showcase Actions
        elif action_id == "showcase_rock":
            self.play_showcase_posture("rock", "Rock Gesture (바위)")
        elif action_id == "showcase_paper":
            self.play_showcase_posture("paper", "Paper Gesture (보)")
        elif action_id == "showcase_scissors":
            self.play_showcase_posture("scissors", "Scissors Gesture (가위)")
        elif action_id == "showcase_neutral":
            self.play_showcase_posture("neutral", "Open Neutral (편 손)")
        elif action_id == "showcase_middle":
            self.play_showcase_middle_finger()
        elif action_id == "showcase_thumbs_up":
            self.play_showcase_posture("thumbs_up", "Thumbs Up (엄지 척)")
        elif action_id == "showcase_ok":
            self.play_showcase_posture("ok_sign", "OK Sign (OK 사인)")
        elif action_id == "showcase_pointing":
            self.play_showcase_posture("pointing", "Pointing (가리키기)")
        elif action_id == "showcase_rock_on":
            self.play_showcase_posture("rock_on", "Rock On (락앤롤)")
        elif action_id == "showcase_finger_wave":
            self.play_showcase_animation("finger_wave", "Finger Wave (파도타기 애니메이션)")
        elif action_id == "showcase_wave_hello":
            self.play_showcase_animation("wave_hello", "Wave Hello (손 인사 애니메이션)")

        # Cube Reorientation Actions (simulation only -- never the hand)
        elif action_id.startswith("reorient_"):
            self.handle_reorient_action(action_id)

    def handle_reorient_action(self, action_id: str) -> None:
        """Act on the cube reorientation screen. Touches the simulation only."""
        controller = self.reorient_controller
        if controller is None:
            self.status_message = "Cube reorientation is not loaded."
            self.status_color = COLOR_DANGER
            return

        if action_id.startswith("reorient_goal_"):
            name = action_id.removeprefix("reorient_goal_")
            axis = "xyz".index(name[0])
            degrees = 90.0 if name[1] == "+" else -90.0
            controller.rotate_goal(axis, degrees)
            self.status_message = f"Goal turned {degrees:+.0f} deg about {name[0].upper()}."
            self.status_color = COLOR_ROBOT
        elif action_id == "reorient_random":
            controller.randomize_goal()
            self.status_message = "Random goal orientation set."
            self.status_color = COLOR_ROBOT
        elif action_id == "reorient_reset_goal":
            controller.reset_goal()
            self.status_message = "Goal reset to upright."
            self.status_color = COLOR_ROBOT
        elif action_id == "reorient_adopt":
            controller.adopt_cube_goal()
            self.status_message = "Goal set to the cube's current orientation."
            self.status_color = COLOR_SUCCESS
        elif action_id == "reorient_auto":
            controller.toggle_auto_goal()
            self._set_button_state(
                AppScreen.REORIENT,
                "reorient_auto",
                label=f"Auto Goal: {'ON' if controller.auto_goal else 'OFF'}",
                active=controller.auto_goal,
            )
            self.status_message = (
                "Auto goal on: the training schedule spins the goal away on every "
                "success."
                if controller.auto_goal
                else "Auto goal off: the goal stays where you put it."
            )
            self.status_color = COLOR_PRIMARY
        elif action_id == "reorient_pause":
            controller.toggle_pause()
            self._set_button_state(
                AppScreen.REORIENT,
                "reorient_pause",
                label="Resume Policy" if controller.paused else "Pause Policy",
                active=controller.paused,
            )
            self.status_message = (
                "Policy paused -- the cube keeps its pose."
                if controller.paused
                else "Policy running."
            )
            self.status_color = COLOR_WARNING if controller.paused else COLOR_SUCCESS
        elif action_id == "reorient_reset":
            controller.reset()
            controller.reset_statistics()
            self.status_message = "Cube and hand returned to the home pose."
            self.status_color = COLOR_SUCCESS

    def _set_button_state(
        self,
        screen: AppScreen,
        button_id: str,
        *,
        label: str | None = None,
        active: bool | None = None,
    ) -> None:
        """Update one button's caption or highlight after its state changed."""
        for button in self.buttons.get(screen, []):
            if button.id == button_id:
                if label is not None:
                    button.label = label
                if active is not None:
                    button.active = active
                return

    def arm_robot(self) -> None:
        """Enable hardware torque and activate teleoperation tracking."""
        self.armed = True
        if self.hardware_controller is not None:
            try:
                self.hardware_controller.enable_torque()
            except Exception as e:
                print(f"[HARDWARE ERROR] Failed to enable torque: {e}")
        for btn in self.buttons[AppScreen.TELEOP]:
            if btn.id == "toggle_arm":
                btn.active = True
                btn.label = "ARMED (Tracking Active)"
        self.status_message = "Robot ARMED! Hand movements will now be followed."
        self.status_color = COLOR_SUCCESS

    def disarm_robot(self) -> None:
        """Disable torque and safely park robot hand."""
        self.armed = False
        if self.hardware_controller is not None:
            try:
                self.hardware_controller.emergency_stop()
            except Exception as e:
                print(f"[HARDWARE ERROR] Emergency stop: {e}")
        for btn in self.buttons[AppScreen.TELEOP]:
            if btn.id == "toggle_arm":
                btn.active = False
                btn.label = "ARM / Follow Hand"
        self.status_message = "Robot DISARMED (Holding / Torque Paused)."
        self.status_color = COLOR_WARNING

    def start_rps_countdown(self) -> None:
        """Begin a 3-2-1 Rock-Paper-Scissors match."""
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            try:
                self.hardware_controller.enable_torque()
            except Exception:
                pass

        self.send_robot_posture_degrees(get_hand_posture("neutral"))
        self.rps_state = RpsState.COUNTDOWN
        self.countdown_start_time = time.monotonic()
        self.robot_chosen_move = None
        self.human_detected_move = None
        self.last_round_verdict = None
        self.status_message = "Rock... Paper... Scissors... Shoot!"
        self.status_color = COLOR_SECONDARY

    def play_showcase_posture(self, posture_name: str, display_name: str) -> None:
        self.active_animation = None
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        deg = get_hand_posture(posture_name)
        self.send_robot_posture_degrees(deg)
        self.status_message = f"Postured: {display_name}"
        self.status_color = COLOR_SUCCESS

    def play_showcase_middle_finger(self) -> None:
        self.active_animation = None
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        deg = get_hand_posture("rock").copy()
        deg[4] = 0.0  # mf_mcp_side
        deg[5] = 0.0  # mf_mcp_flex
        deg[6] = 0.0  # mf_pip_flex
        deg[7] = 0.0  # mf_dip_flex
        self.send_robot_posture_degrees(deg)
        self.status_message = "Postured: Middle Finger Extension (중지)"
        self.status_color = COLOR_WARNING

    def play_showcase_animation(self, anim_name: str, display_name: str) -> None:
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        self.active_animation = anim_name
        self.animation_start_time = time.monotonic()
        self.status_message = f"Animation Running: {display_name}"
        self.status_color = COLOR_PRIMARY

    def update_teleop_frame(self, frame: np.ndarray, landmarks_list: list[Any]) -> None:
        """Process teleoperation hand tracking, calibration, and joint commanding."""
        if not landmarks_list:
            return

        landmarks = landmarks_list[0]
        raw_angles = calculate_leap_control_angles(landmarks)

        # Calibration collection
        if self.calib_open_in_progress or self.calib_fist_in_progress:
            now_perf = time.perf_counter()
            completed = self.calibrator.add_sample("Right", raw_angles, now_perf)
            if completed:
                if self.calib_open_in_progress:
                    self.calib_open_in_progress = False
                    self.status_message = "Open Hand Calibration Complete!"
                else:
                    self.calib_fist_in_progress = False
                    self.status_message = "Closed Fist Calibration Complete!"
                self.status_color = COLOR_SUCCESS

        # Joint angle calculation & calibration
        calibrated_angles = self.calibrator.apply("Right", raw_angles)
        smoothed_angles = self.filter.filter(calibrated_angles, time.monotonic())

        self.current_joint_angles = smoothed_angles.copy()
        self.target_joint_angles = smoothed_angles.copy()

        # Direct 1:1 Instantaneous Hardware and Simulation Dispatch (Zero lag!)
        if self.armed:
            if self.hardware_controller is not None and self.hardware_controller.torque_enabled:
                try:
                    self.hardware_controller.command_degrees(smoothed_angles)
                except Exception as e:
                    print(f"[HARDWARE ERROR] Command failed: {e}")

        if self.mujoco_controller is not None:
            self.mujoco_controller.set_target_degrees(smoothed_angles)
            self.mujoco_controller.step_for(0.02)
            self.mujoco_controller.sync_viewer()

    def update_rps_frame(self, frame: np.ndarray, landmarks_list: list[Any]) -> None:
        """Process Rock-Paper-Scissors game loop, gesture detection, and robot moves."""
        now = time.monotonic()

        current_gesture = None
        if landmarks_list:
            classification = classify_rps_gesture(landmarks_list[0])
            current_gesture = classification.label

        if current_gesture == self.consecutive_detected_gesture:
            self.gesture_streak_count += 1
        else:
            self.consecutive_detected_gesture = current_gesture
            self.gesture_streak_count = 1

        if self.gesture_streak_count >= 3:
            self.human_detected_move = current_gesture

        # State Machine
        if self.rps_state == RpsState.COUNTDOWN:
            elapsed = now - self.countdown_start_time
            shake_amp = math.sin(elapsed * 12.0) * 15.0
            shake_pose = get_hand_posture("neutral").copy()
            shake_pose[1] += shake_amp
            shake_pose[5] += shake_amp
            shake_pose[9] += shake_amp
            self.send_robot_posture_degrees(shake_pose)

            if elapsed >= self.countdown_duration:
                self.robot_chosen_move = random.choice(list(MOVE_NAMES))
                robot_posture = get_hand_posture(self.robot_chosen_move)
                self.send_robot_posture_degrees(robot_posture)

                self.rps_state = RpsState.SHOOT
                self.shoot_time = now

        elif self.rps_state == RpsState.SHOOT:
            if now - self.shoot_time >= 0.5:
                if self.human_detected_move in MOVE_NAMES and self.robot_chosen_move is not None:
                    verdict = self.rps_scoreboard.record_round(
                        self.human_detected_move,
                        self.robot_chosen_move,
                    )
                    self.last_round_verdict = verdict
                    self.rps_state = RpsState.RESULT
                    self.shoot_time = now
                elif now - self.shoot_time >= self.shoot_duration:
                    self.last_round_verdict = "no_hand"
                    self.rps_state = RpsState.RESULT
                    self.shoot_time = now

        elif self.rps_state == RpsState.RESULT:
            if now - self.shoot_time >= self.shoot_duration:
                if self.auto_play:
                    self.start_rps_countdown()
                else:
                    self.rps_state = RpsState.IDLE
                    self.send_robot_posture_degrees(get_hand_posture("neutral"))

    def render(self, camera_frame: np.ndarray | None, landmarks_list: list[Any]) -> np.ndarray:
        """Render the complete Kiosk UI canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = COLOR_BG

        now = time.monotonic()
        dt = now - self.last_frame_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

        self._draw_header(canvas)

        if self.current_screen == AppScreen.HOME:
            self._render_home_screen(canvas)
        elif self.current_screen == AppScreen.TELEOP:
            self._render_teleop_screen(canvas, camera_frame, landmarks_list)
        elif self.current_screen == AppScreen.RPS:
            self._render_rps_screen(canvas, camera_frame, landmarks_list)
        elif self.current_screen == AppScreen.SHOWCASE:
            self._render_showcase_screen(canvas, camera_frame)
        elif self.current_screen == AppScreen.REORIENT:
            self._render_reorient_screen(canvas)

        for btn in self.buttons.get(self.current_screen, []):
            btn.draw(canvas, self.mouse_pos)

        if self.current_screen == AppScreen.HOME:
            # A mode card is a button filling its whole rectangle, so anything
            # written inside one has to go on after it -- underneath, the fill
            # paints it out and the card reads as empty.
            self._render_home_card_text(canvas)

        self._draw_footer(canvas)
        return canvas

    def _draw_header(self, canvas: np.ndarray) -> None:
        cv2.rectangle(canvas, (0, 0), (self.width, 65), COLOR_CARD_BG, -1)
        cv2.line(canvas, (0, 65), (self.width, 65), COLOR_CARD_BORDER, 1, cv2.LINE_AA)

        x = 25
        if self.logo is not None:
            x += draw_logo(canvas, self.logo, (x, 13)) + 14
        cv2.putText(canvas, "LEAP HAND BOOTH", (x, 41), cv2.FONT_HERSHEY_DUPLEX,
                    0.72, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        # Status pills, laid out left to right so none of them collide.
        x = 330
        x += badge(canvas, (x, 19), f"MODE  {self.mode.upper()}", COLOR_PRIMARY) + 10
        badge(canvas, (x, 19), f"{self.fps:.0f} FPS", COLOR_TEXT_MUTED)

    def _draw_footer(self, canvas: np.ndarray) -> None:
        y = self.height - 45
        cv2.rectangle(canvas, (0, y), (self.width, self.height), COLOR_CARD_BG, -1)
        cv2.line(canvas, (0, y), (self.width, y), COLOR_CARD_BORDER, 1, cv2.LINE_AA)

        cv2.circle(canvas, (28, y + 23), 5, self.status_color, -1, cv2.LINE_AA)
        cv2.putText(canvas, self.status_message, (44, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

        hint = "H  Home        Q  Exit"
        (hint_w, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, 1)
        cv2.putText(canvas, hint, (self.width - hint_w - 28, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_home_screen(self, canvas: np.ndarray) -> None:
        heading = "Choose a mode"
        sub = "Physical AI, hands on."
        (hw, _), _ = cv2.getTextSize(heading, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1)
        cv2.putText(canvas, heading, ((self.width - hw) // 2, 132),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, TYPE_SUBTITLE, 1)
        cv2.putText(canvas, sub, ((self.width - sw) // 2, 166),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_SUBTITLE, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_home_card_text(self, canvas: np.ndarray) -> None:
        """Write each mode card's description over its button.

        Called after the buttons are drawn, not with the rest of the home
        screen: the card is the button, and a filled rectangle covers whatever
        was underneath it. The button's own centred label acts as the card's
        headline, so the blurb sits above it and the bullets below.
        """
        card_w, card_h, gap, start_x, y_pos = self._home_card_layout()

        # Each card: a coloured title, two lines of what it is, three bullets.
        cards = [
            (
                "Live Hand Tracking",
                COLOR_PRIMARY,
                ("Replicate user hand motion", "in real time, 16 joints."),
                ("- Open / Fist Calibration",
                 "- OneEuro Jitter Filtering",
                 "- Real-time Arm / Disarm"),
            ),
            (
                "Rock-Paper-Scissors",
                COLOR_SECONDARY,
                ("Battle the robot live!", "3-2-1 countdown & vision."),
                ("- Auto Gesture Recognition",
                 "- Real-time Winner Verdict",
                 "- Live Booth Scoreboard"),
            ),
            (
                "Showcase & Demos",
                COLOR_SUCCESS,
                ("One-click gesture demos", "and posture sanity checks."),
                ("- Rock / Paper / Scissors",
                 "- Middle Finger Extension",
                 "- Hardware Limit Check"),
            ),
            (
                "In-Hand Cube Reorient",
                COLOR_ROBOT,
                ("A trained RL policy turns", "the cube to match a goal."),
                ("- MuJoCo Playground Policy",
                 "- Simulation Only, No Hand",
                 "- Set / Random / Auto Goal"),
            ),
        ]

        # The card is one column: an accent rule, the title, the blurb, then the
        # bullets against the card's foot. The button underneath draws only the
        # shape and the key cap, so every word on a card is written here.
        for index, (title, title_color, blurb, bullets) in enumerate(cards):
            cx = start_x + index * (card_w + gap)

            rounded_rect(canvas, (cx + 22, y_pos + 66, 34, 3), title_color, 2, -1)
            cv2.putText(canvas, title, (cx + 22, y_pos + 100), cv2.FONT_HERSHEY_DUPLEX,
                        TYPE_SUBTITLE, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

            for line_index, line in enumerate(blurb):
                cv2.putText(canvas, line, (cx + 22, y_pos + 138 + line_index * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

            cv2.line(canvas, (cx + 22, y_pos + 186), (cx + card_w - 22, y_pos + 186),
                     COLOR_CARD_BORDER, 1, cv2.LINE_AA)
            for line_index, line in enumerate(bullets):
                row_y = y_pos + 214 + line_index * 26
                cv2.circle(canvas, (cx + 26, row_y - 4), 2, title_color, -1, cv2.LINE_AA)
                cv2.putText(canvas, line.lstrip("- "), (cx + 36, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_teleop_screen(
        self,
        canvas: np.ndarray,
        camera_frame: np.ndarray | None,
        landmarks_list: list[Any],
    ) -> None:
        vis_frame = None
        if camera_frame is not None:
            vis_frame = camera_frame.copy()
            for lm in landmarks_list:
                draw_hand(vis_frame, lm, [], True)
        viewport(canvas, VIEWPORT, vis_frame, "No camera stream")

        vx, vy, vw, vh = VIEWPORT
        armed = self.armed
        overlay_label(
            canvas, (vx + 20, vy + 20),
            "ARMED  ·  tracking" if armed else "DISARMED  ·  torque paused",
            COLOR_SUCCESS if armed else COLOR_WARNING,
        )

        if self.calib_open_in_progress or self.calib_fist_in_progress:
            progress = self.calibrator.progress(time.perf_counter())
            title = ("Capturing open neutral pose"
                     if self.calib_open_in_progress else "Capturing closed fist pose")

            pw, ph = 460, 96
            px, py = vx + (vw - pw) // 2, vy + vh - ph - 24
            card(canvas, (px, py, pw, ph))
            cv2.putText(canvas, title, (px + 24, py + 32), cv2.FONT_HERSHEY_SIMPLEX,
                        TYPE_BODY, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

            bar_w = pw - 48
            pill_rect(canvas, (px + 24, py + 48, bar_w, 12), COLOR_SUNKEN, -1)
            filled = max(12, int(bar_w * progress))
            pill_rect(canvas, (px + 24, py + 48, filled, 12), COLOR_PRIMARY, -1)
            cv2.putText(canvas, f"{int(progress * 100)}%", (px + 24, py + 82),
                        cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, COLOR_TEXT_MUTED,
                        1, cv2.LINE_AA)

    def _render_rps_screen(
        self,
        canvas: np.ndarray,
        camera_frame: np.ndarray | None,
        landmarks_list: list[Any],
    ) -> None:
        vis_frame = None
        if camera_frame is not None:
            vis_frame = camera_frame.copy()
            for lm in landmarks_list:
                draw_hand(vis_frame, lm, [], True)
        viewport(canvas, (40, 85, 460, 360), vis_frame, "No camera")

        # What the camera currently believes the visitor is showing.
        gx, gy, gw, gh = 40, 460, 460, 85
        card(canvas, (gx, gy, gw, gh))
        cv2.putText(canvas, "PLAYER GESTURE", (gx + 20, gy + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        detected = self.human_detected_move
        cv2.putText(canvas, detected.upper() if detected else "Show your hand to the camera",
                    (gx + 20, gy + 64), cv2.FONT_HERSHEY_DUPLEX,
                    TYPE_TITLE if detected else TYPE_SUBTITLE,
                    COLOR_HUMAN if detected else COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        ax, ay, aw, ah = 520, 85, 340, 460
        card(canvas, (ax, ay, aw, ah), title="BATTLE ARENA", title_color=COLOR_PRIMARY)

        now = time.monotonic()
        if self.rps_state == RpsState.IDLE:
            cv2.putText(canvas, "Ready to play", (ax + 24, ay + 160),
                        cv2.FONT_HERSHEY_DUPLEX, TYPE_TITLE, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Press Space, or click", (ax + 24, ay + 200),
                        cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "START RPS MATCH.", (ax + 24, ay + 226),
                        cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        elif self.rps_state == RpsState.COUNTDOWN:
            elapsed = now - self.countdown_start_time
            count_val = max(1, 3 - int(elapsed))
            cx, cy = ax + aw // 2, ay + 220
            # The ring breathes so the countdown reads across a busy room.
            radius = 62 + int(math.sin(elapsed * 10.0) * 6.0)
            cv2.circle(canvas, (cx, cy), radius, COLOR_SUNKEN, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), radius, COLOR_PRIMARY, 3, cv2.LINE_AA)
            (nw, nh), _ = cv2.getTextSize(str(count_val), cv2.FONT_HERSHEY_DUPLEX, 2.2, 2)
            cv2.putText(canvas, str(count_val), (cx - nw // 2, cy + nh // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_TEXT_MAIN, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Rock... Paper... Scissors...", (ax + 40, ay + 380),
                        cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, COLOR_PRIMARY, 1, cv2.LINE_AA)

        elif self.rps_state in (RpsState.SHOOT, RpsState.RESULT):
            for label, move, color, row in (
                ("ROBOT", self.robot_chosen_move, COLOR_ROBOT, 0),
                ("YOU", self.human_detected_move, COLOR_HUMAN, 1),
            ):
                row_y = ay + 90 + row * 110
                cv2.putText(canvas, label, (ax + 24, row_y), cv2.FONT_HERSHEY_SIMPLEX,
                            TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
                cv2.putText(canvas, move.upper() if move else "NO HAND",
                            (ax + 24, row_y + 44), cv2.FONT_HERSHEY_DUPLEX,
                            1.1, color if move else COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

            if self.last_round_verdict:
                verdicts = {
                    "win": ("YOU WIN", COLOR_SUCCESS),
                    "loss": ("ROBOT WINS", COLOR_SECONDARY),
                    "tie": ("DRAW", COLOR_WARNING),
                }
                text, color = verdicts.get(
                    self.last_round_verdict, ("No hand detected", COLOR_TEXT_MUTED)
                )
                vx, vy, vw2, vh2 = ax + 16, ay + 330, aw - 32, 86
                card(canvas, (vx, vy, vw2, vh2), fill=COLOR_SUNKEN, shadow=False)
                rounded_rect(canvas, (vx, vy, vw2, vh2), color, RADIUS_CARD, 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, TYPE_TITLE, 1)
                cv2.putText(canvas, text, (vx + (vw2 - tw) // 2, vy + (vh2 + th) // 2),
                            cv2.FONT_HERSHEY_DUPLEX, TYPE_TITLE, color, 1, cv2.LINE_AA)

        # Below the buttons, which end at y=394. It used to start at 390 and the
        # two drew over each other; the shadows made the collision obvious.
        sx, sy, sw, sh = 880, 412, 370, 178
        card(canvas, (sx, sy, sw, sh), title="SCOREBOARD")

        sb = self.rps_scoreboard
        win_rate = (sb.human_wins / max(1, sb.total_rounds)) * 100.0
        rows = (
            ("Rounds", str(sb.total_rounds), COLOR_TEXT_MAIN),
            ("Player wins", str(sb.human_wins), COLOR_SUCCESS),
            ("Robot wins", str(sb.robot_wins), COLOR_SECONDARY),
            ("Draws", f"{sb.ties}   ·   {win_rate:.0f}% win rate", COLOR_TEXT_MUTED),
        )
        for index, (name, value, color) in enumerate(rows):
            row_y = sy + 78 + index * 27
            cv2.putText(canvas, name, (sx + 20, row_y), cv2.FONT_HERSHEY_SIMPLEX,
                        TYPE_BODY, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (sx + 170, row_y), cv2.FONT_HERSHEY_SIMPLEX,
                        TYPE_BODY, color, 1, cv2.LINE_AA)

    def _render_showcase_screen(self, canvas: np.ndarray, camera_frame: np.ndarray | None) -> None:
        sim_frame = None
        if self.mujoco_controller is not None:
            try:
                sim_frame = self.mujoco_controller.render_bgr(480, 640)
            except Exception:
                sim_frame = None

        frame = sim_frame if sim_frame is not None else camera_frame
        viewport(canvas, VIEWPORT, frame, "No 3D simulation stream")

        vx, vy, _, _ = VIEWPORT
        overlay_label(
            canvas, (vx + 20, vy + 20),
            "3D digital twin" if sim_frame is not None else "Camera view",
            COLOR_PRIMARY if sim_frame is not None else COLOR_SUCCESS,
        )

    def _render_reorient_screen(self, canvas: np.ndarray) -> None:
        controller = self.reorient_controller
        sim_frame = None
        if controller is not None:
            try:
                sim_frame = controller.render_bgr(480, 640)
            except Exception:
                sim_frame = None

        viewport(canvas, VIEWPORT, sim_frame, "Cube reorientation unavailable")
        if controller is None or sim_frame is None:
            return

        vx, vy, vw, vh = VIEWPORT
        # The scene puts the goal on the left and the hand's cube on the right,
        # which is the demo's whole story: make one match the other.
        overlay_label(canvas, (vx + 20, vy + 20), "GOAL", COLOR_SECONDARY)
        (hand_w, _), _ = cv2.getTextSize("HAND", cv2.FONT_HERSHEY_SIMPLEX, TYPE_BODY, 1)
        overlay_label(canvas, (vx + vw - hand_w - 50, vy + 20), "HAND", COLOR_PRIMARY)

        error_degrees = controller.goal_error_degrees
        # Training counts a reorientation as solved at 0.1 rad, about 5.7 deg.
        solved = error_degrees <= 5.73
        error_color = COLOR_SUCCESS if solved else (
            COLOR_PRIMARY if error_degrees < 45.0 else COLOR_SECONDARY
        )

        pw, ph = 320, 104
        px, py = vx + (vw - pw) // 2, vy + vh - ph - 22
        card(canvas, (px, py, pw, ph))
        pill_rect(canvas, (px, py + 14, 4, ph - 28), error_color, -1)
        cv2.putText(canvas, "ORIENTATION ERROR", (px + 24, py + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{error_degrees:.1f}", (px + 24, py + 84),
                    cv2.FONT_HERSHEY_DUPLEX, TYPE_DISPLAY, error_color, 1, cv2.LINE_AA)
        (num_w, _), _ = cv2.getTextSize(f"{error_degrees:.1f}", cv2.FONT_HERSHEY_DUPLEX,
                                        TYPE_DISPLAY, 1)
        cv2.putText(canvas, "deg", (px + 34 + num_w, py + 84), cv2.FONT_HERSHEY_SIMPLEX,
                    TYPE_SUBTITLE, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        state = None
        if controller.paused:
            state, state_color = "PAUSED", COLOR_WARNING
        elif solved:
            state, state_color = "SOLVED", COLOR_SUCCESS
        if state is not None:
            (state_w, _), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX,
                                              TYPE_BODY, 1)
            overlay_label(canvas, (vx + (vw - state_w - 30) // 2, vy + 20),
                          state, state_color)

        # Telemetry between the last button and the footer, where the buttons
        # -- drawn after this method -- cannot paint over it.
        tx, ty, strip_w = 880, 634, 370
        cells = [
            ("RATE", f"{1.0 / controller.dt:.0f} Hz"),
            ("SOLVED", str(controller.successes)),
            ("DROPS", str(controller.drops)),
        ]
        if controller.tilt_degrees:
            cells.append(("TILT", f"{controller.tilt_degrees:.0f}"))
        cell_w = strip_w // len(cells)
        for index, (name, value) in enumerate(cells):
            cx = tx + index * cell_w
            cv2.putText(canvas, name, (cx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        TYPE_CAPTION, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, value, (cx, ty + 26), cv2.FONT_HERSHEY_DUPLEX,
                        TYPE_SUBTITLE, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    def run(self) -> int:
        """Main application loop."""
        print("\n" + "=" * 65)
        print("  LEAP HAND INTERACTIVE BOOTH KIOSK APPLICATION")
        print("=" * 65)
        print(f"  Mode: {self.mode.upper()} | Port: {self.port} | Profile: {self.profile}")
        print("  Controls:")
        print("    [1] Teleoperation | [2] RPS Game | [3] Showcase | [4] Cube Reorient")
        print("    [C] Open Calib    | [F] Fist Calib | [A] Arm     | [D] Disarm")
        print("    [Space] RPS Start | [P] Auto Play  | [H] Home    | [Q/ESC] Quit")
        print("    Cube Reorient: [1-3] goal +90 XYZ | [4-6] -90 | [R] random |")
        print("                   [0] upright | [G] match cube | [A] auto | [X] reset")
        print("=" * 65 + "\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        cap = open_camera(self.camera_id)
        if cap is None:
            cap = cv2.VideoCapture(self.camera_id)  # a closed capture to read from
            print(
                f"[BOOTH WARN] No working camera found (tried index {self.camera_id} "
                "and 0-5). Running in GUI-only mode."
            )

        try:
            while True:
                ret = False
                raw_frame = None
                display_frame = None
                landmarks_list = []
                if cap.isOpened():
                    ret, raw_frame = cap.read()
                    if ret and raw_frame is not None:
                        # 1. Detect hand landmarks on unmirrored raw camera frame (essential for correct anatomy!)
                        landmarks_list = self.tracker.process_frame(raw_frame)
                        # 2. Mirror frame for user display
                        display_frame = cv2.flip(raw_frame, 1)

                if self.current_screen == AppScreen.TELEOP:
                    self.update_teleop_frame(raw_frame, landmarks_list)
                elif self.current_screen == AppScreen.RPS:
                    self.update_rps_frame(raw_frame, landmarks_list)

                # Smooth joint interpolation and physics stepping (Showcase and RPS only)
                self.step_smooth_control(0.02)

                # The reorientation policy runs at the rate it was trained at,
                # which is slower than this loop redraws, so it is paced on its
                # own clock rather than by holding the whole kiosk back.
                if (
                    self.current_screen == AppScreen.REORIENT
                    and self.reorient_controller is not None
                ):
                    self.reorient_controller.step_if_due(time.monotonic())

                canvas = self.render(display_frame, landmarks_list)
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[BOOTH] Exit requested by user.")
                    break

                # The current screen gets first refusal: a screen that prints a
                # shortcut on one of its own buttons has to be the screen that
                # receives it. Global navigation used to be tested first, so
                # [1] [2] [3] on the showcase screen navigated away instead of
                # playing the posture their buttons name.
                if not self._handle_screen_key(key):
                    if key == ord("1"):
                        self.handle_action("goto_teleop")
                    elif key == ord("2"):
                        self.handle_action("goto_rps")
                    elif key == ord("3"):
                        self.handle_action("goto_showcase")
                    elif key == ord("4"):
                        self.handle_action("goto_reorient")
                    elif key in (ord("h"), ord("H")):
                        self.handle_action("back_home_teleop")

        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            self.disarm_robot()
            if self.hardware_controller is not None:
                self.hardware_controller.disconnect()
            if self.reorient_controller is not None:
                self.reorient_controller.close()
            print("[BOOTH] Kiosk shutdown complete.")

        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the LEAP Hand Interactive Booth Kiosk Application."
    )
    parser.add_argument(
        "--mode",
        choices=("hardware", "mujoco", "both"),
        default="hardware",
        help="Target controller backend: hardware, mujoco, or both (default: hardware)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for LEAP Hand hardware (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--profile",
        default="jiwoo",
        help="User neutral calibration profile name (default: jiwoo)",
    )
    parser.add_argument(
        "--calib-path",
        type=Path,
        default=DEFAULT_CALIB_PATH,
        help="Path to neutral calibration JSON file (default: calibration/neutral_calibration.json)",
    )
    parser.add_argument(
        "--motor-calib-path",
        type=Path,
        default=DEFAULT_MOTOR_CALIB_PATH,
        help="Path to hardware motor calibration YAML file (default: calibration/hardware_motors.yaml)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=350,
        help="Motor current limit in mA (default: 350)",
    )
    parser.add_argument(
        "--no-mujoco",
        action="store_true",
        help="Disable MuJoCo simulation viewer",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Disable physical hardware controller",
    )
    parser.add_argument(
        "--reorient-policy",
        type=Path,
        default=DEFAULT_REORIENT_POLICY_PATH,
        help=(
            "Path to the trained cube reorientation policy "
            f"(default: {DEFAULT_REORIENT_POLICY_PATH})"
        ),
    )
    parser.add_argument(
        "--playground-root",
        type=Path,
        default=None,
        help=(
            "MuJoCo Playground checkout holding the reorientation scene "
            "(default: importable package, else ~/Projects/mujoco_playground)"
        ),
    )
    parser.add_argument(
        "--reorient-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "Tilt gravity in the reorientation simulation, to ask what a hand "
            "mounted at a different angle would do (default: 0)"
        ),
    )
    parser.add_argument(
        "--no-reorient",
        action="store_true",
        help="Disable the cube reorientation screen",
    )
    parser.add_argument(
        "--no-shadows",
        action="store_true",
        help=(
            "Draw the flat theme without soft shadows. Cheaper per frame, for "
            "a machine that cannot spare them"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_shadows_enabled(not args.no_shadows)
    enable_mujoco = not args.no_mujoco
    enable_hardware = not args.no_hardware and args.mode in ("hardware", "both")

    app = BoothKioskApp(
        mode=args.mode,
        port=args.port,
        profile=args.profile,
        calib_path=args.calib_path,
        motor_calib_path=args.motor_calib_path,
        camera_id=args.camera_id,
        current_limit=args.current_limit,
        enable_mujoco=enable_mujoco,
        enable_hardware=enable_hardware,
        enable_reorient=not args.no_reorient,
        reorient_policy_path=args.reorient_policy,
        playground_root=args.playground_root,
        reorient_tilt_degrees=args.reorient_tilt_deg,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
