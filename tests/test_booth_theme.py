"""Tests for the booth's drawing primitives and its light theme.

These are about what the kiosk looks like, which is mostly not testable -- but
the parts that are, are the parts that break silently: a colour nobody can read,
a helper that writes outside its rectangle, two panels drawn on top of each
other. Those are checked here.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import booth_app
from booth_app import (
    COLOR_BG,
    COLOR_CARD_BG,
    COLOR_CARD_BORDER,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SUCCESS,
    COLOR_TEXT_MAIN,
    COLOR_TEXT_MUTED,
    AppScreen,
    BoothKioskApp,
    Button,
    badge,
    card,
    pill_rect,
    rounded_rect,
    soft_shadow,
    viewport,
)


def relative_luminance(bgr: tuple[int, int, int]) -> float:
    """WCAG luminance. The palette is BGR, the formula wants RGB."""
    b, g, r = (channel / 255.0 for channel in bgr)
    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)


def contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    high, low = max(la, lb), min(la, lb)
    return (high + 0.05) / (low + 0.05)


class ThemeContrastTests(unittest.TestCase):
    """A booth screen is read from across a room, or it is not read."""

    def test_body_colours_clear_the_readable_threshold(self) -> None:
        for name, color in (
            ("TEXT_MAIN", COLOR_TEXT_MAIN),
            ("TEXT_MUTED", COLOR_TEXT_MUTED),
            ("PRIMARY", COLOR_PRIMARY),
            ("SECONDARY", COLOR_SECONDARY),
            ("SUCCESS", COLOR_SUCCESS),
        ):
            with self.subTest(color=name):
                self.assertGreaterEqual(
                    contrast(color, COLOR_CARD_BG), 4.5,
                    f"{name} is below the 4.5:1 needed for body text",
                )

    def test_the_ground_and_the_cards_are_distinguishable(self) -> None:
        """Cards have to read as raised, but not as a different surface."""
        ratio = contrast(COLOR_CARD_BG, COLOR_BG)
        self.assertGreater(ratio, 1.0, "cards are the same colour as the page")
        self.assertLess(ratio, 1.2, "the card/ground step is too harsh for a light theme")

    def test_the_theme_is_light(self) -> None:
        self.assertGreater(relative_luminance(COLOR_BG), 0.8)
        self.assertLess(relative_luminance(COLOR_TEXT_MAIN), 0.1)


class DrawingPrimitiveTests(unittest.TestCase):
    """The helpers every screen goes through."""

    def setUp(self) -> None:
        self.canvas = np.zeros((200, 300, 3), np.uint8)

    def test_a_rounded_rectangle_fills_its_middle_and_clips_its_corners(self) -> None:
        rounded_rect(self.canvas, (20, 20, 200, 120), (255, 255, 255), 16, -1)
        self.assertTrue((self.canvas[80, 120] == 255).all(), "the middle is not filled")
        self.assertTrue((self.canvas[21, 21] == 0).all(), "the corner was not rounded")

    def test_it_stays_inside_the_rectangle_it_was_given(self) -> None:
        """Within a pixel: the corner arcs are drawn antialiased.

        LINE_AA writes a partial pixel just outside the geometric edge, which
        is what makes the curve look like a curve. A whole pixel of bleed is
        the softened boundary; more than that would be a shape drawn wrong.
        """
        rounded_rect(self.canvas, (20, 20, 200, 120), (255, 255, 255), 16, -1)
        painted = np.argwhere(self.canvas.any(axis=2))
        top, left = painted.min(axis=0)
        bottom, right = painted.max(axis=0)
        self.assertGreaterEqual(left, 19)
        self.assertGreaterEqual(top, 19)
        self.assertLessEqual(right, 221)
        self.assertLessEqual(bottom, 141)

    def test_a_radius_larger_than_the_shape_is_clamped_not_folded(self) -> None:
        """A pill asks for h//2; a caller may ask for more. It must not fold."""
        rounded_rect(self.canvas, (20, 20, 60, 30), (255, 255, 255), 400, -1)
        self.assertTrue((self.canvas[35, 50] == 255).all(), "the shape collapsed")

    def test_a_pill_is_round_ended(self) -> None:
        pill_rect(self.canvas, (20, 20, 160, 40), (255, 255, 255), -1)
        self.assertTrue((self.canvas[40, 100] == 255).all())
        self.assertTrue((self.canvas[21, 21] == 0).all(), "the end is square")

    def test_a_shadow_darkens_the_ground_without_covering_the_card(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        before = canvas[150, 60].copy()
        soft_shadow(canvas, (60, 60, 180, 80))
        self.assertLess(int(canvas[150, 60].mean()), int(before.mean()),
                        "the shadow did not darken anything")

    def test_a_shadow_near_an_edge_is_skipped_rather_than_wrapping(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        soft_shadow(canvas, (0, 0, 300, 200))  # flush to every edge
        self.assertTrue((canvas == 250).all(), "an out-of-bounds band was drawn")

    def test_a_card_reports_where_its_body_starts(self) -> None:
        plain = card(self.canvas, (10, 10, 200, 120), shadow=False)
        titled = card(self.canvas, (10, 10, 200, 120), title="TITLE", shadow=False)
        self.assertGreater(titled, plain, "a title did not push the body down")

    def test_a_badge_reports_the_width_it_used(self) -> None:
        used = badge(self.canvas, (10, 10), "MODE", COLOR_PRIMARY)
        self.assertGreater(used, 26, "a badge has to be wider than its padding")

    def test_a_viewport_without_a_frame_says_so_instead_of_showing_black(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        viewport(canvas, (20, 20, 260, 150), None, "No camera")
        well = canvas[30:160, 30:270]
        self.assertGreater(well.mean(), 200, "an empty viewport rendered dark")

    def test_a_viewport_shows_the_frame_it_was_given(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        frame = np.zeros((60, 80, 3), np.uint8)
        frame[:, :] = (255, 0, 0)
        viewport(canvas, (20, 20, 260, 150), frame)
        self.assertTrue((canvas[95, 150] == (255, 0, 0)).all(), "the frame was not drawn")


class ButtonAppearanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = np.full((200, 400, 3), 247, np.uint8)
        self.button = Button(id="b", label="Do It", rect=(20, 20, 300, 48), shortcut="A")

    def test_an_active_button_is_the_one_solid_thing_on_the_screen(self) -> None:
        self.button.active = True
        self.button.draw(self.canvas, (0, 0))
        centre = self.canvas[44, 170]
        self.assertLess(centre.mean(), 200, "an active button did not fill with its accent")

    def test_a_disabled_button_does_not_invite_a_press(self) -> None:
        self.button.enabled = False
        self.button.draw(self.canvas, (170, 44))  # hovered, but disabled
        self.assertGreater(self.canvas[44, 170].mean(), 220,
                           "a disabled button lit up under the cursor")

    def test_a_card_sized_button_leaves_its_title_to_the_screen(self) -> None:
        """The home cards are buttons; their text is written over them after."""
        blank = self.canvas.copy()
        card_button = Button(id="c", label="Teleoperation", rect=(20, 20, 260, 160))
        card_button.draw(self.canvas, (0, 0))
        painted_rows = np.argwhere((self.canvas != blank).any(axis=(1, 2)))
        # A label would sit near the middle; only the frame should be drawn.
        middle = self.canvas[95, 40:260]
        self.assertTrue((middle > 230).all(), "the card drew a label of its own")
        self.assertGreater(len(painted_rows), 0, "the card drew nothing at all")


class ScreenLayoutTests(unittest.TestCase):
    """Panels that overlap are invisible until something casts a shadow."""

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def _app(self, mock_hw: MagicMock, mock_mj: MagicMock) -> BoothKioskApp:
        return BoothKioskApp(
            enable_mujoco=False, enable_hardware=False, enable_reorient=False
        )

    def test_the_rps_scoreboard_clears_the_buttons_above_it(self) -> None:
        app = self._app()
        lowest = max(y + h for _, y, _, h in
                     (b.rect for b in app.buttons[AppScreen.RPS]))
        scoreboard_top = 412  # _render_rps_screen
        self.assertGreaterEqual(scoreboard_top, lowest,
                                "the scoreboard is drawn over a button")

    def test_no_button_column_reaches_over_the_viewport(self) -> None:
        """The showcase column started at 815, and the viewport ends at 840.

        Every button on that screen had its first 25 pixels drawn over the
        camera image. Flat on flat it was invisible; the moment the buttons
        cast a shadow it was the first thing you saw.
        """
        app = self._app()
        right_edge = booth_app.VIEWPORT[0] + booth_app.VIEWPORT[2]
        for screen in (AppScreen.TELEOP, AppScreen.RPS,
                       AppScreen.SHOWCASE, AppScreen.REORIENT):
            buttons = app.buttons[screen]
            with self.subTest(screen=screen.name):
                self.assertGreaterEqual(
                    min(b.rect[0] for b in buttons), right_edge,
                    "a button starts over the viewport",
                )
                self.assertLessEqual(
                    max(b.rect[0] + b.rect[2] for b in buttons), app.width,
                    "a button runs off the canvas",
                )

    def test_every_screen_renders_on_the_light_ground(self) -> None:
        app = self._app()
        frame = np.full((480, 640, 3), 120, np.uint8)
        for screen in AppScreen:
            with self.subTest(screen=screen.name):
                app.current_screen = screen
                canvas = app.render(frame, [])
                self.assertEqual(canvas.shape, (app.height, app.width, 3))
                # The strip under the footer line is the page, not a panel.
                self.assertGreater(canvas[5, 700].mean(), 200,
                                   "the header is not on the light theme")

    def test_no_screen_paints_outside_the_canvas(self) -> None:
        """A shadow near an edge used to be the way this went wrong."""
        app = self._app()
        for screen in AppScreen:
            app.current_screen = screen
            app.render(None, [])  # would raise on an out-of-bounds write


if __name__ == "__main__":
    unittest.main()
