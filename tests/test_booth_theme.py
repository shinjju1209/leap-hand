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
    DEFAULT_LOGO_PATH,
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
    draw_logo,
    load_logo,
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

    def test_a_shadow_darkens_the_ground_under_what_casts_it(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        soft_shadow(canvas, (60, 60, 180, 80))
        # Directly below the middle, where a shadow is densest. The corners
        # fall off to nearly nothing by design.
        self.assertLess(int(canvas[145, 150].mean()), 250,
                        "the shadow did not darken anything")

    def test_a_shadow_is_clipped_to_the_canvas(self) -> None:
        """Flush to every edge: the shadow is cut, not wrapped or refused."""
        canvas = np.full((200, 300, 3), 250, np.uint8)
        soft_shadow(canvas, (0, 0, 300, 200))
        self.assertLessEqual(int(canvas.max()), 250, "a pixel got brighter")

    def test_a_shadow_entirely_off_canvas_draws_nothing(self) -> None:
        canvas = np.full((200, 300, 3), 250, np.uint8)
        soft_shadow(canvas, (900, 900, 100, 40))
        self.assertTrue((canvas == 250).all(), "it drew outside the canvas")

    def test_the_shadow_shape_is_built_once_and_reused(self) -> None:
        """It is the same few button and card sizes, every frame, forever."""
        booth_app._shadow_mask.cache_clear()
        canvas = np.full((200, 300, 3), 250, np.uint8)
        for _ in range(5):
            soft_shadow(canvas, (60, 60, 180, 80))
        info = booth_app._shadow_mask.cache_info()
        self.assertEqual(info.misses, 1, "the falloff was rebuilt")
        self.assertEqual(info.hits, 4)

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


class LogoTests(unittest.TestCase):
    """The mark is a dark logo on a white artboard, not a cut-out."""

    def test_the_mark_ships_with_the_repo(self) -> None:
        self.assertTrue(DEFAULT_LOGO_PATH.is_file(),
                        f"the brand mark is missing at {DEFAULT_LOGO_PATH}")

    def test_it_is_scaled_to_the_height_asked_for(self) -> None:
        logo = load_logo(DEFAULT_LOGO_PATH, 38)
        self.assertIsNotNone(logo)
        self.assertEqual(logo.shape[0], 38)
        self.assertGreater(logo.shape[1], 0, "the aspect ratio collapsed")

    def test_a_missing_file_is_not_fatal(self) -> None:
        """A booth with no logo still has to open."""
        self.assertIsNone(load_logo(booth_app.Path("no/such/logo.png"), 38))

    def test_the_white_artboard_does_not_paint_a_box(self) -> None:
        """Multiplication is what keeps the mark from arriving in a rectangle.

        The file has no alpha channel, so a straight blit would stamp a white
        square onto the header. White multiplies to 1.0 and leaves the ground
        untouched; only the mark itself darkens anything.
        """
        canvas = np.full((80, 300, 3), 247, np.uint8)
        logo = load_logo(DEFAULT_LOGO_PATH, 38)
        used = draw_logo(canvas, logo, (20, 20))
        self.assertGreater(used, 0)
        # A corner of the logo's own box: artboard, so the ground survives.
        self.assertTrue((canvas[21, 21] == 247).all(),
                        "the artboard was stamped onto the header")
        # Somewhere inside the mark, the canvas got darker.
        patch_region = canvas[20:58, 20:20 + used]
        self.assertLess(patch_region.min(), 200, "the mark did not draw")

    def test_it_declines_to_draw_off_the_canvas(self) -> None:
        canvas = np.full((80, 300, 3), 247, np.uint8)
        logo = load_logo(DEFAULT_LOGO_PATH, 38)
        self.assertEqual(draw_logo(canvas, logo, (290, 70)), 0)
        self.assertTrue((canvas == 247).all(), "it drew anyway")


class ShadowCostTests(unittest.TestCase):
    """Shadows were what made the redesign too slow to run a kiosk on."""

    def tearDown(self) -> None:
        booth_app.set_shadows_enabled(True)

    def test_a_shadow_on_flat_ground_is_composited_once(self) -> None:
        """The viewport is the same rectangle, on the same page, every frame."""
        booth_app._shadow_on_ground.cache_clear()
        canvas = np.full((700, 900, 3), COLOR_BG, np.uint8)
        for _ in range(5):
            soft_shadow(canvas, (40, 85, 800, 530), ground=COLOR_BG)
        info = booth_app._shadow_on_ground.cache_info()
        self.assertEqual(info.misses, 1, "the composite was redone")
        self.assertEqual(info.hits, 4)

    def test_the_cached_path_matches_the_computed_one(self) -> None:
        """The fast path has to be the same picture, not merely a fast one."""
        rect = (30, 30, 120, 60)
        slow = np.full((200, 300, 3), COLOR_BG, np.uint8)
        fast = slow.copy()
        soft_shadow(slow, rect)
        soft_shadow(fast, rect, ground=COLOR_BG)
        self.assertLessEqual(int(np.abs(slow.astype(int) - fast.astype(int)).max()), 1,
                             "the cached shadow differs from the computed one")

    def test_shadows_can_be_turned_off_wholesale(self) -> None:
        canvas = np.full((200, 300, 3), COLOR_BG, np.uint8)
        booth_app.set_shadows_enabled(False)
        soft_shadow(canvas, (60, 60, 180, 80))
        self.assertTrue((canvas == COLOR_BG).all(), "a shadow was drawn anyway")

    def test_the_flag_reaches_the_switch(self) -> None:
        args = booth_app.parse_args(["--no-shadows"])
        self.assertTrue(args.no_shadows)
        args = booth_app.parse_args([])
        self.assertFalse(args.no_shadows)

    def test_a_screen_still_renders_without_shadows(self) -> None:
        booth_app.set_shadows_enabled(False)
        with patch("booth_app.MujocoHandController"), \
             patch("booth_app.LeapHandHardwareController"):
            app = BoothKioskApp(enable_mujoco=False, enable_hardware=False,
                                enable_reorient=False)
        for screen in AppScreen:
            app.current_screen = screen
            canvas = app.render(None, [])
            self.assertEqual(canvas.shape, (app.height, app.width, 3))


class CameraFallbackTests(unittest.TestCase):
    """Opening an index is not the same as getting a picture out of it."""

    @staticmethod
    def _capture(opened: bool, reads: bool) -> MagicMock:
        capture = MagicMock()
        capture.isOpened.return_value = opened
        capture.read.return_value = (reads, np.zeros((4, 4, 3), np.uint8))
        return capture

    def test_the_preferred_index_is_used_when_it_works(self) -> None:
        good = self._capture(True, True)
        with patch("booth_app.cv2.VideoCapture", return_value=good) as factory:
            self.assertIs(booth_app.open_camera(0), good)
        factory.assert_called_once_with(0)

    def test_an_index_that_opens_but_reads_nothing_is_passed_over(self) -> None:
        """This machine has one: /dev/video0 opens and never yields a frame.

        The booth defaults to index 0, so without this it sat in GUI-only mode
        on a machine with a working camera plugged in.
        """
        silent, working = self._capture(True, False), self._capture(True, True)
        with patch("booth_app.cv2.VideoCapture",
                   side_effect=[silent, working]) as factory:
            self.assertIs(booth_app.open_camera(0), working)
        self.assertEqual([call.args[0] for call in factory.call_args_list], [0, 1])
        silent.release.assert_called_once()

    def test_a_rejected_capture_is_released(self) -> None:
        """Otherwise the device stays claimed and the next open fails too."""
        dead = self._capture(False, False)
        with patch("booth_app.cv2.VideoCapture", return_value=dead):
            self.assertIsNone(booth_app.open_camera(0, scan_limit=3))
        self.assertEqual(dead.release.call_count, 3)

    def test_nothing_working_reports_none_rather_than_a_dead_capture(self) -> None:
        with patch("booth_app.cv2.VideoCapture",
                   return_value=self._capture(False, False)):
            self.assertIsNone(booth_app.open_camera(2, scan_limit=4))


class ButtonAppearanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = np.full((200, 400, 3), 247, np.uint8)
        self.button = Button(id="b", label="Do It", rect=(20, 20, 300, 48), shortcut="A")

    def test_an_active_button_is_the_one_solid_thing_on_the_screen(self) -> None:
        self.button.active = True
        self.button.draw(self.canvas, (0, 0))
        centre = self.canvas[44, 170]
        self.assertLess(centre.mean(), 200, "an active button did not fill with its accent")

    def test_a_resting_pill_carries_no_shadow(self) -> None:
        """Shadows are for cards, and for whatever is being pointed at.

        One per button per frame was about 4.7 ms of a redraw on the screens
        with a dozen of them, and the site puts shadows under cards anyway.
        """
        canvas = np.full((200, 400, 3), 247, np.uint8)
        self.button.draw(canvas, (0, 0))  # not hovered
        # Just under the button, where a shadow would be densest.
        self.assertTrue((canvas[73, 170] == 247).all(),
                        "a resting button cast a shadow")

    def test_a_pointed_at_button_lifts(self) -> None:
        canvas = np.full((200, 400, 3), 247, np.uint8)
        self.button.draw(canvas, (170, 44))  # hovered
        self.assertTrue((canvas[73, 170] != 247).any(),
                        "a hovered button did not lift off the page")

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
