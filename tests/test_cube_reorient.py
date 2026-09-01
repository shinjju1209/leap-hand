"""Tests for the cube reorientation policy and its booth screen."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from booth_app import AppScreen, BoothKioskApp
from cube_reorient import ori_error
from cube_reorient_controller import DEFAULT_POLICY_PATH, CubeReorientController

REPO_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = REPO_ROOT / DEFAULT_POLICY_PATH


@unittest.skipUnless(POLICY_PATH.is_file(), f"policy not present at {POLICY_PATH}")
class CubeReorientControllerTests(unittest.TestCase):
    """The simulation half: does the policy actually manipulate the cube."""

    @classmethod
    def setUpClass(cls) -> None:
        # Loading the scene and the policy costs about a second, and none of
        # these tests need a private copy of either.
        cls.controller = CubeReorientController(POLICY_PATH)

    def setUp(self) -> None:
        self.controller.reset()
        self.controller.reset_statistics()
        self.controller.reset_goal()
        self.controller.auto_goal = False
        self.controller.paused = False

    def test_the_policy_holds_the_cube_while_it_works(self) -> None:
        """Five seconds of policy without dropping it.

        This is the claim the whole screen rests on: if the cube falls out of
        the hand in simulation, nothing about the booth's wiring can save it.
        """
        for _ in range(100):
            self.controller.step()

        self.assertEqual(self.controller.steps, 100)
        self.assertEqual(self.controller.drops, 0, "the cube fell out of the hand")

    def test_a_goal_button_turns_the_goal_a_quarter_turn(self) -> None:
        before = self.controller.goal
        self.controller.rotate_goal(0, 90.0)
        turned = math.degrees(ori_error(before, self.controller.goal))
        self.assertAlmostEqual(turned, 90.0, places=4)

    def test_turning_back_returns_to_where_it_started(self) -> None:
        self.controller.rotate_goal(1, 90.0)
        self.controller.rotate_goal(1, -90.0)
        self.assertAlmostEqual(
            math.degrees(ori_error(np.array([1.0, 0.0, 0.0, 0.0]), self.controller.goal)),
            0.0,
            places=4,
        )

    def test_matching_the_cube_leaves_no_error_to_correct(self) -> None:
        for _ in range(20):
            self.controller.step()
        self.controller.adopt_cube_goal()
        self.assertAlmostEqual(self.controller.goal_error_degrees, 0.0, places=3)

    def test_pausing_stops_the_policy_without_losing_the_pose(self) -> None:
        for _ in range(10):
            self.controller.step()
        held = self.controller.backend.data.ctrl.copy()
        steps_before = self.controller.steps

        self.controller.toggle_pause()
        for _ in range(10):
            self.controller.step()

        self.assertEqual(self.controller.steps, steps_before, "paused policy still acted")
        np.testing.assert_allclose(self.controller.backend.data.ctrl, held)

    def test_auto_goal_moves_the_goal_on_its_own(self) -> None:
        self.controller.auto_goal = True
        before = self.controller.goal
        for _ in range(10):
            self.controller.step()
        self.assertGreater(
            math.degrees(ori_error(before, self.controller.goal)),
            0.0,
            "the training schedule left the goal where it was",
        )

    def test_the_render_is_the_size_that_was_asked_for(self) -> None:
        image = self.controller.render_bgr(240, 320)
        self.assertEqual(image.shape, (240, 320, 3))

    def test_a_second_size_is_honoured_rather_than_ignored(self) -> None:
        """A renderer is fixed to the size it was built at.

        Kept from the first call, it returned the old size for every later
        request without saying so.
        """
        self.assertEqual(self.controller.render_bgr(480, 640).shape, (480, 640, 3))
        self.assertEqual(self.controller.render_bgr(240, 320).shape, (240, 320, 3))
        self.assertEqual(self.controller.render_bgr(480, 640).shape, (480, 640, 3))

    def test_pacing_follows_the_policy_rate_not_the_display(self) -> None:
        """step_if_due must refuse to run faster than the trained rate."""
        # Not the exact boundary: 100.0 + 0.05 - 100.0 is 0.04999... in binary
        # floating point, so testing it measures the representation rather than
        # the pacing.
        self.assertTrue(self.controller.step_if_due(100.0))
        self.assertFalse(
            self.controller.step_if_due(100.0 + self.controller.dt / 2.0),
            "the policy stepped twice inside one control period",
        )
        self.assertTrue(self.controller.step_if_due(100.0 + self.controller.dt * 1.5))

    def test_nothing_on_this_path_can_reach_a_motor(self) -> None:
        """The demo is simulation only, and that has to be true by reading.

        The policy commands joint targets far faster than the teleop path does
        and was trained against a hand mounted at one specific angle, so the
        hand is not somewhere its output may arrive by accident.
        """
        source = (REPO_ROOT / "cube_reorient_controller.py").read_text()
        for forbidden in (
            "LeapHandHardwareController",
            "command_degrees",
            "enable_torque",
            "dynamixel",
            "serial",
        ):
            self.assertNotIn(
                forbidden,
                source,
                f"the reorientation controller mentions {forbidden!r}",
            )


class BoothReorientScreenTests(unittest.TestCase):
    """The booth half: buttons, keys, and staying away from the hand."""

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def _app(self, mock_hw: MagicMock, mock_mj: MagicMock, **kwargs) -> BoothKioskApp:
        options = {"enable_mujoco": False, "enable_hardware": False}
        options.update(kwargs)
        return BoothKioskApp(**options)

    def test_every_button_on_the_screen_does_something(self) -> None:
        app = self._app(enable_reorient=False)
        handled = {
            "back_home_reorient",
            "reorient_goal_x+",
            "reorient_goal_y+",
            "reorient_goal_z+",
            "reorient_goal_x-",
            "reorient_goal_y-",
            "reorient_goal_z-",
            "reorient_random",
            "reorient_reset_goal",
            "reorient_adopt",
            "reorient_auto",
            "reorient_pause",
            "reorient_reset",
        }
        ids = {button.id for button in app.buttons[AppScreen.REORIENT]}
        self.assertEqual(ids, handled)

    def test_each_shortcut_fires_its_own_button(self) -> None:
        """A caption that promises a key has to be the key that acts on it."""
        app = self._app(enable_reorient=False)
        app.current_screen = AppScreen.REORIENT

        bound = {}
        for codes, action_id in app._screen_key_bindings():
            for code in codes:
                bound[code] = action_id

        for button in app.buttons[AppScreen.REORIENT]:
            if button.id == "back_home_reorient":
                continue  # H is global navigation, not a screen key
            character = " " if button.shortcut == "Space" else button.shortcut
            self.assertEqual(
                bound.get(ord(character.lower())),
                button.id,
                f"[{button.shortcut}] does not fire {button.id}",
            )

    def test_screen_keys_are_not_swallowed_by_navigation(self) -> None:
        """A screen that binds a digit keeps it; the home screen does not.

        Global navigation used to be tested first, so [1] [2] [3] on the
        showcase screen navigated away rather than playing the posture their
        own buttons name.
        """
        app = self._app(enable_reorient=False)

        app.current_screen = AppScreen.SHOWCASE
        self.assertTrue(app._handle_screen_key(ord("1")))

        app.current_screen = AppScreen.HOME
        self.assertFalse(app._handle_screen_key(ord("1")))

    def test_the_reorient_screen_never_commands_the_hand(self) -> None:
        """step_smooth_control is the booth's only route to a motor."""
        app = self._app(enable_reorient=False)
        app.hardware_controller = MagicMock()
        app.hardware_controller.torque_enabled = True

        app.current_screen = AppScreen.REORIENT
        app.step_smooth_control(0.02)
        app.hardware_controller.command_degrees.assert_not_called()

        # ... and the same call does command the hand from a screen that should,
        # so this test fails if the guard is simply deleted.
        app.current_screen = AppScreen.SHOWCASE
        app.step_smooth_control(0.02)
        app.hardware_controller.command_degrees.assert_called()

    def test_entering_the_screen_disarms_the_hand(self) -> None:
        """[4] reaches this screen straight from an armed teleop session.

        Nothing here commands the hand, so an armed hand would sit holding
        torque on its last pose while everyone watches the simulation.
        """
        app = self._app(enable_reorient=False)
        app.hardware_controller = MagicMock()
        app.current_screen = AppScreen.TELEOP
        app.handle_action("toggle_arm")
        self.assertTrue(app.armed)

        app.handle_action("goto_reorient")
        self.assertFalse(app.armed, "the hand stayed armed on the reorient screen")
        app.hardware_controller.emergency_stop.assert_called()

    def test_the_booth_still_runs_without_a_policy(self) -> None:
        """A missing policy disables one screen, not the exhibition."""
        app = self._app(reorient_policy_path=Path("no/such/policy.npz"))
        self.assertIsNone(app.reorient_controller)

        app.handle_action("goto_reorient")
        self.assertEqual(app.current_screen, AppScreen.REORIENT)

        canvas = app.render(None, [])
        self.assertEqual(canvas.shape, (app.height, app.width, 3))

    def test_the_home_screen_has_a_card_for_every_mode(self) -> None:
        app = self._app(enable_reorient=False)
        destinations = {
            button.id
            for button in app.buttons[AppScreen.HOME]
            if button.id.startswith("goto_")
        }
        self.assertEqual(
            destinations,
            {"goto_teleop", "goto_rps", "goto_showcase", "goto_reorient"},
        )

    def test_the_mode_cards_do_not_overlap(self) -> None:
        """Four cards have to still fit the canvas they are centred in."""
        app = self._app(enable_reorient=False)
        cards = sorted(
            (button.rect for button in app.buttons[AppScreen.HOME]
             if button.id.startswith("goto_")),
            key=lambda rect: rect[0],
        )
        for left, right in zip(cards, cards[1:]):
            self.assertLessEqual(left[0] + left[2], right[0], "mode cards overlap")
        self.assertGreaterEqual(cards[0][0], 0)
        self.assertLessEqual(cards[-1][0] + cards[-1][2], app.width)


if __name__ == "__main__":
    unittest.main()
