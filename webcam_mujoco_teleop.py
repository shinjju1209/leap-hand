"""Run MediaPipe webcam tracking and right LEAP Hand MuJoCo teleoperation."""

from webcam_hand_tracking import main


if __name__ == "__main__":
    main(default_mujoco=True)
