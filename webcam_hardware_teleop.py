"""Run MediaPipe webcam tracking and right LEAP Hand v1 hardware teleoperation."""

from webcam_hand_tracking import main


if __name__ == "__main__":
    main(default_hardware=True)
