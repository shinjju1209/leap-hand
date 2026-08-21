"""Safety-oriented hardware API for a 16-motor LEAP Hand v1."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import ceil, pi
from pathlib import Path
from typing import Any

import numpy as np

from hand_angles import ANGLE_NAMES
from hardware_calibration import HardwareMotorCalibration


PROTOCOL_VERSION = 2.0
DEFAULT_BAUDRATE = 4_000_000
DEFAULT_MOTOR_IDS = tuple(range(16))

# XC330 / DYNAMIXEL X-series control table addresses used by LEAP Hand v1.
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_POSITION_D_GAIN = 80
ADDR_POSITION_I_GAIN = 82
ADDR_POSITION_P_GAIN = 84
ADDR_BUS_WATCHDOG = 98
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

CURRENT_BASED_POSITION_MODE = 5
POSITION_SCALE_RADIANS = 2.0 * pi / 4096.0
VELOCITY_SCALE_RADIANS_PER_SECOND = 0.229 * 2.0 * pi / 60.0

# Official LEAPsim safety bounds. Nominal hardware alignment uses a pi-radian
# open pose; a saved HardwareMotorCalibration can override it per motor.
SIM_MIN_RADIANS = np.asarray(
    [
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -0.349, -0.470, -1.200, -1.340,
    ],
    dtype=np.float64,
)
SIM_MAX_RADIANS = np.asarray(
    [
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        2.094, 2.443, 1.900, 1.880,
    ],
    dtype=np.float64,
)


def _vector16(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (16,):
        raise ValueError(f"{name} must contain exactly 16 values, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


def _unsigned_to_signed(value: int, size: int) -> int:
    sign_bit = 1 << (8 * size - 1)
    return value - (1 << (8 * size)) if value & sign_bit else value


def clip_sim_radians(sim_radians: Sequence[float]) -> np.ndarray:
    """Clip 16 relative joint angles to the official LEAPsim bounds."""
    return np.clip(_vector16(sim_radians, "sim_radians"), SIM_MIN_RADIANS, SIM_MAX_RADIANS)


def sim_radians_to_motor_radians(sim_radians: Sequence[float]) -> np.ndarray:
    """Convert zero-open simulation radians to LEAP v1 motor radians."""
    return clip_sim_radians(sim_radians) + pi


def motor_radians_to_sim_radians(motor_radians: Sequence[float]) -> np.ndarray:
    """Convert LEAP v1 motor feedback to zero-open simulation radians."""
    return _vector16(motor_radians, "motor_radians") - pi


@dataclass(frozen=True)
class LeapHandFeedback:
    positions_degrees: np.ndarray
    velocities_degrees_per_second: np.ndarray
    currents_milliamps: np.ndarray


@dataclass(frozen=True)
class LeapHandHealth:
    temperatures_celsius: np.ndarray
    input_voltages: np.ndarray
    hardware_errors: np.ndarray


class LeapHandHardwareController:
    """Connect and command a LEAP Hand v1 without moving on connection.

    ``connect`` verifies IDs and forces torque off. ``configure`` writes the
    low-current control configuration while torque remains off. Only the
    explicit ``enable_torque`` call energizes the motors, after first seeding
    every goal position with its present position to prevent a startup jump.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = DEFAULT_BAUDRATE,
        motor_ids: Sequence[int] | None = None,
        current_limit_milliamps: int = 300,
        position_p_gain: int = 600,
        position_i_gain: int = 0,
        position_d_gain: int = 200,
        side_gain_scale: float = 0.75,
        bus_watchdog_milliseconds: int = 500,
        max_joint_speed_degrees_per_second: float = 120.0,
        max_command_interval_seconds: float = 0.1,
        motor_calibration: HardwareMotorCalibration | str | Path | None = None,
        sdk_module: Any | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not port.strip():
            raise ValueError("port must not be empty")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if motor_calibration is None:
            ids = tuple(
                int(motor_id)
                for motor_id in (DEFAULT_MOTOR_IDS if motor_ids is None else motor_ids)
            )
            calibration = HardwareMotorCalibration.nominal(ids)
        else:
            calibration = (
                HardwareMotorCalibration.load(motor_calibration)
                if isinstance(motor_calibration, (str, Path))
                else motor_calibration
            )
            if not isinstance(calibration, HardwareMotorCalibration):
                raise TypeError("motor_calibration must be a path or HardwareMotorCalibration")
            ids = calibration.motor_ids
            if motor_ids is not None and tuple(int(motor_id) for motor_id in motor_ids) != ids:
                raise ValueError("motor_ids conflicts with the motor calibration file")
        if len(ids) != 16 or len(set(ids)) != 16:
            raise ValueError("motor_ids must contain 16 unique IDs")
        if not 1 <= current_limit_milliamps <= 550:
            raise ValueError("current limit must be between 1 and 550 mA")
        if min(position_p_gain, position_i_gain, position_d_gain) < 0:
            raise ValueError("PID gains cannot be negative")
        if not np.isfinite(side_gain_scale) or not 0.0 < side_gain_scale <= 1.0:
            raise ValueError("side_gain_scale must be in the interval (0, 1]")
        if not 20 <= bus_watchdog_milliseconds <= 2540:
            raise ValueError("bus watchdog must be between 20 and 2540 ms")
        if (
            not np.isfinite(max_joint_speed_degrees_per_second)
            or max_joint_speed_degrees_per_second <= 0.0
        ):
            raise ValueError("max joint speed must be finite and greater than zero")
        if (
            not np.isfinite(max_command_interval_seconds)
            or max_command_interval_seconds <= 0.0
        ):
            raise ValueError(
                "max command interval must be finite and greater than zero"
            )

        self.port = port.strip()
        self.baudrate = int(baudrate)
        self.motor_ids = ids
        self.motor_calibration = calibration
        self._side_motor_ids = {self.motor_ids[index] for index in (0, 4, 8)}
        self.current_limit_milliamps = int(current_limit_milliamps)
        self.position_p_gain = int(position_p_gain)
        self.position_i_gain = int(position_i_gain)
        self.position_d_gain = int(position_d_gain)
        self.side_gain_scale = float(side_gain_scale)
        self.bus_watchdog_milliseconds = int(bus_watchdog_milliseconds)
        self.max_joint_speed_degrees_per_second = float(
            max_joint_speed_degrees_per_second
        )
        self.max_command_interval_seconds = float(max_command_interval_seconds)
        self._clock = clock or time.monotonic

        self._sdk = sdk_module
        self._port_handler = None
        self._packet_handler = None
        self._feedback_reader = None
        self._goal_writer = None
        self._connected = False
        self._configured = False
        self._torque_enabled = False
        self._model_numbers: dict[int, int] = {}
        self._last_command_degrees = np.zeros(16, dtype=np.float64)
        self._last_goal_motor_radians: np.ndarray | None = None
        self._last_command_time_seconds: float | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_configured(self) -> bool:
        return self._configured

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    @property
    def model_numbers(self) -> dict[int, int]:
        return dict(self._model_numbers)

    @property
    def last_command_degrees(self) -> np.ndarray:
        return self._last_command_degrees.copy()

    def connect(self) -> dict[int, int]:
        """Open the port, verify all 16 IDs, and leave every motor torque-off."""
        if self._connected:
            raise RuntimeError("LEAP Hand is already connected")
        if self._sdk is None:
            self._sdk = import_module("dynamixel_sdk")

        port_handler = self._sdk.PortHandler(self.port)
        if not port_handler.openPort():
            raise OSError(f"Failed to open DYNAMIXEL port: {self.port}")
        if not port_handler.setBaudRate(self.baudrate):
            port_handler.closePort()
            raise OSError(f"Failed to set DYNAMIXEL baudrate: {self.baudrate}")

        self._port_handler = port_handler
        self._packet_handler = self._sdk.PacketHandler(PROTOCOL_VERSION)
        self._connected = True
        try:
            missing_ids: list[int] = []
            for motor_id in self.motor_ids:
                model_number, comm_result, dxl_error = self._packet_handler.ping(
                    self._port_handler,
                    motor_id,
                )
                if comm_result != self._sdk.COMM_SUCCESS or dxl_error:
                    missing_ids.append(motor_id)
                else:
                    self._model_numbers[motor_id] = int(model_number)
            if missing_ids:
                raise ConnectionError(
                    f"Missing or unresponsive DYNAMIXEL IDs: {missing_ids}"
                )

            self._set_torque(False)
            self._feedback_reader = self._sdk.GroupSyncRead(
                self._port_handler,
                self._packet_handler,
                ADDR_PRESENT_CURRENT,
                10,
            )
            for motor_id in self.motor_ids:
                if not self._feedback_reader.addParam(motor_id):
                    raise OSError(f"Failed to add feedback reader for ID {motor_id}")
            self._goal_writer = self._sdk.GroupSyncWrite(
                self._port_handler,
                self._packet_handler,
                ADDR_GOAL_POSITION,
                4,
            )
        except Exception:
            self.close()
            raise
        return self.model_numbers

    def configure(self) -> None:
        """Configure official low-current position control while torque is off."""
        self._require_connected()
        if self._torque_enabled:
            raise RuntimeError("Disable torque before configuring motors")

        for motor_id in self.motor_ids:
            self._write_register(motor_id, ADDR_OPERATING_MODE, 1, CURRENT_BASED_POSITION_MODE)
            p_gain = self.position_p_gain
            d_gain = self.position_d_gain
            if motor_id in self._side_motor_ids:
                p_gain = round(p_gain * self.side_gain_scale)
                d_gain = round(d_gain * self.side_gain_scale)
            self._write_register(motor_id, ADDR_POSITION_P_GAIN, 2, p_gain)
            self._write_register(motor_id, ADDR_POSITION_I_GAIN, 2, self.position_i_gain)
            self._write_register(motor_id, ADDR_POSITION_D_GAIN, 2, d_gain)
            self._write_register(
                motor_id,
                ADDR_GOAL_CURRENT,
                2,
                self.current_limit_milliamps,
            )

        watchdog_ticks = min(127, ceil(self.bus_watchdog_milliseconds / 20.0))
        for motor_id in self.motor_ids:
            # Writing zero also clears a latched watchdog error from a prior run.
            self._write_register(motor_id, ADDR_BUS_WATCHDOG, 1, 0)
            self._write_register(motor_id, ADDR_BUS_WATCHDOG, 1, watchdog_ticks)
        self._configured = True

    def enable_torque(self) -> None:
        """Seed present positions as goals, then explicitly enable torque."""
        self._require_connected()
        if not self._configured:
            raise RuntimeError("Call configure() before enabling torque")
        if self._torque_enabled:
            return

        present_motor_radians = self._read_present_motor_radians()
        self._sync_write_motor_radians(present_motor_radians)
        self._last_goal_motor_radians = present_motor_radians.copy()
        try:
            self._set_torque(True)
        except OSError:
            self.emergency_stop()
            raise
        self._torque_enabled = True
        present_sim = self.motor_calibration.motor_to_sim_radians(
            present_motor_radians
        )
        self._last_command_degrees = np.rad2deg(present_sim)
        self._last_command_time_seconds = float(self._clock())

    def command_degrees(self, angles_degrees: Sequence[float]) -> np.ndarray:
        """Clip and slew-limit 16 zero-open angles in ``ANGLE_NAMES`` order."""
        self._require_connected()
        if not self._torque_enabled:
            raise RuntimeError("Torque is disabled; call enable_torque() explicitly")
        requested_radians = np.deg2rad(_vector16(angles_degrees, "angles_degrees"))
        target_degrees = np.rad2deg(clip_sim_radians(requested_radians))

        now_seconds = float(self._clock())
        if not np.isfinite(now_seconds):
            raise RuntimeError("monotonic clock returned a non-finite value")
        if self._last_command_time_seconds is None:
            raise RuntimeError("Command timing was not initialized by enable_torque()")
        elapsed_seconds = max(0.0, now_seconds - self._last_command_time_seconds)
        limited_interval = min(elapsed_seconds, self.max_command_interval_seconds)
        max_change_degrees = (
            self.max_joint_speed_degrees_per_second * limited_interval
        )
        safe_command_degrees = np.clip(
            target_degrees,
            self._last_command_degrees - max_change_degrees,
            self._last_command_degrees + max_change_degrees,
        )

        # The seeded present position can be just outside the nominal model bounds.
        # Move it toward the clipped target at the speed limit instead of snapping it.
        motor_radians = self.motor_calibration.sim_to_motor_radians(
            np.deg2rad(safe_command_degrees)
        )
        self._sync_write_motor_radians(motor_radians)
        self._last_goal_motor_radians = motor_radians.copy()
        self._last_command_degrees = safe_command_degrees
        self._last_command_time_seconds = now_seconds
        return self.last_command_degrees

    def heartbeat(self) -> None:
        """Re-send the exact last raw goal without conversion or limit changes."""
        self._require_connected()
        if not self._torque_enabled or self._last_goal_motor_radians is None:
            raise RuntimeError("Torque is disabled; there is no active goal")
        self._sync_write_motor_radians(self._last_goal_motor_radians)

    def read_feedback(self) -> LeapHandFeedback:
        """Read synchronized position, velocity, and current feedback."""
        self._require_connected()
        if self._feedback_reader is None:
            raise RuntimeError("Feedback reader is not initialized")
        comm_result = self._feedback_reader.txRxPacket()
        self._check_communication(comm_result, operation="feedback sync read")

        motor_positions = np.zeros(16, dtype=np.float64)
        velocities = np.zeros(16, dtype=np.float64)
        currents = np.zeros(16, dtype=np.float64)
        for index, motor_id in enumerate(self.motor_ids):
            if not self._feedback_reader.isAvailable(
                motor_id,
                ADDR_PRESENT_CURRENT,
                10,
            ):
                raise OSError(f"Feedback unavailable for DYNAMIXEL ID {motor_id}")
            raw_current = self._feedback_reader.getData(
                motor_id,
                ADDR_PRESENT_CURRENT,
                2,
            )
            raw_velocity = self._feedback_reader.getData(
                motor_id,
                ADDR_PRESENT_VELOCITY,
                4,
            )
            raw_position = self._feedback_reader.getData(
                motor_id,
                ADDR_PRESENT_POSITION,
                4,
            )
            currents[index] = _unsigned_to_signed(int(raw_current), 2)
            velocities[index] = (
                _unsigned_to_signed(int(raw_velocity), 4)
                * VELOCITY_SCALE_RADIANS_PER_SECOND
            )
            motor_positions[index] = (
                _unsigned_to_signed(int(raw_position), 4)
                * POSITION_SCALE_RADIANS
            )

        return LeapHandFeedback(
            positions_degrees=np.rad2deg(
                self.motor_calibration.motor_to_sim_radians(motor_positions)
            ),
            velocities_degrees_per_second=np.rad2deg(velocities),
            currents_milliamps=currents,
        )

    def read_health(self) -> LeapHandHealth:
        """Read temperature, voltage, and hardware error flags for all motors."""
        self._require_connected()
        temperatures = np.zeros(16, dtype=np.float64)
        voltages = np.zeros(16, dtype=np.float64)
        errors = np.zeros(16, dtype=np.uint8)
        for index, motor_id in enumerate(self.motor_ids):
            errors[index] = self._read_register(
                motor_id,
                ADDR_HARDWARE_ERROR,
                1,
            )
            voltages[index] = 0.1 * self._read_register(
                motor_id,
                ADDR_PRESENT_INPUT_VOLTAGE,
                2,
            )
            temperatures[index] = self._read_register(
                motor_id,
                ADDR_PRESENT_TEMPERATURE,
                1,
            )
        return LeapHandHealth(temperatures, voltages, errors)

    def emergency_stop(self) -> tuple[int, ...]:
        """Request torque-off and return IDs that did not acknowledge it."""
        failed_ids: tuple[int, ...] = ()
        if self._connected:
            failed_ids = self._set_torque(False, tolerate_errors=True)
        self._torque_enabled = bool(failed_ids)
        return failed_ids

    def close(self) -> tuple[int, ...]:
        """Torque off, close the port, and return unacknowledged motor IDs."""
        failed_ids: tuple[int, ...] = ()
        if self._connected:
            failed_ids = self.emergency_stop()
        if self._port_handler is not None:
            self._port_handler.closePort()
        self._connected = False
        self._configured = False
        self._torque_enabled = bool(failed_ids)
        self._feedback_reader = None
        self._goal_writer = None
        self._last_goal_motor_radians = None
        self._last_command_time_seconds = None
        return failed_ids

    def _read_present_motor_radians(self) -> np.ndarray:
        values = np.zeros(16, dtype=np.float64)
        for index, motor_id in enumerate(self.motor_ids):
            raw_position = self._read_register(
                motor_id,
                ADDR_PRESENT_POSITION,
                4,
                signed=True,
            )
            values[index] = raw_position * POSITION_SCALE_RADIANS
        return values

    def read_motor_positions_radians(self) -> np.ndarray:
        """Read raw DYNAMIXEL positions without applying any calibration."""
        self._require_connected()
        return self._read_present_motor_radians().copy()

    def _sync_write_motor_radians(self, motor_radians: Sequence[float]) -> None:
        values = _vector16(motor_radians, "motor_radians")
        if self._goal_writer is None:
            raise RuntimeError("Goal writer is not initialized")
        self._goal_writer.clearParam()
        for motor_id, radians in zip(self.motor_ids, values):
            raw_position = int(round(radians / POSITION_SCALE_RADIANS))
            encoded = int(raw_position & 0xFFFFFFFF).to_bytes(4, "little")
            if not self._goal_writer.addParam(motor_id, encoded):
                self._goal_writer.clearParam()
                raise OSError(f"Failed to stage goal position for ID {motor_id}")
        comm_result = self._goal_writer.txPacket()
        self._goal_writer.clearParam()
        self._check_communication(comm_result, operation="goal position sync write")

    def _set_torque(
        self,
        enabled: bool,
        tolerate_errors: bool = False,
    ) -> tuple[int, ...]:
        failed_ids: list[int] = []
        for motor_id in self.motor_ids:
            try:
                self._write_register(
                    motor_id,
                    ADDR_TORQUE_ENABLE,
                    1,
                    int(enabled),
                )
            except OSError:
                if not tolerate_errors:
                    raise
                failed_ids.append(motor_id)
        self._torque_enabled = bool(enabled) if not failed_ids else True
        return tuple(failed_ids)

    def _write_register(
        self,
        motor_id: int,
        address: int,
        size: int,
        value: int,
    ) -> None:
        self._require_connected()
        method = getattr(self._packet_handler, f"write{size}ByteTxRx")
        comm_result, dxl_error = method(
            self._port_handler,
            motor_id,
            address,
            int(value),
        )
        self._check_communication(
            comm_result,
            dxl_error,
            motor_id,
            f"write address {address}",
        )

    def _read_register(
        self,
        motor_id: int,
        address: int,
        size: int,
        *,
        signed: bool = False,
    ) -> int:
        self._require_connected()
        method = getattr(self._packet_handler, f"read{size}ByteTxRx")
        value, comm_result, dxl_error = method(
            self._port_handler,
            motor_id,
            address,
        )
        self._check_communication(
            comm_result,
            dxl_error,
            motor_id,
            f"read address {address}",
        )
        integer = int(value)
        return _unsigned_to_signed(integer, size) if signed else integer

    def _check_communication(
        self,
        comm_result: int,
        dxl_error: int = 0,
        motor_id: int | None = None,
        operation: str = "DYNAMIXEL operation",
    ) -> None:
        if comm_result != self._sdk.COMM_SUCCESS:
            detail = self._packet_handler.getTxRxResult(comm_result)
            raise OSError(f"{operation} failed for ID {motor_id}: {detail}")
        if dxl_error:
            detail = self._packet_handler.getRxPacketError(dxl_error)
            raise OSError(f"{operation} motor error for ID {motor_id}: {detail}")

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("LEAP Hand is not connected")

    def __enter__(self) -> "LeapHandHardwareController":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        failed_ids = self.close()
        if failed_ids and exc_type is None:
            raise OSError(
                f"Torque-off was not acknowledged by motor IDs: {list(failed_ids)}"
            )


__all__ = [
    "ANGLE_NAMES",
    "DEFAULT_MOTOR_IDS",
    "HardwareMotorCalibration",
    "LeapHandFeedback",
    "LeapHandHardwareController",
    "LeapHandHealth",
    "SIM_MAX_RADIANS",
    "SIM_MIN_RADIANS",
    "clip_sim_radians",
    "motor_radians_to_sim_radians",
    "sim_radians_to_motor_radians",
]
