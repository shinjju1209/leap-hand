import unittest

import numpy as np

from leap_hand_hardware_controller import (
    ADDR_BUS_WATCHDOG,
    ADDR_GOAL_CURRENT,
    ADDR_GOAL_POSITION,
    ADDR_OPERATING_MODE,
    ADDR_PRESENT_CURRENT,
    ADDR_PRESENT_POSITION,
    ADDR_PRESENT_VELOCITY,
    ADDR_TORQUE_ENABLE,
    CURRENT_BASED_POSITION_MODE,
    POSITION_SCALE_RADIANS,
    SIM_MAX_RADIANS,
    SIM_MIN_RADIANS,
    LeapHandHardwareController,
    motor_radians_to_sim_radians,
    sim_radians_to_motor_radians,
)


class FakePortHandler:
    def __init__(self, port):
        self.port = port
        self.is_open = False

    def openPort(self):
        self.is_open = True
        return True

    def setBaudRate(self, baudrate):
        self.baudrate = baudrate
        return True

    def closePort(self):
        self.is_open = False


class FakePacketHandler:
    def __init__(self, sdk):
        self.sdk = sdk

    def ping(self, port, motor_id):
        return 1234, self.sdk.COMM_SUCCESS, 0

    def write1ByteTxRx(self, port, motor_id, address, value):
        self.sdk.registers[(motor_id, address)] = value
        return self.sdk.COMM_SUCCESS, 0

    def write2ByteTxRx(self, port, motor_id, address, value):
        self.sdk.registers[(motor_id, address)] = value
        return self.sdk.COMM_SUCCESS, 0

    def write4ByteTxRx(self, port, motor_id, address, value):
        self.sdk.registers[(motor_id, address)] = value
        return self.sdk.COMM_SUCCESS, 0

    def read1ByteTxRx(self, port, motor_id, address):
        return self.sdk.registers.get((motor_id, address), 0), 0, 0

    def read2ByteTxRx(self, port, motor_id, address):
        return self.sdk.registers.get((motor_id, address), 50), 0, 0

    def read4ByteTxRx(self, port, motor_id, address):
        default = round(np.pi / POSITION_SCALE_RADIANS)
        return self.sdk.registers.get((motor_id, address), default), 0, 0

    @staticmethod
    def getTxRxResult(comm_result):
        return f"communication {comm_result}"

    @staticmethod
    def getRxPacketError(error):
        return f"motor error {error}"


class FakeGroupSyncWrite:
    def __init__(self, sdk, address):
        self.sdk = sdk
        self.address = address
        self.params = {}

    def addParam(self, motor_id, data):
        self.params[motor_id] = bytes(data)
        return True

    def txPacket(self):
        for motor_id, data in self.params.items():
            self.sdk.registers[(motor_id, self.address)] = int.from_bytes(
                data,
                "little",
            )
        return self.sdk.COMM_SUCCESS

    def clearParam(self):
        self.params = {}


class FakeGroupSyncRead:
    def __init__(self, sdk):
        self.sdk = sdk
        self.motor_ids = []

    def addParam(self, motor_id):
        self.motor_ids.append(motor_id)
        return True

    def txRxPacket(self):
        return self.sdk.COMM_SUCCESS

    def isAvailable(self, motor_id, address, size):
        return motor_id in self.motor_ids

    def getData(self, motor_id, address, size):
        if address == ADDR_PRESENT_POSITION:
            return round(np.pi / POSITION_SCALE_RADIANS)
        if address in (ADDR_PRESENT_CURRENT, ADDR_PRESENT_VELOCITY):
            return 0
        return 0


class FakeDynamixelSdk:
    COMM_SUCCESS = 0

    def __init__(self):
        self.registers = {}

    def PortHandler(self, port):
        return FakePortHandler(port)

    def PacketHandler(self, protocol):
        self.protocol = protocol
        return FakePacketHandler(self)

    def GroupSyncWrite(self, port, packet, address, size):
        return FakeGroupSyncWrite(self, address)

    def GroupSyncRead(self, port, packet, address, size):
        return FakeGroupSyncRead(self)


class LeapHandHardwareControllerTests(unittest.TestCase):
    def setUp(self):
        self.sdk = FakeDynamixelSdk()
        self.controller = LeapHandHardwareController(
            "COM_TEST",
            sdk_module=self.sdk,
        )

    def tearDown(self):
        self.controller.close()

    def test_sim_and_motor_conventions_round_trip(self):
        sim = np.linspace(-0.2, 0.8, 16)
        motor = sim_radians_to_motor_radians(sim)
        np.testing.assert_allclose(motor_radians_to_sim_radians(motor), sim)

    def test_conversion_clips_to_official_limits(self):
        motor = sim_radians_to_motor_radians(np.full(16, 100.0))
        np.testing.assert_allclose(motor, SIM_MAX_RADIANS + np.pi)
        motor = sim_radians_to_motor_radians(np.full(16, -100.0))
        np.testing.assert_allclose(motor, SIM_MIN_RADIANS + np.pi)

    def test_connect_verifies_ids_and_leaves_torque_off(self):
        models = self.controller.connect()
        self.assertEqual(set(models), set(range(16)))
        self.assertFalse(self.controller.torque_enabled)
        for motor_id in range(16):
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_TORQUE_ENABLE)],
                0,
            )

    def test_enable_requires_configuration(self):
        self.controller.connect()
        with self.assertRaises(RuntimeError):
            self.controller.enable_torque()

    def test_configuration_uses_low_current_mode_and_watchdog(self):
        self.controller.connect()
        self.controller.configure()
        for motor_id in range(16):
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_OPERATING_MODE)],
                CURRENT_BASED_POSITION_MODE,
            )
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_GOAL_CURRENT)],
                300,
            )
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_BUS_WATCHDOG)],
                25,
            )

    def test_enable_seeds_present_position_before_torque(self):
        self.controller.connect()
        self.controller.configure()
        self.controller.enable_torque()
        expected_raw = round(np.pi / POSITION_SCALE_RADIANS)
        for motor_id in range(16):
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_GOAL_POSITION)],
                expected_raw,
            )
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_TORQUE_ENABLE)],
                1,
            )

    def test_commands_require_torque_and_use_degree_interface(self):
        self.controller.connect()
        self.controller.configure()
        with self.assertRaises(RuntimeError):
            self.controller.command_degrees(np.zeros(16))
        self.controller.enable_torque()
        actual = self.controller.command_degrees(np.full(16, 30.0))
        np.testing.assert_allclose(actual, np.full(16, 30.0))

    def test_heartbeat_reuses_exact_seeded_raw_goal(self):
        outside_safe_range_raw = round((np.pi - 1.5) / POSITION_SCALE_RADIANS)
        for motor_id in range(16):
            self.sdk.registers[(motor_id, ADDR_PRESENT_POSITION)] = (
                outside_safe_range_raw
            )
        self.controller.connect()
        self.controller.configure()
        self.controller.enable_torque()
        self.controller.heartbeat()
        for motor_id in range(16):
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_GOAL_POSITION)],
                outside_safe_range_raw,
            )

    def test_feedback_uses_zero_open_degree_convention(self):
        self.controller.connect()
        feedback = self.controller.read_feedback()
        np.testing.assert_allclose(
            feedback.positions_degrees,
            np.zeros(16),
            atol=0.1,
        )
        np.testing.assert_array_equal(feedback.currents_milliamps, np.zeros(16))

    def test_close_disables_torque(self):
        self.controller.connect()
        self.controller.configure()
        self.controller.enable_torque()
        self.controller.close()
        self.assertFalse(self.controller.is_connected)
        self.assertFalse(self.controller.torque_enabled)
        for motor_id in range(16):
            self.assertEqual(
                self.sdk.registers[(motor_id, ADDR_TORQUE_ENABLE)],
                0,
            )


if __name__ == "__main__":
    unittest.main()
