from __future__ import annotations

import math
import sys
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hat.config import ServoCal, Settings

_BAR_WIDTH = 20


class ServoController(ABC):
    """Motion backend contract: a mouth (openness) and two eyebrows driven
    together as one "raised" value. Implementations clamp their own inputs
    so callers (lipsync, speech) never have to think about valid ranges."""

    @abstractmethod
    def set_mouth(self, openness: float) -> None:
        """openness in [0, 1], 0 = closed, 1 = fully open. Values outside
        [0, 1] are clamped, not rejected."""

    @abstractmethod
    def set_brows(self, raised: float) -> None:
        """raised in [0, 1], 0 = resting, 1 = fully raised. Values outside
        [0, 1] are clamped, not rejected."""

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources. Safe to call more than once."""


class MockServo(ServoController):
    """Dev-machine stand-in: renders a redrawn console bar in place of real
    servo motion, so lip-sync and idle-brow behavior can be eyeballed
    without any hardware."""

    def __init__(self, width: int = _BAR_WIDTH, stream=sys.stdout) -> None:  # noqa: ANN001
        self.width = width
        self.stream = stream
        self._mouth = 0.0
        self._brows = 0.0
        self._closed = False

    def set_mouth(self, openness: float) -> None:
        self._mouth = _clamp01(openness)
        self._render()

    def set_brows(self, raised: float) -> None:
        self._brows = _clamp01(raised)
        self._render()

    def _render(self) -> None:
        if self._closed:
            return
        filled = int(round(self._mouth * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        self.stream.write(f"\r mouth [{bar}] brows {self._brows:.1f}")
        self.stream.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stream.write("\nmouth closed\n")
        self.stream.flush()


class PCA9685Servos(ServoController):
    """Written against Adafruit's documented adafruit-circuitpython-servokit
    API; UNTESTED -- no PCA9685 hardware available yet. Verify calibration
    constants on the bench before trusting this on real servos.

    `board`/`busio`/`adafruit_servokit` are Raspberry-Pi-only packages that
    are not installed on the Mac dev machine, so they are imported lazily
    inside __init__ -- merely importing this module must never fail here.
    """

    def __init__(self, cal: "ServoCal") -> None:
        # Deliberately deferred: importing this module on a Mac (no Pi
        # hardware libs installed) must succeed; only *constructing* this
        # class requires the real hardware stack.
        import board  # type: ignore[import-not-found]
        import busio  # type: ignore[import-not-found]
        from adafruit_servokit import ServoKit  # type: ignore[import-not-found]

        self.cal = cal
        i2c = busio.I2C(board.SCL, board.SDA)
        self.kit = ServoKit(channels=16, i2c=i2c, frequency=cal.pca9685_freq_hz)

        self.kit.servo[cal.mouth_channel].set_pulse_width_range(
            cal.mouth_closed_us, cal.mouth_open_us
        )
        self.kit.servo[cal.brow_left_channel].set_pulse_width_range(
            cal.brow_rest_us, cal.brow_raised_us
        )
        self.kit.servo[cal.brow_right_channel].set_pulse_width_range(
            cal.brow_rest_us, cal.brow_raised_us
        )

        self._mouth_angle = 0.0
        self._brow_angle = 0.0
        self._mouth_last_t = time.monotonic()
        self._brow_last_t = self._mouth_last_t

        # Park in a known-good rest position on startup.
        self.set_mouth(0.0)
        self.set_brows(0.0)

    def set_mouth(self, openness: float) -> None:
        openness = _clamp01(openness)
        target_angle = openness * 180.0
        angle, now = _slew(
            target_angle, self._mouth_angle, self._mouth_last_t, self.cal.max_slew_deg_per_s
        )
        self._mouth_angle = angle
        self._mouth_last_t = now
        self.kit.servo[self.cal.mouth_channel].angle = angle

    def set_brows(self, raised: float) -> None:
        raised = _clamp01(raised)
        target_angle = raised * 180.0
        angle, now = _slew(
            target_angle, self._brow_angle, self._brow_last_t, self.cal.max_slew_deg_per_s
        )
        self._brow_angle = angle
        self._brow_last_t = now
        self.kit.servo[self.cal.brow_left_channel].angle = angle
        self.kit.servo[self.cal.brow_right_channel].angle = angle

    def close(self) -> None:
        # Park closed/resting; leave the I2C bus as-is (no documented
        # "release" call in the ServoKit API).
        try:
            self.set_mouth(0.0)
            self.set_brows(0.0)
        except Exception:
            pass


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _slew(target_angle: float, last_angle: float, last_t: float, max_slew_deg_per_s: float):
    """Limit how far `last_angle` may move toward `target_angle` given the
    time elapsed since `last_t`, at `max_slew_deg_per_s`. Returns (new_angle, now)."""
    now = time.monotonic()
    dt = max(0.0, now - last_t)
    max_step = max_slew_deg_per_s * dt
    delta = target_angle - last_angle
    if abs(delta) <= max_step:
        new_angle = target_angle
    else:
        new_angle = last_angle + math.copysign(max_step, delta)
    return new_angle, now


def make_servo(settings: "Settings") -> ServoController:
    """Factory selecting Mock vs PCA9685 by `settings.servo_backend`."""
    if settings.servo_backend == "pca9685":
        return PCA9685Servos(settings.servo)
    return MockServo()
