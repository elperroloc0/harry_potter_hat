from __future__ import annotations

import logging
import math
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hat.config import ServoCal, Settings

logger = logging.getLogger(__name__)

_BAR_WIDTH = 20

# Background interpolation tick for PCA9685Servos. 200 Hz guarantees at
# least ~60 intermediate steps across a full 0-180 degree sweep at the
# calibrated 600 deg/s slew rate (0.3s / 5ms = 60) -- bench testing found
# that anything coarser than that reads as a jump, not a motion.
_TICK_S = 0.005


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
    """Raw duty-cycle PCA9685 driver for the two MG90S servos (mouth, brows),
    bench-calibrated by hand. Deliberately bypasses adafruit_motor.servo /
    ServoKit's angle API -- on the bench that gave unstable, jittery motion.
    Driving PCA9685.channels[n].duty_cycle directly, within ServoCal's
    calibrated min_duty/max_duty range, is what actually worked.

    Motion is smoothed by a background thread that walks each servo's angle
    toward its latest target at cal.max_slew_deg_per_s, so set_mouth/
    set_brows are cheap, non-blocking target updates safe to call at any
    frequency (including the ~50 Hz lip-sync loop) -- the servo itself never
    sees an instant jump regardless of how often or rarely the caller calls
    in.

    `board`/`busio`/`adafruit_pca9685` are Raspberry-Pi-only packages that
    are not installed on the Mac dev machine, so they are imported lazily
    inside __init__ -- merely importing this module must never fail here.
    """

    def __init__(self, cal: "ServoCal") -> None:
        # Deliberately deferred: importing this module on a Mac (no Pi
        # hardware libs installed) must succeed; only *constructing* this
        # class requires the real hardware stack.
        import board  # type: ignore[import-not-found]
        import busio  # type: ignore[import-not-found]
        from adafruit_pca9685 import PCA9685  # type: ignore[import-not-found]

        self.cal = cal
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c)
        self.pca.frequency = cal.pca9685_freq_hz

        self._mouth_angle = 0.0
        self._brow_angle = 0.0
        self._mouth_target = 0.0
        self._brow_target = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Park in a known-good rest position immediately, before the
        # background thread (or anything else) touches the hardware.
        self._write(cal.mouth_channel, self._mouth_angle)
        self._write(cal.brow_channel, self._brow_angle)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_mouth(self, openness: float) -> None:
        with self._lock:
            self._mouth_target = _clamp01(openness) * 180.0

    def set_brows(self, raised: float) -> None:
        with self._lock:
            self._brow_target = _clamp01(raised) * 180.0

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                mouth_target = self._mouth_target
                brow_target = self._brow_target
            self._mouth_angle = _step(
                self._mouth_angle, mouth_target, self.cal.max_slew_deg_per_s, _TICK_S
            )
            self._brow_angle = _step(
                self._brow_angle, brow_target, self.cal.max_slew_deg_per_s, _TICK_S
            )
            self._write(self.cal.mouth_channel, self._mouth_angle)
            self._write(self.cal.brow_channel, self._brow_angle)
            time.sleep(_TICK_S)

    def _write(self, channel: int, angle: float) -> None:
        self.pca.channels[channel].duty_cycle = _angle_to_duty(angle, self.cal)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        # Explicitly silence both channels. The PCA9685 keeps outputting
        # whatever duty cycle it was last told, on its own, indefinitely --
        # a Python process exiting does not stop it. Leaving a stale signal
        # live is exactly what caused the servo to lurch/spin on the *next*
        # run before this script had done anything; zeroing duty_cycle here
        # is what actually fixed it on the bench.
        try:
            self.pca.channels[self.cal.mouth_channel].duty_cycle = 0
            self.pca.channels[self.cal.brow_channel].duty_cycle = 0
        except Exception:
            logger.exception("Failed to zero PCA9685 channels on close")


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _angle_to_duty(angle: float, cal: "ServoCal") -> int:
    """Map a 0-180 degree servo angle to a 16-bit PCA9685 duty cycle, using
    the bench-calibrated safe range in `cal`. Outside that range the servo
    doesn't hit a mechanical stop -- it loses tracking and spins
    continuously -- so clamping to [0, 180] here is a hard limit, not a
    suggestion."""
    angle = max(0.0, min(180.0, angle))
    duty = cal.min_duty + (angle / 180.0) * (cal.max_duty - cal.min_duty)
    return int(duty * 65535)


def _step(current: float, target: float, max_slew_deg_per_s: float, dt: float) -> float:
    """Move `current` toward `target` by at most `max_slew_deg_per_s * dt`."""
    max_step = max_slew_deg_per_s * dt
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


def make_servo(settings: "Settings") -> ServoController:
    """Factory selecting Mock vs PCA9685 by `settings.servo_backend`."""
    if settings.servo_backend == "pca9685":
        return PCA9685Servos(settings.servo)
    return MockServo()
