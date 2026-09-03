from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hat.config import Settings

logger = logging.getLogger(__name__)


class MotionSensor(ABC):
    """Presence-detection backend: blocks until a visitor approaches.
    Distinct from audio wake-word detection (hat.wake) -- this is a
    physical trigger (PIR sensor), not something heard."""

    @abstractmethod
    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        """Block until motion is detected (returns True), or until
        `timeout` seconds elapse with none (returns False). No timeout
        (None) means wait forever."""

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources. Safe to call more than once."""


class ManualMotionSensor(MotionSensor):
    """Dev-machine / --text stand-in: press Enter to simulate motion. Same
    shape as hat.audio.stub.FakeVoiceInput.wait_for_wake, for parity when
    no PIR hardware is present."""

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        input("[press Enter to simulate motion detected] ")
        return True

    def close(self) -> None:
        pass


class PIRMotionSensor(MotionSensor):
    """Real PIR motion sensor (e.g. HC-SR501) via GPIO, using gpiozero.

    Deliberately gpiozero, not the older RPi.GPIO: Raspberry Pi 5 moved
    GPIO to a new RP1 southbridge chip that RPi.GPIO does not support
    natively, while gpiozero's `lgpio` pin factory does.

    `gpiozero` is a Raspberry-Pi-only package not installed on the Mac dev
    machine, so it's imported lazily inside __init__ -- merely importing
    this module must never fail here (same pattern as PCA9685Servos in
    hat/motion/servos.py and RpicamJpegCamera in hat/vision/camera.py).
    """

    def __init__(self, pin: int) -> None:
        from gpiozero import MotionSensor as GPIOMotionSensor  # type: ignore[import-not-found]

        self.pin = pin
        self._sensor = GPIOMotionSensor(pin)

    def wait_for_motion(self, timeout: Optional[float] = None) -> bool:
        return bool(self._sensor.wait_for_motion(timeout=timeout))

    def close(self) -> None:
        try:
            self._sensor.close()
        except Exception:
            logger.exception("Failed to close PIR motion sensor")


def make_motion_sensor(settings: "Settings") -> MotionSensor:
    """Factory selecting Manual vs PIR by `settings.profile`, mirroring
    make_camera/make_servo. main.py uses the more defensive
    build_motion_sensor() instead, which also honors --text and falls back
    gracefully if the real sensor fails to construct."""
    if settings.profile == "pi":
        return PIRMotionSensor(settings.motion_sensor_pin)
    return ManualMotionSensor()
