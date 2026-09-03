from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hat.config import Settings

logger = logging.getLogger(__name__)

# HC-SR501 (and similar PIR sensors) read unreliably for a warmup period
# right after power-up -- confirmed on the bench: constructing the sensor
# and immediately reading it gives spurious/unstable readings. 30s is what
# was actually tested as sufficient.
_PIR_WARMUP_S = 30.0


class MotionSensor(ABC):
    """Detects the visitor *sitting down* in the chair in front of the hat.

    Range is a HARDWARE setting: the potentiometer on the sensor board is
    turned down to its minimum so the sensor ignores people at a distance and
    only fires when someone is right up against it -- i.e. seated. Never try
    to emulate that range in software with delays, debounce windows or
    thresholds; if the sensor fires too eagerly, the fix is the screwdriver.

    Watched rather than waited on: the visit is a live conversation, so the
    orchestrator polls between turns instead of blocking. Distinct from audio
    wake-word detection (hat.wake) -- this is physical, not something heard.
    """

    @abstractmethod
    def start_watching(self) -> None:
        """Begin detecting in the background. Safe to call more than once."""

    @abstractmethod
    def pending(self) -> bool:
        """True if someone has sat down since the last poll(), without
        consuming the event. Cheap enough to call once per audio frame --
        it is the cancel predicate that cuts a listen short."""

    @abstractmethod
    def poll(self) -> bool:
        """Take the pending event and clear it. True at most once per
        detection."""

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources. Safe to call more than once."""


class _EventSensor(MotionSensor):
    """Shared threading.Event plumbing for both backends."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def pending(self) -> bool:
        return self._event.is_set()

    def poll(self) -> bool:
        if self._event.is_set():
            self._event.clear()
            return True
        return False


class ManualMotionSensor(_EventSensor):
    """Dev-machine stand-in for the PIR sensor, with two ways to fire.

    Under --text the orchestrator calls simulate() when it sees the typed
    `/sit` token, because FakeVoiceInput already owns stdin and a second
    reader would steal its input.

    With a real microphone on a dev machine (no gpiozero, so the PIR falls
    back here) nothing else reads stdin, so `watch_stdin` starts a daemon
    thread that treats a bare Enter as "the visitor just sat down". Without
    it the verdict is unreachable off the Pi: a spoken utterance can never
    transcribe to the `/sit` token.
    """

    def __init__(self, watch_stdin: bool = False) -> None:
        super().__init__()
        self.watch_stdin = watch_stdin
        self._thread: threading.Thread | None = None

    def start_watching(self) -> None:
        if not self.watch_stdin or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._read_stdin, daemon=True)
        self._thread.start()
        logger.info("No PIR sensor: press Enter to signal that the visitor sat down")

    def _read_stdin(self) -> None:
        while True:
            try:
                input()
            except (EOFError, OSError):
                return
            self.simulate()

    def simulate(self) -> None:
        """Pretend someone just sat down."""
        self._event.set()

    def close(self) -> None:
        pass


class PIRMotionSensor(_EventSensor):
    """Real PIR motion sensor (e.g. HC-SR501) via GPIO, using gpiozero.

    Deliberately gpiozero, not the older RPi.GPIO: on the bench, RPi.GPIO
    raised "RuntimeError: Cannot determine SOC peripheral base address" --
    Raspberry Pi 5 moved GPIO to a new RP1 southbridge chip that RPi.GPIO
    doesn't support at all, not just unreliably.

    gpiozero's own pin factory auto-detection was NOT sufficient on this
    Pi 5 either -- confirmed on the bench that it needs to be pointed at
    the RP1 controller explicitly (chip 4 / /dev/gpiochip4) via
    `Device.pin_factory = LGPIOFactory(chip=4)` before constructing any
    gpiozero device. This is process-wide gpiozero state, not specific to
    this one sensor -- if the project ever adds another plain-GPIO device
    (a button, a relay), it rides on the same pin_factory setting and does
    not need to repeat this.

    `gpiozero` is a Raspberry-Pi-only package not installed on the Mac dev
    machine, so it's imported lazily inside __init__ -- merely importing
    this module must never fail here (same pattern as PCA9685Servos in
    hat/motion/servos.py and RpicamJpegCamera in hat/vision/camera.py).
    """

    def __init__(self, pin: int, warmup_s: float = _PIR_WARMUP_S) -> None:
        super().__init__()
        from gpiozero import Device, MotionSensor as GPIOMotionSensor  # type: ignore[import-not-found]
        from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore[import-not-found]

        Device.pin_factory = LGPIOFactory(chip=4)

        self.pin = pin
        self._sensor = GPIOMotionSensor(pin)

        if warmup_s > 0:
            logger.info("PIR sensor warming up for %.0fs before first use...", warmup_s)
            time.sleep(warmup_s)

    def start_watching(self) -> None:
        # gpiozero runs this callback on its own thread; setting an Event is
        # all we do there, and the orchestrator picks it up between turns.
        self._sensor.when_motion = self._on_motion

    def _on_motion(self) -> None:
        logger.debug("PIR fired")
        self._event.set()

    def close(self) -> None:
        try:
            self._sensor.when_motion = None
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
