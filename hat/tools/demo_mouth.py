"""Replay a scripted, synthetic mouth-openness envelope through MockServo,
so you can eyeball the console bar animate without any real audio, TTS
backend, or servo hardware.

Usage:
    python -m hat.tools.demo_mouth [--duration 6.0] [--hz 50]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from hat.motion.lipsync import Envelope
from hat.motion.servos import MockServo

# (start_s, end_s, peak_openness) -- a few scripted "syllable" pulses.
_PULSES = [
    (0.3, 0.6, 0.9),
    (1.0, 1.3, 0.6),
    (1.6, 2.1, 1.0),
    (2.4, 2.6, 0.4),
    (3.0, 3.8, 0.8),
    (4.2, 4.5, 0.5),
    (4.8, 5.6, 1.0),
]


def _synthetic_envelope(duration_s: float, hop_s: float = 0.02) -> Envelope:
    n = max(1, int(duration_s / hop_s))
    t = np.arange(n, dtype=np.float32) * hop_s
    openness = np.zeros(n, dtype=np.float32)
    for start, end, peak in _PULSES:
        mask = (t >= start) & (t < end)
        local = (t[mask] - start) / (end - start)
        openness[mask] = np.maximum(openness[mask], peak * np.sin(np.pi * local))
    return Envelope(openness=np.clip(openness, 0.0, 1.0), hop_s=hop_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=6.0, help="seconds of scripted envelope")
    parser.add_argument("--hz", type=float, default=50.0, help="console update rate")
    args = parser.parse_args(argv)

    envelope = _synthetic_envelope(args.duration)
    servo = MockServo()
    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            if t >= args.duration:
                break
            servo.set_mouth(envelope.at(t))
            time.sleep(1.0 / args.hz)
    finally:
        servo.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
