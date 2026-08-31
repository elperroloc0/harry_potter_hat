from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hat.tts.base import PcmAudio

if TYPE_CHECKING:
    from hat.audio.player import AudioPlayer
    from hat.motion.servos import ServoController

_EPS = 1e-9


@dataclass
class Envelope:
    """A mouth-openness curve over time, one value per hop, in [0, 1]."""

    openness: np.ndarray  # float32, shape (n_hops,)
    hop_s: float

    def at(self, t_s: float) -> float:
        """Linear interpolation between the two nearest samples. Returns 0.0
        for t_s outside [0, len*hop_s)."""
        n = len(self.openness)
        if n == 0:
            return 0.0
        duration = n * self.hop_s
        if t_s < 0.0 or t_s >= duration:
            return 0.0

        # Sample i is centered at time i*hop_s (hop_s is both the frame
        # advance and our sample spacing here).
        pos = t_s / self.hop_s
        i0 = int(np.floor(pos))
        i0 = max(0, min(i0, n - 1))
        i1 = min(i0 + 1, n - 1)
        frac = pos - i0
        if i0 == i1:
            return float(self.openness[i0])
        v0 = float(self.openness[i0])
        v1 = float(self.openness[i1])
        return v0 + (v1 - v0) * frac


def compute_envelope(
    audio: PcmAudio,
    frame_s: float = 0.04,
    hop_s: float = 0.02,
    attack_s: float = 0.03,
    release_s: float = 0.12,
    gate_db: float = -50.0,
    ceil_percentile: float = 95.0,
) -> Envelope:
    """Derive a mouth-openness envelope from audio loudness. Pure numpy, no
    I/O -- safe to unit-test without any audio device.

    Pipeline: framewise RMS (40ms window / 20ms hop) -> dB -> noise gate ->
    per-utterance normalization (ceil_percentile'th percentile RMS -> ~1.0,
    so quiet and loud phrases both animate across a similar visible range)
    -> asymmetric attack/release smoothing -> clip to [0, 1].
    """
    n = len(audio.samples)
    if n == 0:
        return Envelope(openness=np.zeros(0, dtype=np.float32), hop_s=hop_s)

    sr = audio.sample_rate
    x = audio.samples.astype(np.float32) / 32768.0

    frame_len = max(1, int(round(frame_s * sr)))
    hop_len = max(1, int(round(hop_s * sr)))

    if n < frame_len:
        n_hops = 1
    else:
        n_hops = 1 + (n - frame_len) // hop_len

    rms = np.empty(n_hops, dtype=np.float32)
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + frame_len, n)
        frame = x[start:end]
        if frame.size == 0:
            rms[i] = 0.0
        else:
            rms[i] = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))

    # dB, clamped away from log(0).
    db = 20.0 * np.log10(np.maximum(rms, _EPS))

    # Per-utterance ceiling: the ceil_percentile'th percentile RMS frame
    # maps to openness ~1.0, so a whispered phrase and a shouted phrase both
    # animate across roughly the same visible mouth range.
    ceil_rms = float(np.percentile(rms, ceil_percentile))
    ceil_db = 20.0 * np.log10(max(ceil_rms, _EPS))

    # Gate: anything at or below gate_db (relative to full scale) maps to 0;
    # linearly ramp from the gate floor up to the percentile ceiling.
    span_db = max(ceil_db - gate_db, _EPS)
    raw = (db - gate_db) / span_db
    raw = np.clip(raw, 0.0, 1.0).astype(np.float32)

    # Asymmetric one-pole smoothing: fast attack (snaps open on onsets),
    # slower release (eases closed) -- avoids a jittery mouth.
    smoothed = np.zeros_like(raw)
    prev = 0.0
    attack_coef = _one_pole_coef(attack_s, hop_s)
    release_coef = _one_pole_coef(release_s, hop_s)
    for i, target in enumerate(raw):
        coef = attack_coef if target > prev else release_coef
        prev = prev + coef * (float(target) - prev)
        smoothed[i] = prev

    np.clip(smoothed, 0.0, 1.0, out=smoothed)
    return Envelope(openness=smoothed, hop_s=hop_s)


def _one_pole_coef(time_constant_s: float, hop_s: float) -> float:
    """Per-hop update coefficient for a one-pole filter with the given time
    constant, evaluated at the envelope's hop rate."""
    if time_constant_s <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-hop_s / time_constant_s))


class LipSyncDriver:
    """Drives a ServoController's mouth (and, optionally, idle brow wiggle)
    from an Envelope, polling an AudioPlayer's position clock in real time."""

    def __init__(
        self,
        servo: "ServoController",
        update_hz: float = 50.0,
        brow_wiggle: bool = True,
    ) -> None:
        self.servo = servo
        self.update_hz = update_hz
        self.brow_wiggle = brow_wiggle

    def run(self, envelope: Envelope, player: "AudioPlayer") -> None:
        next_wiggle_at = time.monotonic() + random.uniform(1.0, 2.0)
        try:
            while player.is_playing():
                self.servo.set_mouth(envelope.at(player.position_s()))
                if self.brow_wiggle:
                    now = time.monotonic()
                    if now >= next_wiggle_at:
                        self.servo.set_brows(random.uniform(0.0, 1.0))
                        next_wiggle_at = now + random.uniform(1.0, 2.0)
                time.sleep(1.0 / self.update_hz)
        finally:
            self.servo.set_mouth(0.0)
