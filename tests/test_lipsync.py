from __future__ import annotations

import numpy as np
import pytest

from hat.motion.lipsync import Envelope, compute_envelope
from hat.tts.base import PcmAudio

SR = 22050


def _silence(duration_s: float, sr: int = SR) -> PcmAudio:
    n = int(round(duration_s * sr))
    return PcmAudio(samples=np.zeros(n, dtype=np.int16), sample_rate=sr)


def _sine_burst(
    pre_silence_s: float,
    burst_s: float,
    post_silence_s: float,
    freq_hz: float = 300.0,
    amplitude: float = 0.8,
    sr: int = SR,
) -> PcmAudio:
    """Silence, then a loud sine burst, then silence -- for exercising
    attack/release timing."""
    pre = np.zeros(int(round(pre_silence_s * sr)), dtype=np.int16)
    post = np.zeros(int(round(post_silence_s * sr)), dtype=np.int16)
    t = np.arange(int(round(burst_s * sr))) / sr
    burst = (amplitude * 32767.0 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
    samples = np.concatenate([pre, burst, post])
    return PcmAudio(samples=samples, sample_rate=sr)


class TestComputeEnvelopeSilence:
    def test_pure_silence_stays_near_zero(self) -> None:
        audio = _silence(1.0)
        env = compute_envelope(audio)
        assert env.openness.shape[0] > 0
        assert np.all(env.openness <= 1e-6)

    def test_pure_silence_within_unit_range(self) -> None:
        audio = _silence(0.5)
        env = compute_envelope(audio)
        assert np.all(env.openness >= 0.0)
        assert np.all(env.openness <= 1.0)


class TestComputeEnvelopeBurst:
    PRE = 0.3
    BURST = 0.3
    POST = 0.6
    ATTACK_S = 0.03
    RELEASE_S = 0.12

    def _envelope(self) -> tuple[Envelope, PcmAudio]:
        audio = _sine_burst(self.PRE, self.BURST, self.POST)
        env = compute_envelope(audio, attack_s=self.ATTACK_S, release_s=self.RELEASE_S)
        return env, audio

    def test_output_always_in_unit_range(self) -> None:
        env, _ = self._envelope()
        assert np.all(env.openness >= 0.0)
        assert np.all(env.openness <= 1.0)

    def test_quiet_before_burst(self) -> None:
        env, _ = self._envelope()
        # Well before the burst starts, the mouth should be closed.
        t_before = self.PRE - 0.05
        assert env.at(t_before) < 0.1

    def test_rises_within_roughly_the_attack_window(self) -> None:
        env, _ = self._envelope()
        onset = self.PRE
        # Give some slack for the RMS analysis-window latency (frame_s) on
        # top of the attack time constant itself.
        t_after_attack = onset + 0.15
        assert env.at(t_after_attack) > 0.5

    def test_reaches_near_full_openness_during_sustained_burst(self) -> None:
        env, _ = self._envelope()
        # Deep into the burst, well past attack, openness should be close
        # to the self-normalized ceiling (~1.0).
        t_mid_burst = self.PRE + self.BURST * 0.7
        assert env.at(t_mid_burst) > 0.8

    def test_decays_within_roughly_the_release_window_after_burst_ends(self) -> None:
        env, _ = self._envelope()
        burst_end = self.PRE + self.BURST
        # Comfortably more than a few release time constants after the
        # burst ends (plus RMS window latency) -- should be back near 0.
        t_after_release = burst_end + 0.45
        assert env.at(t_after_release) < 0.15

    def test_still_near_zero_at_end_of_clip(self) -> None:
        env, audio = self._envelope()
        t_near_end = audio.duration_s - 0.05
        assert env.at(t_near_end) < 0.1


class TestEnvelopeAtInterpolation:
    def test_interpolates_between_samples(self) -> None:
        env = Envelope(openness=np.array([0.0, 0.5, 1.0], dtype=np.float32), hop_s=0.1)
        assert env.at(0.0) == pytest.approx(0.0)
        assert env.at(0.05) == pytest.approx(0.25)
        assert env.at(0.1) == pytest.approx(0.5)
        assert env.at(0.15) == pytest.approx(0.75)

    def test_returns_zero_before_start(self) -> None:
        env = Envelope(openness=np.array([0.0, 0.5, 1.0], dtype=np.float32), hop_s=0.1)
        assert env.at(-0.01) == 0.0
        assert env.at(-5.0) == 0.0

    def test_returns_zero_past_the_end(self) -> None:
        env = Envelope(openness=np.array([0.0, 0.5, 1.0], dtype=np.float32), hop_s=0.1)
        # Domain is [0, len*hop_s) ~= [0, 0.3). Use a point clearly past the
        # boundary rather than the exact float-imprecise edge (3 * 0.1 !=
        # 0.3 in binary floating point).
        assert env.at(0.35) == 0.0
        assert env.at(1000.0) == 0.0

    def test_empty_envelope_returns_zero_everywhere(self) -> None:
        env = Envelope(openness=np.zeros(0, dtype=np.float32), hop_s=0.02)
        assert env.at(0.0) == 0.0
        assert env.at(1.0) == 0.0
        assert env.at(-1.0) == 0.0

    def test_single_sample_envelope(self) -> None:
        env = Envelope(openness=np.array([0.7], dtype=np.float32), hop_s=0.02)
        assert env.at(0.0) == pytest.approx(0.7)
        assert env.at(0.019) == pytest.approx(0.7)
        assert env.at(0.02) == 0.0  # outside [0, hop_s)


class TestComputeEnvelopeEdgeCases:
    def test_empty_audio_returns_empty_envelope(self) -> None:
        audio = PcmAudio(samples=np.zeros(0, dtype=np.int16), sample_rate=SR)
        env = compute_envelope(audio)
        assert len(env.openness) == 0
        assert env.at(0.0) == 0.0

    def test_very_short_audio_does_not_crash(self) -> None:
        audio = PcmAudio(samples=np.array([100, -100, 200, -200], dtype=np.int16), sample_rate=SR)
        env = compute_envelope(audio)
        assert len(env.openness) >= 1
        assert np.all(env.openness >= 0.0)
        assert np.all(env.openness <= 1.0)
