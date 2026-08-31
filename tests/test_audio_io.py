"""Unit tests for the pure-numpy pieces of hat.audio.io -- specifically the
stateful 48kHz -> 16kHz polyphase decimator, which is fully testable without
a real audio device."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from hat.audio.io import FRAME_SAMPLES, _PolyphaseDecimator


def _tone(freq_hz: float, duration_s: float, rate: int, amplitude: float = 8000.0) -> np.ndarray:
    t = np.arange(int(duration_s * rate)) / rate
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.int16)


def test_decimator_output_length_and_dtype() -> None:
    dec = _PolyphaseDecimator(factor=3, block_out=FRAME_SAMPLES)
    block = _tone(1000, FRAME_SAMPLES * 3 / 48000, rate=48000)
    assert len(block) == FRAME_SAMPLES * 3

    out = dec.process(block)
    assert out.dtype == np.int16
    assert len(out) == FRAME_SAMPLES


def test_decimator_rejects_wrong_block_size() -> None:
    dec = _PolyphaseDecimator(factor=3, block_out=FRAME_SAMPLES)
    with pytest.raises(ValueError):
        dec.process(np.zeros(100, dtype=np.int16))


def test_decimator_passthrough_for_factor_one() -> None:
    dec = _PolyphaseDecimator(factor=1, block_out=FRAME_SAMPLES)
    block = _tone(300, FRAME_SAMPLES / 16000, rate=16000)
    out = dec.process(block)
    np.testing.assert_array_equal(out, block)


def test_decimator_preserves_in_band_tone_amplitude() -> None:
    # A 1kHz tone is well within the post-decimation Nyquist (8kHz), so a
    # correct polyphase decimator should preserve it close to full
    # amplitude (allowing for filter passband ripple / a few percent).
    rate_in = 48000
    factor = 3
    dec = _PolyphaseDecimator(factor=factor, block_out=FRAME_SAMPLES)

    duration_s = 2.0
    full = _tone(1000, duration_s, rate=rate_in)
    block_in = FRAME_SAMPLES * factor
    n_blocks = len(full) // block_in

    out_chunks = []
    for i in range(n_blocks):
        block = full[i * block_in : (i + 1) * block_in]
        out_chunks.append(dec.process(block))
    decimated = np.concatenate(out_chunks)

    # Skip the first block (filter warm-up/history is zero-initialized) and
    # measure RMS amplitude on a steady-state region.
    steady = decimated[FRAME_SAMPLES * 2 :].astype(np.float64)
    rms = np.sqrt(np.mean(steady**2))
    expected_rms = 8000 / np.sqrt(2)
    assert rms > expected_rms * 0.85


def test_decimator_attenuates_out_of_band_content_vs_naive_slicing() -> None:
    # A 9kHz tone is *above* the post-decimation Nyquist (8kHz @ 16kHz sr).
    # Naive "take every 3rd sample" decimation aliases it straight into the
    # passband at (near) full amplitude. A real polyphase decimator with an
    # anti-alias filter must attenuate it substantially.
    rate_in = 48000
    factor = 3
    dec = _PolyphaseDecimator(factor=factor, block_out=FRAME_SAMPLES)

    duration_s = 2.0
    full = _tone(9000, duration_s, rate=rate_in, amplitude=8000.0)
    block_in = FRAME_SAMPLES * factor
    n_blocks = len(full) // block_in

    out_chunks = []
    for i in range(n_blocks):
        block = full[i * block_in : (i + 1) * block_in]
        out_chunks.append(dec.process(block))
    filtered = np.concatenate(out_chunks)[FRAME_SAMPLES * 2 :].astype(np.float64)

    naive = full[::3][: len(np.concatenate(out_chunks))][FRAME_SAMPLES * 2 :].astype(np.float64)

    filtered_rms = np.sqrt(np.mean(filtered**2))
    naive_rms = np.sqrt(np.mean(naive**2))

    # Naive slicing should barely attenuate the out-of-band tone (aliasing).
    assert naive_rms > 5000
    # The polyphase-filtered path should knock it down substantially.
    assert filtered_rms < naive_rms * 0.3


def test_decimator_stateful_chunking_matches_offline_resample_closely() -> None:
    # Processing a signal in fixed blocks (with history carried across
    # calls) should closely match calling resample_poly once on the whole
    # signal -- away from the very first block's startup transient. This
    # is the "not just naive slicing, and not full of seams either" check.
    rate_in = 48000
    factor = 3
    dec = _PolyphaseDecimator(factor=factor, block_out=FRAME_SAMPLES)

    duration_s = 1.0
    full = _tone(700, duration_s, rate=rate_in) + _tone(2500, duration_s, rate=rate_in, amplitude=3000)
    full = full.astype(np.int16)
    block_in = FRAME_SAMPLES * factor
    n_blocks = len(full) // block_in

    out_chunks = []
    for i in range(n_blocks):
        block = full[i * block_in : (i + 1) * block_in]
        out_chunks.append(dec.process(block))
    chunked = np.concatenate(out_chunks).astype(np.float64)

    offline = signal.resample_poly(full.astype(np.float64), up=1, down=factor)
    offline = offline[: len(chunked)]

    # Compare away from the first block (startup transient / zero history).
    a = chunked[FRAME_SAMPLES:]
    b = offline[FRAME_SAMPLES:]
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.98
