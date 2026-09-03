"""Microphone capture and simple audio I/O helpers.

This module owns the boundary between the physical audio device and the
rest of the audio-input pipeline. Everything downstream (VAD, wake-word,
STT) consumes plain int16 mono 16 kHz frames of ``FRAME_SAMPLES`` length
via :class:`MicStream`, regardless of what the underlying hardware
actually supports natively.

On the target Raspberry Pi, the USB sound card may not support opening a
16 kHz input stream directly. In that case we fall back to capturing at
48 kHz and decimating 3:1 down to 16 kHz using a polyphase filter
(:func:`scipy.signal.resample_poly`), applied statefully across callback
boundaries so there is no audible seam/click every 80 ms.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
import wave
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)
from scipy import signal

from hat.audio.types import FRAME_SAMPLES, SAMPLE_RATE

__all__ = [
    "SAMPLE_RATE",
    "FRAME_SAMPLES",
    "MicStream",
    "play_pcm",
    "list_devices",
    "save_wav",
]


class _PolyphaseDecimator:
    """Stateful integer-factor downsampler using polyphase (FIR) filtering.

    Processes fixed-size blocks of raw int16 audio and returns the
    decimated int16 block. Unlike calling ``resample_poly`` fresh on each
    isolated chunk (which implicitly zero-pads at each chunk boundary and
    produces small clicks/discontinuities), this carries a short tail of
    raw samples from the previous block forward as filter context, so the
    anti-alias filter sees real preceding audio instead of a synthetic
    edge on every call.
    """

    def __init__(self, factor: int, block_out: int = FRAME_SAMPLES) -> None:
        if factor < 1:
            raise ValueError("factor must be >= 1")
        self.factor = factor
        self.block_out = block_out
        self.block_in = block_out * factor
        # Context (in raw input samples) carried across calls for filter
        # continuity. Comfortably larger than resample_poly's default
        # Kaiser-windowed FIR half-length (~10 * factor taps), and kept a
        # multiple of `factor` so the decimated output aligns exactly.
        self._context = 80 * factor
        self._history = np.zeros(self._context, dtype=np.float64)

    def process(self, raw_block: np.ndarray) -> np.ndarray:
        if len(raw_block) != self.block_in:
            raise ValueError(
                f"expected {self.block_in} raw samples, got {len(raw_block)}"
            )
        if self.factor == 1:
            return raw_block.astype(np.int16, copy=True)

        x = raw_block.astype(np.float64)
        padded = np.concatenate([self._history, x])
        y = signal.resample_poly(padded, up=1, down=self.factor)
        y_new = y[-self.block_out :]
        self._history = x[-self._context :]
        return np.clip(np.round(y_new), -32768, 32767).astype(np.int16)

    def reset(self) -> None:
        self._history = np.zeros(self._context, dtype=np.float64)


#: How long the queue may stay empty, while not paused, before the input
#: stream is assumed dead and reopened.
_STALL_TIMEOUT_S = 5.0


class MicStream:
    """Live microphone capture producing int16 mono 16 kHz frames.

    Usable as a context manager::

        with MicStream() as mic:
            for frame in mic.frames():
                ...

    Opens a ``sounddevice.InputStream`` at 16 kHz if the device supports
    it; otherwise falls back to 48 kHz and decimates 3:1 in the audio
    callback (see :class:`_PolyphaseDecimator`). Frames are pushed onto a
    thread-safe queue by the audio callback and pulled out by
    :meth:`frames`.

    ``pause()``/``resume()`` implement half-duplex muting: while paused,
    the callback still runs (the stream itself stays open) but newly
    captured audio is dropped instead of being queued, and any audio
    already queued is discarded. This is meant to be used while the hat
    is speaking (TTS playback) so it doesn't transcribe its own voice.
    """

    #: Rates attempted, in order, when no explicit rate is forced.
    _FALLBACK_RATES = (SAMPLE_RATE, 48000)

    def __init__(
        self,
        device: Optional[int | str] = None,
        force_rate: Optional[int] = None,
        queue_maxsize: int = 64,
    ) -> None:
        self.device = device
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_maxsize)
        self._paused = False
        self._closed = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._decimator: Optional[_PolyphaseDecimator] = None
        self._native_rate = SAMPLE_RATE
        self._overflow_count = 0
        self._open(force_rate)

    # -- setup -----------------------------------------------------------

    def _open(self, force_rate: Optional[int]) -> None:
        rates = (force_rate,) if force_rate else self._FALLBACK_RATES
        last_err: Optional[BaseException] = None
        for rate in rates:
            if rate != SAMPLE_RATE and rate % SAMPLE_RATE != 0:
                raise ValueError(
                    f"rate {rate} is not an integer multiple of {SAMPLE_RATE}"
                )
            factor = rate // SAMPLE_RATE
            blocksize = FRAME_SAMPLES * factor
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="int16",
                    blocksize=blocksize,
                    device=self.device,
                    callback=self._callback,
                )
                stream.start()
            except sd.PortAudioError as exc:
                last_err = exc
                continue
            self._native_rate = rate
            self._decimator = _PolyphaseDecimator(factor) if factor > 1 else None
            self._stream = stream
            return
        raise sd.PortAudioError(
            f"could not open an input stream at any of {rates}"
        ) from last_err

    # -- audio thread callback -------------------------------------------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
        if status:
            self._overflow_count += 1
        raw = np.asarray(indata[:, 0], dtype=np.int16)
        frame = self._decimator.process(raw) if self._decimator is not None else raw

        if self._paused or self._closed.is_set():
            return

        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            # Drop the oldest frame rather than blocking the audio thread.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame.copy())
            except queue.Full:
                pass

    # -- public API --------------------------------------------------------

    @property
    def native_rate(self) -> int:
        """The sample rate the underlying hardware stream was opened at."""
        return self._native_rate

    def frames(self) -> Iterator[np.ndarray]:
        """Yield int16 mono 16 kHz frames of ``FRAME_SAMPLES`` length, blocking
        until each one is available. Stops when the stream is closed.

        A dead PortAudio stream is silent in every sense: the callback simply
        stops being invoked, the queue stays empty, and without the watchdog
        below this loop would spin here forever while the hat appeared to be
        running and simply never heard anything again. So a long enough
        starvation, while not deliberately paused, is treated as a broken
        stream and the device is reopened.
        """
        starved = 0.0
        while not self._closed.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._paused:
                    starved = 0.0
                    continue
                starved += 0.2
                if starved >= _STALL_TIMEOUT_S:
                    starved = 0.0
                    self._reopen()
                continue
            starved = 0.0
            yield frame

    def _reopen(self) -> None:
        """Tear the input stream down and open it again. Best effort: if it
        fails we keep the old stream and try again after the next stall,
        because a hat that is deaf for a few more seconds beats one that
        crashes mid-conversation."""
        logger.warning("No audio for %.0fs; restarting the input stream", _STALL_TIMEOUT_S)
        old, self._stream = self._stream, None
        try:
            if old is not None:
                old.stop()
                old.close()
        except Exception:
            logger.debug("Failed closing the stalled stream", exc_info=True)
        try:
            self._open(None)
            logger.info("Input stream restarted")
        except Exception:
            self._stream = old
            logger.exception("Could not restart the input stream")

    def pause(self) -> None:
        """Half-duplex mute: stop feeding new frames and drop anything queued."""
        self._paused = True
        self._drain()

    def resume(self) -> None:
        """Clear any stale queued audio and resume feeding new frames."""
        self._drain()
        self._paused = False

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._closed.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "MicStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def play_pcm(pcm: np.ndarray, sample_rate: int, blocking: bool = True) -> None:
    """Blocking (by default) playback of a PCM array via the default output
    device. This is a simple probe/debug helper for this subsystem's CLIs —
    the real TTS/motion playback+lipsync player lives elsewhere."""
    sd.play(pcm, samplerate=sample_rate)
    if blocking:
        sd.wait()


def list_devices() -> str:
    """Human-readable dump of the available audio devices."""
    return str(sd.query_devices())


def save_wav(path: str | Path, pcm: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    """Write mono int16 PCM to a WAV file using the stdlib ``wave`` module."""
    pcm16 = np.asarray(pcm).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm16.tobytes())


# -- CLI -------------------------------------------------------------------


def _rms_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
    if rms < 1.0:
        return -120.0
    return 20.0 * np.log10(rms / 32768.0)


def _monitor(force_rate: Optional[int], device: Optional[int]) -> None:
    with MicStream(device=device, force_rate=force_rate) as mic:
        print(
            f"Mic open (native_rate={mic.native_rate} Hz, "
            f"decimating={mic.native_rate != SAMPLE_RATE}). Ctrl+C to stop.",
            file=sys.stderr,
        )
        bar_width = 40
        floor_db, ceil_db = -60.0, 0.0
        try:
            for frame in mic.frames():
                db = _rms_dbfs(frame)
                level = (db - floor_db) / (ceil_db - floor_db)
                filled = int(np.clip(level, 0.0, 1.0) * bar_width)
                bar = "#" * filled + "-" * (bar_width - filled)
                sys.stdout.write(f"\r[{bar}] {db:7.1f} dBFS")
                sys.stdout.flush()
        except KeyboardInterrupt:
            print()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hat.audio.io")
    parser.add_argument(
        "--monitor", action="store_true", help="open the mic and show a live RMS VU meter"
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="force opening the mic at this native rate (e.g. 48000 to exercise the decimation fallback path)",
    )
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument(
        "--list-devices", action="store_true", help="print available audio devices and exit"
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        print(list_devices())
        return

    if args.monitor:
        _monitor(force_rate=args.rate, device=args.device)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
