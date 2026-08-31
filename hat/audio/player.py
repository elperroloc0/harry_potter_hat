from __future__ import annotations

import threading

import numpy as np
import sounddevice as sd

from hat.config import settings
from hat.tts.base import PcmAudio


class AudioPlayer:
    """Non-blocking int16 mono playback, plus a `position_s()` clock that
    lip-sync polls from an ordinary thread.

    The sounddevice callback runs on a realtime audio thread and must never
    do anything expensive -- no Python-heavy work, no I2C/servo calls, ever.
    It only memcpy's a slice of the pending samples into `outdata` and bumps
    a frame cursor; everything else (position tracking, completion
    signalling) happens outside the callback.
    """

    def __init__(
        self,
        device: int | str | None = None,
        latency_offset_s: float = settings.output_latency_s,
        blocksize: int = 512,
    ) -> None:
        self.device = device
        self.latency_offset_s = latency_offset_s
        self.blocksize = blocksize

        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._samples: np.ndarray | None = None
        self._frame_pos = 0  # next source frame to copy (audio thread only)
        self._frames_written = 0  # frames handed to the device so far (shared, lock-protected)
        self._sample_rate = 0
        self._finished = threading.Event()
        self._finished.set()  # nothing playing yet

    def play(self, audio: PcmAudio) -> None:
        """Start playback of `audio`. Returns immediately; playback happens
        on a background audio thread managed by PortAudio/sounddevice."""
        self.stop()  # tear down any previous stream first

        self._samples = audio.samples
        self._sample_rate = audio.sample_rate
        self._frame_pos = 0
        with self._lock:
            self._frames_written = 0
        self._finished.clear()

        def callback(outdata: np.ndarray, frames: int, time_info, status) -> None:  # noqa: ANN001
            samples = self._samples
            pos = self._frame_pos
            remaining = 0 if samples is None else len(samples) - pos
            n = max(0, min(frames, remaining))
            if n > 0:
                outdata[:n, 0] = samples[pos : pos + n]
                self._frame_pos = pos + n
            if n < frames:
                outdata[n:, 0] = 0
            with self._lock:
                self._frames_written += n
            if n < frames:
                raise sd.CallbackStop()

        def finished_callback() -> None:
            self._finished.set()

        stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
            finished_callback=finished_callback,
        )
        self._stream = stream
        stream.start()

    def position_s(self) -> float:
        """What the listener is actually hearing right now, in seconds from
        the start of the current clip -- the clock lip-sync polls."""
        if self._sample_rate == 0:
            return 0.0
        with self._lock:
            frames = self._frames_written
        return max(0.0, frames / self._sample_rate - self.latency_offset_s)

    def is_playing(self) -> bool:
        return self._stream is not None and not self._finished.is_set()

    def wait(self) -> None:
        """Block the calling thread until the current clip finishes."""
        self._finished.wait()

    def stop(self) -> None:
        """Abort playback immediately (if any) and release the stream."""
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        self._finished.set()

    def close(self) -> None:
        self.stop()
