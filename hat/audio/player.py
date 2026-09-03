from __future__ import annotations

import threading

import logging

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from hat.config import settings
from hat.tts.base import PcmAudio

logger = logging.getLogger(__name__)

# Tried in order when the device refuses the clip's own rate. Both are
# near-universal on consumer hardware; 48k first because that is what the
# USB adapters and PipeWire on this rig actually run at.
_FALLBACK_RATES = (48000, 44100)


def default_output_device() -> int | str | None:
    """Prefer PipeWire's "pulse" device over whatever PortAudio calls the
    default, which on the Pi is the raw ALSA card (hw:2,0).

    Two things go wrong when playback opens the raw card directly. It only
    accepts its own hardware rates -- 44100 and 48000 here -- so ElevenLabs'
    22050 Hz PCM is rejected outright with "Invalid sample rate". And it is
    a specific card, so the sound comes out of the USB adapter no matter
    which sink the system is actually pointed at, which is how you end up
    talking to a dongle instead of the Bluetooth speaker. Going through
    pulse hands both problems to PipeWire, which resamples transparently
    and follows the current default sink.
    """
    try:
        for index, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0 and dev["name"].strip().lower() == "pulse":
                return index
    except Exception:
        logger.debug("Could not enumerate audio devices; using the system default", exc_info=True)
    return None


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Rate-convert int16 mono audio, clipped back into int16 range."""
    factor = np.gcd(int(from_rate), int(to_rate))
    resampled = resample_poly(samples.astype(np.float32), to_rate // factor, from_rate // factor)
    return np.clip(resampled, -32768, 32767).astype(np.int16)



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
        self.device = (
            settings.audio_output_device or default_output_device() if device is None else device
        )
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

        samples, rate = audio.samples, audio.sample_rate
        if not self._supports(rate):
            fallback = next((r for r in _FALLBACK_RATES if self._supports(r)), None)
            if fallback is None:
                raise sd.PortAudioError(
                    f"output device accepts neither {rate} Hz nor any of {_FALLBACK_RATES}"
                )
            logger.info("Device refused %d Hz; resampling to %d Hz", rate, fallback)
            samples, rate = _resample(samples, rate, fallback), fallback

        self._samples = samples
        self._sample_rate = rate
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

    def _supports(self, rate: int) -> bool:
        try:
            sd.check_output_settings(
                device=self.device, samplerate=rate, channels=1, dtype="int16"
            )
            return True
        except Exception:
            return False

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
