"""Wake-word detection via openWakeWord.

IMPORTANT (macOS): the ONNX inference backend in openWakeWord returns
near-zero scores on macOS regardless of the input audio -- a known
upstream issue with onnxruntime's CoreML/CPU execution path for these
particular models on Apple Silicon. The "tflite" backend (via
``ai_edge_litert``) works correctly and is what this module uses
unconditionally; ``inference_framework`` is exposed as a constructor
parameter for forwards-compatibility, but "tflite" is the only backend
verified to work and should not be changed to "onnx" on macOS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from hat.audio.types import WakeEvent
from hat.config import settings

__all__ = ["WakeWordDetector"]


def _wakeword_files_present(model_name: str, inference_framework: str, models_dir: Path) -> bool:
    import openwakeword

    suffix = ".tflite" if inference_framework == "tflite" else ".onnx"
    candidates = [
        Path(meta["download_url"]).name
        for meta in openwakeword.MODELS.values()
        if model_name in Path(meta["download_url"]).name
    ]
    candidates = [c for c in candidates if c.endswith(suffix)]
    if not candidates:
        # Not one of the bundled pretrained names -- can't pre-check; let
        # openwakeword.Model() itself raise a clear error if it's missing.
        return True
    return all((models_dir / c).exists() for c in candidates)


def _feature_and_vad_files_present(models_dir: Path) -> bool:
    import openwakeword

    required = [Path(m["download_url"]).name for m in openwakeword.FEATURE_MODELS.values()]
    return models_dir.exists() and all((models_dir / f).exists() for f in required)


def _ensure_models_downloaded(model_name: str, inference_framework: str) -> None:
    """Download the pretrained wake-word + shared feature model files on
    first run, if they aren't already present on disk. Needs network."""
    import openwakeword

    models_dir = Path(openwakeword.__file__).parent / "resources" / "models"
    have_wakeword = _wakeword_files_present(model_name, inference_framework, models_dir)
    have_features = _feature_and_vad_files_present(models_dir)

    if have_wakeword and have_features:
        return

    print(
        f"[hat.wake] openWakeWord model files not found locally for "
        f"'{model_name}' -- downloading now (needs network access, first run only)...",
        file=sys.stderr,
    )
    from openwakeword.utils import download_models

    download_models(model_names=[model_name])


class WakeWordDetector:
    """Streaming wake-word detector wrapping ``openwakeword.Model``.

    Feed it one FRAME_SAMPLES (1280-sample, 80ms) int16 mono 16kHz frame at
    a time via :meth:`process`. Returns a :class:`WakeEvent` when the
    model's score for ``model`` crosses ``threshold`` and at least
    ``refractory_s`` seconds have elapsed since the last event, else
    ``None``.
    """

    def __init__(
        self,
        model: str = settings.wake_model,
        threshold: float = 0.5,
        inference_framework: str = "tflite",
        refractory_s: float = 2.0,
    ) -> None:
        self.model_name = model
        self.threshold = threshold
        self.refractory_s = refractory_s
        self.inference_framework = inference_framework

        _ensure_models_downloaded(model, inference_framework)

        from openwakeword.model import Model

        self._oww = Model(wakeword_models=[model], inference_framework=inference_framework)
        self._last_event_at: Optional[float] = None
        self._last_score: float = 0.0

    @property
    def last_score(self) -> float:
        """The most recent raw score for ``model_name`` (updated on every
        call to :meth:`process`, whether or not it crossed threshold).
        Mainly useful for live monitoring/debugging."""
        return self._last_score

    def process(self, frame: np.ndarray) -> Optional[WakeEvent]:
        preds = self._oww.predict(frame)
        score = preds.get(self.model_name)
        if score is None:
            # Defensive fallback: openwakeword's dict keys are normally the
            # model name we passed in, but take the max across whatever
            # labels are present rather than silently reporting 0.
            score = max(preds.values()) if preds else 0.0
        score = float(score)
        self._last_score = score

        now = time.monotonic()
        if score >= self.threshold:
            if self._last_event_at is None or (now - self._last_event_at) >= self.refractory_s:
                self._last_event_at = now
                return WakeEvent(model_name=self.model_name, score=score, at=now)
        return None

    def reset(self) -> None:
        """Clear openWakeWord's internal streaming/prediction buffers. Call
        this after every recorded phrase (e.g. from VoiceInput.hold()'s
        exit) so stale audio in its buffers can't cause a spurious
        re-trigger. Does NOT reset the refractory timer."""
        self._oww.reset()


# -- CLI ---------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    from hat.audio.io import MicStream

    parser = argparse.ArgumentParser(prog="python -m hat.wake.detector")
    parser.add_argument("--model", default=settings.wake_model)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--refractory-s", type=float, default=2.0)
    args = parser.parse_args(argv)

    detector = WakeWordDetector(
        model=args.model, threshold=args.threshold, refractory_s=args.refractory_s
    )
    print(
        f"Listening for wake word '{args.model}' (threshold={args.threshold}, "
        f"framework={detector.inference_framework}). Ctrl+C to stop.",
        file=sys.stderr,
    )

    last_print_t = 0.0
    try:
        with MicStream() as mic:
            for frame in mic.frames():
                event = detector.process(frame)
                now = time.monotonic()
                if event is not None:
                    print(
                        f"\n*** WAKE EVENT: model={event.model_name} "
                        f"score={event.score:.3f} at={event.at:.2f}"
                    )
                    last_print_t = now
                elif detector.last_score > 0.1 or (now - last_print_t) > 1.0:
                    sys.stdout.write(f"\rscore={detector.last_score:.3f}   ")
                    sys.stdout.flush()
                    last_print_t = now
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
