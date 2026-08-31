from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import cv2

logger = logging.getLogger(__name__)

MAX_DIMENSION = 896
JPEG_QUALITY = 85


@runtime_checkable
class Camera(Protocol):
    def capture_jpeg(self) -> bytes | None:
        """Capture one frame and return it JPEG-encoded. Never raises —
        any failure (device busy, permission not granted, empty frame) is
        caught internally and reported as None."""
        ...

    def close(self) -> None: ...


def _downscale_if_needed(frame, max_dimension: int = MAX_DIMENSION):
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return frame
    scale = max_dimension / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


class OpenCVWebcam:
    """Live webcam capture via OpenCV, macOS (AVFoundation) backend. This is
    the dev-machine camera; the Pi target uses Picamera2Camera instead."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        warmup_frames: int = 5,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self._cap = None
        try:
            self._cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if self._cap is not None and self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            logger.exception("Failed to open webcam at index %s", index)
            self._cap = None

    def capture_jpeg(self) -> bytes | None:
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Webcam at index %s is not open (permission denied or no device)", self.index)
            return None
        try:
            for _ in range(self.warmup_frames):
                self._cap.read()  # discard — let auto-exposure settle
            ok, frame = self._cap.read()
            if not ok or frame is None or frame.size == 0:
                logger.warning("Webcam read returned an empty frame")
                return None
            frame = _downscale_if_needed(frame)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                logger.warning("JPEG encode failed")
                return None
            return buf.tobytes()
        except Exception:
            logger.exception("Webcam capture failed")
            return None

    def close(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            logger.exception("Failed to release webcam")


class StaticImageStub:
    """Camera backed by a fixed image file on disk, for dev/testing without
    live hardware (--image on hat.chat / hat.main)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def capture_jpeg(self) -> bytes | None:
        try:
            img = cv2.imread(self.path)
            if img is None:
                logger.warning("Could not read image at %s", self.path)
                return None
            height, width = img.shape[:2]
            if max(height, width) <= MAX_DIMENSION:
                with open(self.path, "rb") as f:
                    return f.read()
            img = _downscale_if_needed(img)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                logger.warning("JPEG encode failed for %s", self.path)
                return None
            return buf.tobytes()
        except Exception:
            logger.exception("Failed to load static image %s", self.path)
            return None

    def close(self) -> None:
        pass


class Picamera2Camera:
    """Raspberry Pi CSI camera via picamera2. Not testable on the Mac — the
    picamera2 library only installs on Pi OS. Lazy-imported so importing
    this module never fails on the dev machine."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720) -> None:
        import picamera2  # noqa: F401  (Pi-only; lazy import keeps this module Mac-safe)

        raise NotImplementedError(
            "Picamera2Camera: build and test on the Pi once hardware arrives"
        )

    def capture_jpeg(self) -> bytes | None:
        raise NotImplementedError(
            "Picamera2Camera: build and test on the Pi once hardware arrives"
        )

    def close(self) -> None:
        pass


def make_camera(settings, image_path: str | None = None) -> Camera:
    if image_path:
        return StaticImageStub(image_path)
    if settings.profile == "mac":
        return OpenCVWebcam(settings.camera_index)
    if settings.profile == "pi":
        return Picamera2Camera()
    raise ValueError(f"Unknown profile: {settings.profile!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam = OpenCVWebcam(0)
    jpeg = cam.capture_jpeg()
    if jpeg is None:
        print(
            "capture_jpeg() returned None. This is expected if: (a) macOS has not yet "
            "granted this terminal/process camera permission (System Settings > Privacy "
            "& Security > Camera — the OS shows a one-time prompt on first access from a "
            "given app, which may not appear/complete in a non-interactive shell), or "
            "(b) no webcam is attached at index 0, or (c) another process holds the camera."
        )
    else:
        print(f"capture_jpeg() succeeded: {len(jpeg)} bytes")
    cam.close()
