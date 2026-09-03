from __future__ import annotations

import logging
import os
import subprocess
import tempfile
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
    the dev-machine camera; the Pi target uses RpicamJpegCamera instead."""

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


class RpicamJpegCamera:
    """Raspberry Pi CSI camera (Arducam OV5647) via the rpicam-apps CLI --
    `rpicam-jpeg`, the Bookworm+ rename of libcamera-jpeg. Shells out rather
    than using the picamera2 library: picamera2 has no PyPI wheel (apt-only,
    needs --system-site-packages or the system Python), while rpicam-jpeg is
    a plain CLI already on the OS image and is what was actually bench-
    tested (`rpicam-jpeg -o snapshot.jpg --width 800 --height 600
    --nopreview`).
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        timeout_s: float = 15.0,
        settle_ms: int = 500,
    ) -> None:
        # 800x600 (down from the sensor's native 2592x1944) is the
        # bench-chosen resolution -- plenty for "describe the person",
        # and faster to shoot/transfer/decode than full res.
        self.width = width
        self.height = height
        self.timeout_s = timeout_s
        # rpicam-jpeg's own default is -t 5000: five seconds of AGC settling
        # before it takes the shot. That is five seconds of dead silence in
        # the middle of a conversation with a child, so it is cut right down.
        # Measured on this rig across 200/500/1000/2000/5000ms: exposure is
        # already correct at 200ms (mean brightness 100.8 vs 102.6 at the
        # default, contrast flat), and only fine detail keeps improving up to
        # about 1000ms. 500ms costs 0.87s instead of 5.30s and leaves margin
        # for dimmer light than the room this was measured in -- raise it if
        # the party turns out to be darker.
        self.settle_ms = settle_ms

    def capture_jpeg(self) -> bytes | None:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            result = subprocess.run(
                [
                    "rpicam-jpeg",
                    "-o", path,
                    "--width", str(self.width),
                    "--height", str(self.height),
                    "--nopreview",
                    "-t", str(self.settle_ms),
                ],
                capture_output=True,
                timeout=self.timeout_s,
            )
            if result.returncode != 0:
                logger.warning(
                    "rpicam-jpeg exited %s: %s",
                    result.returncode,
                    result.stderr.decode(errors="replace").strip(),
                )
                return None
            with open(path, "rb") as f:
                data = f.read()
            return data or None
        except FileNotFoundError:
            logger.warning("rpicam-jpeg not found on PATH -- is rpicam-apps installed?")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("rpicam-jpeg timed out after %.1fs", self.timeout_s)
            return None
        except Exception:
            logger.exception("rpicam-jpeg capture failed")
            return None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def close(self) -> None:
        pass


def make_camera(settings, image_path: str | None = None) -> Camera:
    if image_path:
        return StaticImageStub(image_path)
    if settings.profile == "mac":
        return OpenCVWebcam(settings.camera_index)
    if settings.profile == "pi":
        return RpicamJpegCamera()
    raise ValueError(f"Unknown profile: {settings.profile!r}")


if __name__ == "__main__":
    from hat.config import settings

    logging.basicConfig(level=logging.INFO)
    cam = make_camera(settings)
    print(f"profile={settings.profile!r} -> using {type(cam).__name__}")
    jpeg = cam.capture_jpeg()
    if jpeg is None:
        if settings.profile == "mac":
            print(
                "capture_jpeg() returned None. This is expected if: (a) macOS has not yet "
                "granted this terminal/process camera permission (System Settings > Privacy "
                "& Security > Camera — the OS shows a one-time prompt on first access from a "
                "given app, which may not appear/complete in a non-interactive shell), or "
                "(b) no webcam is attached at index 0, or (c) another process holds the camera."
            )
        else:
            print(
                "capture_jpeg() returned None. Check: (a) `rpicam-jpeg` is on PATH and "
                "`rpicam-hello --list-cameras` sees the sensor, (b) nothing else is holding "
                "the camera, (c) the ribbon cable is seated. Re-run with logging already on "
                "above to see rpicam-jpeg's own stderr."
            )
    else:
        print(f"capture_jpeg() succeeded: {len(jpeg)} bytes")
    cam.close()
