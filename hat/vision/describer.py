from __future__ import annotations

import base64
import logging

import httpx

from hat.config import settings

logger = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "Describe the person in this photo in 2-3 short factual sentences in English: "
    "apparent age group, hair, glasses if any, clothing and its colors, and one "
    "notable accessory if present. Mention only what is clearly visible. Do not "
    "guess identity, do not evaluate attractiveness, do not mention the photo or "
    "camera."
)


class OllamaDescriber:
    """Local, privacy-preserving vision description via Ollama. The JPEG never
    leaves the machine running Ollama; only the short text description this
    produces goes into the cloud LLM prompt. Degrades to None on any failure —
    Ollama not installed/running is an expected dev-machine state, not a bug."""

    def __init__(
        self,
        base_url: str = settings.ollama_url,
        model: str = settings.vision_model,
        timeout_s: float = settings.vision_timeout_s,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def describe(self, jpeg: bytes) -> str | None:
        try:
            b64 = base64.standard_b64encode(jpeg).decode()
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": DESCRIBE_PROMPT,
                            "images": [b64],
                        }
                    ],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0.2, "num_predict": 120},
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            content = content.strip()
            return content or None
        except Exception:
            logger.warning("OllamaDescriber.describe failed; degrading to None", exc_info=True)
            return None

    def warm_up(self) -> None:
        """Best-effort model preload so the first real describe() call isn't
        slowed by Ollama loading weights. Swallows all errors."""
        try:
            httpx.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": [], "keep_alive": "30m"},
                timeout=min(self.timeout_s, 5.0),
            )
        except Exception:
            logger.debug("OllamaDescriber.warm_up failed (non-fatal)", exc_info=True)


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    describer = OllamaDescriber()
    print(f"Pointing at {describer.base_url}, model={describer.model}")
    start = time.monotonic()
    result = describer.describe(b"not a real jpeg, just checking graceful failure")
    elapsed = time.monotonic() - start
    print(f"describe() returned {result!r} in {elapsed:.2f}s (expected: None, fast, no raise)")
