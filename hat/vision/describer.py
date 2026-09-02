from __future__ import annotations

import base64
import logging

import httpx

from hat.config import settings

logger = logging.getLogger(__name__)

NO_PERSON_SENTINEL = "NONE"

DESCRIBE_PROMPT = (
    "Look at this photo. If, and only if, a person is clearly visible in it, "
    "describe ONLY that person in 2-3 short factual sentences in English: "
    "apparent age group, hair, glasses if any, clothing and its colors, and "
    "one notable accessory if present. Describe only the person -- do not "
    "mention any objects, equipment, furniture, or background details, even "
    "if visible. Mention only what is clearly visible on the person. Do not "
    "guess identity, do not evaluate attractiveness, do not mention the "
    "photo or camera. If no person is clearly visible -- for example an "
    "empty room, a floor, a wall, furniture, or anything else with nobody "
    f"in frame -- reply with exactly the single word {NO_PERSON_SENTINEL} "
    "and nothing else. Do not guess or invent a description of a person who "
    "might be just out of frame; only describe someone you can actually see."
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
                    # Default model (qwen2.5vl:7b) does not "think" -- no
                    # hidden reasoning pass, so num_predict only needs to
                    # cover the actual 2-3 sentence reply. Measured live
                    # against a real, deliberately hard photo (dim light,
                    # out of focus, awkward angle): 1.7-2.1s, 40-60 tokens.
                    # 300 leaves comfortable headroom. If VISION_MODEL is
                    # overridden to a thinking model (e.g. qwen3-vl, which
                    # was the original default -- see project memory for why
                    # it was dropped: 3-6x slower and no reliable way found
                    # to suppress its reasoning pass), this budget is too
                    # small and describe() will silently degrade to None via
                    # the empty-content check below -- raise this back up if
                    # you switch models.
                    "options": {"temperature": 0.2, "num_predict": 300},
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["message"]["content"].strip()
            if not content:
                logger.warning(
                    "OllamaDescriber.describe got an empty reply (done_reason=%s) -- "
                    "likely the model spent its whole token budget on its hidden "
                    "reasoning pass before writing an answer; degrading to None",
                    body.get("done_reason"),
                )
                return None
            # Bare-word check, not a substring check: a real description could
            # legitimately contain "none" (e.g. "no glasses"), so only treat
            # this as the sentinel when the whole reply -- once trimmed of
            # trailing punctuation -- is just that one word.
            if content.rstrip(".!").strip().upper() == NO_PERSON_SENTINEL:
                logger.info("OllamaDescriber.describe: no person visible in frame")
                return None
            return content
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
