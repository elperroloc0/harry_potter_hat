from __future__ import annotations

import base64
import logging

import httpx

from hat.config import settings

logger = logging.getLogger(__name__)

NO_PERSON_SENTINEL = "NONE"

DESCRIBE_PROMPT = (
    "Look at this photo. If, and only if, a person is clearly visible in it, "
    "describe them in 2-3 short factual sentences in English: apparent age "
    "group, hair, glasses if any, clothing and its colors, and one notable "
    "accessory if present. Mention only what is clearly visible. Do not guess "
    "identity, do not evaluate attractiveness, do not mention the photo or "
    "camera. If no person is clearly visible -- for example an empty room, a "
    "floor, a wall, furniture, or anything else with nobody in frame -- reply "
    f"with exactly the single word {NO_PERSON_SENTINEL} and nothing else. Do "
    "not guess or invent a description of a person who might be just out of "
    "frame; only describe someone you can actually see."
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
                    # qwen3-vl:8b is a "thinking" model: it spends part of
                    # this budget on a hidden reasoning pass (message.thinking)
                    # before it ever writes message.content. Confirmed live
                    # that Ollama's think:false does not suppress that pass
                    # here. How much it needs varies per photo, not a fixed
                    # cost: measured live on the Pi against real photos of a
                    # real person, one finished cleanly at 424 total tokens
                    # (26.9s), another exceeded 500 and got cut off with
                    # empty content (done_reason "length") before writing any
                    # answer. 1000 leaves real headroom above both; a
                    # sufficiently complex photo can still exhaust it, in
                    # which case describe() degrades to None same as any
                    # other failure (see the empty-content check below).
                    "options": {"temperature": 0.2, "num_predict": 1000},
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
