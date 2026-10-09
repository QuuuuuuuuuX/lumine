"""The cloud (or local-server) VLM brain.

Deliberately built on ``urllib`` rather than a vendor SDK: the request shape is identical
across every OpenAI-compatible provider, and a raw HTTP call cannot be broken by an SDK
major-version bump in the middle of a five-hour run.

Failure policy
--------------
:meth:`VLMBrain.think` never raises.  Transport errors, HTTP errors, truncated responses
and unparseable JSON all degrade to a safe ``wait`` intent and are counted in
``self.stats.failures`` so the evaluation report can attribute them.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from ..config import BrainConfig, resolve_api_key
from ..types import Intent, Observation
from .base import Brain, BrainContext
from .prompt import build_messages, parse_intent
from .providers import get_provider


class VLMBrain(Brain):
    """A vision-language model over an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, cfg: BrainConfig, frame_width: int = 640, frame_height: int = 360) -> None:
        super().__init__()
        self.cfg = cfg
        self.spec = get_provider(cfg.provider)
        self.base_url = (cfg.base_url or self.spec.base_url).rstrip("/")
        self.model = cfg.model or self.spec.default_model
        self.width = frame_width
        self.height = frame_height
        self.name = f"vlm:{self.spec.key}/{self.model}"

        self.api_key = resolve_api_key(cfg.api_key_env, self.spec.key)
        if self.spec.key == "ollama":
            self.api_key = self.api_key or "ollama"
        if self.spec.key != "none" and not self.api_key:
            raise RuntimeError(
                f"no API key for provider {self.spec.key!r}. Set {cfg.api_key_env} "
                f"(get one at {self.spec.console}) or switch provider. "
                f"Run 'python -m lumine.cli providers' to see the options."
            )

    # -- request building -----------------------------------------------------------------

    def _body(self, obs: Observation, ctx: BrainContext) -> dict:
        messages = build_messages(obs.frame.to_data_url(), obs.hud, ctx)

        if not self.spec.vision:
            # Text-only provider: drop the image part and say so, so the model knows it is
            # flying blind rather than hallucinating a screen.
            for m in messages:
                if isinstance(m.get("content"), list):
                    parts = [p for p in m["content"] if p.get("type") == "text"]
                    parts.append({
                        "type": "text",
                        "text": "(No image available: this provider is text-only. Reason from "
                                "the structured state and history only, and prefer conservative skills.)",
                    })
                    m["content"] = parts

        if self.spec.requires_base64:
            for m in messages:
                if isinstance(m.get("content"), list):
                    for part in m["content"]:
                        if part.get("type") == "image_url" and not part["image_url"]["url"].startswith("data:"):
                            raise RuntimeError(f"{self.spec.key} requires base64 image data")

        body: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        # Reasoning models burn the whole completion budget on hidden reasoning tokens and
        # can then return an empty `content`. Measured on deepseek-flash: 2.0 s at "low"
        # versus 2.4 s at the default, with 342 versus 407 reasoning tokens.
        if self.cfg.reasoning_effort and self.spec.key in ("deepseek", "zai", "glm-cn"):
            body["reasoning_effort"] = self.cfg.reasoning_effort
        body.update(self.spec.extra_body)
        return body

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "lumine-agent/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # -- Brain API ------------------------------------------------------------------------

    def think(self, obs: Observation, ctx: BrainContext) -> Intent:
        body = self._body(obs, ctx)

        last_err = ""
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                payload = self._post(body)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                last_err = f"HTTP {exc.code}: {detail}"
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.cfg.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - network is hostile by default
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt < self.cfg.max_retries:
                    time.sleep(1.0)
                    continue
                break
            finally:
                self.stats.total_seconds += time.monotonic() - t0

            self.stats.calls += 1
            return self._handle(payload, obs)

        self.stats.calls += 1
        self.stats.failures += 1
        self.stats.last_error = last_err
        return Intent(skill="wait", reasoning=f"brain unavailable ({last_err})",
                      params={"seconds": 1.0})

    def _handle(self, payload: dict, obs: Observation) -> Intent:
        usage = payload.get("usage") or {}
        self.stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("completion_tokens_details") or {}
        self.stats.reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)

        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            self.stats.failures += 1
            self.stats.last_error = f"malformed payload: {str(payload)[:200]}"
            return Intent(skill="wait", reasoning="malformed response",
                          params={"seconds": 1.0})

        text = msg.get("content") or ""
        if not text.strip():
            # A reasoning model that exhausted max_tokens leaves its work in
            # reasoning_content. Salvage the JSON from there instead of losing the call.
            text = msg.get("reasoning_content") or ""

        intent = parse_intent(text, self.width, self.height, self.cfg.ground_grid)
        if intent.skill == "wait" and not intent.reasoning:
            self.stats.failures += 1
        return intent

    # -- diagnostics ----------------------------------------------------------------------

    def list_models(self) -> list[str]:
        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return [m.get("id", "") for m in json.loads(resp.read()).get("data", [])]

    def probe_vision(self, timeout: float = 60.0) -> tuple[bool, str]:
        """Send a synthetic image and report whether the model can actually read it.

        Cheap insurance: several providers advertise vision and then silently ignore the
        image part, which is indistinguishable from a broken agent unless you test it.
        """
        import base64
        import io

        import numpy as np
        from PIL import Image, ImageDraw

        code = "LUM-7391"
        img = Image.fromarray(np.full((120, 320, 3), 24, dtype=np.uint8))
        ImageDraw.Draw(img).text((12, 50), f"CODE {code}", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Reply with only the code shown in the image."},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
            "max_tokens": 900,
            "temperature": 0.0,
        }
        try:
            payload = self._post(body)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"

        msg = payload.get("choices", [{}])[0].get("message", {})
        text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
        found = code in text
        return found, re.sub(r"\s+", " ", text)[:200]
