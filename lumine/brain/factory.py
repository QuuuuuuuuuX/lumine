"""Brain construction: config in, :class:`Brain` out.

Kept separate from :mod:`lumine.brain.base` so that importing the protocol never drags in
HTTP or the provider table.
"""

from __future__ import annotations

from ..config import BrainConfig
from .base import Brain
from .providers import RECOMMENDED, PROVIDERS, get_provider, render_table
from .scripted import ScriptedBrain
from .vlm import VLMBrain


def build_brain(cfg: BrainConfig, frame_width: int = 640, frame_height: int = 360) -> Brain:
    """Instantiate the configured brain, degrading to scripted play when that is the only
    thing that can actually run."""
    if cfg.provider == "none":
        return ScriptedBrain()

    try:
        return VLMBrain(cfg, frame_width=frame_width, frame_height=frame_height)
    except RuntimeError as exc:
        if not cfg.fallback_to_scripted:
            raise
        print(f"[lumine] VLM unavailable ({exc}); falling back to the scripted brain.")
        return ScriptedBrain()


def describe_providers() -> str:
    lines = [render_table(), "", "## Recommended, in order", ""]
    for key, why in RECOMMENDED:
        spec = PROVIDERS[key]
        lines.append(f"- **`{key}`** (`{spec.default_model}`) — {why}")
    lines.append("")
    lines.append("Set the key in the environment, then pick the provider in your config:")
    lines.append("")
    lines.append("```bash")
    lines.append('export DEEPSEEK_API_KEY=sk-...      # https://platform.deepseek.com/api_keys')
    lines.append('lumine play --provider deepseek --model deepseek-flash')
    lines.append("```")
    return "\n".join(lines)


def provider_notes(key: str) -> str:
    return get_provider(key).notes
