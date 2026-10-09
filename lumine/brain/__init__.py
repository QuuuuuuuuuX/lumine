"""Brains: anything that can turn an observation into an ``Intent``.

The agent never imports a concrete brain.  It asks :func:`build_brain` for one and gets
back something satisfying :class:`Brain`.  That is what makes ``config.brain.provider``
a one-line swap between a 2-second cloud VLM, a local Ollama model, and a zero-dependency
scripted controller.
"""

from .base import Brain, BrainStats
from .factory import build_brain, describe_providers

__all__ = ["Brain", "BrainStats", "build_brain", "describe_providers"]
