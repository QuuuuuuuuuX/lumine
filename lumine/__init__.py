"""Lumine: a generalist agent for 3D open worlds.

A reimplementation of the architecture in *Lumine: Building Generalist Agents in 3D Open
Worlds* (ByteDance Seed & NTU, arXiv 2511.08892), with a built-in 3D sandbox standing in
for the game so the whole thing runs on one machine.

The three rates the whole design hangs off::

    perceive  5 Hz    raw pixels
    reason   ~0.5 Hz  a vision-language model, invoked only when necessary
    control  30 Hz    keyboard and mouse

Quick start::

    from lumine import SimEnv, Agent, LumineConfig

    env = SimEnv()
    with Agent(env=env) as agent:
        result = agent.run_episode("combat_defeat_and_chest")
        print(result.success, result.reason)
"""

from .agent import Agent, AgentConfig
from .config import LumineConfig
from .types import Action, EpisodeResult, Frame, HUDState, Intent, Observation, TaskSpec

__version__ = "1.0.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "LumineConfig",
    "Action",
    "Observation",
    "Frame",
    "HUDState",
    "Intent",
    "TaskSpec",
    "EpisodeResult",
    "SimEnv",
    "TASKS",
    "__version__",
]


def __getattr__(name: str):
    """Expose the sandbox lazily so importing the agent never requires the renderer."""
    if name in ("SimEnv", "TASKS", "TASK_LIST", "task_by_id"):
        from . import sim

        return getattr(sim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
