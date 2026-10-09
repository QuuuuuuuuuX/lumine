"""The ``Brain`` protocol and its accounting.

Every brain is measured, not trusted.  :class:`BrainStats` tracks wall-clock latency and
call count so the evaluation report can state honestly what fraction of the agent's
decision budget the slow brain consumed.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field

from ..types import Intent, Observation


@dataclass
class BrainStats:
    calls: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    last_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    last_error: str = ""

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "mean_seconds": round(self.mean_seconds, 3),
            "total_seconds": round(self.total_seconds, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "last_error": self.last_error,
        }


@dataclass
class BrainContext:
    """Everything a brain may look at.  Assembled once per reasoning call."""

    instruction: str
    subgoal: str = ""
    history: list[str] = field(default_factory=list)
    last_result: str = ""
    progress: tuple[int, int] = (0, 0)
    seconds_elapsed: float = 0.0
    hints: tuple[str, ...] = ()
    region: str = ""
    #: Set when the caller already knows the goal is unreachable and wants a re-plan.
    retry: bool = False


class Brain(abc.ABC):
    """Turns (observation, context) into an intent.

    Implementations must be *slow-brain shaped*: allowed to take seconds, must not be
    called at control rate, and must never raise on a bad model response — return a safe
    ``wait`` intent instead so one malformed JSON blob cannot kill a 5-hour run.
    """

    name: str = "brain"

    def __init__(self) -> None:
        self.stats = BrainStats()

    @abc.abstractmethod
    def think(self, obs: Observation, ctx: BrainContext) -> Intent:
        """Produce the next intent.  Blocking; callers run this off the control thread."""

    def warmup(self) -> None:
        """Optional: pay connection setup cost before the timed episode starts."""

    def close(self) -> None:
        """Release sockets/processes."""

    # -- shared timing helper ------------------------------------------------------------

    def _timed(self, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            out = fn(*args, **kwargs)
            self.stats.calls += 1
            return out
        except Exception as exc:  # noqa: BLE001 - a brain failure must never be fatal
            self.stats.failures += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            self.stats.calls += 1
            return None
        finally:
            dt = time.monotonic() - t0
            self.stats.last_seconds = dt
            self.stats.total_seconds += dt

    def __enter__(self) -> "Brain":
        self.warmup()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
