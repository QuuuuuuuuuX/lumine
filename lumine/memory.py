"""Memory: what survives across a reasoning call, and across a five-hour mission.

Two horizons, deliberately separate:

``WorkingMemory``   seconds.  Recent actions and their outcomes, so the VLM does not repeat
                    a move that just failed.  Bounded, cheap, wiped per episode.
``MissionMemory``   hours.  Acts, subgoals, discovered places, and a failure ledger that
                    stops the agent from re-attempting the same dead end forever.

The failure ledger is the piece that actually matters.  A 2-second-latency brain with no
memory of its own failures will loop until the episode timer expires; recording
``(subgoal, why it failed)`` and feeding it back is the cheapest possible fix and it is what
makes hours-long unattended play viable.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from .types import Intent, Observation


@dataclass
class WorkingMemory:
    max_actions: int = 12

    actions: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    outcomes: deque[str] = field(default_factory=lambda: deque(maxlen=12))
    last_intent: Intent | None = None
    last_result: str = ""
    seen_entities: dict[str, float] = field(default_factory=dict)

    def record(self, intent: Intent, result: str) -> None:
        label = f"{intent.skill}({intent.target})" if intent.target else intent.skill
        self.actions.append(label)
        self.outcomes.append(result)
        self.last_intent = intent
        self.last_result = result

    def observe(self, obs: Observation) -> None:
        now = obs.t
        for e in obs.hud.nearby_entities:
            name = str(e.get("name", ""))
            if name:
                self.seen_entities[name] = now

    def history(self) -> list[str]:
        out = []
        for a, o in zip(self.actions, self.outcomes):
            out.append(f"{a} -> {o}" if o else a)
        return out

    def recent_failures(self, within: int = 6) -> list[str]:
        recent = list(zip(self.actions, self.outcomes))[-within:]
        return [a for a, o in recent if o.startswith("failed")]


@dataclass
class MissionMemory:
    """Long-horizon state for missions that outlive any single reasoning call."""

    mission: str = ""
    act: int = 1
    total_acts: int = 3
    started: float = field(default_factory=time.monotonic)

    subgoals: list[str] = field(default_factory=list)
    completed_subgoals: list[str] = field(default_factory=list)
    #: (skill, target) -> number of failed attempts.
    failures: Counter = field(default_factory=Counter)
    #: Facts the agent has established, e.g. "chest@3 needs pyro".
    facts: dict[str, Any] = field(default_factory=dict)
    acts_done: list[int] = field(default_factory=list)

    def add_subgoal(self, text: str) -> None:
        if text and text not in self.subgoals:
            self.subgoals.append(text)
            self.subgoals = self.subgoals[-24:]

    def complete(self, text: str) -> None:
        if not text:
            return
        if text not in self.completed_subgoals:
            self.completed_subgoals.append(text)
        if text in self.subgoals:
            self.subgoals.remove(text)

    def note_failure(self, intent: Intent, reason: str) -> int:
        key = f"{intent.skill}:{intent.target or '-'}"
        self.failures[key] += 1
        self.facts[f"fail:{key}"] = reason
        return self.failures[key]

    def is_dead_end(self, intent: Intent, limit: int = 3) -> bool:
        return self.failures.get(f"{intent.skill}:{intent.target or '-'}", 0) >= limit

    def advance_act(self) -> None:
        self.acts_done.append(self.act)
        self.act += 1

    def summary(self) -> str:
        stuck = [k for k, v in self.failures.items() if v >= 2]
        parts = [f"act {self.act}/{self.total_acts}"]
        if self.subgoals:
            parts.append(f"open subgoals: {', '.join(self.subgoals[-4:])}")
        if stuck:
            parts.append(f"repeatedly failing: {', '.join(stuck[:4])} -- do something different")
        return "; ".join(parts)

    def failure_hint(self) -> str:
        """Text injected into the next prompt so the brain can route around dead ends."""
        dead = [k for k, v in self.failures.items() if v >= 2]
        if not dead:
            return ""
        return ("These approaches have already failed repeatedly and must not be repeated: "
                + ", ".join(dead[:6]))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started
