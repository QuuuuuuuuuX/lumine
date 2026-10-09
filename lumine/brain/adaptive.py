"""Adaptive reasoning: deciding when two seconds of VLM time is worth spending.

This module is the direct implementation of Lumine's "adaptively invokes reasoning only
when necessary".  The agent only reaches 30 Hz because almost every tick is *not* a
reasoning tick; the whole trick is choosing the rare ticks that are.

Triggers, cheapest first
------------------------
======================  =========================================================
instruction_changed     a new standing order arrived
skill_finished          the current skill succeeded, failed, or timed out
progress_changed        the objective counter moved (a chest opened, an enemy died)
hp_dropped              the agent just took a serious hit
surprise                the screen changed far more than recent history predicts
stuck                   position has barely moved for N seconds
budget                  the rolling call-rate budget has room and the last call is stale
keepalive               a hard ceiling on how long the agent may act unsupervised
======================  =========================================================

``surprise`` uses a tiny running model of the frame: each capture is reduced to a coarse
grayscale vector, and a spike in L2 distance against the running mean is a cheap novelty
detector.  It is deliberately not learned -- it has to work on the very first frame of the
very first episode.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..config import ReasonConfig
from ..types import Observation


@dataclass
class ReasonDecision:
    should: bool
    because: str = ""
    urgency: float = 0.0


@dataclass
class AdaptiveReasoner:
    """Stateful trigger logic.  One instance per agent, one ``should_reason`` per tick."""

    cfg: ReasonConfig = field(default_factory=ReasonConfig)

    _last_call: float = 0.0
    _last_instruction: str = ""
    _last_progress: tuple[int, int] = (0, 0)
    _last_hp: float = -1.0
    _skill_started: float = 0.0
    _skill_name: str = ""
    _skill_ended: bool = False
    _end_reason: str = ""
    #: Edge-trigger latches, so a persistent condition is reported once, not 30 times a second.
    _stuck_fired: bool = False
    _skill_timeout_fired: bool = False
    #: (time, xz-position) samples for the stuck detector.
    _track: deque = field(default_factory=lambda: deque(maxlen=240))
    _features: deque = field(default_factory=lambda: deque(maxlen=60))
    _calls: deque = field(default_factory=lambda: deque(maxlen=64))
    _started: float = field(default_factory=time.monotonic)

    # -- lifecycle ------------------------------------------------------------------------

    def start_skill(self, name: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._skill_name = name
        self._skill_started = now
        self._skill_ended = False
        self._end_reason = ""
        self._skill_timeout_fired = False

    def end_skill(self, reason: str, now: float | None = None) -> None:
        self._skill_ended = True
        self._end_reason = reason

    def note_call(self, now: float | None = None) -> None:
        self._last_call = now if now is not None else time.monotonic()
        self._calls.append(self._last_call)
        self._skill_ended = False
        self._end_reason = ""

    def note_progress(self, progress: tuple[int, int]) -> None:
        self._last_progress = progress

    # -- the decision ---------------------------------------------------------------------

    def should_reason(self, obs: Observation, instruction: str,
                      has_skill: bool = True) -> ReasonDecision:
        now = obs.t
        h = obs.hud

        # Baselines advance on *every* call, including the ones that return early below.
        # Skipping this on the instruction-change path was a real bug: it meant the first
        # hit taken right after a new order went undetected, because there was no previous
        # HP value to compare against.
        prev_hp, prev_progress = self._last_hp, self._last_progress
        self._last_hp = h.hp

        # 1. New standing order -- highest priority, always obeyed, cooldown or not.
        if self.cfg.on_instruction_change and instruction != self._last_instruction:
            self._last_instruction = instruction
            self._last_progress = tuple(h.objective_progress)
            return ReasonDecision(True, "instruction_changed", 1.0)

        # 1b. Hard floor. Several triggers below describe a persistent *state* rather than an
        # event, so without this they re-fire on every tick and the agent spends its entire
        # budget re-deciding the same thing.
        if self._calls and (now - self._last_call) < self.cfg.min_interval_s:
            return ReasonDecision(False, "cooldown")

        # 1c. Idle: no skill is executing, so there is nothing for the controller to do.
        #     This must live *behind* the cooldown, not in front of it. An earlier version
        #     special-cased the idle case in the agent loop and bypassed the trigger logic
        #     entirely, which let a finished skill re-plan on the very next tick, forever.
        if not has_skill:
            return ReasonDecision(True, "no_skill", 0.95)

        # 2. The current skill finished, one way or another.
        if self.cfg.on_skill_end and self._skill_ended:
            return ReasonDecision(True, f"skill_{self._end_reason or 'ended'}", 0.9)

        # 3. Objective progress moved.
        if self.cfg.on_progress_change and tuple(h.objective_progress) != tuple(prev_progress):
            self._last_progress = tuple(h.objective_progress)
            return ReasonDecision(True, "progress_changed", 0.8)

        # 4. Took a real hit.
        if prev_hp >= 0 and prev_hp > 0:
            drop = (prev_hp - h.hp) / max(1e-6, h.max_hp)
            if drop >= self.cfg.hp_drop_frac:
                return ReasonDecision(True, f"hp_dropped({drop:.0%})", 0.85)

        # 5. Visual surprise.
        if obs.frame_is_new:
            self._track.append((now, h.position))
            z = self._observe_frame(obs.frame.image)
            if z is not None and z >= self.cfg.surprise_z:
                return ReasonDecision(True, f"surprise(z={z:.1f})", 0.6)

        # 6. Stuck -- edge-triggered. Firing once when the agent *becomes* stuck is useful;
        #    firing again on every tick while it remains stuck is a livelock.
        if self._is_stuck(now):
            if not self._stuck_fired:
                self._stuck_fired = True
                return ReasonDecision(True, "stuck", 0.7)
        else:
            self._stuck_fired = False

        # 7. Skill ran too long -- also edge-triggered, for the same reason.
        if now - self._skill_started > self.cfg.max_skill_seconds:
            if not self._skill_timeout_fired:
                self._skill_timeout_fired = True
                return ReasonDecision(True, "skill_timeout", 0.5)
        else:
            self._skill_timeout_fired = False

        # 8. Rate budget says we may, and the last decision is stale.
        if self._budget_ok(now) and now - self._last_call > max(self.cfg.keepalive_s / 4.0, 1.0):
            return ReasonDecision(True, "budget", 0.2)

        # 9. Hard keepalive.
        if now - self._last_call > self.cfg.keepalive_s:
            return ReasonDecision(True, "keepalive", 0.3)

        return ReasonDecision(False)

    # -- detectors ------------------------------------------------------------------------

    def _observe_frame(self, image: np.ndarray) -> float | None:
        """Return the novelty z-score of this frame, or None while warming up."""
        small = image[::12, ::12].mean(axis=2).astype(np.float32)
        feat = small.reshape(-1)
        if feat.size == 0:
            return None
        self._features.append(feat)
        if len(self._features) < 12:
            return None
        hist = np.stack(list(self._features)[:-1])
        mu = hist.mean(axis=0)
        sd = hist.std(axis=0) + 1e-3
        z = float(np.abs((feat - mu) / sd).mean())
        return z

    def _is_stuck(self, now: float) -> bool:
        window = [p for t, p in self._track if now - t <= self.cfg.stuck_seconds]
        if len(window) < 4:
            return False
        pts = np.asarray(window, dtype=np.float64)
        # Horizontal displacement only: being lifted by a wind current is not "stuck".
        spread = float(np.linalg.norm(pts[:, [0, 2]].max(axis=0) - pts[:, [0, 2]].min(axis=0)))
        return spread < self.cfg.stuck_distance

    def _budget_ok(self, now: float) -> bool:
        recent = [t for t in self._calls if now - t <= 10.0]
        return len(recent) < max(1, int(self.cfg.hz * 10.0))

    # -- introspection --------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        elapsed = max(1e-6, time.monotonic() - self._started)
        return {
            "reason_calls": len(self._calls),
            "reason_hz": round(len(self._calls) / elapsed, 3),
            "last_reason": self._end_reason or self._skill_name,
        }
