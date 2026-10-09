"""The agent: three rates, one loop, and the threading that keeps them honest.

    perceive   5 Hz    a frame arrives
    reason   ~0.5 Hz   the VLM is asked what to do (optionally on a worker thread)
    control  30 Hz     a key/mouse action leaves for the game

The threading question is not incidental.  A two-second VLM call that blocks the control
loop freezes the game for two seconds, which in a real game means death.  Two modes exist
because the two settings want opposite things:

``sync``  (default for the sandbox) the world waits while the agent thinks.  Deterministic,
          reproducible, and it makes evaluation results comparable across runs.
``async`` (default for a real game) reasoning runs on a worker thread while the current
          skill keeps executing, so input never stalls.

The loop is written so that both modes share every other line of code.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .brain import Brain, build_brain
from .brain.adaptive import AdaptiveReasoner
from .brain.base import BrainContext
from .config import DataConfig, LumineConfig
from .controller import DONE, FAILED, SkillController
from .data import Recorder
from .memory import MissionMemory, WorkingMemory
from .perception import Perceiver, build_source
from .skills import SkillCall, validate
from .types import Action, EpisodeResult, Intent, Observation, TaskSpec


# --------------------------------------------------------------------------------------
# Async reasoning
# --------------------------------------------------------------------------------------


@dataclass
class ReasoningRequest:
    obs: Observation
    ctx: BrainContext


class ReasoningWorker:
    """Runs the slow brain off the control thread.  Queue depth is one: newer wins.

    Dropping a stale request is correct -- by the time a two-second call returns, the
    situation that motivated it may be gone, and reasoning about a stale frame is worse
    than not reasoning at all.
    """

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self._in: queue.Queue[ReasoningRequest | None] = queue.Queue(maxsize=1)
        self._out: queue.Queue[tuple[Intent, float]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="lumine-reason")
        self._busy = threading.Event()
        self._stop = threading.Event()
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self, req: ReasoningRequest) -> bool:
        if self._busy.is_set():
            return False
        try:
            self._in.put_nowait(req)
            return True
        except queue.Full:
            return False

    def poll(self) -> tuple[Intent, float] | None:
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                req = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            if req is None:
                break
            self._busy.set()
            t0 = time.monotonic()
            try:
                intent = self.brain.think(req.obs, req.ctx)
            except Exception as exc:  # noqa: BLE001 - worker must not die
                intent = Intent(skill="wait", reasoning=f"worker error: {exc}",
                                params={"seconds": 1.0})
            finally:
                self._busy.clear()
            self._out.put((intent, time.monotonic() - t0))

    def close(self) -> None:
        self._stop.set()
        try:
            self._in.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------


@dataclass
class AgentConfig:
    """Knobs that are about the agent's behaviour rather than any one subsystem."""

    #: "sync" (deterministic, sandbox default) or "async" (real game default).
    reasoning_mode: str = "sync"
    #: Stop an episode once the task succeeds, rather than running to the time limit.
    stop_on_success: bool = True
    #: Record every (frame, intent, action) triplet.  Off for pure evaluation.
    record: bool = False
    record_dir: str = "artifacts/data"
    #: Which curriculum stage the recording belongs to.
    record_stage: str = "pretrain"
    #: Samples per second written to disk.  Decoupled from the 30 Hz control rate, because
    #: storing 30 frames/s of 160x90 costs ~1.5 GB per hour of play.
    record_hz: float = 10.0
    #: Print a line per reasoning call.
    verbose: bool = True
    #: Hard ceiling on reasoning calls per episode; a runaway guard, not a policy.
    max_reason_calls: int = 200
    #: Give up on a skill after this long and force a re-plan.
    skill_timeout_s: float = 45.0
    #: Keep every captured frame in memory so an episode can be written to video.
    capture_frames: bool = False


class Agent:
    """The generalist agent: perceive, reason, act."""

    def __init__(
        self,
        cfg: LumineConfig | None = None,
        env: Any = None,
        brain: Brain | None = None,
        agent_cfg: AgentConfig | None = None,
    ) -> None:
        self.cfg = cfg or LumineConfig()
        self.agent_cfg = agent_cfg or AgentConfig()
        self.env = env

        self.perceiver = Perceiver(cfg=self.cfg.perception, env=env)
        if env is None and self.cfg.perception.source != "sim":
            self.perceiver.source = build_source(self.cfg.perception)

        self.brain = brain or build_brain(
            self.cfg.brain, self.cfg.perception.width, self.cfg.perception.height)
        self.reasoner = AdaptiveReasoner(self.cfg.reason)
        self.controller = SkillController(cfg=self.cfg.control, perception=self.cfg.perception)
        self.memory = MissionMemory()
        self.working = WorkingMemory()

        self._worker: ReasoningWorker | None = None
        if self.agent_cfg.reasoning_mode == "async":
            self._worker = ReasoningWorker(self.brain)

        self._instructions: list[str] = []
        self._transcript: list[dict[str, Any]] = []
        self._reason_calls = 0
        #: Environment clock at the start of the current episode.
        self._episode_t0: float = 0.0

        self.recorder: Recorder | None = None
        self.frames_captured: list[np.ndarray] = []
        if self.agent_cfg.record:
            data_cfg = DataConfig(root=self.agent_cfg.record_dir)
            self.recorder = Recorder(
                cfg=data_cfg, stage=self.agent_cfg.record_stage,
                frame_width=self.cfg.perception.policy_width,
                frame_height=self.cfg.perception.policy_height,
                record_hz=self.agent_cfg.record_hz,
                source=getattr(self.brain, "name", "unknown"),
            )

    # -- public API ------------------------------------------------------------------------

    def run_episode(self, task: TaskSpec | str, seed: int | None = None,
                    region: str | None = None) -> EpisodeResult:
        """Play one task to completion, success, death or timeout.

        ``region`` overrides where the task is played, which is the entire point of the
        out-of-distribution evaluation: the same order, in a world the agent has never
        seen. It must be threaded through to the environment -- an earlier version
        accepted the region and dropped it, so the 'held-out' column was measuring the
        training world twice.
        """
        if self.env is None:
            raise RuntimeError("Agent.run_episode needs an environment")

        obs = self.env.reset(task=task, region=region, seed=seed)
        spec: TaskSpec = self.env.task or TaskSpec(task_id=str(task), category="?", instruction=str(task))
        instruction = spec.instruction

        self.memory = MissionMemory(mission=spec.task_id)
        self.working = WorkingMemory()
        if self.recorder is not None:
            self.recorder.new_episode()
        self.reasoner = AdaptiveReasoner(self.cfg.reason)
        self.controller.reset_episode()
        self._transcript = []
        self._reason_calls = 0
        for h in spec.hints:
            self.memory.add_subgoal(h)

        # Two clocks, and they must not be confused. ``t_start`` is wall clock and is only
        # used for reporting; ``sim_t0`` is the environment's clock and is what every
        # deadline, timeout and elapsed-time figure inside the episode is measured against.
        t_start = time.monotonic()
        sim_t0 = obs.t
        self._episode_t0 = obs.t
        dt = 1.0 / self.cfg.control.hz
        vlm_seconds = 0.0
        frames = 0
        last_frame_t = 0.0

        # The very first decision is always worth making.
        self.reasoner._last_instruction = ""
        obs = self.perceiver.observe(obs, force=True)

        while True:
            # ---- perception (5 Hz) ----------------------------------------------------
            if obs.frame_is_new:
                frames += 1
                self.working.observe(obs)
                last_frame_t = obs.t
                if self.agent_cfg.capture_frames:
                    self.frames_captured.append(obs.frame.image.copy())

            # ---- reasoning (adaptive, ~0.5 Hz) ----------------------------------------
            vlm_seconds += self._maybe_reason(obs, instruction, spec, sim_t0)

            # ---- control (30 Hz) ------------------------------------------------------
            if self.controller.call is None:
                action = Action(dt=dt)
            else:
                action = self.controller.tick(obs)

            if self.recorder is not None:
                self.recorder.maybe_add(
                    obs.frame.image, action, obs.t,
                    instruction=instruction,
                    reasoning=(self.working.last_intent.reasoning if self.working.last_intent else ""),
                    skill=(self.working.last_intent.skill if self.working.last_intent else ""),
                )

            if self.controller.status in (DONE, FAILED):
                reason = self.controller.reason
                intent = self.working.last_intent or Intent(skill="wait")
                self.working.record(intent, f"{self.controller.status}: {reason}")
                if self.controller.status == DONE:
                    self.memory.complete(intent.subgoal or str(intent.target or ""))
                else:
                    n = self.memory.note_failure(intent, reason)
                    if n >= 3:
                        self.controller.call = None
                        self.reasoner.end_skill("dead_end")
                self.reasoner.end_skill(self.controller.status)
                self.controller.call = None

            obs = self.env.step(action)

            # ---- termination ----------------------------------------------------------
            success = bool(obs.info.get("success")) or self.env.success()
            if success and self.agent_cfg.stop_on_success:
                return self._result(spec, True, t_start, frames, vlm_seconds, "task complete")
            if obs.done or not obs.hud.alive:
                return self._result(spec, False, t_start, frames, vlm_seconds,
                                    "player defeated" if not obs.hud.alive else "episode ended")
            if obs.t - sim_t0 > spec.max_seconds:
                return self._result(spec, False, t_start, frames, vlm_seconds, "timed out")
            if obs.t - sim_t0 > spec.max_seconds * 3:
                return self._result(spec, False, t_start, frames, vlm_seconds, "hard timeout")

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        if self._worker is not None:
            self._worker.close()
        self.perceiver.close()
        self.brain.close()
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- reasoning -------------------------------------------------------------------------

    def _maybe_reason(self, obs: Observation, instruction: str, spec: TaskSpec,
                      t_start: float) -> float:
        """Return the wall-clock seconds this tick spent waiting on the brain."""
        if self._reason_calls >= self.agent_cfg.max_reason_calls:
            return 0.0

        # Pick up an async result.
        if self._worker is not None:
            got = self._worker.poll()
            if got is not None:
                intent, _dt = got
                self._install(intent, obs, t_start)
                self.reasoner.note_call(obs.t)
            if self.controller.call is None and not self._worker.busy:
                decision = self.reasoner.should_reason(obs, instruction)
                if decision.should:
                    self._worker.submit(ReasoningRequest(obs, self._context(instruction, spec, t_start, obs.t)))
            return 0.0

        # Synchronous: the world waits.  Deterministic, and the pause is accounted for.
        #
        # Both branches must consult the trigger logic. The original version computed the
        # decision in the idle branch and then ignored it, so an agent with no active skill
        # called the brain on *every* tick -- up to 30 Hz, sixty times the design rate, and
        # the single most expensive bug in this file.
        decision = self.reasoner.should_reason(
            obs, instruction, has_skill=self.controller.call is not None)
        if not decision.should:
            return 0.0

        ctx = self._context(instruction, spec, t_start, obs.t)
        t0 = time.monotonic()
        intent = self.brain.think(obs, ctx)
        spent = time.monotonic() - t0
        self._install(intent, obs, t_start)
        self.reasoner.note_call(obs.t)
        return spent

    def _context(self, instruction: str, spec: TaskSpec, t_start: float,
                 now: float) -> BrainContext:
        fail_hint = self.memory.failure_hint()
        history = self.working.history()
        if fail_hint:
            history = history + [fail_hint]
        return BrainContext(
            instruction=instruction,
            subgoal=self.working.last_intent.subgoal if self.working.last_intent else "",
            history=history,
            last_result=self.working.last_result,
            progress=tuple(self.memory.facts.get("progress", (0, 0))) or (0, 0),  # type: ignore[arg-type]
            seconds_elapsed=max(0.0, now - (self._episode_t0 or now)),
            hints=spec.hints,
            region=spec.region,
            retry=bool(fail_hint),
        )

    def _install(self, intent: Intent, obs: Observation, t_start: float) -> None:
        """Validate a brain decision and hand it to the controller."""
        self._reason_calls += 1

        if self.memory.is_dead_end(intent):
            intent = Intent(skill="switch_character", target="anemo",
                            params={"element": "anemo"},
                            reasoning="previous approach is a known dead end; re-approach",
                            subgoal="change approach")
        ok, why = validate(intent.skill, intent.params)
        if not ok:
            intent = Intent(skill="wait", params={"seconds": 1.0}, reasoning=f"rejected: {why}")

        deadline = obs.t + self.agent_cfg.skill_timeout_s
        self.controller.set_skill(
            SkillCall(intent.skill, dict(intent.params), obs.t, deadline,
                      source="vlm", target_px=intent.target_px),
            obs,
        )
        self.reasoner.start_skill(intent.skill, obs.t)
        self.working.last_intent = intent
        self.memory.add_subgoal(intent.subgoal)

        if self.agent_cfg.verbose:
            print(f"  [{obs.t - t_start:6.1f}s] {intent.skill:16s} "
                  f"target={str(intent.target):14.14s} :: {intent.reasoning[:80]}")

        if self.agent_cfg.record:
            self._transcript.append({
                "t": round(obs.t - t_start, 3),
                "skill": intent.skill,
                "target": intent.target,
                "reasoning": intent.reasoning,
                "subgoal": intent.subgoal,
            })

    # -- results ---------------------------------------------------------------------------

    def _rate_note(self, reason: str, sim_seconds: float) -> str:
        """Qualify the reasoning rate so a short episode cannot mislead.

        The rate is a *mean over the episode*. Two calls in a 2.4 s conversation is 0.83 Hz,
        which reads as exceeding the 0.5 Hz ceiling even though the instantaneous rate is
        correctly capped at one call per `min_interval_s`. Below ~20 s of game time the mean
        says more about episode length than about the agent, so report the count instead.
        """
        if sim_seconds < 20.0:
            return f"{reason} [{self._reason_calls} model calls in {sim_seconds:.1f}s]"
        return f"{reason} [mean {self._reason_calls / sim_seconds:.2f} Hz over {sim_seconds:.0f}s]"

    def _result(self, spec: TaskSpec, success: bool, t_start: float, frames: int,
                vlm_seconds: float, reason: str) -> EpisodeResult:
        # Per-episode figures, not the brain's lifetime counters. Reporting the running total
        # made the reasoning rate climb across a suite and hid the real number entirely.
        sim_seconds = max(1e-6, float(getattr(self.env, "sim_time", 0.0) or 0.0))
        return EpisodeResult(
            task=spec,
            success=success,
            seconds=time.monotonic() - t_start,
            frames=frames,
            vlm_calls=self._reason_calls,
            vlm_seconds=round(vlm_seconds, 3),
            controller=self.cfg.control.controller,
            reason=self._rate_note(reason, sim_seconds),
            faults=self.controller.faults,
            faults_by_kind=self.controller.faults_by_kind(),
            fault_examples=dict(self.controller.fault_examples),
            transcript=list(self._transcript),
        )
