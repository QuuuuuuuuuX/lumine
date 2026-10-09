"""Live mode: drive a real game through the same interface as the sandbox.

This is the piece that makes the sandbox more than a toy.  :class:`LiveEnv` presents a real
screen and a real keyboard/mouse as an object with ``reset`` / ``step`` / ``success``, so
:class:`~lumine.agent.Agent` runs unmodified against Genshin, or any other game, by changing
one line of config::

    cfg.perception.source = "screen"     # capture the real display
    cfg.control.backend   = "uinput"     # inject real key events

Two honest caveats, stated here rather than discovered later:

* **There is no success signal.**  A real game does not tell you when you won.  ``success()``
  always returns False and termination comes from the instruction's time budget, from the
  user pressing Ctrl-C, or from a task-specific checker you supply via ``checker``.
* **Frame pacing is real.**  The sandbox can run thousands of logical steps per second; a
  real game runs at its own speed, so ``step`` blocks to hold a true 30 Hz control loop.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .config import LumineConfig
from .input_backend import ActionBackend, build_backend
from .perception import FrameSource, Perceiver, build_source
from .types import Action, HUDState, Observation, TaskSpec

DEFAULT_INSTRUCTION = "Play the game well. Pursue whatever objective is currently on screen."


class LiveEnv:
    """A real game, wearing the sandbox's clothes."""

    def __init__(
        self,
        cfg: LumineConfig | None = None,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        source: FrameSource | None = None,
        backend: ActionBackend | None = None,
        checker: Callable[[np.ndarray], bool] | None = None,
        max_seconds: float = 3600.0,
        hud_provider: Callable[[], HUDState] | None = None,
    ) -> None:
        self.cfg = cfg or LumineConfig()
        self.task = TaskSpec(
            task_id="live",
            category="live",
            instruction=instruction,
            region="live",
            max_seconds=max_seconds,
        )
        self.source = source or build_source(self.cfg.perception)
        self.backend = backend or build_backend(
            self.cfg.control.backend, self.cfg.control.mouse_sensitivity)
        self.checker = checker
        self.hud_provider = hud_provider
        self.perceiver = Perceiver(cfg=self.cfg.perception, source=self.source)

        self._t0 = 0.0
        self._ticks = 0
        self._last_step = 0.0
        self._done = False

    # -- env interface ---------------------------------------------------------------------

    def reset(self, task: TaskSpec | str | None = None, region: str | None = None,
              seed: int | None = None) -> Observation:
        if isinstance(task, TaskSpec):
            self.task = task
        elif isinstance(task, str):
            self.task = TaskSpec(task_id="live", category="live", instruction=task,
                                 region="live", max_seconds=self.task.max_seconds)
        self._t0 = time.monotonic()
        self._last_step = self._t0
        self._ticks = 0
        self._done = False
        obs = self.perceiver.observe(None, force=True)
        obs.hud = self._hud(obs)
        return obs

    def step(self, action: Action) -> Observation:
        period = 1.0 / max(1e-6, self.cfg.control.hz)
        now = time.monotonic()
        # Hold the real 30 Hz cadence.  A real game cannot be stepped faster than it runs.
        sleep = period - (now - self._last_step)
        if sleep > 0:
            time.sleep(sleep)
        self._last_step = time.monotonic()
        self._ticks += 1

        self.backend.send(action)

        obs = self.perceiver.observe(None)
        obs.t = time.monotonic() - self._t0
        hud = self._hud(obs)
        obs.hud = hud
        obs.done = self._done
        obs.info.setdefault("success", False)
        obs.info["region"] = "live"
        obs.info["task_id"] = self.task.task_id
        if self.checker is not None and obs.frame_is_new:
            hit = False
            try:
                hit = bool(self.checker(obs.frame.image))
            except Exception:  # noqa: BLE001 - a user checker must not kill the run
                hit = False
            obs.info["success"] = hit
            if hit:
                self._done = True
        return obs

    def render_frame(self) -> np.ndarray:
        return self.perceiver.observe(None, force=True).frame.image

    def success(self) -> bool:
        return self._done

    def close(self) -> None:
        self.backend.close()
        self.perceiver.close()

    # -- internals -------------------------------------------------------------------------

    def _hud(self, obs: Observation) -> HUDState:
        """A real game gives us no structured state.  A caller may still supply a reader.

        With no reader, return a *neutral* HUD: HP is reported at full and no objective
        progress is claimed.  This matters -- the adaptive reasoner triggers on HP drops and
        on progress changes, and inventing values would either spam the model or blind it.
        """
        if self.hud_provider is not None:
            try:
                return self.hud_provider()
            except Exception:  # noqa: BLE001
                pass
        return HUDState(region="live", objective=self.task.instruction, position=(0.0, 0.0, 0.0))
