"""``SimEnv``: the sandbox behind the same interface a real game would present.

The contract is deliberately tiny, because ``lumine.live.LiveEnv`` implements it too and the
agent must not be able to tell the two apart::

    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    while True:
        obs = env.step(action)        # one 30 Hz control tick
        if obs.done: break

Two performance decisions matter:

* **Stepping never renders.** :meth:`World.step` is scalar maths over a few dozen entities;
  the rasteriser is asked for pixels at the *perception* rate (5 Hz), and
  ``Observation.frame_is_new`` tells the agent whether it is looking at a fresh image or the
  previous one.  That is what lets a laptop run the whole loop far faster than real time.
* **Stepping never sleeps** unless ``realtime=True``.  Data collection wants thousands of
  logical steps per second; evaluation wants reproducibility, which a sleep would destroy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import PerceptionConfig, SimConfig
from ..types import Action, Frame, HUDState, Observation, TaskSpec
from .render import Renderer
from .tasks import TASK_LIST, TASKS, task_by_id, task_success
from .world import World

__all__ = ["SimEnv", "TASK_LIST", "TASKS", "task_by_id"]


class SimEnv:
    """A deterministic 3D open world, stepped at 30 Hz and rendered at 5 Hz."""

    def __init__(
        self,
        sim: SimConfig | None = None,
        perception: PerceptionConfig | None = None,
        render: bool = True,
        realtime: bool = False,
    ) -> None:
        self.sim = sim or SimConfig()
        self.perception = perception or PerceptionConfig()
        self.do_render = render
        self.realtime = realtime

        self._world: World | None = None
        self._task: TaskSpec | None = None
        self._renderer: Renderer | None = None
        self._frame: Frame | None = None
        self._t = 0.0
        self._since_render = 1e9          # force a render on the first step
        self._done = False
        self._render_cost = 0.0

        if render:
            self._renderer = Renderer(width=self.perception.width,
                                      height=self.perception.height,
                                      fog=self.sim.fog)

    # -- properties ------------------------------------------------------------------------

    @property
    def world(self) -> World:
        if self._world is None:
            raise RuntimeError("SimEnv.reset() must be called before using the world")
        return self._world

    @property
    def task(self) -> TaskSpec | None:
        return self._task

    @property
    def renderer(self) -> Renderer | None:
        return self._renderer

    @property
    def sim_time(self) -> float:
        """Logical seconds elapsed in the episode. This is the agent's clock."""
        return self._t

    # -- lifecycle -------------------------------------------------------------------------

    def reset(self, task: TaskSpec | str | None = None, region: str | None = None,
              seed: int | None = None) -> Observation:
        """Build a fresh world for ``task`` and return the first observation."""
        spec = self._resolve_task(task)
        self._task = spec
        self._t = 0.0
        self._since_render = 1e9
        self._done = False
        self._frame = None

        self._world = World(config=self.sim, task=spec, region=region, seed=seed)
        if self._renderer is not None:
            self._renderer.entity_boxes = []

        obs = self._observe(fresh=True, force=True)
        return obs

    def step(self, action: Action) -> Observation:
        """Advance one control tick by ``action``."""
        if self._world is None:
            raise RuntimeError("SimEnv.reset() must be called before step()")

        dt = 1.0 / max(1e-6, 30.0)
        # dt is owned by the environment, not by the caller's Action, so a malformed action
        # cannot slow the world down or speed it up.
        self._t += dt
        self._world.step(action, dt)
        self._since_render += dt

        if not self._world.player.alive or self._world.done_reason:
            self._done = True
        if self._task is not None and self._t > self._task.max_seconds * 4.0:
            self._done = True

        fresh = self._should_render()
        obs = self._observe(fresh=fresh)
        if self.realtime:
            time.sleep(dt)
        return obs

    def render_frame(self) -> np.ndarray:
        """Force a render and return the pixels (used by ``PerceptionConfig.source='sim'``)."""
        self._observe(fresh=True, force=True)
        assert self._frame is not None
        return self._frame.image

    def success(self) -> bool:
        """Score the episode from the world's flags, against the task's criteria."""
        if self._world is None:
            return False
        return task_success(self._task, self._world.flags)

    def close(self) -> None:
        self._renderer = None
        self._frame = None

    # -- internals -------------------------------------------------------------------------

    def _resolve_task(self, task: TaskSpec | str | None) -> TaskSpec | None:
        if isinstance(task, TaskSpec):
            return task
        if isinstance(task, str):
            return task_by_id(task)
        return None

    def _should_render(self) -> bool:
        if not self.do_render or self._renderer is None:
            return False
        period = 1.0 / max(1e-6, self.perception.hz)
        # The epsilon matters. Accumulating 1/30 six times gives 0.19999999999999998, which is
        # < 0.2, so a naive comparison renders every seventh tick -- 4.3 Hz instead of the
        # specified 5. Perception rate is one of the three rates that define the architecture;
        # being 14% under it is not a rounding detail.
        if self._since_render >= period - 1e-9:
            self._since_render = 0.0
            return True
        return False

    def _observe(self, fresh: bool, force: bool = False) -> Observation:
        assert self._world is not None
        world = self._world

        if force or (fresh and self.do_render):
            if self._renderer is not None:
                t0 = time.perf_counter()
                image = self._renderer.render(world)
                self._render_cost = time.perf_counter() - t0
                self._frame = Frame(image=image, t=self._t)
            elif self._frame is None:
                # Rendering disabled: hand back a blank frame of the right shape so the
                # policy still has something to chew on.
                blank = np.zeros((self.perception.height, self.perception.width, 3), np.uint8)
                self._frame = Frame(image=blank, t=self._t)
        elif self._frame is None:
            blank = np.zeros((self.perception.height, self.perception.width, 3), np.uint8)
            self._frame = Frame(image=blank, t=self._t)

        hud: HUDState = world.hud_state()
        boxes = list(self._renderer.entity_boxes) if (fresh or force) and self._renderer else []
        success = task_success(self._task, world.flags)

        return Observation(
            frame=self._frame,
            hud=hud,
            t=self._t,
            reward=self._reward(success),
            done=self._done,
            frame_is_new=bool(fresh or force),
            info={
                "success": bool(success),
                "flags": dict(world.flags),
                "entity_boxes": boxes,
                "region": world.region,
                "task_id": self._task.task_id if self._task else None,
                "render_seconds": round(self._render_cost, 5),
                "sim_time": round(self._t, 3),
            },
        )

    def _reward(self, success: bool) -> float:
        """Sparse task reward plus shaping from progress.

        Shaping exists only so that the scripted teacher's recordings carry a usable signal;
        success itself is always judged from ``flags``, never from this number.
        """
        if self._world is None:
            return 0.0
        flags = self._world.flags
        r = 2.0 * float(flags.get("chests_opened", 0))
        r += 1.0 * float(flags.get("enemies_defeated", 0))
        r += 0.25 * float(flags.get("monuments_activated", 0))
        r += 1.0 * float(flags.get("anemoculus_collected", 0))
        r += 5.0 * len(flags.get("boss_defeated", ()) or ())
        if success:
            r += 50.0
        return r


# --------------------------------------------------------------------------------------
# Demo / preview generation
# --------------------------------------------------------------------------------------


def _preview(out_dir: str = "artifacts/sim_previews", steps: int = 120) -> int:
    import cv2

    cfg = PerceptionConfig()
    sim = SimConfig()
    env = SimEnv(sim=sim, perception=cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'task':30s} {'category':10s} {'boxes':>5s}  note")
    written = 0
    for spec in TASK_LIST:
        try:
            obs = env.reset(task=spec, seed=0)
            for i in range(steps):
                # Nudge forward so the camera is not staring at the spawn point.
                act = Action(keys={"W"} if i % 30 < 20 else set())
                obs = env.step(act)
            img = env.render_frame()
            path = out / f"{spec.task_id}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            boxes = len(env.renderer.entity_boxes) if env.renderer else 0
            unique = len(np.unique(img.reshape(-1, 3), axis=0))
            print(f"{spec.task_id:30s} {spec.category:10s} {boxes:5d}  "
                  f"{unique:5d} colours  {'OK' if unique > 200 else 'FLAT!'}")
            written += 1
        except Exception as exc:  # noqa: BLE001
            print(f"{spec.task_id:30s} {spec.category:10s}     -  FAILED: "
                  f"{type(exc).__name__}: {exc}")
    env.close()
    print(f"\n{written} previews written to {out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(_preview())
