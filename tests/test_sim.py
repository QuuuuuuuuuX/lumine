"""Tests for the built-in 3D sandbox.

    python -m tests.test_sim
    pytest tests/test_sim.py -q

These are integration tests: they assert the contract ``lumine.agent`` actually consumes
(``reset``/``step``/``success``, the HUD fields, the frame-pacing behaviour, entity boxes)
rather than the renderer's aesthetics. A separate check asserts the frame is not blank,
because a renderer that silently returns flat colour would make the entire VLM pipeline look
like a model failure.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumine.config import PerceptionConfig, SimConfig
from lumine.sim import TASK_LIST, TASKS, SimEnv, task_by_id
from lumine.sim.render import Renderer
from lumine.sim.tasks import task_success
from lumine.sim.world import PARTY, World
from lumine.types import Action, HUDState, Observation, TaskSpec


def _env(**kw) -> SimEnv:
    return SimEnv(sim=SimConfig(), perception=PerceptionConfig(width=320, height=180), **kw)


# --------------------------------------------------------------------------------------


def test_task_registry_shape() -> None:
    assert len(TASK_LIST) >= 20, "the suite should cover six categories plus the missions"
    assert len(TASKS) == len(TASK_LIST)
    cats = Counter(t.category for t in TASK_LIST)
    for required in ("combat", "boss", "puzzle", "npc", "gui", "icl"):
        assert cats[required] >= 3, f"{required} has only {cats[required]} tasks"
    for t in TASK_LIST:
        assert t.instruction and t.success_criteria
        assert t.max_seconds > 0
    assert task_by_id("combat_defeat_and_chest").category == "combat"


def test_icl_tasks_carry_a_decomposition() -> None:
    # The in-context-learning category is defined by *having* a step-by-step decomposition.
    for t in TASK_LIST:
        if t.category == "icl":
            assert len(t.hints) >= 2, f"{t.task_id} has no decomposition"
        else:
            assert not t.hints, f"{t.task_id} should not carry hints"


def test_mission_task_is_out_of_region() -> None:
    liyue = task_by_id("mission_liyue_act1")
    assert liyue.region == "liyue", "the Liyue mission must run in the held-out region"


# --------------------------------------------------------------------------------------


def test_reset_returns_a_well_formed_observation() -> None:
    env = _env()
    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    assert isinstance(obs, Observation)
    assert isinstance(obs.hud, HUDState)
    assert obs.frame.image.shape == (180, 320, 3)
    assert obs.frame.image.dtype == np.uint8
    assert obs.t == 0.0
    assert obs.frame_is_new
    assert obs.hud.alive
    assert obs.hud.max_hp > 0
    assert obs.hud.region == "mondstadt"
    assert env.task is not None and env.task.task_id == "combat_defeat_and_chest"
    assert env.task.instruction and env.task.category == "combat"
    env.close()


def test_step_advances_time_and_paces_rendering() -> None:
    """Rendering is the expensive half; it must run at the perception rate, not the tick rate."""
    env = _env()
    env.reset(task="combat_defeat_and_chest", seed=0)
    fresh = 0
    ticks = 300                       # 10 s of game time at 30 Hz
    for _ in range(ticks):
        obs = env.step(Action())
        fresh += int(obs.frame_is_new)
    assert abs(obs.t - ticks / 30.0) < 1e-6
    # 10 s at 5 Hz is 50 renders. Allow slack for the first-step forced render and rounding.
    assert 40 <= fresh <= 60, f"{fresh} renders in {ticks} ticks; expected ~50"
    env.close()


def test_stepping_does_not_sleep() -> None:
    """Data collection runs the world thousands of ticks per second; only realtime=True sleeps."""
    import time

    env = _env()
    env.reset(task="combat_defeat_and_chest", seed=0)
    t0 = time.perf_counter()
    for _ in range(300):
        env.step(Action())
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"300 ticks took {elapsed:.2f}s; the world is sleeping"
    env.close()


def test_observation_info_contract() -> None:
    env = _env()
    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    for key in ("success", "flags", "entity_boxes", "region", "task_id"):
        assert key in obs.info, f"Observation.info is missing {key!r}"
    assert obs.info["task_id"] == "combat_defeat_and_chest"
    assert isinstance(obs.info["flags"], dict)
    env.close()


def test_entity_boxes_are_populated_and_well_formed() -> None:
    """The controller steers by these boxes. Without them the agent is blind."""
    env = _env()
    env.reset(task="combat_defeat_and_chest", seed=0)
    boxes = env.renderer.entity_boxes
    assert boxes, "a fresh render produced no entity boxes"
    kinds = {b["kind"] for b in boxes}
    assert kinds, "boxes carry no kind"
    for b in boxes[:20]:
        assert set(b) >= {"name", "kind", "bbox", "distance", "screen_center", "visible"}
        x0, y0, x1, y1 = b["bbox"]
        assert x1 >= x0 and y1 >= y0
        assert b["distance"] >= 0.0
    env.close()


def test_frame_is_not_flat_and_looks_like_a_scene() -> None:
    """Guard against the worst renderer failure mode: geometry drawn in black.

    The entity builders originally took a colour argument and discarded it while the caller
    supplied ``None`` as the fill, so every tree, rock and enemy rendered as a solid black
    silhouette. It looked like a scene, it passed a "not flat" check, and a vision-language
    model read it as an empty room. Hence the explicit black-pixel budget below.
    """
    env = _env()
    env.reset(task="combat_defeat_and_chest", seed=0)
    for _ in range(60):
        env.step(Action(keys={"W"}))
    img = env.render_frame()
    total = img.shape[0] * img.shape[1]

    colours = len(np.unique(img.reshape(-1, 3), axis=0))
    assert colours > 150, f"only {colours} unique colours; the renderer produced a flat frame"

    black = int((img.astype(np.int16).sum(axis=2) < 30).sum())
    assert black < 0.05 * total, f"{black / total:.1%} of the frame is black; geometry lost its colour"

    blue = int(((img[:, :, 2].astype(int) - img[:, :, 0]) > 25).sum())
    assert blue > 500, f"only {blue} sky/water pixels; the camera may be inside the terrain"
    env.close()


def test_hud_is_readable_by_a_vision_model() -> None:
    """The HUD carries the quest text a VLM is expected to read off the screen."""
    env = _env()
    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    assert obs.hud.objective, "the objective line is empty"
    assert obs.hud.quest, "the quest title is empty"
    assert obs.hud.party and len(obs.hud.party) == len(PARTY)
    assert obs.hud.element in {e for _n, e in PARTY}
    env.close()


# --------------------------------------------------------------------------------------


def test_determinism_same_seed_same_trajectory() -> None:
    """(seed, region, task, action sequence) must fully determine an episode.

    Recorded behaviour-cloning data is worthless if this does not hold.
    """
    def rollout() -> tuple[float, float, float]:
        env = _env()
        env.reset(task="combat_defeat_and_chest", seed=7)
        for i in range(200):
            env.step(Action(keys={"W"} if i % 3 else {"W", "J"}))
        pos = env.world.player.position
        hp = env.world.player.hp
        env.close()
        return pos[0], pos[2], hp

    a, b = rollout(), rollout()
    assert a == b, f"same seed produced different trajectories: {a} vs {b}"


def test_different_seeds_produce_different_worlds() -> None:
    env = _env()
    env.reset(task="combat_defeat_and_chest", seed=1)
    first = [(e.kind, round(e.x, 3), round(e.z, 3)) for e in env.world.entities]
    env.reset(task="combat_defeat_and_chest", seed=2)
    second = [(e.kind, round(e.x, 3), round(e.z, 3)) for e in env.world.entities]
    assert first != second, "the seed has no effect on world generation"
    env.close()


def test_liyue_is_a_different_world() -> None:
    env = _env()
    env.reset(task="mission_mondstadt_act1", region="mondstadt", seed=3)
    mon = [(e.kind, round(e.x, 2), round(e.z, 2)) for e in env.world.entities]
    mon_npcs = {e.name for e in env.world.entities if e.kind == "npc"}
    env.reset(task="mission_liyue_act1", region="liyue", seed=3)
    liy = [(e.kind, round(e.x, 2), round(e.z, 2)) for e in env.world.entities]
    liy_npcs = {e.name for e in env.world.entities if e.kind == "npc"}
    assert mon != liy, "the held-out region is the same world with a different name"
    assert mon_npcs != liy_npcs, "the regions share their NPC cast; Liyue is not a real hold-out"
    env.close()


# --------------------------------------------------------------------------------------


def test_every_task_can_reset_and_run() -> None:
    """No task may crash the sandbox. Broken tasks would silently read as agent failures."""
    env = _env()
    failures = []
    for spec in TASK_LIST:
        try:
            env.reset(task=spec, seed=0)
            for _ in range(90):
                env.step(Action(keys={"W"}))
            env.success()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{spec.task_id}: {type(exc).__name__}: {exc}")
    env.close()
    assert not failures, "tasks crashed the sandbox:\n  " + "\n  ".join(failures)


def test_success_is_judged_from_flags() -> None:
    """Task scoring must read the world's progress flags, not the renderer or a timer."""
    flags = {"chests_opened": 1, "enemies_defeated": 3}
    task = task_by_id("combat_defeat_and_chest")
    assert task_success(task, flags) in (True, False)
    empty = dict.fromkeys(flags, 0)
    assert not task_success(task, empty), "an untouched world must not score as success"


def test_reward_is_finite_and_monotonic_in_progress() -> None:
    env = _env()
    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    r0 = obs.reward
    env.world.flags["chests_opened"] = 1
    obs2 = env.step(Action())
    assert np.isfinite(obs2.reward)
    assert obs2.reward > r0, "opening a chest did not increase the shaped reward"
    env.close()


# --------------------------------------------------------------------------------------


def test_renderer_projects_the_world_consistently() -> None:
    world = World(config=SimConfig(), task=task_by_id("combat_defeat_and_chest"), seed=0)
    r = Renderer(width=320, height=180)
    r.render(world)
    # The player is behind the camera, so a point just in front of the player must project
    # near the middle of the frame and be closer than a point far behind.
    p = world.player
    import math

    fx, fz = math.sin(p.yaw), math.cos(p.yaw)
    near = r.project((p.x + fx * 2.0, p.y + 1.6, p.z + fz * 2.0))
    behind = r.project((p.x - fx * 20.0, p.y + 1.6, p.z - fz * 20.0))
    assert near is not None
    assert 0 <= near[0] <= 320
    assert behind is None or behind[2] > near[2]


def test_render_throughput_is_usable() -> None:
    """Rendering happens 5x per second; at 200 ms a frame that would be a third of the CPU."""
    import time

    world = World(config=SimConfig(), task=task_by_id("combat_defeat_and_chest"), seed=0)
    r = Renderer(width=640, height=360)
    r.render(world)
    t0 = time.perf_counter()
    for _ in range(5):
        r.render(world)
    per_frame = (time.perf_counter() - t0) / 5.0
    assert per_frame < 0.25, f"{per_frame * 1000:.0f} ms per 640x360 frame is too slow"


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
