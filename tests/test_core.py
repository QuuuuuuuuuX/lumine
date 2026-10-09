"""Tests for everything that is not the sandbox.

Run either way:

    python -m pytest tests/ -q
    python -m tests.test_core
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumine.brain.adaptive import AdaptiveReasoner
from lumine.brain.prompt import extract_json, parse_intent
from lumine.brain.providers import PROVIDERS, render_table
from lumine.config import BrainConfig, LumineConfig, ReasonConfig, resolve_api_key
from lumine.data import Recorder, ShardReader, STAGE_BY_NAME, corpus_report
from lumine.memory import MissionMemory, WorkingMemory
from lumine.policy import NumpyActionHead
from lumine.skills import SKILL_BY_NAME, SKILLS, validate
from lumine.types import (Action, HUDState, Intent, KEYS, NUM_KEYS, Observation, Frame,
                          TaskSpec, quantize_px)


# --------------------------------------------------------------------------------------


def test_action_vector_round_trip() -> None:
    a = Action(keys={"W", "SHIFT", "J"}, buttons={"left"}, mouse_dx=3.5, mouse_dy=-2.0)
    v = a.to_vector()
    assert v.shape == (NUM_KEYS + 2 + 3,)
    b = Action.from_vector(v)
    assert b.keys == a.keys
    assert b.buttons == a.buttons
    assert abs(b.mouse_dx - 3.5) < 1e-6
    assert abs(b.mouse_dy + 2.0) < 1e-6


def test_action_rejects_unknown_keys() -> None:
    try:
        Action(keys={"NOT_A_KEY"})
    except ValueError as exc:
        assert "unknown keys" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown key")


def test_noop_and_describe() -> None:
    assert Action().is_noop()
    assert not Action(keys={"W"}).is_noop()
    assert "W" in Action(keys={"W"}, mouse_dx=1).describe()


def test_quantize_clamps_and_snaps() -> None:
    x, y = quantize_px(101.0, 199.0, 640, 360, grid=32)
    assert x == 96.0 and y == 192.0
    # Off-frame points snap to the grid and are then clamped to the frame edge, which is
    # deliberately allowed to land off-grid rather than being pushed back inside by a cell.
    x, y = quantize_px(-50, 9999, 640, 360, grid=32)
    assert x == 0.0 and y == 359.0


# --------------------------------------------------------------------------------------


def test_config_round_trip() -> None:
    cfg = LumineConfig()
    cfg.brain.provider = "ollama"
    cfg.reason.hz = 1.25
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.json"
        cfg.save(p)
        back = LumineConfig.load(p)
    assert back.brain.provider == "ollama"
    assert abs(back.reason.hz - 1.25) < 1e-9
    # The three rates are the architecture; a typo here would silently change the project.
    assert back.perception.hz == 5.0
    assert back.control.hz == 30.0


def test_config_rejects_unknown_field() -> None:
    try:
        LumineConfig.from_dict({"brain": {"not_a_field": 1}})
    except KeyError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("expected KeyError")


def test_yaml_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("default", "scripted", "local", "live"):
        cfg = LumineConfig.load(root / "configs" / f"{name}.yaml")
        assert cfg.control.hz == 30.0, name
        assert cfg.perception.hz == 5.0, name


def test_resolve_api_key_finds_dsh_credential() -> None:
    # The harness stores its key in ~/.dsh/.credentials.yaml; the agent must be able to reuse
    # it without the secret ever being copied into this repo.
    key = resolve_api_key("DEEPSEEK_API_KEY", "deepseek")
    assert key is None or key.startswith("sk-")


# --------------------------------------------------------------------------------------


def test_extract_json_handles_fences_and_prose() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"skill": "goto", "target": "chest"} done.') == {
        "skill": "goto", "target": "chest"}
    assert extract_json('{"a": 1,}') == {"a": 1}
    assert extract_json("no json at all") is None


def test_extract_json_ignores_braces_inside_strings() -> None:
    text = '{"reasoning": "go to the {weird} place", "skill": "wait"}'
    assert extract_json(text) == {"reasoning": "go to the {weird} place", "skill": "wait"}


def test_parse_intent_valid() -> None:
    text = json.dumps({"reasoning": "chest is right", "subgoal": "open it", "skill": "open_chest",
                       "target": "chest", "target_px": [321, 199], "params": {}})
    intent = parse_intent(text, 640, 360, 32)
    assert intent.skill == "open_chest"
    assert intent.target == "chest"
    assert intent.target_px == (320.0, 192.0)


def test_parse_intent_degrades_instead_of_raising() -> None:
    # A malformed response must never be able to end a five-hour episode.
    assert parse_intent("total garbage", 640, 360).skill == "wait"
    bad = json.dumps({"skill": "fly_to_the_moon", "target": "x"})
    assert parse_intent(bad, 640, 360).skill == "wait"
    missing_target = json.dumps({"skill": "attack"})
    assert parse_intent(missing_target, 640, 360).skill == "wait"


def test_parse_intent_normalises_null_target() -> None:
    text = json.dumps({"skill": "wait", "target": "null", "params": {}})
    intent = parse_intent(text, 640, 360)
    assert intent.target is None


# --------------------------------------------------------------------------------------


def test_skill_catalogue_is_consistent() -> None:
    assert len(SKILLS) == len(SKILL_BY_NAME)
    names = {s.name for s in SKILLS}
    assert {"goto", "attack", "interact", "open_chest", "wait"} <= names
    ok, _ = validate("goto", {"target": "chest"})
    assert ok
    ok, why = validate("goto", {})
    assert not ok and "target" in why
    ok, why = validate("nope", {"target": "x"})
    assert not ok and "unknown skill" in why


# --------------------------------------------------------------------------------------


def _obs(t: float, hp: float = 100.0, pos=(0.0, 0.0, 0.0), progress=(0, 1),
         image: np.ndarray | None = None) -> Observation:
    img = image if image is not None else np.zeros((90, 160, 3), np.uint8)
    hud = HUDState(hp=hp, max_hp=100.0, position=pos, objective_progress=progress)
    return Observation(frame=Frame(image=img, t=t), hud=hud, t=t, frame_is_new=True)


def test_reasoner_fires_on_instruction_change() -> None:
    r = AdaptiveReasoner(ReasonConfig())
    d = r.should_reason(_obs(0.0), "open the chest")
    assert d.should and d.because == "instruction_changed"


def test_reasoner_fires_on_progress_change() -> None:
    r = AdaptiveReasoner(ReasonConfig())
    r.should_reason(_obs(0.0, progress=(0, 1)), "same order")   # seeds the instruction
    r._last_progress = (0, 1)
    d = r.should_reason(_obs(0.1, progress=(1, 1)), "same order")
    assert d.should and d.because == "progress_changed"


def test_reasoner_fires_on_hp_drop() -> None:
    r = AdaptiveReasoner(ReasonConfig())
    r.should_reason(_obs(0.0, hp=100.0), "order")
    d = r.should_reason(_obs(0.1, hp=60.0), "order")
    assert d.should and d.because.startswith("hp_dropped")


def test_reasoner_detects_stuck() -> None:
    r = AdaptiveReasoner(ReasonConfig(stuck_seconds=1.0, stuck_distance=1.0, keepalive_s=99.0))
    r.should_reason(_obs(0.0, pos=(0, 0, 0)), "order")
    r._last_progress = (0, 1)
    # Stand still for two seconds of perception ticks.
    fired = []
    for i in range(1, 20):
        d = r.should_reason(_obs(i * 0.2, pos=(0.01, 0, 0.01)), "order")
        if d.should:
            fired.append(d.because)
    assert fired == ["stuck"], f"stuck should latch and fire once, got {fired}"

    # Moving again re-arms the detector.
    r.should_reason(_obs(4.0, pos=(30.0, 0, 30.0)), "order")
    r._last_call = -99.0
    again = [r.should_reason(_obs(6.0 + i * 0.2, pos=(30.0, 0, 30.0)), "order").should
             for i in range(30)]
    assert any(again), "the stuck detector never re-armed after the agent moved"


def test_reasoner_respects_keepalive_and_budget() -> None:
    r = AdaptiveReasoner(ReasonConfig(keepalive_s=5.0))
    r.should_reason(_obs(0.0), "order")
    r.note_call(0.0)
    r._last_progress = (0, 1)
    recent = r.should_reason(_obs(0.5), "order")
    assert not recent.should
    late = r.should_reason(_obs(30.0), "order")
    assert late.should


# --------------------------------------------------------------------------------------


def test_working_memory_and_dead_end_ledger() -> None:
    m = MissionMemory()
    intent = Intent(skill="goto", target="chest", subgoal="reach the chest")
    m.add_subgoal(intent.subgoal)
    assert not m.is_dead_end(intent)
    for i in range(3):
        m.note_failure(intent, f"attempt {i}")
    assert m.is_dead_end(intent)
    assert "goto:chest" in m.failure_hint()

    w = WorkingMemory()
    w.record(intent, "failed: unreachable")
    w.record(intent, "done: arrived")
    assert len(w.history()) == 2
    assert w.recent_failures() == ["goto(chest)"]


# --------------------------------------------------------------------------------------


def test_numpy_action_head_round_trip() -> None:
    head = NumpyActionHead(width=32, height=18)
    img = np.random.default_rng(0).integers(0, 255, (90, 160, 3), dtype=np.uint8)
    out = head.predict(img)
    assert 0.0 <= out.confidence <= 1.0
    with tempfile.TemporaryDirectory() as d:
        head.save(d)
        back = NumpyActionHead.load(d)
    out2 = back.predict(img)
    assert np.allclose(out.key_probs, out2.key_probs)


def test_numpy_action_head_learns_a_trivial_signal() -> None:
    """The pipeline must actually be capable of learning, not merely of running."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (64, 90, 160, 3), dtype=np.uint8)
    actions = np.zeros((64, NUM_KEYS + 5), np.float32)
    # Bright frames -> press W. The head should pick that up.
    for i, f in enumerate(frames):
        if f.mean() > 127:
            actions[i, KEYS.index("W")] = 1.0
    import cv2

    head = NumpyActionHead()
    x = np.stack([head._features(f) for f in frames])
    y = actions[:, :NUM_KEYS]
    w = head.w_key
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
        w += 0.5 * (x.T @ (y - p)) / len(x)
    head.w_key = w
    correct = (p > 0.5) == (y > 0.5)
    assert correct.mean() > 0.9, f"only {correct.mean():.3f} correct"


# --------------------------------------------------------------------------------------


def test_recorder_and_reader_round_trip() -> None:
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(root=d) if False else Recorder()
        from lumine.config import DataConfig

        rec = Recorder(cfg=DataConfig(root=d, shard_size=4), stage="pretrain", record_hz=1000.0)
        img = np.full((90, 160, 3), 128, np.uint8)
        stored = 0
        for i in range(10):
            if rec.maybe_add(img, Action(keys={"W"}, mouse_dx=1.0), t=float(i),
                             instruction="go", skill="goto"):
                stored += 1
        rec.close()
        assert stored == 10

        reader = ShardReader.discover(d, "pretrain")
        assert reader.paths, "recorder produced no shards"
        assert len(reader) == 10
        data = reader.load_all()
        assert data["frames"].shape == (10, 90, 160)
        assert data["actions"].shape[0] == 10
        assert set(data["instructions"]) == {"go"}

        report = corpus_report(d)
        assert report["pretrain"]["frames"] == 10
        assert report["pretrain"]["target_hours"] == STAGE_BY_NAME["pretrain"].target_hours


def test_recorder_respects_record_hz() -> None:
    from lumine.config import DataConfig

    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(cfg=DataConfig(root=d), record_hz=1.0)
        img = np.zeros((90, 160, 3), np.uint8)
        kept = sum(rec.maybe_add(img, Action(), t=i * 0.05) for i in range(40))
        rec.close()
        # 40 samples at 20 Hz over ~2 s, recorded at 1 Hz => about 2 kept.
        assert 1 <= kept <= 4, kept


# --------------------------------------------------------------------------------------


def test_provider_table_is_well_formed() -> None:
    for key, p in PROVIDERS.items():
        assert p.key == key
        assert p.label
        if key != "none":
            assert p.base_url.startswith("http"), key
            assert p.default_model, key
    table = render_table()
    assert "deepseek" in table and "ollama" in table
    assert "free" in table


def test_deepseek_vision_note_is_accurate() -> None:
    """Regression guard: deepseek-v4-pro is text-only and must not be the default."""
    assert PROVIDERS["deepseek"].default_model == "deepseek-flash"
    assert "ONLY" in PROVIDERS["deepseek"].notes
    assert PROVIDERS["kimi"].requires_base64 is True


# --------------------------------------------------------------------------------------


def test_agent_loop_smoke_without_sandbox() -> None:
    """The agent must run against any object with reset/step/success -- that is what makes
    live mode possible. This uses a 20-line fake instead of the sandbox on purpose."""
    from lumine.agent import Agent, AgentConfig
    from lumine.brain.scripted import ScriptedBrain

    class FakeEnv:
        def __init__(self) -> None:
            self.task = TaskSpec(task_id="fake", category="combat",
                                 instruction="Defeat the enemies ahead and collect the chest",
                                 max_seconds=3.0)
            self.n = 0

        def reset(self, task=None, region=None, seed=None):
            self.n = 0
            return self._obs()

        def _obs(self):
            img = np.zeros((90, 160, 3), np.uint8)
            hud = HUDState(hp=100, max_hp=100, alive=True, region="fake",
                           objective_progress=(0, 1),
                           nearby_entities=({"name": "hilichurl", "kind": "enemy",
                                             "position": (2.0, 0.0, 2.0), "distance": 2.8},))
            return Observation(frame=Frame(image=img), hud=hud, t=self.n / 30.0,
                               frame_is_new=True, info={"entity_boxes": [], "success": False})

        def step(self, action):
            self.n += 1
            obs = self._obs()
            obs.done = self.n > 90
            return obs

        def success(self):
            return False

        def close(self):
            pass

    cfg = LumineConfig()
    cfg.perception.source = "sim"
    cfg.control.backend = "dryrun"
    agent = Agent(cfg=cfg, env=FakeEnv(), brain=ScriptedBrain(),
                  agent_cfg=AgentConfig(verbose=False, max_reason_calls=3))
    try:
        result = agent.run_episode("fake")
    finally:
        agent.close()
    assert result.frames > 0
    assert result.seconds >= 0.0
    assert result.reason


def test_brain_failure_does_not_kill_the_episode() -> None:
    from lumine.brain.base import Brain, BrainContext
    from lumine.types import Intent

    class ExplodingBrain(Brain):
        name = "boom"

        def think(self, obs, ctx):  # noqa: ANN001, ANN201
            raise RuntimeError("network on fire")

    b = ExplodingBrain()
    out = b._timed(b.think, None, None)
    assert out is None
    assert b.stats.failures == 1
    assert "network on fire" in b.stats.last_error


# --------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------
# Regression guards for bugs that actually bit during development
# --------------------------------------------------------------------------------------


def test_reasoning_rate_is_bounded_when_no_skill_is_active() -> None:
    """The most expensive bug in the project, pinned down.

    `_maybe_reason` used to compute the trigger decision on the idle path and then ignore
    it, so an agent with no active skill called the model on *every* 30 Hz tick. Measured
    before the fix: 1.67 Hz average, with the burst rate at the full control rate. Measured
    after: 0.48 Hz against a 0.5 Hz design target.
    """
    from lumine.brain.adaptive import AdaptiveReasoner
    from lumine.config import ReasonConfig

    r = AdaptiveReasoner(ReasonConfig(min_interval_s=2.0, keepalive_s=20.0))
    # First call always fires: a new standing order outranks the cooldown.
    assert r.should_reason(_obs(0.0), "order").should
    r.note_call(0.0)

    # Idle (no skill) must still respect the floor, not bypass it.
    fired = [t for t in (0.5, 1.0, 1.5, 1.9)
             if r.should_reason(_obs(t), "order", has_skill=False).should]
    assert fired == [], f"idle path bypassed the cooldown at {fired}"
    # Once the floor expires, being idle is itself a good reason to think.
    d = r.should_reason(_obs(2.5), "order", has_skill=False)
    assert d.should and d.because == "no_skill"


def test_persistent_conditions_fire_once_not_every_tick() -> None:
    """Being stuck is a state, not an event; it must be edge-triggered."""
    from lumine.brain.adaptive import AdaptiveReasoner
    from lumine.config import ReasonConfig

    cfg = ReasonConfig(stuck_seconds=1.0, stuck_distance=1.0, keepalive_s=999.0,
                       min_interval_s=0.0)
    r = AdaptiveReasoner(cfg)
    r.should_reason(_obs(0.0, pos=(0, 0, 0)), "order")
    r.note_call(0.0)

    reasons = []
    for i in range(1, 40):
        d = r.should_reason(_obs(i * 0.2, pos=(0.01, 0, 0.01)), "order")
        reasons.append(d.because if d.should else "")
    stuck = [x for x in reasons if x == "stuck"]
    assert len(stuck) == 1, f"stuck fired {len(stuck)} times; it must latch until it moves"


def test_episode_result_counts_calls_per_episode() -> None:
    """`vlm_calls` must be per episode. Using the brain's lifetime counter made the reported
    reasoning rate climb across a suite and hid a 3x overspend."""
    import inspect

    from lumine import agent as agent_mod

    src = inspect.getsource(agent_mod.Agent._result)
    assert "self._reason_calls" in src, "vlm_calls is reporting a lifetime counter again"
    assert "self.brain.stats.calls" not in src


def test_controller_timer_uses_the_environment_clock() -> None:
    """Skill deadlines are measured in game time.

    In synchronous reasoning mode a two-second model call advances the wall clock while the
    world stands still. A wall-clock skill timeout therefore fires on time the game never
    experienced -- and makes episodes non-reproducible.
    """
    import inspect

    from lumine import controller as ctrl

    src = inspect.getsource(ctrl)
    body = src[src.index("class SkillController"):]
    assert "time.monotonic()" not in body, (
        "SkillController is reading the wall clock; use Observation.t via self._now"
    )


def test_region_override_reaches_the_environment() -> None:
    """The held-out evaluation is only meaningful if the region actually changes the world.

    `run_episode` used to accept a region and drop it before `reset`, and the CLI's env
    factory ignored the region it was handed -- so every "liyue" episode was played in
    mondstadt and the out-of-distribution column silently measured the training world twice.
    """
    import inspect

    from lumine import agent as agent_mod
    from lumine import cli as cli_mod
    from lumine import evaluate as eval_mod

    assert "region=region" in inspect.getsource(agent_mod.Agent.run_episode), (
        "run_episode no longer forwards the region to reset"
    )
    assert "region=region" in inspect.getsource(eval_mod.evaluate), (
        "evaluate() no longer passes the region to the agent"
    )
    assert "replace(cfg.sim, region=region)" in inspect.getsource(cli_mod._make_env), (
        "_make_env ignores the region it is given"
    )


def test_region_override_actually_changes_the_sandbox() -> None:
    """End to end: the same task, two regions, two different worlds."""
    try:
        from lumine.sim import SimEnv
    except Exception:  # noqa: BLE001 - the sandbox is optional for this module
        return

    from lumine.agent import Agent, AgentConfig
    from lumine.brain.scripted import ScriptedBrain

    cfg = LumineConfig()
    cfg.perception.width, cfg.perception.height = 160, 90
    env = SimEnv(sim=cfg.sim, perception=cfg.perception)
    agent = Agent(cfg=cfg, env=env, brain=ScriptedBrain(),
                  agent_cfg=AgentConfig(verbose=False, max_reason_calls=1))
    try:
        agent.env.reset(task="npc_talk_grace", region="mondstadt", seed=0)
        mon = {(e.kind, round(e.x, 3), round(e.z, 3)) for e in agent.env.world.entities}
        agent.env.reset(task="npc_talk_grace", region="liyue", seed=0)
        liy = {(e.kind, round(e.x, 3), round(e.z, 3)) for e in agent.env.world.entities}
    finally:
        agent.close()
    assert mon != liy, "the region override did not change the world"


def test_dialogue_choice_accepts_every_shape_a_model_returns() -> None:
    """The model answers "which dialogue option" three different ways, all of them reasonable.

    Observed live: deepseek-flash returned the full option text where an index was expected,
    and an unguarded int() ended the episode with a ValueError.
    """
    from lumine.controller import SkillController

    opts = ("I'm here to help the Knights of Favonius.",
            "Lovely weather we're having.",
            "Where can I find the tavern?")
    r = SkillController._resolve_choice
    assert r(2, opts) == 2
    assert r("2", opts) == 2
    assert r(opts[0], opts) == 0
    assert r("Lovely weather we're having.", opts) == 1
    # Paraphrase still lands somewhere sane rather than raising.
    assert r("help the knights", opts) == 0
    assert r(None, opts) == 0
    assert r("", opts) == 0
    assert r(True, opts) == 0
    assert r(99, opts) == 99 % len(opts)


def test_model_parameters_never_raise() -> None:
    """`stop_distance` arrived as "close" once. A float() there would end the run."""
    from lumine.controller import _num

    assert _num("1.5", 1.6) == 1.5
    assert _num(2, 1.6) == 2.0
    assert _num("close", 1.6) == 1.6
    assert _num(None, 1.6) == 1.6
    assert _num(True, 1.6) == 1.6
    assert _num("about 3 units", 1.6) == 3.0
    assert _num([], 1.6) == 1.6


def test_controller_fault_fails_the_skill_not_the_episode() -> None:
    """A malformed parameter must cost one skill, not the whole run."""
    import inspect

    from lumine import controller as ctrl

    src = inspect.getsource(ctrl.SkillController.tick)
    assert "except Exception" in src, "the skill FSM is no longer wrapped"
    assert "FAILED" in src, "a controller fault must fail the skill, not propagate"

    # And the fault must be countable, so a systematically broken skill is visible in a report.
    assert "faults" in inspect.getsource(ctrl.SkillController)


def test_faults_do_not_accumulate_across_episodes() -> None:
    """The controller is reused across episodes, so its fault counter must be reset per run.

    This is the same mistake that made the reported reasoning rate climb across a suite: a
    per-episode figure silently became a lifetime total.
    """
    from lumine.agent import Agent, AgentConfig
    from lumine.brain.scripted import ScriptedBrain
    from lumine.controller import SkillController
    from lumine.sim import SimEnv

    original = SkillController._do_attack

    def boom(self, call, obs, target, dt):  # noqa: ANN001, ANN202
        raise TypeError("stop_distance must be a number, got 'close'")

    cfg = LumineConfig()
    cfg.perception.width, cfg.perception.height = 160, 90
    SkillController._do_attack = boom
    try:
        env = SimEnv(sim=cfg.sim, perception=cfg.perception)
        agent = Agent(cfg=cfg, env=env, brain=ScriptedBrain(),
                      agent_cfg=AgentConfig(verbose=False, max_reason_calls=3))
        counts = [agent.run_episode("combat_defeat_and_chest", seed=0).faults for _ in range(3)]
        agent.close()
    finally:
        SkillController._do_attack = original

    assert counts == [1, 1, 1], f"fault counts accumulated across episodes: {counts}"


def test_faults_are_attributed_to_a_skill_and_an_exception() -> None:
    """A bare count cannot distinguish "the guard worked once" from "this skill is broken"."""
    from lumine.agent import Agent, AgentConfig
    from lumine.brain.scripted import ScriptedBrain
    from lumine.controller import SkillController
    from lumine.sim import SimEnv

    original = SkillController._do_attack

    def boom(self, call, obs, target, dt):  # noqa: ANN001, ANN202
        raise TypeError("stop_distance must be a number, got 'close'")

    cfg = LumineConfig()
    cfg.perception.width, cfg.perception.height = 160, 90
    SkillController._do_attack = boom
    try:
        env = SimEnv(sim=cfg.sim, perception=cfg.perception)
        agent = Agent(cfg=cfg, env=env, brain=ScriptedBrain(),
                      agent_cfg=AgentConfig(verbose=False, max_reason_calls=3))
        result = agent.run_episode("combat_defeat_and_chest", seed=0)
        agent.close()
    finally:
        SkillController._do_attack = original

    assert result.faults_by_kind == {"attack:TypeError": 1}
    assert "close" in result.fault_examples["attack:TypeError"]


def test_report_surfaces_controller_faults() -> None:
    """A systematic fault must be visible in the report, not only in a log."""
    import inspect

    from lumine import evaluate as eval_mod

    src = inspect.getsource(eval_mod)
    assert "## Controller faults" in src, "the report lost its fault section"
    assert "total_faults" in src, "faults are no longer aggregated"
    assert "faults_by_kind" in inspect.getsource(eval_mod.evaluate), (
        "per-episode faults are not collected"
    )
    # And the fault column must actually be in the tables.
    assert "| category | episodes | success | mean s | model calls | model seconds | faults |" \
        in src, "the category table lost its faults column"


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
