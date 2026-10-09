"""Command line interface.

    python -m lumine providers            what brains can I plug in?
    python -m lumine probe                does this key actually see images?
    python -m lumine tasks                what can the agent be asked to do?
    python -m lumine play --task ...      watch it play, optionally to an mp4
    python -m lumine collect --minutes 10 gather behaviour-cloning data
    python -m lumine train --stage pretrain
    python -m lumine eval --seeds 0 1     the task suite, with an out-of-distribution region
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from .agent import Agent, AgentConfig
from .brain.factory import describe_providers
from .brain.vlm import VLMBrain
from .config import LumineConfig
from .data import CURRICULUM, ShardReader, corpus_report, format_corpus_report
from .evaluate import evaluate, write_report
from .policy import load_action_head
from .types import Action


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _load_config(args: argparse.Namespace) -> LumineConfig:
    cfg = LumineConfig.load(getattr(args, "config", None))
    for attr, path in (("provider", ("brain", "provider")), ("model", ("brain", "model")),
                       ("base_url", ("brain", "base_url")), ("api_key_env", ("brain", "api_key_env")),
                       ("region", ("sim", "region")), ("observation", ("observation_mode",)),
                       ("controller", ("control", "controller"))):
        val = getattr(args, attr, None)
        if val is None:
            continue
        if len(path) == 1:
            setattr(cfg, path[0], val)
        else:
            setattr(getattr(cfg, path[0]), path[1], val)
    if getattr(args, "backend", None):
        cfg.control.backend = args.backend
    if getattr(args, "perception_source", None):
        cfg.perception.source = args.perception_source
    if getattr(args, "checkpoint", None):
        cfg.control.checkpoint = args.checkpoint
    if getattr(args, "reasoning_mode", None):
        cfg.__dict__["_reasoning_mode"] = args.reasoning_mode
    return cfg


def _make_env(cfg: LumineConfig, region: str, seed: int = 0):  # noqa: ANN202
    """Build the environment: the sandbox, or a real game when perception points at one."""
    if cfg.perception.source in ("screen", "video"):
        from .live import LiveEnv

        return LiveEnv(cfg)
    from .sim import SimEnv

    # The region override is what makes the held-out evaluation real; without it every
    # 'liyue' episode was played in mondstadt.
    sim = replace(cfg.sim, region=region)
    return SimEnv(sim=sim, perception=cfg.perception)


def _task_list(args: argparse.Namespace):  # noqa: ANN202
    from .sim import TASK_LIST, task_by_id

    if getattr(args, "task", None):
        return [task_by_id(args.task)]
    cats = getattr(args, "category", None)
    tasks = list(TASK_LIST)
    if cats:
        tasks = [t for t in tasks if t.category in cats]
    per = getattr(args, "per_category", None)
    if per:
        # A balanced-but-fast suite: N tasks per category rather than all 23, so an evaluation
        # finishes in minutes instead of an hour without silently over-weighting one category.
        kept, seen = [], {}
        for t in tasks:
            seen[t.category] = seen.get(t.category, 0) + 1
            if seen[t.category] <= per:
                kept.append(t)
        tasks = kept
    return tasks


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def cmd_providers(args: argparse.Namespace) -> int:
    print(describe_providers())
    print()
    print("Full registry:")
    print()
    from .brain.providers import render_table

    print(render_table())
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    print(f"probing {cfg.brain.provider} / {cfg.brain.model} ...")
    brain = VLMBrain(cfg.brain, cfg.perception.width, cfg.perception.height)
    try:
        models = brain.list_models()
        print(f"  models visible to this key: {', '.join(models) if models else '(none listed)'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not list models: {type(exc).__name__}: {exc}")
    ok, text = brain.probe_vision()
    print(f"  vision: {'WORKS' if ok else 'FAILED'}")
    print(f"  model said: {text!r}")
    return 0 if ok else 1


def cmd_tasks(args: argparse.Namespace) -> int:
    from .sim import TASK_LIST

    by_cat: dict[str, list] = {}
    for t in TASK_LIST:
        by_cat.setdefault(t.category, []).append(t)
    for cat in sorted(by_cat):
        print(f"\n{cat.upper()}")
        for t in by_cat[cat]:
            print(f"  {t.task_id:28s} [{t.region:10s}] {t.instruction}")
            if t.hints:
                for h in t.hints:
                    print(f"      - {h}")
    print(f"\n{len(TASK_LIST)} tasks in {len(by_cat)} categories.")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    task_id = args.task
    mode = cfg.__dict__.pop("_reasoning_mode", None) or args.reasoning_mode

    env = _make_env(cfg, cfg.sim.region, args.seed)
    agent_cfg = AgentConfig(reasoning_mode=mode, verbose=not args.quiet,
                            record=args.record, record_stage=args.stage,
                            record_dir=cfg.data.root,
                            capture_frames=bool(args.video))
    agent = Agent(cfg=cfg, env=env, agent_cfg=agent_cfg)
    if cfg.control.controller == "learned" and cfg.control.checkpoint:
        agent.controller.policy = load_action_head(cfg.control.checkpoint)

    try:
        print(f"task       : {task_id}")
        print(f"brain      : {agent.brain.name}")
        print(f"controller : {cfg.control.controller}")
        print(f"reasoning  : {mode}")
        print()

        result = agent.run_episode(task_id, seed=args.seed)
        if args.video:
            _save_video(args.video, agent.frames_captured)
        _print_result(result)
        return 0 if result.success else 2
    finally:
        agent.close()


def _save_video(path: str, frames: list) -> None:
    if not frames:
        return
    import cv2

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"video -> {path}")


def _print_result(result) -> None:  # noqa: ANN001
    print()
    print(f"success : {result.success}")
    print(f"reason  : {result.reason}")
    print(f"seconds : {result.seconds:.1f}")
    print(f"frames  : {result.frames}")
    print(f"model   : {result.vlm_calls} calls, {result.vlm_seconds:.1f}s")
    if result.faults:
        print(f"faults  : {result.faults} -> {result.faults_by_kind}")


def cmd_collect(args: argparse.Namespace) -> int:
    """Drive the sandbox with the scripted teacher and record (frame, action) pairs."""
    cfg = _load_config(args)
    cfg.brain.provider = "none"          # the teacher is the heuristic brain, not a model
    cfg.brain.fallback_to_scripted = True

    tasks = _task_list(args)
    deadline = time.monotonic() + args.minutes * 60.0
    env = _make_env(cfg, cfg.sim.region, args.seed)
    agent_cfg = AgentConfig(reasoning_mode="sync", verbose=args.verbose,
                            record=True, record_stage=args.stage,
                            record_dir=cfg.data.root, record_hz=args.record_hz,
                            stop_on_success=False)
    agent = Agent(cfg=cfg, env=env, agent_cfg=agent_cfg)

    episodes = 0
    try:
        while time.monotonic() < deadline:
            task = tasks[episodes % len(tasks)]
            print(f"\n--- episode {episodes + 1}: {task.task_id}")
            result = agent.run_episode(task, seed=args.seed + episodes)
            episodes += 1
            print(f"    {'PASS' if result.success else 'FAIL'} ({result.reason})")
    finally:
        agent.close()

    report = corpus_report(cfg.data.root)
    print()
    print(format_corpus_report(report))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    cfg.train.stage = args.stage
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.train.device = args.device
    from .train import train_action_head

    train_action_head(cfg.train, cfg.data)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    tasks = _task_list(args)
    regions = tuple(args.regions) if args.regions else (cfg.sim.region,)
    seeds = tuple(args.seeds)

    agent_cfg = AgentConfig(reasoning_mode=args.reasoning_mode, verbose=not args.quiet,
                            stop_on_success=True)
    report = evaluate(
        cfg, tasks, regions=regions, seeds=seeds, agent_cfg=agent_cfg,
        env_factory=lambda region, seed: _make_env(cfg, region, seed),
        verbose=not args.quiet,
    )
    jpath, mpath = write_report(report, cfg.evaluation)
    print()
    print(f"overall success: {report.overall.get('success_rate', 0) * 100:.1f}%")
    for reg, m in report.by_region.items():
        print(f"  {reg:12s} {m['success_rate'] * 100:5.1f}%  ({int(m['episodes'])} episodes)")
    print(f"report -> {mpath}")
    print(f"json   -> {jpath}")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    report = corpus_report(cfg.data.root)
    print(format_corpus_report(report))
    print()
    for stage in CURRICULUM:
        r = report[stage.name]
        print(f"{stage.name:9s} {stage.description}")
        if r["frames"]:
            reader = ShardReader.discover(cfg.data.root, stage.name)
            sample = reader.load_all(limit=1)
            if len(sample["frames"]):
                print(f"          sample frame {sample['frames'][0].shape}, "
                      f"action dim {sample['actions'][0].shape}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    env = _make_env(cfg, cfg.sim.region, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .sim import TASK_LIST

    import cv2

    n = 0
    for task in TASK_LIST[: args.count]:
        obs = env.reset(task=task, seed=args.seed)
        for _ in range(args.steps):
            obs = env.step(Action())
        img = env.render_frame()
        path = out / f"{task.task_id}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        n += 1
        print(f"  {task.task_id:28s} -> {path}")
    print(f"{n} previews in {out}")
    return 0


def cmd_rates(args: argparse.Namespace) -> int:
    """Print the three rates and check that they are actually being achieved."""
    cfg = _load_config(args)
    print("configured rates")
    print(f"  perception : {cfg.perception.hz:6.1f} Hz")
    print(f"  control    : {cfg.control.hz:6.1f} Hz")
    print(f"  reasoning  : {cfg.reason.hz:6.1f} Hz (ceiling; adaptive triggers gate it)")
    print()
    env = _make_env(cfg, cfg.sim.region, 0)
    obs = env.reset(task="combat_defeat_and_chest", seed=0)
    n = 900
    t0 = time.monotonic()
    fresh = 0
    for _ in range(n):
        obs = env.step(Action())
        fresh += int(obs.frame_is_new)
    wall = time.monotonic() - t0
    game = n / cfg.control.hz

    print("measured on the sandbox")
    print(f"  logical control : {n / wall:8.1f} steps/s wall  "
          f"({n} steps in {wall:.2f}s = {game:.1f}s of game time)")
    print(f"  speedup         : {game / wall:8.1f}x real time")
    print()
    print("  both rates, expressed per second of *game* time (what the architecture means):")
    print(f"    control    : {n / game:8.1f} Hz   target {cfg.control.hz:.1f}")
    print(f"    perception : {fresh / game:8.1f} Hz   target {cfg.perception.hz:.1f}")
    print()
    print(f"  render cost     : {float(obs.info.get('render_seconds', 0)) * 1000:8.1f} ms/frame "
          f"at {cfg.perception.width}x{cfg.perception.height}")
    print("  reasoning is not measured here -- it is gated by the adaptive triggers and by")
    print("  model latency, both of which need a live brain: see `lumine play`.")
    return 0


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lumine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="path to a JSON/YAML config overriding the defaults")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        # SUPPRESS, not None: the subparser shares the top-level namespace, so a plain
        # default would overwrite a --config given *before* the subcommand with None.
        sp.add_argument("--config", default=argparse.SUPPRESS,
                        help="path to a JSON/YAML config (may also precede the subcommand)")
        sp.add_argument("--provider", help="brain provider key (see 'providers')")
        sp.add_argument("--model", help="model id")
        sp.add_argument("--base-url", dest="base_url")
        sp.add_argument("--api-key-env", dest="api_key_env")
        sp.add_argument("--region", default=None, help="sandbox region: mondstadt | liyue")
        sp.add_argument("--observation", choices=["pixels", "pixels+hud"])
        sp.add_argument("--controller", choices=["rules", "learned", "hybrid"])
        sp.add_argument("--checkpoint", help="action-head checkpoint directory")
        sp.add_argument("--backend", choices=["sim", "uinput", "dryrun"])
        sp.add_argument("--perception-source", dest="perception_source",
                        choices=["sim", "screen", "video"])
        sp.add_argument("--reasoning-mode", dest="reasoning_mode", choices=["sync", "async"],
                        default="sync")
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("providers", help="list brain providers and how they compare")
    sp.set_defaults(func=cmd_providers)

    sp = sub.add_parser("probe", help="verify that a key/model pair really accepts images")
    common(sp)
    sp.set_defaults(func=cmd_probe)

    sp = sub.add_parser("tasks", help="list the evaluation task suite")
    sp.add_argument("--category", action="append")
    sp.set_defaults(func=cmd_tasks)

    sp = sub.add_parser("play", help="run one task")
    common(sp)
    sp.add_argument("--task", default="combat_defeat_and_chest")
    sp.add_argument("--quiet", action="store_true")
    sp.add_argument("--record", action="store_true", help="write behaviour-cloning data")
    sp.add_argument("--stage", default="pretrain")
    sp.add_argument("--video", help="write an mp4 of the episode")
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("collect", help="drive the scripted teacher and record data")
    common(sp)
    sp.add_argument("--task", default=None)
    sp.add_argument("--category", action="append")
    sp.add_argument("--minutes", type=float, default=5.0)
    sp.add_argument("--stage", default="pretrain", choices=[s.name for s in CURRICULUM])
    sp.add_argument("--record-hz", dest="record_hz", type=float, default=10.0)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("train", help="train the action head")
    common(sp)
    sp.add_argument("--stage", default="pretrain", choices=[s.name for s in CURRICULUM])
    sp.add_argument("--epochs", type=int)
    sp.add_argument("--device", choices=["auto", "cpu", "cuda"])
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("eval", help="run the evaluation suite")
    common(sp)
    sp.add_argument("--task", default=None)
    sp.add_argument("--category", action="append")
    sp.add_argument("--regions", nargs="*", default=None,
                    help="regions to evaluate on; include 'liyue' for the OOD test")
    sp.add_argument("--seeds", nargs="*", type=int, default=[0])
    sp.add_argument("--per-category", dest="per_category", type=int,
                    help="take only N tasks per category (keeps the suite balanced and fast)")
    sp.add_argument("--quiet", action="store_true")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("corpus", help="report on the collected data")
    common(sp)
    sp.set_defaults(func=cmd_corpus)

    sp = sub.add_parser("preview", help="render one frame per task to PNG")
    common(sp)
    sp.add_argument("--out", default="artifacts/sim_previews")
    sp.add_argument("--count", type=int, default=99)
    sp.add_argument("--steps", type=int, default=90)
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("rates", help="measure the sandbox against the 5/30/0.5 Hz design")
    common(sp)
    sp.set_defaults(func=cmd_rates)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
