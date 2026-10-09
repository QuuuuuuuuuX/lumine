"""Evaluation: the task suite, and an honest report of what the agent actually did.

The suite mirrors the categories Lumine reports on -- combat, boss, puzzle, NPC
interaction, GUI manipulation and in-context learning -- and, critically, it evaluates on a
region the agent has never played.  Lumine's Liyue result is the interesting one because it
is out of distribution; a number that only covers the training region measures memorisation.

Every report states, per episode: success, wall-clock seconds, how many frames the agent
saw, how many times it called the model, and what fraction of wall-clock time it spent
waiting on that model.  The last number is the one that determines whether the architecture
is viable at all: an agent that spends 80% of its life blocked on a network call is not a
30 Hz agent no matter what the config says.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .agent import Agent, AgentConfig
from .config import EvalConfig, LumineConfig
from .types import EpisodeResult, TaskSpec


@dataclass
class EvalReport:
    started: str
    provider: str
    model: str
    controller: str
    observation_mode: str
    episodes: list[dict[str, Any]] = field(default_factory=list)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_region: dict[str, dict[str, float]] = field(default_factory=dict)
    overall: dict[str, float] = field(default_factory=dict)
    brain_stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"episodes": 0, "success_rate": 0.0}
    succ = [1.0 if r["success"] else 0.0 for r in rows]
    return {
        "episodes": float(len(rows)),
        "successes": float(sum(succ)),
        "success_rate": round(statistics.fmean(succ), 4),
        "mean_seconds": round(statistics.fmean(r["seconds"] for r in rows), 2),
        "mean_vlm_calls": round(statistics.fmean(r["vlm_calls"] for r in rows), 1),
        "mean_vlm_seconds": round(statistics.fmean(r["vlm_seconds"] for r in rows), 2),
        # Faults are aggregated because a *systematic* controller fault is a bug, whereas a
        # single one is the guard working as intended. The count alone cannot tell them
        # apart; the per-kind breakdown in the report is what does.
        "total_faults": float(sum(r.get("faults", 0) for r in rows)),
        "episodes_with_faults": float(sum(1 for r in rows if r.get("faults", 0))),
    }


def evaluate(
    cfg: LumineConfig,
    tasks: Iterable[TaskSpec],
    *,
    regions: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = (0,),
    agent_cfg: AgentConfig | None = None,
    env_factory: Callable[..., Any] | None = None,
    verbose: bool = True,
) -> EvalReport:
    """Run the suite and return a full report."""
    env_factory = env_factory or _default_env_factory
    tasks = list(tasks)
    regions = regions or (cfg.sim.region,)

    agent_cfg = agent_cfg or AgentConfig(verbose=verbose, stop_on_success=True)
    rows: list[dict[str, Any]] = []
    t_start = time.monotonic()

    total = len(tasks) * len(seeds) * len(regions)
    done = 0

    for region in regions:
        env = env_factory(region=region, seed=seeds[0])
        agent = Agent(cfg=cfg, env=env, agent_cfg=agent_cfg)
        try:
            for task in tasks:
                for seed in seeds:
                    done += 1
                    label = f"[{done}/{total}] {region}/{task.task_id} seed={seed}"
                    if verbose:
                        print(f"\n=== {label} :: {task.instruction}")
                    result: EpisodeResult = agent.run_episode(task, seed=seed, region=region)
                    in_dist = region == cfg.sim.region
                    rows.append({
                        "brain": agent.brain.name,
                        "task_id": task.task_id,
                        "category": task.category,
                        "region": region,
                        "in_distribution": in_dist,
                        "seed": seed,
                        "success": result.success,
                        "reason": result.reason,
                        "seconds": round(result.seconds, 2),
                        "frames": result.frames,
                        "vlm_calls": result.vlm_calls,
                        "vlm_seconds": round(result.vlm_seconds, 2),
                        "faults": result.faults,
                        "faults_by_kind": result.faults_by_kind,
                        "fault_examples": result.fault_examples,
                        "controller": result.controller,
                        "transcript": result.transcript,
                    })
                    if verbose:
                        mark = "PASS" if result.success else "FAIL"
                        print(f"    -> {mark} ({result.reason}) in {result.seconds:.1f}s, "
                              f"{result.vlm_calls} model calls")
        finally:
            agent.close()

    by_cat: dict[str, dict[str, float]] = {}
    for cat in sorted({r["category"] for r in rows}):
        by_cat[cat] = _aggregate([r for r in rows if r["category"] == cat])

    by_region: dict[str, dict[str, float]] = {}
    for reg in sorted({r["region"] for r in rows}):
        by_region[reg] = _aggregate([r for r in rows if r["region"] == reg])

    report = EvalReport(
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        provider=cfg.brain.provider,
        model=cfg.brain.model,
        # The brain that actually ran: with provider "none", or a fallback, these differ.

        controller=cfg.control.controller,
        observation_mode=cfg.observation_mode,
        episodes=rows,
        by_category=by_cat,
        by_region=by_region,
        overall=_aggregate(rows),
        brain_stats={},
        seconds=round(time.monotonic() - t_start, 2),
    )
    return report


def _default_env_factory(region: str, seed: int):  # noqa: ANN202
    from .sim import SimEnv

    return SimEnv()


def write_report(report: EvalReport, cfg: EvalConfig) -> tuple[Path, Path]:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jpath = Path(cfg.report_json)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))

    mpath = Path(cfg.report_md)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(render_markdown(report))
    return jpath, mpath


def render_markdown(report: EvalReport) -> str:
    o = report.overall
    lines = [
        "# Lumine agent evaluation",
        "",
        f"- started: {report.started}",
        f"- brain: `{report.episodes[0].get('brain', report.provider) if report.episodes else report.provider}`"
        f" (configured: `{report.provider}` / `{report.model}`)",
        f"- controller: `{report.controller}` ({report.observation_mode} observation)",
        f"- episodes: {int(o.get('episodes', 0))}",
        f"- **overall success: {o.get('success_rate', 0) * 100:.1f}%**",
        f"- wall clock: {report.seconds:.0f}s",
        "",
        "## By category",
        "",
        "| category | episodes | success | mean s | model calls | model seconds | faults |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat, m in report.by_category.items():
        lines.append(f"| {cat} | {int(m['episodes'])} | {m['success_rate'] * 100:.0f}% | "
                     f"{m['mean_seconds']:.1f} | {m['mean_vlm_calls']:.1f} | "
                     f"{m['mean_vlm_seconds']:.1f} | {int(m.get('total_faults', 0))} |")

    lines += ["", "## By region (generalisation)", "",
              "| region | episodes | success | note |", "|---|---|---|---|"]
    for reg, m in report.by_region.items():
        note = "in distribution" if reg == "mondstadt" else "**held out / out of distribution**"
        lines.append(f"| {reg} | {int(m['episodes'])} | {m['success_rate'] * 100:.0f}% | {note} |")

    lines += ["", "## Episodes", "",
              "| task | category | region | seed | result | reason | s | calls | faults |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in report.episodes:
        lines.append(f"| `{r['task_id']}` | {r['category']} | {r['region']} | {r['seed']} | "
                     f"{'PASS' if r['success'] else 'FAIL'} | {r['reason']} | "
                     f"{r['seconds']:.1f} | {r['vlm_calls']} | {r.get('faults', 0)} |")

    if report.overall.get("total_faults"):
        lines += ["", "## Controller faults", "",
                  "A fault is a malformed model parameter the controller absorbed instead of "
                  "crashing on: one is the guard working as intended, the same one over and "
                  "over is a bug in the skill's parameter handling.", ""]
        kinds: dict[str, int] = {}
        examples: dict[str, str] = {}
        for r in report.episodes:
            for kind, n in (r.get("faults_by_kind") or {}).items():
                kinds[kind] = kinds.get(kind, 0) + n
                if kind not in examples:
                    ex = (r.get("fault_examples") or {}).get(kind, "")
                    if ex:
                        examples[kind] = ex.replace("|", "\\|")[:110]
        lines += ["| skill : exception | occurrences | example |", "|---|---|---|"]
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{kind}` | {n} | {examples.get(kind, '')} |")
        lines += ["",
                  f"{int(report.overall['total_faults'])} faults across "
                  f"{int(report.overall.get('episodes_with_faults', 0))} of "
                  f"{int(report.overall.get('episodes', 0))} episodes.", ""]

    lines += ["", "## Reading this report", "",
              "The `model seconds` column is wall-clock time the agent spent blocked on (or",
              "concurrently waiting for) its vision-language model. It, not the success rate,",
              "is the number that decides whether the architecture is deployable at 30 Hz:",
              "the controller keeps emitting keys throughout, which is exactly why the brain",
              "is a planner over skills rather than a source of keystrokes.", ""]
    return "\n".join(lines)
