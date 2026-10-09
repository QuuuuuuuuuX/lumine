"""The task registry: every evaluable episode the sandbox can host.

The list below mirrors the categories of the Lumine report (arXiv 2511.08892) -- combat,
boss, puzzle, npc, gui and *in-context learning* (icl) -- plus the two storyline missions.
Each :class:`~lumine.types.TaskSpec` carries the natural-language instruction that is shown
in the HUD banner *and* a machine-readable ``success_criteria`` string, so the very same
episode can be scored with or without a model in the loop.

The criteria mini-language
--------------------------
``success_criteria`` is a conjunction (``and``) of clauses::

    enemies_defeated >= 3          numeric comparison against a flag
    chests_opened > 0              the same, with a strict operator
    npcs_talked:grace              membership -- ``grace`` in the flag's list value
    time_trial_cleared             bare flag, truthy test

Lists compare by length for the ordering operators, so ``npcs_talked >= 2`` means "at least
two NPCs were talked to".  :func:`criterion_met` evaluates a string against a flag dict and
:func:`criterion_progress` extracts a ``(current, target)`` pair for the HUD counter, which
is how the on-screen quest tracker is kept honest.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..types import TaskSpec

__all__ = [
    "TASK_LIST",
    "TASKS",
    "task_by_id",
    "criterion_met",
    "criterion_progress",
    "task_success",
    "objective_text",
    "TASK_IDS",
]


def _spec(task_id: str, category: str, instruction: str, criteria: str, **kw: Any) -> TaskSpec:
    """Build one :class:`TaskSpec` with an explicit success criterion."""
    return TaskSpec(
        task_id=task_id,
        category=category,
        instruction=instruction,
        success_criteria=criteria,
        **kw,
    )


# --------------------------------------------------------------------------------------
# Combat
# --------------------------------------------------------------------------------------

TASK_LIST: list[TaskSpec] = [
    _spec(
        "combat_defeat_and_chest",
        "combat",
        "Defeat the enemies ahead and collect the chest",
        "enemies_defeated >= 2 and chests_opened >= 1",
        max_seconds=120.0,
        seed=11,
    ),
    _spec(
        "combat_domain",
        "combat",
        "Complete the Domain",
        "enemies_defeated >= 4 and chests_opened >= 1",
        max_seconds=180.0,
        seed=12,
    ),
    _spec(
        "combat_daily",
        "combat",
        "Complete the Daily Commission: Defeat all enemies",
        "enemies_defeated >= 3",
        max_seconds=150.0,
        seed=13,
    ),
    # -- Boss ---------------------------------------------------------------------------
    _spec(
        "boss_hypostasis_electro",
        "boss",
        "Defeat the Electro Hypostasis",
        "boss_defeated:electro_hypostasis",
        max_seconds=240.0,
        seed=21,
    ),
    _spec(
        "boss_stormterror",
        "boss",
        "Defeat Stormterror",
        "boss_defeated:stormterror",
        max_seconds=240.0,
        seed=22,
    ),
    _spec(
        "boss_hypostasis_anemo",
        "boss",
        "Defeat the Anemo Hypostasis",
        "boss_defeated:anemo_hypostasis",
        max_seconds=240.0,
        seed=23,
    ),
    # -- Puzzle -------------------------------------------------------------------------
    _spec(
        "puzzle_anemoculus_wind",
        "puzzle",
        "Fly along the Wind Current to collect the Anemoculus",
        "anemoculus_collected >= 1 and wind_current_activated >= 1",
        max_seconds=150.0,
        seed=31,
    ),
    _spec(
        "puzzle_slime_chest",
        "puzzle",
        "After defeating the floating Anemo Slime, open the chest",
        "enemies_defeated >= 1 and chests_opened >= 1",
        max_seconds=120.0,
        seed=32,
    ),
    _spec(
        "puzzle_boulder_chest",
        "puzzle",
        "Break the boulder ahead and open the chest",
        "boulders_broken >= 1 and chests_opened >= 1",
        max_seconds=120.0,
        seed=33,
    ),
    _spec(
        "puzzle_thorn_chest",
        "puzzle",
        "Open the chest wrapped in thorns ahead",
        "thorns_burned >= 1 and chests_opened >= 1",
        max_seconds=120.0,
        seed=34,
    ),
    _spec(
        "puzzle_monument",
        "puzzle",
        "Activate the Elemental Monument using the corresponding element",
        "monuments_activated >= 1",
        max_seconds=120.0,
        seed=35,
    ),
    _spec(
        "puzzle_time_trial",
        "puzzle",
        "Complete the Time Trial Challenge ahead: Open the chest within the time limit",
        "time_trial_cleared",
        max_seconds=120.0,
        seed=36,
    ),
    # -- NPC ----------------------------------------------------------------------------
    _spec(
        "npc_talk_grace",
        "npc",
        "Talk to NPC Grace",
        "npcs_talked:grace",
        max_seconds=90.0,
        seed=41,
    ),
    _spec(
        "npc_talk_monroe",
        "npc",
        "Talk to NPC Monroe",
        "npcs_talked:monroe",
        max_seconds=90.0,
        seed=42,
    ),
    _spec(
        "npc_talk_sayid",
        "npc",
        "Talk to NPC Sayid",
        "npcs_talked:sayid",
        max_seconds=90.0,
        seed=43,
    ),
    # -- GUI ----------------------------------------------------------------------------
    _spec(
        "gui_cook_sweet_madame",
        "gui",
        "Cook Sweet Madame",
        "cooked_dishes:sweet_madame",
        max_seconds=90.0,
        seed=51,
    ),
    _spec(
        "gui_teleport_waypoint",
        "gui",
        "Teleport using a Teleport Waypoint",
        "teleported",
        max_seconds=90.0,
        seed=52,
    ),
    _spec(
        "gui_change_weapon",
        "gui",
        "Change the character's weapon",
        "weapon_changed",
        max_seconds=90.0,
        seed=53,
    ),
    # -- In-context learning (hints are handed to the agent) -----------------------------
    _spec(
        "icl_climb_pillar",
        "icl",
        "Climb the stone pillar on the right and, once you reach the top, collect the blue "
        "Anemoculus floating in the air on the left",
        "anemoculus_collected >= 1 and climbed",
        max_seconds=180.0,
        seed=61,
        hints=(
            "Walk to the tall stone pillar on the right side of the path.",
            "Face the pillar and hold W to climb it; keep climbing until you reach the top.",
            "Walk off the top toward the left and fall/glide through the blue Anemoculus "
            "floating in the air to collect it.",
        ),
    ),
    _spec(
        "icl_kaeya_freeze",
        "icl",
        "Switch to Kaeya, continuously use his Elemental Skill (E Skill) to freeze the water "
        "surface, and collect the Anemoculus floating ahead",
        "anemoculus_collected >= 1 and ice_created >= 1",
        max_seconds=180.0,
        seed=62,
        hints=(
            "Press 2 to switch to Kaeya (cryo).",
            "Walk to the water's edge and press E repeatedly to freeze the surface into "
            "walkable ice.",
            "Walk across the ice and touch the Anemoculus floating beyond the water.",
        ),
    ),
    _spec(
        "icl_wind_barrier",
        "icl",
        "Collect the Wind Anemograna to activate a Wind Current, then enter the Wind Barrier "
        "to open the chest",
        "anemograna_collected >= 1 and wind_barrier_broken >= 1 and chests_opened >= 1",
        max_seconds=180.0,
        seed=63,
        hints=(
            "Walk into the glowing Wind Anemograna ahead to collect it.",
            "Enter the Wind Current it activates and ride the updraft forward.",
            "Step into the Wind Barrier -- it dissipates while you carry Anemo energy -- then "
            "press F on the chest inside.",
        ),
    ),
    # -- Storyline missions -------------------------------------------------------------
    _spec(
        "mission_mondstadt_act1",
        "mission",
        "Complete Act I of the Mondstadt storyline",
        "quest_stage >= 3",
        region="mondstadt",
        max_seconds=300.0,
        seed=71,
    ),
    _spec(
        "mission_liyue_act1",
        "mission",
        "Complete Act I of the Liyue storyline",
        "quest_stage >= 3",
        region="liyue",
        max_seconds=300.0,
        seed=72,
    ),
]

#: Task id -> spec, the canonical lookup table used by the harness and the CLI.
TASKS: dict[str, TaskSpec] = {t.task_id: t for t in TASK_LIST}

#: Every legal task id, in registry order.
TASK_IDS: tuple[str, ...] = tuple(t.task_id for t in TASK_LIST)

#: Ordered categories; useful for grouping in reports and the preview demo.
CATEGORIES: tuple[str, ...] = ("combat", "boss", "puzzle", "npc", "gui", "icl", "mission")

#: Sub-objective chains for the two storyline missions: ``(hud text, criteria)``.
MISSION_CHAINS: dict[str, tuple[tuple[str, str], ...]] = {
    "mission_mondstadt_act1": (
        ("Talk to Grace at the Mondstadt gate", "npcs_talked:grace"),
        ("Defeat the hilichurls on the road", "enemies_defeated >= 3"),
        ("Open the reward chest", "chests_opened >= 1"),
    ),
    "mission_liyue_act1": (
        ("Talk to Bao'er at the harbor", "npcs_talked:bao'er"),
        ("Defeat the Treasure Hoarders", "enemies_defeated >= 3"),
        ("Open the chest on the terrace", "chests_opened >= 1"),
    ),
}


def task_by_id(task_id: str) -> TaskSpec:
    """Return the spec for ``task_id``.

    Raises:
        KeyError: when the id is not in the registry, with the full legal list attached so a
            failed rollout is self-diagnosing.
    """
    try:
        return TASKS[task_id]
    except KeyError as exc:  # pragma: no cover - defensive, exercised by tests
        raise KeyError(f"unknown task {task_id!r}; known ids: {', '.join(TASK_IDS)}") from exc


# --------------------------------------------------------------------------------------
# Success-criteria mini-language
# --------------------------------------------------------------------------------------

_CLAUSE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_']*)"
    r"(?:\s*(?P<op>>=|<=|>|<|==|=)\s*(?P<num>-?\d+(?:\.\d+)?)"
    r"|:(?P<member>[^\s]+))?\s*$"
)


def _split_clauses(criteria: str) -> list[str]:
    """Split a criteria string on ``and`` (and on commas), dropping empty fragments."""
    parts = re.split(r"\band\b|,", criteria or "")
    return [p for p in (s.strip() for s in parts) if p]


def _numeric(value: Any) -> float:
    """Coerce a flag value to a number: lists/tuples count their length, bools are 0/1."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple, set, dict, str)):
        return float(len(value))
    return 0.0


def _eval_clause(clause: str, flags: Mapping[str, Any]) -> bool:
    """Evaluate one clause of the criteria mini-language against ``flags``."""
    m = _CLAUSE_RE.match(clause)
    if m is None:
        raise ValueError(f"cannot parse success criterion clause {clause!r}")
    name = m.group("name")
    value = flags.get(name)
    num = m.group("num")
    if num is not None:
        lhs = _numeric(value)
        rhs = float(num)
        op = m.group("op")
        if op in (">=", "=>"):
            return lhs >= rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == "<":
            return lhs < rhs
        return lhs == rhs
    member = m.group("member")
    if member is not None:
        if value is None:
            return False
        if isinstance(value, (list, tuple, set)):
            return any(str(v).lower() == member.lower() for v in value)
        if isinstance(value, dict):
            return member.lower() in {str(k).lower() for k in value}
        return str(value).lower() == member.lower()
    return bool(value)


def criterion_met(criteria: str, flags: Mapping[str, Any]) -> bool:
    """True when every conjunct of ``criteria`` holds for ``flags``."""
    clauses = _split_clauses(criteria)
    if not clauses:
        return False
    return all(_eval_clause(c, flags) for c in clauses)


def criterion_progress(criteria: str, flags: Mapping[str, Any]) -> tuple[int, int]:
    """Return a ``(current, target)`` counter for the HUD quest tracker.

    The bottleneck conjunct wins: if the task needs two enemies *and* one chest and the
    player has killed two enemies but opened no chest, the tracker shows ``0/1``.
    """
    clauses = _split_clauses(criteria)
    if not clauses:
        return (0, 0)
    best: tuple[float, int, int] | None = None
    for c in clauses:
        m = _CLAUSE_RE.match(c)
        if m is None:
            continue
        value = flags.get(m.group("name"))
        if m.group("num") is not None:
            cur, target = _numeric(value), float(m.group("num"))
        elif m.group("member") is not None:
            cur, target = (1.0 if _eval_clause(c, flags) else 0.0), 1.0
        else:
            cur, target = (1.0 if value else 0.0), 1.0
        ratio = cur / target if target > 0 else 1.0
        cand = (ratio, int(round(cur)), int(round(target)))
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:  # pragma: no cover - unreachable with a non-empty clause list
        return (0, 0)
    return (best[1], best[2])


def task_success(task: TaskSpec | str | None, flags: Mapping[str, Any]) -> bool:
    """Judge a task from the world's flag dictionary."""
    if task is None:
        return False
    spec = task if isinstance(task, TaskSpec) else task_by_id(task)
    if not spec.success_criteria:
        return False
    return criterion_met(spec.success_criteria, flags)


def objective_text(criteria: str, flags: Mapping[str, Any]) -> str:
    """Human-readable objective line, e.g. ``Defeat the enemies 2/3``.

    Used for the quest-tracker panel so a VLM can read numeric progress off the screen.
    """
    clauses = _split_clauses(criteria)
    if not clauses:
        return "No objective"
    bits: list[str] = []
    for c in clauses:
        m = _CLAUSE_RE.match(c)
        if m is None:
            continue
        name = m.group("name")
        label = name.replace("_", " ").capitalize()
        if m.group("member") is not None:
            bits.append(f"{label}: {m.group('member')}")
        else:
            cur, target = criterion_progress(c, flags)
            bits.append(f"{label} {cur}/{target}")
    return "  |  ".join(bits)
