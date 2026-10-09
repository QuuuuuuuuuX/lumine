"""The skill library: the interface between a slow language brain and fast control.

Why this file exists
--------------------
A VLM answers in ~2 seconds.  The game needs an answer in 33 milliseconds.  You cannot
reconcile those by making the VLM faster or the game slower, so the system is split in
two and this module is the seam:

    VLM   ->  Intent(skill="goto", target="chest", ...)      ~0.5 Hz, language
    here  ->  SkillSpec catalogue, exposed to the VLM as its tool list
    ctrl  ->  30 Hz keystrokes that actually execute the skill

That is the same decomposition Lumine trains: first master action primitives, then ground
those primitives in language.  The controller is the primitives; this catalogue is the
language binding.

Adding a skill is a two-line change here plus one branch in ``lumine/controller.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import SkillSpec

# --------------------------------------------------------------------------------------
# The catalogue handed to the VLM
# --------------------------------------------------------------------------------------

SKILLS: tuple[SkillSpec, ...] = (
    SkillSpec(
        "goto",
        "Walk toward a named entity or landmark (chest, npc, enemy, waypoint, monument, "
        "anemoculus, cooking_pot). The controller pathfinds and avoids water. Use for any "
        "travel that does not need a specific route.",
        params=("target", "stop_distance"),
        category="navigation",
    ),
    SkillSpec(
        "follow_route",
        "Walk to an explicit list of world-ish waypoints given as screen offsets, used only "
        "when 'goto' has no nameable target (e.g. following a wind current upward).",
        params=("waypoints",),
        category="navigation",
    ),
    SkillSpec(
        "attack",
        "Approach the nearest hostile within range and attack it until it dies. Automatically "
        "switches to an element the enemy is weak to when one is available.",
        params=("target",),
        category="combat",
    ),
    SkillSpec(
        "dodge",
        "Sprint out of the telegraphed AoE circle the agent is currently standing in. Call this "
        "immediately when a red ring appears under the player.",
        params=(),
        category="combat",
    ),
    SkillSpec(
        "use_skill",
        "Use the elemental skill (E). With element='pyro' it burns thorn bushes, 'cryo' freezes "
        "water into walkable ice, 'anemo' activates wind currents, 'electro' breaks electro "
        "shields. Switch the party member first if the wrong element is active.",
        params=("element",),
        category="combat",
    ),
    SkillSpec(
        "use_burst",
        "Use the elemental burst (Q) when the energy meter is full.",
        params=(),
        category="combat",
    ),
    SkillSpec(
        "interact",
        "Press F on the nearest interactable (chest, npc, waypoint, monument, cooking pot, "
        "time-trial trigger) and advance any dialogue by choosing the option most consistent "
        "with the current objective.",
        params=("target", "dialogue_choice"),
        category="interaction",
    ),
    SkillSpec(
        "open_chest",
        "Walk to a chest, deal with whatever guards it (thorns, boulder, enemies) and open it.",
        params=("target",),
        category="interaction",
    ),
    SkillSpec(
        "climb",
        "Jump and climb a vertical surface (stone pillar, cliff) to reach a higher ledge.",
        params=("target",),
        category="navigation",
    ),
    SkillSpec(
        "glide",
        "Deploy the glider to descend slowly and cross a gap or reach a floating collectible.",
        params=("target",),
        category="navigation",
    ),
    SkillSpec(
        "switch_character",
        "Switch to the party member whose element is needed (1-4). Kaeya is cryo, Amber is "
        "pyro, Lisa is electro, Lumine is anemo.",
        params=("element", "slot"),
        category="party",
    ),
    SkillSpec(
        "open_menu",
        "Open an in-game GUI panel: 'map', 'cook', 'weapon' or 'bag'.",
        params=("menu",),
        category="gui",
    ),
    SkillSpec(
        "menu_select",
        "Inside an open GUI panel, move the cursor to a named entry and confirm it. Use for "
        "'Teleport', 'Sweet Madame', or a weapon name.",
        params=("entry",),
        category="gui",
    ),
    SkillSpec(
        "collect",
        "Pick up a nearby collectible (anemoculus, anemograna, dropped loot).",
        params=("target",),
        category="interaction",
    ),
    SkillSpec(
        "wait",
        "Do nothing for a moment. Use only when genuinely blocked or when waiting for an "
        "animation, a lift, or a boss phase to finish.",
        params=("seconds",),
        category="general",
    ),
)

SKILL_BY_NAME: dict[str, SkillSpec] = {s.name: s for s in SKILLS}

#: Rendered into the system prompt.  Kept terse on purpose: every token here is paid for on
#: every one of the agent's ~0.5 Hz calls.
SKILL_MENU: str = "\n".join(
    f"- {s.name}({', '.join(s.params) if s.params else ''}): {s.description}" for s in SKILLS
)


def skill_names() -> list[str]:
    return [s.name for s in SKILLS]


def validate(skill: str, params: dict) -> tuple[bool, str]:
    """Check a VLM-produced skill call before it reaches the controller."""
    spec = SKILL_BY_NAME.get(skill)
    if spec is None:
        return False, f"unknown skill {skill!r}; valid: {', '.join(skill_names())}"
    missing = [p for p in spec.params if p not in params]
    # Only 'target' is treated as mandatory; the rest have sane controller defaults.
    if "target" in missing:
        return False, f"skill {skill!r} requires param 'target'"
    return True, ""


@dataclass(slots=True)
class SkillCall:
    """A validated, executable skill invocation."""

    skill: str
    params: dict
    issued_at: float
    deadline: float
    source: str = "vlm"          # "vlm" | "rules" | "icl"
    #: Where the brain said the target is, in frame pixels.  Used as the *fallback* steering
    #: source when the on-screen detector cannot name the target.  Horizontally this is
    #: reliable; vertically it is noisy, which is fine because steering is horizontal.
    target_px: tuple[float, float] | None = None

    @property
    def target(self) -> str | None:
        return self.params.get("target") or self.params.get("menu") or self.params.get("element")
