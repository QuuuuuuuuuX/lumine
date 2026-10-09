"""Prompt construction for the slow brain.

Two design rules, both learned the hard way:

1. **The output schema is stated once, as JSON, with a worked example.**  Vision models
   drift into prose if you let them; every extra sentence of prose is a parse failure.
2. **Coordinates are coarse on purpose.**  VLMs ground regions well and pixels badly, so we
   ask for a point on a 32 px grid and let the controller refine it with classical CV.
"""

from __future__ import annotations

import json

from ..skills import SKILL_MENU
from ..types import HUDState, Intent
from .base import BrainContext

SYSTEM = f"""You are Lumine, an agent that plays a 3D open-world action game from raw screen pixels.

You are the SLOW brain. You are called rarely (about twice per second at most), so spend
your call on deciding *what to do next*, never on low-level control. A separate 30 Hz
controller turns your decision into keystrokes; it already knows how to walk, aim, attack
and dodge. Your job is to pick the right skill and the right target.

# How to read the screen
- Bottom-left bars are your HP (red) and stamina (green). If HP is low, disengage.
- Top-left is the quest tracker: title, current objective, and a progress counter.
- Top-centre is the standing order you were given.
- Top-right is the minimap: red dots are enemies, yellow are chests, cyan are NPCs,
  blue are waypoints.
- A red ring on the ground is an incoming area attack. Call `dodge` immediately.
- `[F] ...` near the bottom centre means something is in interact range.

# Available skills
{SKILL_MENU}

# Output
Reply with ONE JSON object and nothing else. No markdown fence, no commentary, no trailing text.
{{
  "reasoning": "<one or two sentences: what you see and why this move>",
  "subgoal": "<short imperative phrase for your own future reference>",
  "skill": "<one skill name from the list above>",
  "target": "<entity or landmark name you can see, or null>",
  "target_px": [<x>, <y>],
  "params": {{}},
  "expected_done": false
}}

Rules:
- "skill" MUST be exactly one of the skill names listed above.
- "target_px" is the pixel of the thing you are acting on, in the image you were shown
  (origin top-left). Give your best estimate; it does not need to be exact.
- Set "expected_done" to true only when the standing order itself is complete.
- If you are mid-way through something that is working, keep the same skill and say so.
- Never repeat an action that the history shows just failed. Change approach instead.
"""

EXAMPLE = """Worked example.

Standing order: "Defeat the enemies ahead and collect the chest"
Screen: two hilichurls at centre-right, a chest with a gold outline further right, HP 78/100.

{"reasoning": "Two hilichurls block the path to the chest; the nearest is to the right of the crosshair and HP is healthy, so I should clear them first.", "subgoal": "kill the nearest hilichurl", "skill": "attack", "target": "hilichurl", "target_px": [384, 192], "params": {}, "expected_done": false}
"""


def build_messages(image_data_url: str, hud: HUDState, ctx: BrainContext) -> list[dict]:
    """Assemble the chat-completions messages.

    ``hud`` is only used when the caller explicitly opted into privileged state; in the
    faithful pixels-only configuration the caller passes a blanked ``HUDState``.
    """
    lines: list[str] = [f"Standing order: {ctx.instruction}"]

    if ctx.subgoal:
        lines.append(f"Your current subgoal: {ctx.subgoal}")
    if ctx.hints:
        lines.append("Given decomposition (follow it):")
        lines.extend(f"  {i + 1}. {h}" for i, h in enumerate(ctx.hints))
    lines.append(f"Region: {ctx.region or 'unknown'}")
    lines.append(f"Elapsed: {ctx.seconds_elapsed:.0f}s   Progress: {ctx.progress[0]}/{ctx.progress[1]}")

    if ctx.last_result:
        lines.append(f"Result of your last action: {ctx.last_result}")
    if ctx.retry:
        lines.append("NOTE: the plan just failed. You MUST choose a different approach.")

    if ctx.history:
        lines.append("Recent action history (oldest first):")
        lines.extend(f"  - {h}" for h in ctx.history[-8:])

    if hud.region:
        # Privileged block, only populated in --observation=pixels+hud mode.
        lines.append(
            "Structured state (privileged): "
            f"hp={hud.hp:.0f}/{hud.max_hp:.0f} stamina={hud.stamina:.0f} "
            f"pos=({hud.position[0]:.1f},{hud.position[1]:.1f},{hud.position[2]:.1f}) "
            f"menu={hud.menu} dialogue={hud.dialogue_open}"
        )

    lines.append("What is your next move? Reply with JSON only.")

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE},
        {"role": "user", "content": "Standing order: \"Defeat the enemies ahead and collect the chest\"\nAcknowledge with JSON only."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "\n".join(lines)},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]


# --------------------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model response.

    Handles the three failure modes seen in practice: a ```json fence, prose before the
    object, and trailing commas.
    """
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    import re

                    repaired = re.sub(r",\s*([}\]])", r"\1", blob)
                    repaired = repaired.replace("'", '"')
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        return None
    return None


def parse_intent(text: str, width: int, height: int, grid: int = 32) -> Intent:
    """Turn raw model text into a validated :class:`Intent`.

    A malformed response degrades to ``wait`` rather than raising: a single bad JSON blob
    must not be able to end a five-hour episode.
    """
    from ..skills import validate
    from ..types import quantize_px

    data = extract_json(text)
    if data is None:
        return Intent(skill="wait", reasoning="unparseable model response",
                      params={"seconds": 1.0}, raw=text[:500])

    skill = str(data.get("skill") or "wait").strip().lower()
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    target = data.get("target")
    if isinstance(target, str) and target.strip().lower() in ("", "null", "none"):
        target = None
    if target:
        params.setdefault("target", target)

    ok, why = validate(skill, params)
    if not ok:
        return Intent(skill="wait", reasoning=f"invalid skill call ({why})",
                      params={"seconds": 1.0}, raw=text[:500])

    px = data.get("target_px")
    target_px = None
    if isinstance(px, (list, tuple)) and len(px) == 2:
        try:
            target_px = quantize_px(float(px[0]), float(px[1]), width, height, grid)
        except (TypeError, ValueError):
            target_px = None

    return Intent(
        skill=skill,
        target=target,
        target_px=target_px,
        params=params,
        reasoning=str(data.get("reasoning", ""))[:600],
        subgoal=str(data.get("subgoal", ""))[:200],
        expected_done=bool(data.get("expected_done", False)),
        raw=text[:1000],
    )
