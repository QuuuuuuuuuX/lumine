"""The scripted brain: a competent heuristic that needs no model, no key and no network.

Two jobs:

1. **Fallback.**  When the VLM is unreachable, or the user has no key at all, the agent
   still has to walk, fight, open chests and finish tasks.  Without this the demo dies on
   a plane's wifi.
2. **Data generator.**  Lumine's stage-0 corpus is 1731 hours of *human* gameplay.  We have
   no humans, so a behaviour-cloned controller needs a behaviour to clone.  This brain
   drives a scripted policy that plays the sandbox well enough to produce honest
   ``(frame, action)`` pairs for the action head to imitate.

It reads the structured HUD, which the VLM deliberately does not get.  That asymmetry is
intentional: the scripted brain is a *teacher*, and teachers are allowed privileged state.
"""

from __future__ import annotations

import re
import time

from ..types import Intent, Observation
from .base import Brain, BrainContext

#: Instruction keyword -> skill.  Ordered; first match wins.
_ROUTES: tuple[tuple[str, str], ...] = (
    (r"\btalk to\b|\bspeak (?:to|with)\b", "interact"),
    (r"\bdefeat\b|\bkill\b|\bcomplete the domain\b|\bdaily commission\b", "attack"),
    (r"\bteleport\b", "open_map"),
    (r"\bcook\b|\bsweet madame\b", "open_cook"),
    (r"\bweapon\b", "open_weapon"),
    (r"\bmonument\b", "use_skill"),
    (r"\bboulder\b|\bbreak\b", "attack"),
    (r"\bthorn", "use_skill"),
    (r"\bwind current\b|\banemograna\b|\bwind barrier\b", "collect_anemograna"),
    (r"\banemoculus\b", "collect"),
    (r"\bchest\b", "open_chest"),
    (r"\bclimb\b|\bstone pillar\b", "climb"),
)

_NPC_RE = re.compile(r"\b(?:npc|talk to|speak to)\s+([A-Z][a-zA-Z']+)", re.IGNORECASE)


class ScriptedBrain(Brain):
    """A finite-state heuristic over the structured HUD."""

    name = "scripted"

    def __init__(self) -> None:
        super().__init__()
        self._mode = ""
        self._mode_since = 0.0

    def think(self, obs: Observation, ctx: BrainContext) -> Intent:
        t0 = time.monotonic()
        try:
            intent = self._decide(obs, ctx)
        finally:
            self.stats.calls += 1
            self.stats.total_seconds += time.monotonic() - t0
            self.stats.last_seconds = time.monotonic() - t0
        return intent

    # ---------------------------------------------------------------------------------

    def _decide(self, obs: Observation, ctx: BrainContext) -> Intent:
        h = obs.hud
        now = obs.t

        # --- modal states first: a menu or dialogue swallows everything ------------------
        if h.dialogue_open:
            choice = self._best_dialogue_option(h.dialogue_options, ctx.instruction)
            return Intent(
                skill="interact",
                target=h.dialogue_speaker or "npc",
                params={"dialogue_choice": choice, "target": h.dialogue_speaker or "npc"},
                subgoal=f"pick dialogue option {choice}",
                reasoning="Dialogue is open; advance it with the option that matches the order.",
            )

        if h.menu:
            entries = self._menu_entries(h)
            want = self._wanted_menu_entry(ctx.instruction, h.menu)
            return Intent(
                skill="menu_select",
                target=want,
                params={"entry": want, "menu": h.menu},
                subgoal=f"select {want!r} in the {h.menu} menu",
                reasoning=f"The {h.menu} menu is open and the order implies {want!r}.",
            )

        if not h.alive:
            return Intent(skill="wait", params={"seconds": 1.0}, reasoning="Player is down.")

        # --- what does the order actually want? ------------------------------------------
        route = self._route(ctx.instruction)

        # --- explicit interaction prompt in range ---------------------------------------
        #
        # An interaction prompt only pre-empts the order when it *serves* that order. Without
        # this check the teacher stood next to an NPC forever: talking is always available, so
        # "talk to Grace" beat "defeat the enemies" on every single replan, and the agent
        # never advanced the actual objective.
        if h.interact_prompt and self._prompt_serves(h.interact_prompt, route, h, ctx.instruction):
            low = h.interact_prompt.lower()
            # Prefer the entity the game itself names in the prompt over the nearest one.
            target = self._target_from_prompt(h, ctx.instruction) or self._pick_target(h, ctx.instruction)
            if "talk" in low or "dialogue" in low:
                return Intent(skill="interact", target=target,
                              params={"target": target},
                              subgoal="talk to the NPC in range",
                              reasoning=f"Prompt '{h.interact_prompt}' means an NPC is in range.")
            if "chest" in low:
                return Intent(skill="open_chest", target=target, params={"target": target},
                              subgoal="open the chest in range",
                              reasoning=f"Prompt '{h.interact_prompt}' means a chest is in range.")
            return Intent(skill="interact", target=target, params={"target": target},
                          subgoal="use what is in range",
                          reasoning=f"Prompt '{h.interact_prompt}' is available now.")

        # --- read the order ---------------------------------------------------------------
        target = self._pick_target(h, ctx.instruction)

        if route == "attack":
            foe = self._nearest(h, kinds=("enemy", "boss"))
            if foe is None:
                return Intent(skill="goto", target="chest",
                              params={"target": "chest", "stop_distance": 2.0},
                              subgoal="find remaining enemies",
                              reasoning="No hostile in range; sweep toward the objective marker.")
            return Intent(
                skill="attack",
                target=foe["name"],
                params={"target": foe["name"]},
                subgoal=f"kill {foe['name']}",
                reasoning=f"{foe['name']} is {foe['distance']:.1f} units away and the order is to fight.",
            )

        if route == "use_skill":
            elem = self._element_for(ctx.instruction, h)
            return Intent(skill="use_skill", target=elem, params={"element": elem},
                          subgoal=f"apply {elem} to the obstacle",
                          reasoning=f"The instruction needs the {elem} element.")

        if route in ("open_map", "open_cook", "open_weapon"):
            menu = {"open_map": "map", "open_cook": "cook", "open_weapon": "weapon"}[route]
            return Intent(skill="open_menu", target=menu, params={"menu": menu},
                          subgoal=f"open the {menu} menu",
                          reasoning=f"The order requires the {menu} interface.")

        if route == "collect_anemograna":
            mote = self._nearest(h, kinds=("anemograna",))
            if mote:
                return Intent(skill="collect", target=mote["name"], params={"target": mote["name"]},
                              subgoal="collect the anemograna",
                              reasoning="An anemograna is in range; collecting it starts a wind current.")
            wind = self._nearest(h, kinds=("wind_current",))
            if wind:
                return Intent(skill="goto", target=wind["name"], params={"target": wind["name"]},
                              subgoal="ride the wind current",
                              reasoning="A wind current is active; move into it to gain height.")
            return Intent(skill="goto", target="anemograna",
                          params={"target": "anemograna", "stop_distance": 1.2},
                          subgoal="find an anemograna",
                          reasoning="No anemograna in range yet; search for one.")

        if route in ("open_chest", "collect"):
            chest = self._nearest(h, kinds=("chest",))
            if chest and chest.get("locked"):
                guard = self._nearest(h, kinds=("enemy", "boss"))
                if guard:
                    return Intent(skill="attack", target=guard["name"], params={"target": guard["name"]},
                                  subgoal=f"clear the guard on the chest",
                                  reasoning="The chest is locked; its guard must die first.")
            name = (chest or {}).get("name", "chest")
            return Intent(skill="open_chest", target=name, params={"target": name},
                          subgoal="open the chest",
                          reasoning="The objective is a chest; go and open it.")

        if route == "climb":
            return Intent(skill="climb", target="stone_pillar",
                          params={"target": "stone_pillar"},
                          subgoal="climb the pillar",
                          reasoning="The order names a vertical obstacle.")

        # --- default: advance toward whatever the order names -----------------------------
        if target:
            return Intent(skill="goto", target=target, params={"target": target, "stop_distance": 2.0},
                          subgoal=f"reach {target}",
                          reasoning=f"Nothing urgent; close the distance to {target}.")
        return Intent(skill="wait", params={"seconds": 0.5},
                      reasoning="Nothing actionable is visible yet; hold position briefly.")

    # -- helpers ---------------------------------------------------------------------------

    def _route(self, instruction: str) -> str:
        low = instruction.lower()
        for pat, route in _ROUTES:
            if re.search(pat, low):
                return route
        return ""

    def _pick_target(self, hud, instruction: str) -> str | None:
        ents = list(hud.nearby_entities)
        if not ents:
            return None
        # An NPC named in the instruction beats a generic nearest-entity choice.
        m = _NPC_RE.search(instruction)
        if m:
            want = m.group(1).lower()
            for e in ents:
                if want in str(e.get("name", "")).lower():
                    return str(e["name"])
        for kind in ("npc", "chest", "enemy", "boss", "monument", "waypoint", "cooking_pot",
                     "anemoculus", "anemograna", "wind_current", "boulder", "thorn_bush"):
            if kind in instruction.lower().replace(" ", "_"):
                best = self._nearest(hud, kinds=(kind,))
                if best:
                    return best["name"]
        return str(min(ents, key=lambda e: e.get("distance", 1e9)).get("name"))

    @staticmethod
    def _prompt_serves(prompt: str, route: str, hud, instruction: str) -> bool:  # noqa: ANN001
        """Does the on-screen ``[F]`` prompt advance the standing order?

        ``route`` is what the instruction asked for; the prompt is what happens to be within
        arm's reach.  Conflating the two is how an agent ends up having the same conversation
        for two hours while a chest sits unopened behind it.
        """
        low = prompt.lower()
        is_npc = "talk" in low or "dialogue" in low
        is_chest = "chest" in low
        is_monument = "monument" in low
        is_waypoint = "waypoint" in low or "map" in low
        is_cook = "cook" in low

        if route == "interact":
            return is_npc
        if route in ("open_chest", "collect"):
            return is_chest
        if route == "use_skill":
            return is_monument
        if route == "open_map":
            return is_waypoint
        if route == "open_cook":
            return is_cook
        if route == "attack":
            # Only fall back to a prompt once there is nothing left to fight.
            hostiles = [e for e in hud.nearby_entities if e.get("kind") in ("enemy", "boss")]
            return not hostiles and (is_chest or is_npc)
        # No explicit route: take whatever is offered.
        return True

    @staticmethod
    def _target_from_prompt(hud, instruction: str) -> str | None:
        """Recover the entity name the game put in its own ``[F] ...`` prompt."""
        prompt = hud.interact_prompt or ""
        # Prompts read like "[F] Talk to Sayid", "[F] Open Chest 1", "[F] Use Monument".
        tail = re.sub(r"^\[F\]\s*", "", prompt).strip()
        for verb in ("Talk to ", "Open ", "Use ", "Activate ", "Cook ", "Teleport to "):
            if tail.startswith(verb):
                name = tail[len(verb):].strip()
                for e in hud.nearby_entities:
                    if str(e.get("name", "")).lower() == name.lower():
                        return str(e["name"])
                return name or None
        for e in hud.nearby_entities:
            if str(e.get("name", "")).lower() in tail.lower():
                return str(e["name"])
        return None

    @staticmethod
    def _nearest(hud, kinds: tuple[str, ...]) -> dict | None:
        cands = [e for e in hud.nearby_entities if e.get("kind") in kinds]
        if not cands:
            return None
        return dict(min(cands, key=lambda e: e.get("distance", 1e9)))

    @staticmethod
    def _element_for(instruction: str, hud) -> str:
        low = instruction.lower()
        for elem in ("pyro", "cryo", "electro", "hydro", "geo", "dendro", "anemo"):
            if elem in low:
                return elem
        # Fall back to whatever element is not currently active.
        for elem in ("pyro", "cryo", "electro", "anemo"):
            if elem != hud.element:
                return elem
        return "anemo"

    @staticmethod
    def _best_dialogue_option(options: tuple[str, ...], instruction: str) -> int:
        if not options:
            return 0
        words = set(re.findall(r"[a-z]{4,}", instruction.lower()))
        best, best_score = 0, -1
        for i, opt in enumerate(options):
            score = len(words & set(re.findall(r"[a-z]{4,}", opt.lower())))
            if score > best_score:
                best, best_score = i, score
        return best

    @staticmethod
    def _menu_entries(hud) -> list[str]:
        return []

    def _wanted_menu_entry(self, instruction: str, menu: str) -> str:
        low = instruction.lower()
        if menu == "cook":
            return "Sweet Madame"
        if menu == "map":
            return "Teleport"
        if menu == "weapon":
            for w in ("sword", "claymore", "bow", "catalyst", "polearm"):
                if w in low:
                    return w.title()
            return "Sword"
        return "Confirm"
