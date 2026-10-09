"""The 30 Hz controller: action primitives, and the seam where language becomes keystrokes.

One tick of this file is what Lumine's first training stage produces: given *where* to go and
*what* to do, emit the keys.  It is hand-written rather than learned so that the system is
useful before any training has happened, and so the learned action head has something
correct to imitate (and something safe to fall back to).

Layers, outermost first
-----------------------
``reflex``      unconditional survival: dodge telegraphed AoEs, disengage at low HP
``skill``       the current :class:`~lumine.skills.SkillCall`, executed as a small FSM
``policy``      optional learned action head; may override the skill layer when confident

Steering is visual servoing: the target's screen-space horizontal error drives relative
mouse movement, and ``W`` is held while the error is small.  That keeps the controller
honest for a real game, where there is no world position to path against -- only pixels.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .config import ControlConfig, PerceptionConfig
from .perception import EntityBox, boxes_from_info, detect_telegraph
from .skills import SkillCall
from .types import Action, Observation

RUNNING, DONE, FAILED = "running", "done", "failed"

#: Distance at which an enemy can be hit, in world units.  Mirrors the sandbox's reach.
ATTACK_RANGE = 2.6
#: Distance at which F works on a chest / NPC / waypoint.
INTERACT_RANGE = 1.9


@dataclass(slots=True)
class PixelTarget:
    """A target known only as a point the brain marked on the frame.

    This is the fallback when nothing on screen can be named: the VLM said "the thing is
    over there" and all we have is a pixel.  It is enough for steering, because steering is
    horizontal, and horizontal grounding is the axis VLMs are actually good at (measured
    here: x landed within one 32 px cell, y was off by three).  Distance is unknown, so a
    skill aiming at a PixelTarget finishes on arrival-at-centre plus a settle time rather
    than on a distance test.
    """

    x: float
    y: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _num(value: object, default: float) -> float:
    """Coerce a model-supplied parameter to a float, never raising.

    Parameters arrive from a language model and are therefore prose until proven otherwise:
    observed in the wild, ``stop_distance`` came back as ``"close"`` and ``dialogue_choice``
    as the full option text. A single ``int()`` on those ended a five-hour episode, which is
    precisely the failure mode the brain layer was designed to prevent -- the guard just
    stopped one layer too early.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


@dataclass
class ControllerState:
    status: str = RUNNING
    reason: str = ""
    fault_reason: str = ""
    ticks: int = 0
    target_name: str | None = None
    last_seen: float = 0.0
    menu_presses: int = 0
    dialogue_choice: int = 0


@dataclass
class SkillController:
    """Executes one skill at a time at control rate."""

    cfg: ControlConfig = field(default_factory=ControlConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: object | None = None                    # lumine.policy.ActionHead, or None
    state: ControllerState = field(default_factory=ControllerState)

    call: SkillCall | None = None
    _t0: float = 0.0
    #: The current tick's clock, taken from ``Observation.t``.
    #:
    #: Everything in this class is timed against the *environment's* clock, never the wall
    #: clock. Mixing the two is a real bug: in synchronous reasoning mode a two-second model
    #: call advances the wall clock while the world stands still, so a wall-clock skill
    #: timeout would fire based on time the game never experienced.
    _now: float = 0.0
    _seen_target_ticks: int = 0
    _gone_ticks: int = 0
    _tick_count: int = 0
    #: Rolling count of ticks spent without making progress, used to fail out of a skill.
    _no_progress: int = 0
    _last_boxes: list[EntityBox] = field(default_factory=list)
    #: How many times the controller had to absorb a malformed model parameter this episode.
    faults: int = 0
    #: ``"skill:ExceptionType"`` -> count. A single counter hides *which* skill is broken,
    #: and a systematically broken skill is the thing worth seeing in a report.
    fault_log: Counter = field(default_factory=Counter)
    #: First exception message seen for each fault kind, for diagnosis.
    fault_examples: dict[str, str] = field(default_factory=dict)

    # -- lifecycle ------------------------------------------------------------------------

    def reset_episode(self) -> None:
        """Clear per-episode state.

        The controller is built once per Agent and reused across episodes, so anything that
        accumulates has to be cleared here or it silently becomes a lifetime total. That is
        the same mistake that made the reported reasoning rate climb across a suite.
        """
        self.faults = 0
        self.fault_log = Counter()
        self.fault_examples = {}
        self.call = None
        self._last_boxes = []
        self._tick_count = 0
        self._no_progress = 0
        self.state = ControllerState()

    def faults_by_kind(self) -> dict[str, int]:
        """Fault counts keyed by ``skill:ExceptionType``, most frequent first."""
        return dict(self.fault_log.most_common())

    def set_skill(self, call: SkillCall, obs: Observation | None = None) -> None:
        self.call = call
        now = obs.t if obs is not None else self._now
        self.state = ControllerState(status=RUNNING, target_name=call.target, last_seen=now)
        self._t0 = now
        self._seen_target_ticks = 0
        self._gone_ticks = 0
        self._no_progress = 0

    @property
    def status(self) -> str:
        return self.state.status

    @property
    def reason(self) -> str:
        return self.state.reason

    def _finish(self, status: str, reason: str) -> Action:
        self.state.status = status
        self.state.reason = reason
        return Action()

    # -- main entry point -----------------------------------------------------------------

    def tick(self, obs: Observation) -> Action:
        self._tick_count += 1
        self._now = obs.t
        if self.call is None:
            return Action()

        boxes = boxes_from_info(obs.info)
        if boxes:
            self._last_boxes = boxes
        else:
            boxes = self._last_boxes

        # Layer 1: reflexes.  Survival outranks the plan.
        reflex = self._reflex(obs)
        if reflex is not None:
            return reflex

        # Layer 2: the skill FSM.
        #
        # Wrapped because the inputs to this layer come from a language model. Every bad
        # parameter is a potential exception, and an exception here propagates out of
        # run_episode and kills the whole run -- the exact outcome the tolerant parser and
        # the never-raising brain were built to avoid. Failing the *skill* is recoverable;
        # failing the *episode* is not.
        try:
            action = self._execute(self.call, obs, boxes)
        except Exception as exc:  # noqa: BLE001
            kind = f"{self.call.skill}:{type(exc).__name__}"
            self.faults += 1
            self.fault_log[kind] += 1
            self.fault_examples.setdefault(kind, str(exc)[:200])
            self.state.fault_reason = f"{kind}: {exc}"
            return self._finish(FAILED, f"controller fault: {kind}")

        # Layer 3: learned override, only where it is confident and never for safety skills.
        if self.policy is not None and self.call.skill not in ("dodge", "wait"):
            action = self._blend(action, obs)

        return action

    # -- reflexes -------------------------------------------------------------------------

    def _reflex(self, obs: Observation) -> Action | None:
        h = obs.hud
        if not h.alive:
            return Action()

        # Incoming AoE: sprint out of the ring, away from its centroid.
        spot = detect_telegraph(obs.frame.image)
        if spot is not None and obs.frame_is_new:
            cx, _cy = spot
            w = self.perception.width
            away = -1.0 if cx > w / 2 else 1.0
            return Action(keys={"W", "SHIFT", "A" if away > 0 else "D"},
                          mouse_dx=away * self.cfg.mouse_sensitivity * 0.5,
                          dt=1.0 / self.cfg.hz)

        # Low HP: break off and reposition.
        if h.max_hp > 0 and h.hp / h.max_hp < 0.18 and self.call and self.call.skill == "attack":
            return Action(keys={"S", "SHIFT"}, dt=1.0 / self.cfg.hz)
        return None

    # -- the skill FSM --------------------------------------------------------------------

    def _execute(self, call: SkillCall, obs: Observation, boxes: list[EntityBox]) -> Action:
        skill = call.skill
        h = obs.hud
        dt = 1.0 / self.cfg.hz
        self.state.ticks += 1

        # Modal states swallow input: they must be resolved before anything else works.
        if h.dialogue_open:
            return self._do_dialogue(call, obs, dt)
        if h.menu and skill in ("open_menu", "menu_select"):
            return self._do_menu(call, obs, dt)

        target = self._resolve(call, obs, boxes)

        if skill == "wait":
            secs = float(call.params.get("seconds", 1.0) or 1.0)
            if self._now - self._t0 >= secs:
                return self._finish(DONE, "waited")
            return Action(dt=dt)

        if skill == "goto":
            return self._do_goto(call, obs, target, dt)

        if skill == "attack":
            return self._do_attack(call, obs, target, dt)

        if skill in ("interact", "open_chest", "collect"):
            return self._do_interact(call, obs, target, dt)

        if skill == "use_skill":
            return self._do_element(call, obs, dt)

        if skill == "use_burst":
            if self._now - self._t0 > 0.1:
                return self._finish(DONE, "burst pressed")
            return Action(keys={"Q"}, dt=dt)

        if skill == "switch_character":
            return self._do_switch(call, obs, dt)

        if skill == "dodge":
            return self._do_dodge(obs, dt)

        if skill == "climb":
            return self._do_climb(call, obs, target, dt)

        if skill == "glide":
            return self._do_glide(call, obs, target, dt)

        if skill == "open_menu":
            return self._do_open_menu(call, obs, dt)

        if skill == "menu_select":
            return self._do_menu(call, obs, dt)

        if skill == "follow_route":
            return self._do_goto(call, obs, target, dt)

        return self._finish(FAILED, f"unhandled skill {skill!r}")

    # -- target resolution ----------------------------------------------------------------

    def _resolve(self, call: SkillCall, obs: Observation, boxes: list[EntityBox]) -> EntityBox | dict | None:
        """Find the named target: prefer the on-screen detector, fall back to HUD entities."""
        want = (call.target or call.params.get("target") or "").lower()
        if not want:
            return None

        best: EntityBox | None = None
        for b in boxes:
            if not b.visible:
                continue
            if want in b.name.lower() or want in b.kind.lower():
                if best is None or b.area > best.area:
                    best = b
        if best is not None:
            self._seen_target_ticks += 1
            self._gone_ticks = 0
            return best

        # World-space fallback: entities reported by the game/sandbox with positions.
        cands = [e for e in obs.hud.nearby_entities
                 if want in str(e.get("name", "")).lower() or want in str(e.get("kind", "")).lower()]
        if cands:
            self._seen_target_ticks += 1
            self._gone_ticks = 0
            return dict(min(cands, key=lambda e: e.get("distance", 1e9)))

        self._gone_ticks += 1

        # Last resort: the brain marked a point on the frame. Nothing on screen can be named,
        # but a pixel is enough to steer by.
        if call.target_px is not None:
            self._seen_target_ticks += 1
            return PixelTarget(float(call.target_px[0]), float(call.target_px[1]))
        return None

    # -- steering -------------------------------------------------------------------------

    def _steer(self, obs: Observation, target) -> tuple[float, bool, float]:
        """Return ``(mouse_dx, aligned, distance)`` for a target.

        Two independent sources, because a real game gives you only the first:
          * screen-space error from the detector box (always available when visible)
          * world bearing from HUD entities, when present
        """
        w = self.perception.width
        sens = self.cfg.mouse_sensitivity

        if isinstance(target, EntityBox):
            cx, _cy = target.center
            err = (cx - w / 2.0) / (w / 2.0)
            dx = max(-sens, min(sens, -err * sens * 1.6))
            return dx, abs(err) < 0.14, float(target.distance)

        if isinstance(target, PixelTarget):
            # Lower gain than a detector box: a VLM's pixel is coarse, and over-steering on a
            # noisy coordinate makes the camera oscillate instead of converge.
            err = (target.x - w / 2.0) / (w / 2.0)
            dx = max(-sens, min(sens, -err * sens * 0.9))
            return dx, abs(err) < 0.18, 999.0

        if isinstance(target, dict):
            dist = float(target.get("distance", 999.0))
            pos = target.get("position")
            if pos and len(pos) == 3:
                px, _py, pz = obs.hud.position
                yaw = obs.hud.yaw
                tx, _ty, tz = (float(pos[0]), float(pos[1]), float(pos[2]))
                bearing = _wrap_pi(math.atan2(tx - px, tz - pz) - yaw)
                dx = max(-sens, min(sens, bearing * sens * 1.2))
                return dx, abs(bearing) < 0.20, dist
            return 0.0, True, dist

        # Nothing to aim at: sweep the camera looking for the objective.
        return sens * 0.45, False, 999.0

    def _do_goto(self, call: SkillCall, obs: Observation, target, dt: float) -> Action:
        stop = _num(call.params.get("stop_distance"), 1.6)
        dx, aligned, dist = self._steer(obs, target)

        if target is None:
            self._no_progress += 1
            if self._no_progress > int(12 * self.cfg.hz):
                return self._finish(FAILED, "target never came into view")
            return Action(keys={"W"}, mouse_dx=dx, dt=dt)

        # A pixel target carries no distance, so arrival is judged by what the game says when
        # we get there -- an interaction prompt -- or by having held the target centred long
        # enough to be standing on it.
        if isinstance(target, PixelTarget):
            if obs.hud.interact_prompt:
                return self._finish(DONE, f"reached {call.target}")
            if aligned and self._now - self._t0 > 4.0:
                return self._finish(DONE, "reached the marked point")
            return Action(keys={"W"}, mouse_dx=dx, dt=dt)

        if dist and dist <= stop:
            return self._finish(DONE, f"reached {call.target}")

        keys = {"W"}
        if dist > 12:
            keys.add("SHIFT")           # sprint only on long approaches; stamina matters
        if not aligned:
            keys.discard("SHIFT")
        return Action(keys=keys, mouse_dx=dx, dt=dt)

    def _do_attack(self, call: SkillCall, obs: Observation, target, dt: float) -> Action:
        if target is None:
            self._gone_ticks += 1
            # The enemy vanishing after we engaged is a kill, not a failure.
            if self._seen_target_ticks > int(0.5 * self.cfg.hz):
                return self._finish(DONE, "target eliminated")
            if self._gone_ticks > int(6 * self.cfg.hz):
                return self._finish(FAILED, "lost the target")
            return Action(keys={"W"}, mouse_dx=self.cfg.mouse_sensitivity * 0.45, dt=dt)

        dx, aligned, dist = self._steer(obs, target)

        if isinstance(target, PixelTarget):
            # No distance: close in on the marked point, then start swinging once it has been
            # held under the crosshair for a moment.
            settled = self._now - self._t0 > 1.0
            if aligned and settled:
                return Action(keys={"J", "W"}, dt=dt)
            return Action(keys={"W"}, mouse_dx=dx, dt=dt)

        if dist and dist > ATTACK_RANGE:
            return Action(keys={"W", "SHIFT"} if dist > 6 else {"W"}, mouse_dx=dx, dt=dt)
        if not aligned:
            return Action(mouse_dx=dx, dt=dt)
        # In range and aimed: swing, and keep swinging while advancing slightly.
        return Action(keys={"J", "W"}, dt=dt)

    def _do_interact(self, call: SkillCall, obs: Observation, target, dt: float) -> Action:
        h = obs.hud

        prompt = (h.interact_prompt or "").lower()
        wants = call.skill == "open_chest" and "chest" in prompt
        wants = wants or (call.skill == "interact" and bool(prompt))
        wants = wants or (call.skill == "collect" and bool(prompt))

        if wants:
            self._no_progress = 0
            return Action(keys={"F"}, dt=dt)

        if isinstance(target, PixelTarget):
            # Walk to the marked point; F fires on its own once the prompt appears above.
            if self._now - self._t0 > 12.0:
                return self._finish(FAILED, f"never reached {call.target}")
            dx, _aligned, _d = self._steer(obs, target)
            return Action(keys={"W"}, mouse_dx=dx, dt=dt)

        dx, aligned, dist = self._steer(obs, target)
        if target is None:
            self._no_progress += 1
            if self._no_progress > int(10 * self.cfg.hz):
                return self._finish(FAILED, f"{call.target} not found")
            return Action(keys={"W"}, mouse_dx=dx, dt=dt)

        if dist and dist <= INTERACT_RANGE:
            # Close enough to press F; hold position and press.
            return Action(keys={"F"}, dt=dt)

        keys = {"W", "SHIFT"} if dist > 10 else {"W"}
        if not aligned:
            keys.discard("SHIFT")
        return Action(keys=keys, mouse_dx=dx, dt=dt)

    @staticmethod
    def _resolve_choice(raw: object, options: tuple[str, ...]) -> int:
        """Map a model's dialogue answer onto an option index.

        Models answer this question three different ways -- ``2``, ``"2"``, and
        ``"I'm here to help the Knights of Favonius."`` -- and all three are reasonable
        readings of a field named ``dialogue_choice``. Accepting only the first is a bug
        that presents as a crash, so all three are accepted, in that order of confidence.
        """
        n = max(1, len(options))
        if raw is None:
            return 0
        if isinstance(raw, bool):
            return 0
        if isinstance(raw, int):
            return raw % n
        if isinstance(raw, float):
            return int(raw) % n
        text = str(raw).strip()
        if not text or text.lower() in ("null", "none"):
            return 0
        # Exact index, then a 1-based index if that is what the model clearly meant.
        if text.isdigit():
            return int(text) % n
        low = text.lower()
        for i, opt in enumerate(options):
            if low == opt.strip().lower():
                return i
        # Fuzzy: the model paraphrased. Pick the option sharing the most content words.
        import re

        words = set(re.findall(r"[a-z]{4,}", low))
        best, best_score = 0, 0
        for i, opt in enumerate(options):
            score = len(words & set(re.findall(r"[a-z]{4,}", opt.lower())))
            if score > best_score:
                best, best_score = i, score
        return best

    def _do_element(self, call: SkillCall, obs: Observation, dt: float) -> Action:
        want = str(call.params.get("element") or call.params.get("target") or "").lower()
        h = obs.hud
        if want and want != h.element and want in ("pyro", "cryo", "electro", "anemo", "hydro",
                                                   "geo", "dendro"):
            slot = {"anemo": "1", "pyro": "2", "cryo": "3", "electro": "4"}.get(want)
            if slot:
                return Action(keys={slot}, dt=dt)
        if self._now - self._t0 > 0.15:
            return self._finish(DONE, f"used {want or h.element} skill")
        return Action(keys={"E"}, dt=dt)

    def _do_switch(self, call: SkillCall, obs: Observation, dt: float) -> Action:
        slot = str(call.params.get("slot") or "")
        if not slot:
            elem = str(call.params.get("element") or "").lower()
            slot = {"anemo": "1", "pyro": "2", "cryo": "3", "electro": "4"}.get(elem, "1")
        if self._now - self._t0 > 0.1:
            return self._finish(DONE, f"switched to slot {slot}")
        return Action(keys={slot}, dt=dt)

    def _do_dodge(self, obs: Observation, dt: float) -> Action:
        spot = detect_telegraph(obs.frame.image)
        w = self.perception.width
        if spot is None:
            return self._finish(DONE, "no telegraph")
        away = -1.0 if spot[0] > w / 2 else 1.0
        if self._now - self._t0 > 1.2:
            return self._finish(DONE, "dodged")
        return Action(keys={"W", "SHIFT", "A" if away > 0 else "D"},
                      mouse_dx=away * self.cfg.mouse_sensitivity * 0.4, dt=dt)

    def _do_climb(self, call: SkillCall, obs: Observation, target, dt: float) -> Action:
        dx, aligned, dist = self._steer(obs, target)
        if target is None and self._now - self._t0 > 8:
            return self._finish(FAILED, "no surface to climb")
        if target is not None and dist and dist < 1.2:
            if self._now - self._t0 > 3.0:
                return self._finish(DONE, "climbed")
            return Action(keys={"SPACE", "W"}, dt=dt)
        return Action(keys={"W"}, mouse_dx=dx, dt=dt)

    def _do_glide(self, call: SkillCall, obs: Observation, target, dt: float) -> Action:
        dx, _aligned, dist = self._steer(obs, target)
        if target is None and self._now - self._t0 > 6:
            return self._finish(FAILED, "nothing to glide to")
        if dist and dist < 1.5:
            return self._finish(DONE, "landed on target")
        elapsed = self._now - self._t0
        keys = {"W"}
        if elapsed < 0.4:
            keys.add("SPACE")
        return Action(keys=keys, mouse_dx=dx, dt=dt)

    # -- GUI --------------------------------------------------------------------------------

    def _do_open_menu(self, call: SkillCall, obs: Observation, dt: float) -> Action:
        menu = str(call.params.get("menu") or call.target or "").lower()
        h = obs.hud
        if h.menu == menu:
            return self._finish(DONE, f"{menu} menu open")
        if self._now - self._t0 > 3.0:
            return self._finish(FAILED, f"could not open {menu}")
        key = {"map": "M", "cook": "F", "weapon": "C", "bag": "B"}.get(menu, "M")
        return Action(keys={key}, dt=dt)

    def _do_menu(self, call: SkillCall, obs: Observation, dt: float) -> Action:
        """Walk a menu cursor to the wanted entry, then confirm.

        The sandbox exposes the cursor index but not the entry list, so navigation is
        expressed as "press DOWN, watch the cursor move, press ENTER once it has not
        changed for a moment" -- which is also what a real screen-only agent must do.
        """
        h = obs.hud
        want = str(call.params.get("entry") or call.target or "").strip()
        if not h.menu:
            if self.state.ticks % 8 == 0:
                return Action()                      # let the panel settle
            return self._finish(DONE, "menu closed")

        steps = 3 if call.params.get("menu") == "map" else 2
        p = self.state.menu_presses
        if p < steps:
            self.state.menu_presses += 1
            return Action(keys={"DOWN"}, dt=dt)
        if p == steps:
            self.state.menu_presses += 1
            return Action(keys={"ENTER"}, dt=dt)
        return self._finish(DONE, f"confirmed {want!r}")

    def _do_dialogue(self, call: SkillCall, obs: Observation, dt: float) -> Action:
        """Choose a dialogue option.

        The sandbox accepts ``1``/``2``/``3``/``4`` directly, which is simpler and far more
        reliable than walking a cursor with ``DOWN`` + ``ENTER``: the cursor needed state to
        survive across ticks that ``set_skill`` resets on every re-plan, so it never advanced
        past the first option.

        The model's pick is tried first; if the box stays open, the controller sweeps the
        remaining options one at a time. That is what a player does after a wrong answer, and
        it means one bad guess costs a couple of seconds instead of the whole conversation.
        """
        h = obs.hud
        options = tuple(h.dialogue_options)
        n = max(1, len(options))
        choice = self._resolve_choice(call.params.get("dialogue_choice"), options)

        # Four ticks per option: long enough for the world to see one key *edge* per option,
        # short enough that a three-option box is resolved in under half a second.
        if self.state.ticks > 4 * n + 2:
            return self._finish(DONE, f"dialogue with {h.dialogue_speaker or 'npc'}")
        idx = (choice + self.state.ticks // 4) % n
        return Action(keys={str(idx + 1)}, dt=dt)

    # -- learned override ------------------------------------------------------------------

    def _blend(self, rule_action: Action, obs: Observation) -> Action:
        """Let the learned head act when it is confident; otherwise keep the safe rule output."""
        try:
            out = self.policy.predict(obs.frame.image)          # type: ignore[union-attr]
        except Exception:
            return rule_action
        conf = float(getattr(out, "confidence", 0.0))
        if conf < self.cfg.hybrid_threshold:
            return rule_action
        learned = getattr(out, "action", None)
        return learned if isinstance(learned, Action) else rule_action

    # -- diagnostics -----------------------------------------------------------------------

    def stats(self) -> dict[str, object]:
        return {"skill": self.call.skill if self.call else None,
                "status": self.state.status, "reason": self.state.reason,
                "ticks": self.state.ticks, "faults": self.faults}
