"""Core data types shared by every layer of Lumine.

Design constraint that drives everything here
---------------------------------------------
Lumine (arXiv 2511.08892) is defined by three rates that must not be conflated:

    perception  5 Hz    raw pixels in
    control    30 Hz    keyboard / mouse out
    reasoning  ~0.5 Hz  a VLM, invoked *only when necessary*

The types below keep those three rates physically separate.  A slow planner never
touches per-tick control, and a fast controller never blocks on the network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------------------

#: Canonical key vocabulary.  Anything a 30 Hz controller may press must be listed here so
#: that every backend (simulator, uinput, dataset, learned policy) agrees on the alphabet.
KEYS: tuple[str, ...] = (
    "W", "A", "S", "D",            # locomotion, camera-relative
    "SPACE", "SHIFT", "CTRL",      # jump / sprint / crouch
    "J", "K", "L",                 # normal attack / charged attack / plunge
    "E", "Q",                      # elemental skill / burst
    "F", "R", "T",                 # interact / aim / elemental sight
    "1", "2", "3", "4",            # party slots
    "M", "B", "C", "ESC",          # map / bag / character / cancel
    "ENTER", "UP", "DOWN", "LEFT", "RIGHT",   # GUI navigation
    "TAB", "X", "Z",
)

#: Mouse buttons.
BUTTONS: tuple[str, ...] = ("left", "right", "middle")

KEY_INDEX: dict[str, int] = {k: i for i, k in enumerate(KEYS)}
NUM_KEYS: int = len(KEYS)


@dataclass(slots=True)
class Action:
    """One 30 Hz control tick.

    Held keys are *edge-triggered by the executor*: a controller returns the set of keys
    that should be down for this tick and the executor presses/releases the difference.
    That matches how a human plays — you hold ``W`` across many ticks — and it removes
    an entire class of key-repeat bugs.
    """

    keys: set[str] = field(default_factory=set)
    mouse_dx: float = 0.0            # relative camera movement, in counts
    mouse_dy: float = 0.0
    buttons: set[str] = field(default_factory=set)
    dt: float = 1.0 / 30.0

    def __post_init__(self) -> None:
        bad = self.keys - set(KEYS)
        if bad:
            raise ValueError(f"unknown keys {sorted(bad)}; add them to lumine.types.KEYS")
        bad_b = self.buttons - set(BUTTONS)
        if bad_b:
            raise ValueError(f"unknown buttons {sorted(bad_b)}")

    # -- conversions used by the learner and the recorder ---------------------------------

    def to_vector(self) -> np.ndarray:
        """Flat float32 vector: [key multi-hot | mouse_dx | mouse_dy | button multi-hot]."""
        v = np.zeros(NUM_KEYS + 2 + len(BUTTONS), dtype=np.float32)
        for k in self.keys:
            v[KEY_INDEX[k]] = 1.0
        v[NUM_KEYS] = self.mouse_dx
        v[NUM_KEYS + 1] = self.mouse_dy
        for i, b in enumerate(BUTTONS):
            if b in self.buttons:
                v[NUM_KEYS + 2 + i] = 1.0
        return v

    @classmethod
    def from_vector(cls, v: Sequence[float], threshold: float = 0.5, dt: float = 1.0 / 30.0) -> "Action":
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        keys = {KEYS[i] for i in range(NUM_KEYS) if v[i] > threshold}
        buttons = {BUTTONS[i] for i in range(len(BUTTONS)) if v[NUM_KEYS + 2 + i] > threshold}
        return cls(keys=keys, mouse_dx=float(v[NUM_KEYS]), mouse_dy=float(v[NUM_KEYS + 1]),
                   buttons=buttons, dt=dt)

    def is_noop(self) -> bool:
        return not (self.keys or self.buttons or self.mouse_dx or self.mouse_dy)

    def describe(self) -> str:
        parts = sorted(self.keys) + [f"{b}-click" for b in sorted(self.buttons)]
        if self.mouse_dx or self.mouse_dy:
            parts.append(f"mouse({self.mouse_dx:+.0f},{self.mouse_dy:+.0f})")
        return "+".join(parts) if parts else "idle"


def merge_actions(*actions: Action, dt: float = 1.0 / 30.0) -> Action:
    """Union of several simultaneous intentions (e.g. walk + attack)."""
    keys: set[str] = set()
    buttons: set[str] = set()
    dx = dy = 0.0
    for a in actions:
        if a is None:
            continue
        keys |= a.keys
        buttons |= a.buttons
        dx += a.mouse_dx
        dy += a.mouse_dy
    return Action(keys=keys, buttons=buttons, mouse_dx=dx, mouse_dy=dy, dt=dt)


# --------------------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Frame:
    """A raw RGB screenshot, exactly what the VLM would be shown."""

    image: np.ndarray                      # (H, W, 3) uint8, RGB
    t: float = field(default_factory=time.monotonic)

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[0], self.image.shape[1]

    def to_png_bytes(self) -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(self.image).save(buf, format="PNG")
        return buf.getvalue()

    def to_data_url(self) -> str:
        import base64

        return "data:image/png;base64," + base64.b64encode(self.to_png_bytes()).decode()


@dataclass(slots=True)
class HUDState:
    """Structured read-out of the on-screen UI.

    This is *privileged* information.  The default agent runs pixels-only, precisely
    because Lumine's whole point is that a VLM reads the screen.  It is emitted anyway so
    the evaluation harness can (a) score task progress without a model in the loop and
    (b) run a "HUD-oracle" ablation against the pixels-only configuration.
    """

    hp: float = 100.0
    max_hp: float = 100.0
    stamina: float = 100.0
    max_stamina: float = 100.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    region: str = "mondstadt"
    quest: str = ""
    objective: str = ""
    objective_progress: tuple[int, int] = (0, 0)
    interact_prompt: str | None = None
    dialogue_open: bool = False
    dialogue_speaker: str = ""
    dialogue_options: tuple[str, ...] = ()
    menu: str | None = None                # "map" | "cook" | "weapon" | "bag" | None
    menu_cursor: int = 0
    party: tuple[str, ...] = ()
    active_party_index: int = 0
    element: str = "anemo"
    nearby_entities: tuple[Mapping[str, Any], ...] = ()
    alive: bool = True
    time_of_day: float = 12.0


@dataclass(slots=True)
class Observation:
    """Everything the agent receives for one control tick."""

    frame: Frame
    hud: HUDState
    t: float = field(default_factory=time.monotonic)
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    #: True when ``frame`` is a fresh capture and False when the controller is reusing the
    #: previous image between 5 Hz captures.  The adaptive reasoner keys off this.
    frame_is_new: bool = True

    def summary(self) -> str:
        h = self.hud
        return (f"t={self.t:8.2f} region={h.region} hp={h.hp:.0f}/{h.max_hp:.0f} "
                f"pos=({h.position[0]:.1f},{h.position[1]:.1f},{h.position[2]:.1f}) "
                f"quest={h.quest!r} obj={h.objective!r} progress={h.objective_progress}")


# --------------------------------------------------------------------------------------
# Language level
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class Intent:
    """The slow brain's output: *what to achieve*, not *which key to hold*.

    This split is the single most important engineering decision in this codebase.  A VLM
    that answers in 2 s cannot emit 30 Hz control, and a 30 Hz controller cannot read a
    quest log.  The VLM therefore speaks in skills + targets; a pretrained controller
    grounds those into keystrokes.  It mirrors Lumine's own two-stage curriculum
    (action primitives -> instruction following).
    """

    skill: str = "wait"
    target: str | None = None
    target_px: tuple[float, float] | None = None      # (x, y) in frame pixel space
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    subgoal: str = ""
    expected_done: bool = False
    raw: str = ""

    def __str__(self) -> str:  # pragma: no cover - debug helper
        t = f" target={self.target!r}" if self.target else ""
        p = f" px={self.target_px}" if self.target_px else ""
        return f"<Intent {self.skill}{t}{p}>"


@dataclass(slots=True)
class SkillSpec:
    """A named, parameterised macro-action the fast controller knows how to execute.

    ``description`` is fed verbatim to the VLM as part of the tool list, so it doubles as
    the prompt for instruction following.
    """

    name: str
    description: str
    params: tuple[str, ...] = ()
    category: str = "general"


# --------------------------------------------------------------------------------------
# Tasks and missions
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class TaskSpec:
    """One evaluable task, mirroring the categories in the Lumine report."""

    task_id: str
    category: str                  # combat | boss | puzzle | npc | gui | icl
    instruction: str
    region: str = "mondstadt"
    seed: int = 0
    max_seconds: float = 120.0
    #: Extra decomposition handed to the agent for the in-context-learning category.
    hints: tuple[str, ...] = ()
    success_criteria: str = ""

    def __str__(self) -> str:
        return f"[{self.category}/{self.region}] {self.instruction}"


@dataclass(slots=True)
class EpisodeResult:
    task: TaskSpec
    success: bool
    seconds: float
    frames: int
    vlm_calls: int
    vlm_seconds: float
    controller: str
    reason: str = ""
    #: Malformed model parameters the controller had to absorb. Non-zero is not a failure by
    #: itself -- it means the guard did its job -- but a *systematic* fault on one skill is a
    #: bug that would otherwise only be visible by reading logs.
    faults: int = 0
    faults_by_kind: dict[str, int] = field(default_factory=dict)
    #: One example message per fault kind, so the report is diagnosable.
    fault_examples: dict[str, str] = field(default_factory=dict)
    transcript: list[dict[str, Any]] = field(default_factory=list)


def quantize_px(x: float, y: float, width: int, height: int, grid: int = 32) -> tuple[float, float]:
    """Snap a pixel coordinate to a coarse grid.

    VLMs are bad at exact coordinates and good at coarse regions; a 32 px grid at 640x360
    gives ~20x11 cells, which is reliably groundable, and the controller then refines the
    target with classical CV.
    """
    gx = round(x / grid) * grid
    gy = round(y / grid) * grid
    return float(np.clip(gx, 0, width - 1)), float(np.clip(gy, 0, height - 1))


def flatten(seq: Iterable[Iterable[Any]]) -> list[Any]:
    return [x for sub in seq for x in sub]
