"""The built-in 3D open-world sandbox: terrain, physics, entities and game logic.

Genshin Impact cannot run on this machine, so Lumine ships its own world.  Everything the
agent can see or do in an episode lives in this module:

* a deterministic heightmap over ``[-60, 60]^2`` with bilinear sampling,
* a capsule player with gravity, terrain following, sprint, swim, climb and wind lift,
* two regions -- ``mondstadt`` (in-distribution) and ``liyue`` (the out-of-distribution
  hold-out the paper tests on) -- that differ in palette, terrain shape, landmark layout
  and NPC cast while sharing every mechanic,
* the full entity zoo: chests, slimes, hilichurls, whopperflowers, hypostasis and
  Stormterror bosses, NPCs, monuments, waypoints, cooking pots, Anemoculi, Anemograna,
  wind currents, wind barriers, boulders, thorn bushes, pillars and scenery,
* combat (normal / charged / elemental skill / burst), puzzles, dialogue, the map / cook /
  weapon / bag GUIs, damage numbers, particles and ground telegraphs.

Two invariants matter for the rest of the project:

1. **Determinism.**  Every random draw comes from one ``numpy.random.default_rng(seed)``
   consumed in a fixed order, so ``(seed, region, task, action sequence)`` fully determines
   the episode.  That makes recorded datasets reproducible.
2. **Cheapness.**  :meth:`World.step` is pure Python scalar maths over a few dozen entities;
   it never touches the rasteriser.  The 30 Hz control loop must not wait for pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..config import SimConfig
from ..types import Action, HUDState, TaskSpec
from . import tasks as _tasks

__all__ = ["World", "Entity", "Player", "TERRAIN_HALF", "WATER_LEVEL", "PARTY", "WEAPONS", "ELEMENT_RGB"]

# --------------------------------------------------------------------------------------
# World constants
# --------------------------------------------------------------------------------------

#: Terrain covers ``[-TERRAIN_HALF, TERRAIN_HALF]`` on both X (east) and Z (north).
TERRAIN_HALF: float = 60.0
#: Heightmap sampling step, in world units.
GRID_STEP: float = 1.0
#: Number of heightmap samples along one axis.
GRID_N: int = int(2.0 * TERRAIN_HALF / GRID_STEP) + 1
#: The water plane sits at ``y = 0``; anything below it is a lake.
WATER_LEVEL: float = 0.0

PLAYER_RADIUS: float = 0.4
PLAYER_EYE_HEIGHT: float = 1.55
GRAVITY: float = 22.0
JUMP_SPEED: float = 8.2
WALK_SPEED: float = 5.0
SPRINT_SPEED: float = 8.4
SWIM_SPEED: float = 2.3
CLIMB_SPEED: float = 3.4
WIND_LIFT_SPEED: float = 6.0
SPRINT_STAMINA_DRAIN: float = 18.0
SWIM_STAMINA_DRAIN: float = 9.0
STAMINA_REGEN: float = 22.0
CLIMB_STAMINA_DRAIN: float = 12.0

ATTACK_RANGE: float = 2.2
ATTACK_HALF_ARC: float = math.radians(55.0)
ATTACK_DAMAGE: float = 22.0
ATTACK_CD: float = 0.35
CHARGED_RANGE: float = 3.0
CHARGED_DAMAGE: float = 42.0
CHARGED_CD: float = 1.2
SKILL_CD: float = 6.0
BURST_COST: float = 100.0
BURST_RADIUS: float = 6.0
BURST_DAMAGE: float = 70.0
ENERGY_PER_HIT: float = 8.0
INTERACT_RANGE: float = 1.8
TIME_TRIAL_SECONDS: float = 40.0

MOUSE_YAW_GAIN: float = 0.0032
MOUSE_PITCH_GAIN: float = 0.0032
KEY_TURN_SPEED: float = 2.1
KEY_PITCH_SPEED: float = 1.4

#: The player's party: ``(name, element)``.  Kaeya is cryo by design -- the ``icl_kaeya_freeze``
#: task depends on it -- and the elemental monument puzzles always require one of these four.
PARTY: tuple[tuple[str, str], ...] = (
    ("Lumine", "anemo"),
    ("Kaeya", "cryo"),
    ("Amber", "pyro"),
    ("Lisa", "electro"),
)

#: Equippable weapons, changed through the ``C`` character menu.
WEAPONS: tuple[dict[str, Any], ...] = (
    {"name": "Dull Blade", "kind": "sword", "atk": 39, "element": "anemo"},
    {"name": "Harbinger of Dawn", "kind": "sword", "atk": 51, "element": "cryo"},
    {"name": "Amos' Bow", "kind": "bow", "atk": 46, "element": "pyro"},
    {"name": "Black Tassel", "kind": "polearm", "atk": 44, "element": "electro"},
    {"name": "Skyward Spine", "kind": "polearm", "atk": 58, "element": "electro"},
)

#: Cookable recipes; the first entry is the target of ``gui_cook_sweet_madame``.
RECIPES: tuple[str, ...] = ("Sweet Madame", "Mondstadt Hash Brown", "Satisfying Salad")

#: Pantry contents shown in the cook menu.
INGREDIENTS: tuple[tuple[str, int], ...] = (
    ("Fowl", 6),
    ("Sweet Flower", 4),
    ("Flour", 8),
    ("Raw Meat", 5),
    ("Bird Egg", 3),
)

#: Backpack contents shown in the bag menu.
STARTING_INVENTORY: tuple[tuple[str, int], ...] = (
    ("Sweet Madame", 2),
    ("Apple", 5),
    ("Mushroom", 3),
    ("Anemoculus Fragment", 1),
)

#: Skills and their cooldowns, mirrored into the HUD icon shading.
SKILL_COOLDOWNS: dict[str, float] = {"E": SKILL_CD, "Q": 0.0}

#: RGB used for particles and elemental effects, shared with the renderer so the HUD and the
#: world never disagree about what "pyro orange" looks like.
ELEMENT_RGB: dict[str, tuple[int, int, int]] = {
    "anemo": (120, 232, 190),
    "pyro": (250, 130, 60),
    "cryo": (150, 220, 250),
    "electro": (190, 140, 250),
    "hydro": (80, 160, 250),
    "dendro": (140, 210, 90),
    "geo": (240, 200, 90),
    "physical": (240, 240, 240),
}

#: Per-kind default body metrics, so :meth:`World.add` stays a one-liner at call sites.
KIND_DEFAULTS: dict[str, dict[str, Any]] = {
    "tree": {"radius": 0.95, "height": 7.0, "solid": True},
    "rock": {"radius": 0.85, "height": 1.2, "solid": True},
    "boulder": {"radius": 1.35, "height": 1.6, "solid": True, "hp": 60.0},
    "chest": {"radius": 0.75, "height": 1.05, "interactable": True},
    "enemy": {"radius": 0.7, "height": 1.6, "hp": 60.0},
    "boss": {"radius": 2.3, "height": 4.6, "hp": 620.0},
    "npc": {"radius": 0.5, "height": 1.8, "solid": True, "interactable": True},
    "monument": {"radius": 0.95, "height": 3.2, "solid": True},
    "waypoint": {"radius": 0.9, "height": 4.6, "interactable": True},
    "cooking_pot": {"radius": 0.65, "height": 1.1, "solid": True, "interactable": True},
    "anemoculus": {"radius": 0.5, "height": 0.5, "fly": True},
    "anemograna": {"radius": 0.35, "height": 0.35, "fly": True},
    "wind_current": {"radius": 1.9, "height": 12.0},
    "wind_barrier": {"radius": 3.0, "height": 6.0, "solid": True},
    "thorn_bush": {"radius": 1.15, "height": 1.4, "solid": True, "hp": 30.0},
    "stone_pillar": {"radius": 1.25, "height": 12.0, "solid": True},
    "time_trial": {"radius": 0.8, "height": 2.4, "solid": True, "interactable": True},
    "player": {"radius": PLAYER_RADIUS, "height": 1.75},
}

#: Enemy archetypes: hp, speed, contact damage, reach, aggro radius, element.
ENEMY_TYPES: dict[str, dict[str, Any]] = {
    "slime": {"hp": 55.0, "speed": 2.0, "damage": 6.0, "reach": 1.8, "aggro": 15.0, "element": "anemo"},
    "hilichurl": {"hp": 85.0, "speed": 2.7, "damage": 8.0, "reach": 2.0, "aggro": 16.0, "element": "physical"},
    "whopperflower": {"hp": 70.0, "speed": 1.1, "damage": 11.0, "reach": 2.4, "aggro": 14.0, "element": "pyro"},
}

#: Boss archetypes: hp, contact damage and the element that breaks their shield fastest.
BOSS_TYPES: dict[str, dict[str, Any]] = {
    "electro_hypostasis": {"hp": 620.0, "damage": 16.0, "weak": "electro", "speed": 0.0},
    "anemo_hypostasis": {"hp": 620.0, "damage": 15.0, "weak": "anemo", "speed": 0.0},
    "stormterror": {"hp": 720.0, "damage": 22.0, "weak": "anemo", "speed": 3.4},
}

#: NPC dialogue scripts.  ``correct`` indexes the option that advances the quest.
DIALOGUES: dict[str, dict[str, Any]] = {
    "grace": {
        "line": "Traveler! The winds are restless today. What brings you to Mondstadt?",
        "options": (
            "I'm here to help the Knights of Favonius.",
            "Lovely weather we're having.",
            "Where can I find the tavern?",
        ),
        "correct": 0,
        "reply": "Then take the Anemo Archon's blessing. The road ahead is dangerous!",
    },
    "monroe": {
        "line": "You look like you can handle a sword. Care to hear about the commission?",
        "options": (
            "Actually, I collect mushrooms.",
            "Tell me about the commission.",
            "I have no time for this.",
        ),
        "correct": 1,
        "reply": "Good. Clear the hilichurl camp east of here and the reward is yours.",
    },
    "sayid": {
        "line": "Merchants whisper of treasure guarded by thorny vines. Interested?",
        "options": (
            "Thorns don't scare me.",
            "I only trade in stories.",
            "Burn it all, I always say.",
        ),
        "correct": 0,
        "reply": "Ha! Pyro makes short work of thorns. Off you go, then.",
    },
    "bao'er": {
        "line": "Hey there, outlander! Looking for work at the harbor?",
        "options": (
            "I'm just passing through Liyue.",
            "Yes -- who's causing trouble?",
            "The sea air is wonderful.",
        ),
        "correct": 1,
        "reply": "The Treasure Hoarders are sniffing around the terrace. Deal with them.",
    },
    "xingxiu": {
        "line": "The stone forest hides more than ore, you know.",
        "options": (
            "Tell me about the monuments.",
            "Ore is all I care about.",
            "I should be going.",
        ),
        "correct": 0,
        "reply": "Anemo answers Anemo. Match the element and the monument awakens.",
    },
    "yinian": {
        "line": "Careful on the cliffs -- the wind here bites.",
        "options": (
            "Thanks, I'll manage.",
            "Do you have anything to warm me up?",
            "I'll be off, then.",
        ),
        "correct": 1,
        "reply": "Take this recipe. A hot meal beats the cold every time.",
    },
}

#: Quest-tracker titles, per region (and per act for the missions).
QUEST_TITLES: dict[str, str] = {
    "mondstadt": "Mondstadt: Winds of Freedom",
    "liyue": "Liyue: The Harbor's Tale",
}
MISSION_TITLES: dict[str, str] = {
    "mission_mondstadt_act1": "Mondstadt -- Act I: The Outlander Who Caught the Wind",
    "mission_liyue_act1": "Liyue -- Act I: Of the Land Amidst Monoliths",
}


def _wrap_angle(a: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _dist2d(ax: float, az: float, bx: float, bz: float) -> float:
    """Horizontal (XZ) distance between two points."""
    return math.hypot(ax - bx, az - bz)


# --------------------------------------------------------------------------------------
# Entities and player
# --------------------------------------------------------------------------------------


@dataclass
class Entity:
    """One object in the world.

    ``x``/``z`` are the ground-plane position; ``y`` is the base of the body (terrain height
    for grounded entities, free-floating for anything with ``fly=True``).  Gameplay state
    that only some kinds need lives in ``data`` so the class stays small.
    """

    kind: str
    name: str = ""
    x: float = 0.0
    z: float = 0.0
    y: float = 0.0
    radius: float = 0.6
    height: float = 1.5
    solid: bool = False
    interactable: bool = False
    fly: bool = False
    alive: bool = True
    hp: float = 1.0
    max_hp: float = 1.0
    element: str | None = None
    variant: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> tuple[float, float, float]:
        """World-space position of the entity's base."""
        return (self.x, self.y, self.z)

    @property
    def center(self) -> tuple[float, float, float]:
        """World-space position of the entity's visual centre."""
        return (self.x, self.y + 0.5 * self.height, self.z)

    def horizontal_radius(self) -> float:
        """Collision radius; canopies and domes are thinner at the base than visually."""
        base = float(self.data.get("collide_scale", 0.75))
        return self.radius * base


@dataclass
class Player:
    """The player capsule and its per-episode state."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    hp: float = 100.0
    max_hp: float = 100.0
    stamina: float = 100.0
    max_stamina: float = 100.0
    grounded: bool = True
    in_water: bool = False
    climbing: bool = False
    sprinting: bool = False
    alive: bool = True
    energy: float = 0.0
    party_index: int = 0

    @property
    def position(self) -> tuple[float, float, float]:
        """Player feet position."""
        return (self.x, self.y, self.z)

    @property
    def eye(self) -> tuple[float, float, float]:
        """Head position, used by the renderer for a third-person camera."""
        return (self.x, self.y + PLAYER_EYE_HEIGHT, self.z)


# --------------------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------------------


class World:
    """A deterministic open-world sandbox for one episode.

    Args:
        config: simulation config (region, seed, day length, fog).
        task: the task being evaluated; drives which entities are spawned near the player.
        region: overrides ``config.region``/``task.region`` (used for the Liyue hold-out).
        seed: overrides ``config.seed``.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        task: TaskSpec | None = None,
        region: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.cfg = config or SimConfig()
        self.task = task
        self.task_id: str | None = task.task_id if task is not None else None
        self.region: str = region or (task.region if task is not None else self.cfg.region) or "mondstadt"
        self.seed: int = int(self.cfg.seed if seed is None else seed)
        self.rng = np.random.default_rng(self.seed)

        # Simulation state.
        self.time: float = 0.0
        self.time_of_day: float = 9.0
        self.entities: list[Entity] = []
        self.player = Player()
        self.done_reason: str = ""

        # Presentation state consumed by the renderer.
        self.particles: list[dict[str, Any]] = []
        self.damage_numbers: list[dict[str, Any]] = []
        self.telegraphs: list[dict[str, Any]] = []
        self.ice_patches: list[dict[str, Any]] = []
        self.hit_flash: float = 0.0
        self.attack_anim: float = 0.0
        self.skill_anim: float = 0.0
        self.attack_facing: tuple[float, float] = (0.0, 1.0)
        self.toast: str = ""
        self.toast_ttl: float = 0.0

        # GUI state.
        self.menu: str | None = None
        self.menu_cursor: int = 0
        self.dialogue: dict[str, Any] | None = None
        self.weapon: str = WEAPONS[0]["name"]
        self.inventory: dict[str, int] = {k: v for k, v in STARTING_INVENTORY}
        self.energy: float = 0.0
        self.skill_cd: dict[str, float] = {"E": 0.0, "Q": 0.0}
        self.charged_cd: float = 0.0
        self.normal_cd: float = 0.0
        self.trial_timer: float = 0.0
        self._prev_keys: set[str] = set()

        # Flags: the machine-readable progress record every task is scored from.
        self.flags: dict[str, Any] = {
            "chests_opened": 0,
            "enemies_defeated": 0,
            "npcs_talked": [],
            "monuments_activated": 0,
            "cooked": 0,
            "cooked_dishes": [],
            "teleported": False,
            "weapon_changed": False,
            "boss_defeated": [],
            "anemoculus_collected": 0,
            "time_trial_cleared": False,
            "time_trial_started": False,
            "boulders_broken": 0,
            "thorns_burned": 0,
            "anemograna_collected": 0,
            "wind_current_activated": 0,
            "wind_barrier_broken": 0,
            "climbed": False,
            "ice_created": 0,
            "waypoints_activated": [],
            "quest_stage": 0,
            "chests_total": 0,
        }

        # Language surface.
        self.instruction: str = task.instruction if task is not None else ""
        self.quest_title: str = MISSION_TITLES.get(
            self.task_id or "", QUEST_TITLES.get(self.region, "Adventure")
        )
        self.objective_text: str = "Explore the open world"
        self.objective_progress: tuple[int, int] = (0, 0)
        self.last_event: str = ""

        self._build()
        self._refresh_objectives()

    # -- construction -------------------------------------------------------------------

    def _build(self) -> None:
        """Generate terrain, scenery, landmarks and the task's entity cluster."""
        self.height = self._generate_terrain()
        self._place_player()
        self._scatter_scenery()
        self._place_landmarks()
        if self.task is not None:
            self._populate_task()

    def _noise(self, freq: int, rng: np.random.Generator) -> np.ndarray:
        """Smooth value noise in ``[-1, 1]`` on the heightmap grid, at ``freq`` cells per axis."""
        n = GRID_N
        g = rng.random((freq + 1, freq + 1)) * 2.0 - 1.0
        t = np.linspace(0.0, float(freq), n)
        i0 = np.clip(np.floor(t).astype(np.int64), 0, freq - 1)
        f = t - i0
        f = f * f * (3.0 - 2.0 * f)
        fy = f[:, None]
        fx = f[None, :]
        a = g[np.ix_(i0, i0)]
        b = g[np.ix_(i0, i0 + 1)]
        c = g[np.ix_(i0 + 1, i0)]
        d = g[np.ix_(i0 + 1, i0 + 1)]
        top = a * (1.0 - fx) + b * fx
        bot = c * (1.0 - fx) + d * fx
        return top * (1.0 - fy) + bot * fy

    def _generate_terrain(self) -> np.ndarray:
        """Build the region's heightmap.

        Mondstadt is rolling green hills; Liyue is sharper amber ridges with sparser water,
        which is exactly the distribution shift the OOD evaluation wants.
        """
        rng = self.rng
        if self.region == "liyue":
            ridge = 1.0 - np.abs(self._noise(3, rng))
            h = 13.0 * np.power(np.clip(ridge, 0.0, 1.0), 1.7)
            h += 5.0 * self._noise(6, rng) + 1.8 * self._noise(13, rng)
            # 16% of Liyue sits below the waterline: sparser lakes than Mondstadt.
            water_percentile = 16.0
        else:
            h = 5.6 * self._noise(3, rng) + 2.7 * self._noise(6, rng) + 0.9 * self._noise(13, rng)
            water_percentile = 26.0
        h = h - float(np.percentile(h, water_percentile))
        # Flatten a small plateau at the origin so spawns are never on a cliff face.
        xs = np.linspace(-TERRAIN_HALF, TERRAIN_HALF, GRID_N)
        X, Z = np.meshgrid(xs, xs)
        r = np.hypot(X, Z)
        w = np.clip((r - 4.0) / 9.0, 0.0, 1.0)
        centre = float(h[GRID_N // 2, GRID_N // 2])
        h = h * w + centre * (1.0 - w)
        h = np.clip(h, -9.0, 26.0)
        return h.astype(np.float64)

    # -- terrain queries ----------------------------------------------------------------

    def sample_heights(self, xs: Any, zs: Any) -> np.ndarray:
        """Bilinearly sample terrain height at arbitrary ``(x, z)``.

        Both arguments broadcast, so passing ``xs`` of shape ``(nx,)`` and ``zs`` of shape
        ``(nz, 1)`` yields an ``(nz, nx)`` height grid in one vectorised call -- that is how
        the renderer builds its quad mesh without a per-pixel loop.
        """
        xa = np.asarray(xs, dtype=np.float64)
        za = np.asarray(zs, dtype=np.float64)
        gx = np.clip((xa + TERRAIN_HALF) / GRID_STEP, 0.0, GRID_N - 1.0)
        gz = np.clip((za + TERRAIN_HALF) / GRID_STEP, 0.0, GRID_N - 1.0)
        ix0 = np.clip(np.floor(gx).astype(np.int64), 0, GRID_N - 2)
        iz0 = np.clip(np.floor(gz).astype(np.int64), 0, GRID_N - 2)
        fx = gx - ix0
        fz = gz - iz0
        h00 = self.height[iz0, ix0]
        h10 = self.height[iz0, ix0 + 1]
        h01 = self.height[iz0 + 1, ix0]
        h11 = self.height[iz0 + 1, ix0 + 1]
        top = h00 + (h10 - h00) * fx
        bot = h01 + (h11 - h01) * fx
        return top + (bot - top) * fz

    def height_at(self, x: float, z: float) -> float:
        """Terrain height at one point."""
        return float(self.sample_heights(x, z))

    def slope_at(self, x: float, z: float) -> float:
        """Approximate terrain steepness (``|dh/dx| + |dh/dz|``) at one point."""
        e = 0.5
        dx = self.height_at(x + e, z) - self.height_at(x - e, z)
        dz = self.height_at(x, z + e) - self.height_at(x, z - e)
        return float(abs(dx) / (2.0 * e) + abs(dz) / (2.0 * e))

    def normal_at(self, x: float, z: float) -> tuple[float, float, float]:
        """Unit terrain normal at one point (Y-up, right-handed)."""
        e = 0.6
        dx = (self.height_at(x + e, z) - self.height_at(x - e, z)) / (2.0 * e)
        dz = (self.height_at(x, z + e) - self.height_at(x, z - e)) / (2.0 * e)
        n = np.array([-dx, 1.0, -dz], dtype=np.float64)
        n /= max(float(np.linalg.norm(n)), 1e-9)
        return (float(n[0]), float(n[1]), float(n[2]))

    def ground_height(self, x: float, z: float) -> float:
        """Walkable surface height: terrain, raised to ``y=0`` over any frozen water."""
        g = self.height_at(x, z)
        if g < WATER_LEVEL:
            for patch in self.ice_patches:
                if _dist2d(x, z, patch["x"], patch["z"]) <= patch["r"]:
                    return WATER_LEVEL
        return g

    def in_water_at(self, x: float, z: float) -> bool:
        """True when the terrain at ``(x, z)`` lies below the water plane and is not frozen."""
        return self.ground_height(x, z) < WATER_LEVEL - 0.15

    # -- spawning -----------------------------------------------------------------------

    def add(self, kind: str, name: str = "", x: float = 0.0, z: float = 0.0, y: float | None = None,
            **kw: Any) -> Entity:
        """Create an entity, snap it to the terrain unless it flies, and register it.

        Keyword arguments override :data:`KIND_DEFAULTS`; ``data=...`` attaches gameplay
        state, and ``hp`` also sets ``max_hp`` when only one of the two is given.
        """
        defaults = dict(KIND_DEFAULTS.get(kind, {}))
        data = kw.pop("data", None)
        defaults.update(kw)
        ent = Entity(kind=kind, name=name or kind, x=float(x), z=float(z), **defaults)
        if data:
            ent.data.update(data)
        if y is not None:
            ent.y = float(y)
        elif ent.fly:
            ent.y = float(self.height_at(ent.x, ent.z)) + float(ent.data.get("hover", 1.4))
        else:
            ent.y = float(self.height_at(ent.x, ent.z))
        if "max_hp" not in defaults:
            ent.max_hp = max(ent.hp, 1.0)
        self.entities.append(ent)
        return ent

    def _place_player(self) -> None:
        """Choose a dry, gentle start whose *forward corridor* is walkable land.

        The camera opens looking toward ``+Z`` and every task cluster is laid out ahead of
        the player, so a start that faces a lake would make half the benchmark unplayable.
        The whole candidate set is scored in one vectorised pass.
        """
        radius = np.array([4.0, 6.0, 8.0, 11.0, 14.0], dtype=np.float64)
        ang = np.linspace(0.0, 2.0 * math.pi, 25, endpoint=False)
        X = (radius[:, None] * np.cos(ang)[None, :]).ravel()
        Z = (radius[:, None] * np.sin(ang)[None, :]).ravel()
        dist = np.repeat(radius, ang.size)
        H = np.asarray(self.sample_heights(X, Z), dtype=np.float64)
        slope = (
            np.abs(self.sample_heights(X + 0.5, Z) - self.sample_heights(X - 0.5, Z))
            + np.abs(self.sample_heights(X, Z + 0.5) - self.sample_heights(X, Z - 0.5))
        )
        # Forward corridor: the strip of land the task cluster will occupy.
        ahead = np.array([4.0, 7.0, 10.0, 13.0, 17.0, 21.0, 26.0], dtype=np.float64)
        side = np.array([-7.0, -3.5, 0.0, 3.5, 7.0], dtype=np.float64)
        CX = X[:, None, None] + side[None, None, :]
        CZ = Z[:, None, None] + ahead[None, :, None]
        CH = np.asarray(self.sample_heights(CX, CZ), dtype=np.float64)
        water_frac = (CH < 0.6).mean(axis=(1, 2))
        low = np.clip(0.9 - CH, 0.0, None).mean(axis=(1, 2))
        score = -12.0 * water_frac - 1.6 * low + 0.2 * H - 3.0 * slope - 0.25 * dist
        best_i = int(np.argmax(score))
        bx, bz = float(X[best_i]), float(Z[best_i])
        self.start_position = (bx, bz)
        self.player.x = bx
        self.player.z = bz
        self.player.y = self.height_at(bx, bz) + 0.05
        self.player.yaw = 0.0
        self.player.pitch = -0.05

    def _dry_near(self, x: float, z: float, min_height: float = 0.6) -> tuple[float, float]:
        """Nudge a placement to the nearest dry spot so nothing spawns in a lake."""
        if self.height_at(x, z) >= min_height:
            return (float(x), float(z))
        for r in (1.0, 2.0, 3.0, 4.5, 6.0):
            for k in range(12):
                a = 2.0 * math.pi * k / 12.0
                nx, nz = x + r * math.cos(a), z + r * math.sin(a)
                if self.height_at(nx, nz) >= min_height:
                    return (float(nx), float(nz))
        return (float(x), float(z))

    def front_point(self, distance: float, lateral: float = 0.0, dry: bool = True) -> tuple[float, float]:
        """A point ``distance`` ahead of the player's start, ``lateral`` to its right.

        Task clusters are laid out with this so that everything the instruction mentions is
        inside the initial camera frustum -- the agent must be able to *see* its objective.
        With ``dry=True`` the point is nudged onto land, which keeps objectives reachable.
        """
        x = self.start_position[0] + lateral
        z = self.start_position[1] + distance
        if not dry:
            return (float(x), float(z))
        return self._dry_near(x, z)

    def _scatter_scenery(self) -> None:
        """Scatter trees, rocks and boulders across the region."""
        dense = self.region != "liyue"
        n_trees = 190 if dense else 110
        n_rocks = 70 if dense else 95
        for _ in range(n_trees):
            spot = self._random_land_point(min_height=0.4)
            if spot is None:
                continue
            x, z = spot
            h = self.height_at(x, z)
            ent = self.add("tree", "tree", x, z)
            ent.height = float(5.5 + 2.4 * self.rng.random())
            ent.radius = 0.75 + 0.35 * self.rng.random()
            ent.data["crown"] = float(1.6 + 0.8 * self.rng.random())
            ent.data["tint"] = float(self.rng.random())
            if self.region == "liyue":
                ent.data["palette"] = "dark"
            ent.y = h
        for _ in range(n_rocks):
            spot = self._random_land_point(min_height=-0.4)
            if spot is None:
                continue
            x, z = spot
            ent = self.add("rock", "rock", x, z)
            ent.radius = 0.5 + 0.7 * self.rng.random()
            ent.height = 0.6 + 0.9 * self.rng.random()
            ent.data["tint"] = float(self.rng.random())
        for _ in range(24 if dense else 40):
            spot = self._random_land_point(min_height=0.2)
            if spot is None:
                continue
            x, z = spot
            ent = self.add("boulder", "boulder", x, z)
            ent.data["tint"] = float(self.rng.random())
            ent.data["bonus"] = True

    def _random_land_point(self, min_height: float = 0.0) -> tuple[float, float] | None:
        """Uniform rejection sample over the terrain, avoiding water and the spawn plateau."""
        for _ in range(24):
            x = float(self.rng.uniform(-TERRAIN_HALF + 3.0, TERRAIN_HALF - 3.0))
            z = float(self.rng.uniform(-TERRAIN_HALF + 3.0, TERRAIN_HALF - 3.0))
            if self.height_at(x, z) < min_height:
                continue
            if _dist2d(x, z, self.start_position[0], self.start_position[1]) < 3.0:
                continue
            return x, z
        return None

    def _place_landmarks(self) -> None:
        """Region-specific waypoints, monuments, cooking pot and NPC cast."""
        if self.region == "liyue":
            npcs = (("Bao'er", -4.5), ("Xingxiu", 6.0), ("Yinian", 2.5))
            waypoints = (("Liyue Harbor", 7.0, -9.0), ("Mt. Tianheng", -22.0, 16.0), ("Guili Plains", 26.0, 22.0))
            monuments = (("Electro Monument", 13.0, 4.0, "electro"),)
            pot = (4.5, 3.0)
        else:
            npcs = (("Grace", 3.2), ("Monroe", -5.0), ("Sayid", 6.5))
            waypoints = (("Mondstadt Gate", 6.0, 9.0), ("Windrise", -24.0, 18.0), ("Starfell Lake", 20.0, -16.0))
            monuments = (("Anemo Monument", 12.0, -6.0, "anemo"),)
            pot = (3.0, 5.0)

        for name, lateral in npcs:
            x, z = self.front_point(7.0 + abs(lateral) * 0.3, lateral)
            ent = self.add("npc", name, x, z)
            ent.element = "anemo" if self.region != "liyue" else "geo"
            ent.data["tint"] = float(self.rng.random())
            ent.data["script"] = name.lower()
            ent.data["talked"] = False
        for name, x, z in waypoints:
            ent = self.add("waypoint", name, x, z)
            ent.data["activated"] = False
        for name, x, z, elem in monuments:
            ent = self.add("monument", name, x, z)
            ent.element = elem
            ent.data["activated"] = False
        px, pz = pot
        self.add("cooking_pot", "Cooking Pot", px, pz)
        # A guaranteed pond for the cryo task: carve one in front of the player.
        if self.task_id == "icl_kaeya_freeze":
            self._carve_pond()

    def _carve_pond(self) -> None:
        """Lower the terrain ahead of the player into a shallow pond (cryo task only)."""
        cx, cz = self.front_point(12.0)
        xs = np.linspace(-TERRAIN_HALF, TERRAIN_HALF, GRID_N)
        X, Z = np.meshgrid(xs, xs)
        d = np.hypot(X - cx, Z - cz)
        w = np.clip(1.0 - d / 9.0, 0.0, 1.0)
        self.height = self.height - 4.5 * (w * w)
        self.height = np.clip(self.height, -9.0, 26.0)
        self.carved = (cx, cz)

    # -- task population ----------------------------------------------------------------

    def _populate_task(self) -> None:
        """Spawn exactly the entities the active task needs, within ~25 units of the player."""
        tid = self.task_id or ""
        if tid in ("combat_defeat_and_chest", "combat_domain", "combat_daily", "mission_mondstadt_act1",
                   "mission_liyue_act1"):
            n = {"combat_defeat_and_chest": 3, "combat_domain": 4, "combat_daily": 3}.get(tid, 3)
            for i in range(n):
                x, z = self.front_point(11.0 + 2.2 * i, -3.0 + 2.0 * i)
                variant = ("hilichurl", "slime", "hilichurl", "whopperflower")[i % 4]
                self._spawn_enemy(variant, x, z)
            x, z = self.front_point(20.0, 1.0)
            key = "enemies" if tid != "combat_daily" else "none"
            self._spawn_chest(x, z, locked=(key != "none"), key=key)
        elif tid.startswith("boss_"):
            variant = tid.replace("boss_", "")
            if variant == "hypostasis_electro":
                variant = "electro_hypostasis"
            elif variant == "hypostasis_anemo":
                variant = "anemo_hypostasis"
            x, z = self.front_point(16.0)
            self._spawn_boss(variant, x, z)
            x, z = self.front_point(9.0, 6.0)
            self.add("stone_pillar", "cover pillar", x, z)
        elif tid == "puzzle_anemoculus_wind":
            x, z = self.front_point(12.0)
            wc = self.add("wind_current", "Wind Current", x, z)
            wc.data["active"] = False
            x, z = self.front_point(12.0)
            self.add("anemoculus", "Anemoculus", x, z, y=self.height_at(x, z) + 8.5)
            x, z = self.front_point(22.0, -5.0)
            wc2 = self.add("wind_current", "Wind Current", x, z)
            wc2.data["active"] = False
        elif tid == "puzzle_slime_chest":
            x, z = self.front_point(11.0)
            slime = self._spawn_enemy("slime", x, z)
            slime.fly = True
            slime.y = self.height_at(x, z) + 1.7
            slime.data["hover"] = 1.7
            slime.data["floating"] = True
            x, z = self.front_point(15.0)
            self._spawn_chest(x, z, locked=True, key="enemies")
        elif tid == "puzzle_boulder_chest":
            x, z = self.front_point(9.0)
            b = self.add("boulder", "Boulder", x, z, data={"tint": 0.4, "blocks": True})
            x, z = self.front_point(12.5)
            self._spawn_chest(x, z, locked=True, key="boulder")
            b.data["chest"] = self.entities[-1].name
        elif tid == "puzzle_thorn_chest":
            x, z = self.front_point(9.5)
            th = self.add("thorn_bush", "Thorn Bush", x, z, data={"tint": 0.3})
            x, z = self.front_point(11.5)
            self._spawn_chest(x, z, locked=True, key="pyro")
            th.data["chest"] = self.entities[-1].name
        elif tid == "puzzle_monument":
            elem = "anemo" if self.region != "liyue" else "electro"
            x, z = self.front_point(11.0)
            m = self.add("monument", "Elemental Monument", x, z)
            m.element = elem
            m.data["activated"] = False
        elif tid == "puzzle_time_trial":
            x, z = self.front_point(6.0)
            t = self.add("time_trial", "Time Trial", x, z)
            t.data["started"] = False
            x, z = self.front_point(30.0, 4.0)
            c = self._spawn_chest(x, z, locked=True, key="time_trial")
            c.data["trial_target"] = True
        elif tid.startswith("npc_talk_"):
            wanted = tid.replace("npc_talk_", "")
            moved = False
            for e in self.entities:
                if e.kind != "npc":
                    continue
                e.data["script"] = e.name.lower()
                e.data["talked"] = False
                if e.name.lower() != wanted or moved:
                    continue
                fx, fz = self.front_point(6.0, 0.6)
                e.x, e.z = fx, fz
                e.y = self.height_at(fx, fz)
                moved = True
            if not moved:
                # The region cast does not contain the requested name; spawn it.
                fx, fz = self.front_point(6.0, 0.6)
                ent = self.add("npc", wanted.capitalize(), fx, fz)
                ent.data["script"] = wanted
                ent.data["talked"] = False
        elif tid == "gui_cook_sweet_madame":
            x, z = self.front_point(4.5)
            self.add("cooking_pot", "Cooking Pot", x, z)
        elif tid == "gui_teleport_waypoint":
            x, z = self.front_point(5.0, 2.0)
            ent = self.add("waypoint", "Teleport Waypoint", x, z)
            ent.data["activated"] = False
        elif tid == "icl_climb_pillar":
            x, z = self.front_point(9.0, 4.0)
            pillar = self.add("stone_pillar", "Stone Pillar", x, z)
            pillar.height = 12.0
            # The Anemoculus hangs where a fall off the pillar's forward-left edge passes.
            # Solving the ballistics keeps the task solvable "climb, then walk off".
            lateral_off = 1.2
            forward_off = 3.6
            path = math.hypot(lateral_off, forward_off)
            t_fall = max((path - pillar.radius) / WALK_SPEED, 0.05)
            drop = 0.5 * GRAVITY * t_fall * t_fall
            ox, oz = self.front_point(9.0 + forward_off, 4.0 - lateral_off)
            self.add("anemoculus", "Anemoculus", ox, oz, y=pillar.y + pillar.height - drop)
        elif tid == "icl_kaeya_freeze":
            # The Anemoculus floats over the carved pond, so it can only be reached on ice.
            cx, cz = getattr(self, "carved", self.front_point(12.0, dry=False))
            x, z = cx, cz + 1.5
            self.add("anemoculus", "Anemoculus", x, z, y=WATER_LEVEL + 1.35)
        elif tid == "icl_wind_barrier":
            x, z = self.front_point(7.0)
            self.add("anemograna", "Wind Anemograna", x, z, y=self.height_at(x, z) + 1.1)
            x, z = self.front_point(14.0)
            wc = self.add("wind_current", "Wind Current", x, z)
            wc.data["active"] = False
            x, z = self.front_point(23.0)
            barrier = self.add("wind_barrier", "Wind Barrier", x, z, data={"broken": False})
            cx, cz = self.front_point(23.0)
            chest = self._spawn_chest(cx, cz, locked=True, key="barrier")
            chest.data["barrier"] = barrier.name
            barrier.data["chest"] = chest.name
        elif tid == "gui_change_weapon":
            pass
        else:  # pragma: no cover - registry and this method are kept in sync by tests
            pass
        self.flags["chests_total"] = sum(1 for e in self.entities if e.kind == "chest")

    def _region_npcs(self) -> tuple[tuple[str, float, float], ...]:
        """The cast of the current region as ``(name, x, z)`` triples."""
        if self.region == "liyue":
            names = ("Bao'er", "Xingxiu", "Yinian")
        else:
            names = ("Grace", "Monroe", "Sayid")
        out: list[tuple[str, float, float]] = []
        for i, name in enumerate(names):
            x, z = self.front_point(6.5 + 1.5 * i, -4.0 + 4.0 * i)
            out.append((name, x, z))
        return tuple(out)

    def _spawn_enemy(self, variant: str, x: float, z: float) -> Entity:
        """Create one hostile creature at ``(x, z)``."""
        stats = ENEMY_TYPES[variant]
        ent = self.add(
            "enemy",
            variant,
            x,
            z,
            hp=stats["hp"],
            radius=0.75 if variant != "whopperflower" else 0.85,
            height=1.6 if variant != "slime" else 1.1,
            data={
                "variant": variant,
                "speed": stats["speed"],
                "damage": stats["damage"],
                "reach": stats["reach"],
                "aggro": stats["aggro"],
                "atk_cd": float(self.rng.uniform(0.6, 2.4)),
                "telegraph": 0.0,
                "flash": 0.0,
                "home": (x, z),
                "shield": 0.0,
                "wander": float(self.rng.uniform(0.0, 6.28)),
            },
        )
        ent.variant = variant
        ent.element = str(stats["element"])
        return ent

    def _spawn_boss(self, variant: str, x: float, z: float) -> Entity:
        """Create the episode's boss at ``(x, z)``."""
        stats = BOSS_TYPES[variant]
        ent = self.add(
            "boss",
            variant,
            x,
            z,
            hp=stats["hp"],
            radius=2.4,
            height=4.6 if variant != "stormterror" else 3.6,
            fly=variant == "stormterror",
            y=None,
            data={
                "variant": variant,
                "damage": stats["damage"],
                "weak": stats["weak"],
                "speed": stats["speed"],
                "phase": "shield" if "hypostasis" in variant else "fly",
                "phase_t": 0.0,
                "aoe_cd": 2.5,
                "flash": 0.0,
                "home": (x, z),
                "hover": 9.0,
                "atk_cd": 3.0,
                "telegraph": 0.0,
                "swoop": 0.0,
            },
        )
        ent.variant = variant
        ent.element = "electro" if "electro" in variant else "anemo"
        if variant == "stormterror":
            ent.y = self.height_at(x, z) + 9.0
        return ent

    def _spawn_chest(self, x: float, z: float, locked: bool = False, key: str = "none") -> Entity:
        """Create a chest at ``(x, z)``, optionally locked behind a puzzle or enemies."""
        name = f"Chest {self.flags['chests_total'] + 1}"
        ent = self.add(
            "chest",
            name,
            x,
            z,
            data={"locked": locked, "key": key, "opened": False, "tint": float(self.rng.random())},
        )
        return ent

    # -- public queries -----------------------------------------------------------------

    def entities_of(self, kind: str) -> list[Entity]:
        """All entities of one kind, alive or not."""
        return [e for e in self.entities if e.kind == kind]

    def alive_of(self, kind: str) -> list[Entity]:
        """All *living* entities of one kind."""
        return [e for e in self.entities if e.kind == kind and e.alive]

    def nearby_entities(self, radius: float = 25.0) -> tuple[Mapping[str, Any], ...]:
        """Compact read-out of everything around the player, nearest first.

        Returns:
            A tuple of dicts with ``name``, ``kind``, ``position``, ``distance`` and ``hp``.
        """
        p = self.player
        out: list[Mapping[str, Any]] = []
        for e in self.entities:
            if not e.alive:
                continue
            d = math.sqrt((e.x - p.x) ** 2 + (e.y - p.y) ** 2 + (e.z - p.z) ** 2)
            if d > radius:
                continue
            out.append(
                {
                    "name": e.name,
                    "kind": e.kind,
                    "position": (round(e.x, 2), round(e.y, 2), round(e.z, 2)),
                    "distance": round(d, 2),
                    "hp": round(e.hp, 1),
                }
            )
        out.sort(key=lambda m: float(m["distance"]))
        return tuple(out)

    def interact_prompt(self) -> str | None:
        """The ``[F] ...`` prompt for the nearest interactable, or a monument hint."""
        target = self._interact_target()
        if target is not None:
            if target.kind == "chest":
                if target.data.get("opened"):
                    return None
                if target.data.get("locked"):
                    key = target.data.get("key", "none")
                    hints = {
                        "pyro": "Chest is wrapped in thorns (needs Pyro)",
                        "boulder": "Chest is blocked by a boulder (break it)",
                        "enemies": "Chest is guarded (defeat the enemies)",
                        "time_trial": "Chest is sealed (start the Time Trial)",
                        "barrier": "Chest is inside the Wind Barrier",
                    }
                    return hints.get(str(key), "Chest is locked")
                return "[F] Open Chest"
            if target.kind == "npc":
                return f"[F] Talk to {target.name}"
            if target.kind == "waypoint":
                return "[F] Activate Waypoint" if not target.data.get("activated") else f"[M] Map: {target.name}"
            if target.kind == "cooking_pot":
                return "[F] Cook"
            if target.kind == "time_trial":
                return "[F] Start Time Trial"
        for m in self.entities:
            if m.kind != "monument" or not m.alive or m.data.get("activated"):
                continue
            if _dist2d(m.x, m.z, self.player.x, self.player.z) < 4.0:
                return f"[E] Activate Monument ({m.element})"
        return None

    # -- stepping -----------------------------------------------------------------------

    def step(self, action: Action, dt: float = 1.0 / 30.0) -> None:
        """Advance physics, AI and game logic by ``dt`` seconds (default 1/30).

        Args:
            action: the keys/mouse for this control tick.  Movement keys are *held*; menu,
                dialogue and one-shot action keys are edge-triggered internally.
            dt: tick length, clamped to ``[1e-4, 0.1]`` so a stalled caller cannot tunnel
                the player through the terrain.
        """
        dt = float(min(max(dt, 1e-4), 0.1))
        keys = set(action.keys)
        pressed = keys - self._prev_keys
        self._prev_keys = keys

        self.time += dt
        self.time_of_day = (self.time_of_day + dt * 24.0 / max(self.cfg.day_length_s, 1e-3)) % 24.0
        self._decay(dt)

        busy = self.dialogue is not None or self.menu is not None
        if busy:
            self._handle_gui(action, keys, pressed, dt)
        else:
            self._handle_look(action, keys, dt)
            if self.player.alive:
                self._handle_actions(keys, pressed, dt)
                self._handle_movement(keys, dt)
        self._integrate(dt)
        self._update_world(dt, paused=busy)
        self._refresh_objectives()

    # -- input --------------------------------------------------------------------------

    def _handle_look(self, action: Action, keys: set[str], dt: float) -> None:
        """Mouse look plus keyboard camera turning (arrows are free outside menus)."""
        p = self.player
        p.yaw = _wrap_angle(p.yaw + action.mouse_dx * MOUSE_YAW_GAIN)
        p.pitch = float(np.clip(p.pitch - action.mouse_dy * MOUSE_PITCH_GAIN, -1.2, 1.2))
        turn = 0.0
        if "RIGHT" in keys:
            turn += 1.0
        if "LEFT" in keys:
            turn -= 1.0
        p.yaw = _wrap_angle(p.yaw + turn * KEY_TURN_SPEED * dt)
        pitch = 0.0
        if "UP" in keys:
            pitch += 1.0
        if "DOWN" in keys:
            pitch -= 1.0
        p.pitch = float(np.clip(p.pitch + pitch * KEY_PITCH_SPEED * dt, -1.2, 1.2))

    def _handle_gui(self, action: Action, keys: set[str], pressed: set[str], dt: float) -> None:
        """Route input to the open dialogue box or menu panel."""
        if self.dialogue is not None:
            self._dialogue_input(pressed)
            return
        entries = self.menu_entries()
        n = max(1, len(entries))
        if "UP" in pressed or "LEFT" in pressed:
            self.menu_cursor = (self.menu_cursor - 1) % n
        if "DOWN" in pressed or "RIGHT" in pressed:
            self.menu_cursor = (self.menu_cursor + 1) % n
        if "ESC" in pressed or "M" in pressed or "B" in pressed or "C" in pressed:
            self._close_menu()
        elif "ENTER" in pressed:
            self._menu_confirm()

    def _dialogue_input(self, pressed: set[str]) -> None:
        """Pick a dialogue option with ``1``/``2``/``3`` or ``UP``/``DOWN`` + ``ENTER``."""
        dlg = self.dialogue
        if dlg is None:
            return
        options: tuple[str, ...] = tuple(dlg["options"])
        if "ESC" in pressed:
            self.dialogue = None
            return
        if "UP" in pressed:
            dlg["cursor"] = (int(dlg["cursor"]) - 1) % len(options)
        if "DOWN" in pressed:
            dlg["cursor"] = (int(dlg["cursor"]) + 1) % len(options)
        choice: int | None = None
        for i, key in enumerate(("1", "2", "3", "4")):
            if key in pressed and i < len(options):
                choice = i
                break
        if "ENTER" in pressed:
            choice = int(dlg["cursor"])
        if choice is None:
            return
        npc: Entity | None = dlg.get("npc")
        if choice == int(dlg["correct"]):
            speaker = str(dlg["speaker"]).lower()
            if speaker not in self.flags["npcs_talked"]:
                self.flags["npcs_talked"].append(speaker)
            if npc is not None:
                npc.data["talked"] = True
            self.last_event = f"Talked to {dlg['speaker']}"
            self._toast(str(dlg.get("reply", "Quest updated.")))
            self._burst(npc.x, npc.y + 1.6, npc.z, 1.2, (255, 240, 170), n=14)
            self.dialogue = None
        else:
            dlg["line"] = f"{dlg['speaker']}: ...that is not what I asked. Try again."
            dlg["cursor"] = 0

    def _menu_confirm(self) -> None:
        """Activate the highlighted menu entry."""
        entries = self.menu_entries()
        if not entries:
            return
        idx = min(self.menu_cursor, len(entries) - 1)
        if self.menu == "map":
            wps = self.alive_of("waypoint")
            if not wps:
                self._toast("No waypoints discovered")
                self._close_menu()
                return
            wp = wps[idx % len(wps)]
            self.player.x = wp.x + 1.6
            self.player.z = wp.z + 1.6
            self.player.y = self.height_at(self.player.x, self.player.z) + 0.1
            self.player.vx = self.player.vy = self.player.vz = 0.0
            self.flags["teleported"] = True
            self.last_event = f"Teleported to {wp.name}"
            self._toast(f"Teleported to {wp.name}")
            self._burst(self.player.x, self.player.y + 1.0, self.player.z, 2.2, (150, 220, 255), n=18)
            self._close_menu()
        elif self.menu == "weapon":
            name = entries[idx]
            self.weapon = name
            self.flags["weapon_changed"] = True
            self.last_event = f"Equipped {name}"
            self._toast(f"Equipped {name}")
            self._close_menu()
        elif self.menu == "cook":
            recipe = entries[idx]
            self.flags["cooked"] = int(self.flags["cooked"]) + 1
            key = recipe.lower().replace(" ", "_").replace("'", "")
            dishes = self.flags["cooked_dishes"]
            if key not in dishes:
                dishes.append(key)
            self.inventory[recipe] = self.inventory.get(recipe, 0) + 1
            self.last_event = f"Cooked {recipe}"
            self._toast(f"Cooked {recipe}")
            self._close_menu()
        elif self.menu == "bag":
            entry = entries[idx]
            item = entry.rsplit(" x", 1)[0]
            if self.inventory.get(item, 0) > 0 and item in ("Sweet Madame", "Apple"):
                self.inventory[item] -= 1
                self.player.hp = min(self.player.max_hp, self.player.hp + 25.0)
                self._toast(f"Used {item} (+25 HP)")
            else:
                self._toast(f"{item}: nothing happens")
            self._close_menu()

    def _close_menu(self) -> None:
        """Close whatever panel is open."""
        self.menu = None
        self.menu_cursor = 0

    def menu_entries(self) -> list[str]:
        """Labels for the currently open menu (empty when none is open)."""
        if self.menu == "map":
            return [f"Teleport: {w.name}" for w in self.alive_of("waypoint")]
        if self.menu == "weapon":
            return [str(w["name"]) for w in WEAPONS]
        if self.menu == "bag":
            return [f"{k} x{v}" for k, v in self.inventory.items()]
        if self.menu == "cook":
            return list(RECIPES)
        return []

    def _handle_actions(self, keys: set[str], pressed: set[str], dt: float) -> None:
        """One-shot inputs: attacks, skills, interact, party switch and menu opening."""
        p = self.player
        if "SPACE" in pressed and p.grounded and not p.in_water:
            p.vy = JUMP_SPEED
            p.grounded = False
        if "1" in pressed:
            p.party_index = 0
        if "2" in pressed:
            p.party_index = 1
        if "3" in pressed:
            p.party_index = 2
        if "4" in pressed:
            p.party_index = 3
        if "M" in pressed:
            self.menu = "map"
            self.menu_cursor = 0
        elif "C" in pressed:
            self.menu = "weapon"
            self.menu_cursor = 0
        elif "B" in pressed:
            self.menu = "bag"
            self.menu_cursor = 0
        if "F" in pressed:
            self._do_interact()
        if "J" in keys and self.normal_cd <= 0.0:
            self._normal_attack()
        if "K" in keys and self.charged_cd <= 0.0:
            self._charged_attack()
        if "E" in keys and self.skill_cd["E"] <= 0.0:
            self._elemental_skill()
        if "Q" in keys and self.skill_cd["Q"] <= 0.0 and self.energy >= BURST_COST:
            self._elemental_burst()

    def _handle_movement(self, keys: set[str], dt: float) -> None:
        """Camera-relative locomotion, sprint, swim and climb."""
        p = self.player
        fx, fz = math.sin(p.yaw), math.cos(p.yaw)
        rx, rz = math.cos(p.yaw), -math.sin(p.yaw)
        mx = mz = 0.0
        if "W" in keys:
            mx += fx
            mz += fz
        if "S" in keys:
            mx -= fx
            mz -= fz
        if "D" in keys:
            mx += rx
            mz += rz
        if "A" in keys:
            mx -= rx
            mz -= rz
        norm = math.hypot(mx, mz)
        if norm > 1e-6:
            mx /= norm
            mz /= norm
        p.in_water = self.in_water_at(p.x, p.z) and p.y <= WATER_LEVEL + 0.3
        sprint = "SHIFT" in keys and p.stamina > 1.0 and not p.in_water and norm > 0.0
        p.sprinting = sprint
        if sprint:
            p.stamina = max(0.0, p.stamina - SPRINT_STAMINA_DRAIN * dt)
        elif not p.climbing:
            p.stamina = min(p.max_stamina, p.stamina + STAMINA_REGEN * dt)
        speed = SPRINT_SPEED if sprint else WALK_SPEED
        if p.in_water:
            speed = SWIM_SPEED
            p.stamina = max(0.0, p.stamina - SWIM_STAMINA_DRAIN * dt)
        if not p.grounded:
            speed *= 0.85
        p.vx = mx * speed
        p.vz = mz * speed

        # Climbing: hold W against a pillar to scale it.
        p.climbing = False
        for e in self.entities:
            if e.kind != "stone_pillar" or not e.alive:
                continue
            if _dist2d(e.x, e.z, p.x, p.z) <= e.radius + 0.85 and p.y < e.y + e.height - 0.4:
                if "W" in keys and norm > 0.0:
                    p.climbing = True
                    p.vy = CLIMB_SPEED
                    p.stamina = max(0.0, p.stamina - CLIMB_STAMINA_DRAIN * dt)
                    p.grounded = False
                    if p.y >= e.y + e.height - 1.5:
                        if not self.flags["climbed"]:
                            self.last_event = f"Climbed {e.name}"
                        self.flags["climbed"] = True
                break

    # -- combat -------------------------------------------------------------------------

    def _facing(self) -> tuple[float, float]:
        """The player's XZ forward unit vector."""
        return (math.sin(self.player.yaw), math.cos(self.player.yaw))

    def _combat_targets(self) -> list[Entity]:
        """Every hostile entity in the world."""
        return [e for e in self.entities if e.kind in ("enemy", "boss") and e.alive]

    def _normal_attack(self) -> None:
        """``J``: a short cone swing in front of the player."""
        self.normal_cd = ATTACK_CD
        self.attack_anim = 0.18
        fx, fz = self._facing()
        self.attack_facing = (fx, fz)
        p = self.player
        for e in self._combat_targets():
            d = _dist2d(e.x, e.z, p.x, p.z)
            if d > ATTACK_RANGE + e.radius:
                continue
            if abs((e.y + e.height * 0.5) - (p.y + 0.9)) > 1.5 + e.height * 0.5:
                continue
            ang = math.acos(max(-1.0, min(1.0, ((e.x - p.x) * fx + (e.z - p.z) * fz) / max(d, 1e-6))))
            if ang > ATTACK_HALF_ARC and d > 1.2:
                continue
            self.damage_entity(e, ATTACK_DAMAGE * self._weapon_multiplier(), "physical", crit=False)
        for bush in self.alive_of("thorn_bush"):
            if _dist2d(bush.x, bush.z, p.x, p.z) <= ATTACK_RANGE + bush.radius:
                self.damage_entity(bush, ATTACK_DAMAGE, "physical")
        self.energy = min(BURST_COST, self.energy + ENERGY_PER_HIT)
        self._add_damage_overlay(fx, fz)

    def _charged_attack(self) -> None:
        """``K``: a wide heavy swing that also shatters boulders."""
        self.charged_cd = CHARGED_CD
        self.attack_anim = 0.32
        fx, fz = self._facing()
        self.attack_facing = (fx, fz)
        p = self.player
        for e in self._combat_targets():
            if _dist2d(e.x, e.z, p.x, p.z) <= CHARGED_RANGE + e.radius:
                self.damage_entity(e, CHARGED_DAMAGE * self._weapon_multiplier(), "physical", crit=True)
        for b in self.alive_of("boulder"):
            if _dist2d(b.x, b.z, p.x, p.z) <= CHARGED_RANGE + b.radius:
                self._break_boulder(b)
        for bush in self.alive_of("thorn_bush"):
            if _dist2d(bush.x, bush.z, p.x, p.z) <= CHARGED_RANGE + bush.radius:
                self.damage_entity(bush, CHARGED_DAMAGE, "pyro")
        self.energy = min(BURST_COST, self.energy + ENERGY_PER_HIT * 2.0)
        self._burst(p.x + fx * 1.4, p.y + 0.7, p.z + fz * 1.4, 2.4, (255, 235, 190), n=16)

    def _elemental_skill(self) -> None:
        """``E``: the active character's elemental skill.

        Pyro burns thorns, cryo freezes water into ice, anemo drives wind currents and breaks
        anemo shields, electro overloads an electro shield.
        """
        p = self.player
        element = PARTY[p.party_index][1]
        self.skill_cd["E"] = SKILL_CD
        self.skill_anim = 0.4
        fx, fz = self._facing()
        cx, cz = p.x + fx * 2.2, p.z + fz * 2.2
        rgb = ELEMENT_RGB.get(element, (255, 255, 255))
        self._burst(cx, p.y + 1.0, cz, 3.4, rgb, n=26)
        self.energy = min(BURST_COST, self.energy + 12.0)

        radius = 4.2
        for e in self._combat_targets():
            if _dist2d(e.x, e.z, p.x, p.z) > radius + e.radius:
                continue
            dmg = 26.0
            if e.kind == "boss" and e.data.get("weak") == element:
                dmg *= 2.2
                if e.data.get("phase") == "shield":
                    e.data["phase"] = "core"
                    e.data["phase_t"] = 0.0
                    self._toast(f"{element.title()} broke the shield!")
            if element == "anemo" and float(e.data.get("shield", 0.0)) > 0.0:
                e.data["shield"] = 0.0
            self.damage_entity(e, dmg, element)
        if element == "pyro":
            for bush in self.alive_of("thorn_bush"):
                if _dist2d(bush.x, bush.z, p.x, p.z) <= radius + 2.0:
                    self._burn_thorns(bush)
        elif element == "cryo":
            self._freeze_surface(cx, cz, radius)
        elif element == "anemo":
            for wc in self.alive_of("wind_current"):
                if _dist2d(wc.x, wc.z, p.x, p.z) <= 14.0 and not wc.data.get("active"):
                    wc.data["active"] = True
                    self.flags["wind_current_activated"] = int(self.flags["wind_current_activated"]) + 1
                    self._toast("Wind Current activated")
        for m in self.entities:
            if m.kind != "monument" or not m.alive or m.data.get("activated"):
                continue
            if _dist2d(m.x, m.z, p.x, p.z) <= radius:
                if m.element == element:
                    m.data["activated"] = True
                    self.flags["monuments_activated"] = int(self.flags["monuments_activated"]) + 1
                    self.last_event = f"Activated {m.name}"
                    self._toast(f"{m.name} activated")
                    self._burst(m.x, m.y + m.height, m.z, 2.4, ELEMENT_RGB[str(m.element)], n=24)
                else:
                    self._toast(f"Wrong element: it needs {m.element}")

    def _elemental_burst(self) -> None:
        """``Q``: spend a full energy meter on a large elemental nova."""
        p = self.player
        element = PARTY[p.party_index][1]
        self.energy = 0.0
        self.skill_cd["Q"] = 1.0
        rgb = ELEMENT_RGB.get(element, (255, 255, 255))
        self._burst(p.x, p.y + 1.0, p.z, BURST_RADIUS, rgb, n=40)
        self._toast(f"{element.title()} Burst!")
        for e in self._combat_targets():
            if _dist2d(e.x, e.z, p.x, p.z) <= BURST_RADIUS + e.radius:
                self.damage_entity(e, BURST_DAMAGE, element, crit=True)
        if element == "pyro":
            for bush in self.alive_of("thorn_bush"):
                if _dist2d(bush.x, bush.z, p.x, p.z) <= BURST_RADIUS:
                    self._burn_thorns(bush)
        elif element == "cryo":
            self._freeze_surface(p.x, p.z, BURST_RADIUS)

    def _weapon_multiplier(self) -> float:
        """Damage scale from the equipped weapon (``1.0`` for the starting blade)."""
        for w in WEAPONS:
            if w["name"] == self.weapon:
                return 0.8 + float(w["atk"]) / 100.0
        return 1.0

    def damage_entity(self, e: Entity, amount: float, element: str = "physical", crit: bool = False) -> None:
        """Apply damage to an entity, handling shields, death, numbers and particles.

        Args:
            e: target entity (enemy, boss, boulder or thorn bush).
            amount: raw damage before shield and phase multipliers.
            element: element of the hit, used for boss shield interactions.
            crit: draw the damage number in the "critical" style.
        """
        if not e.alive:
            return
        mult = 1.0
        if e.kind == "boss":
            phase = str(e.data.get("phase", "shield"))
            if phase in ("shield", "fly"):
                mult = 0.25
            elif phase in ("core", "swoop"):
                mult = 1.5
            if e.data.get("weak") == element:
                mult *= 1.6
        if e.kind == "boulder" and element != "charged":
            mult = 0.0
        if e.kind == "boulder" and crit:
            self._break_boulder(e)
            return
        dmg = max(0.0, float(amount) * mult)
        if dmg > 0.0:
            e.hp -= dmg
            e.data["flash"] = 0.16
            self.damage_numbers.append(
                {
                    "value": dmg,
                    "text": f"{dmg:.0f}",
                    "x": e.x,
                    "y": e.y + e.height * 0.9 + 0.3,
                    "z": e.z,
                    "age": 0.0,
                    "ttl": 1.0,
                    "color": (255, 170, 60) if crit else (255, 255, 255),
                    "drift": float(self.rng.uniform(-0.35, 0.35)),
                }
            )
            rgb = ELEMENT_RGB.get(element if element != "charged" else "physical", (255, 255, 255))
            self._burst(e.x, e.y + e.height * 0.6, e.z, 0.7, rgb, n=8)
        if e.hp <= 0.0:
            self._kill(e)

    def _kill(self, e: Entity) -> None:
        """Retire a dead entity and record the corresponding progress flag."""
        e.alive = False
        e.hp = 0.0
        if e.kind == "enemy":
            self.flags["enemies_defeated"] = int(self.flags["enemies_defeated"]) + 1
            self.last_event = f"Defeated {e.variant}"
            self._burst(e.x, e.y + 0.8, e.z, 1.8, (255, 210, 120), n=22)
            self._unlock_guarded_chest()
        elif e.kind == "boss":
            defeated = self.flags["boss_defeated"]
            if e.variant not in defeated:
                defeated.append(e.variant)
            self.last_event = f"Defeated {e.variant}"
            self._burst(e.x, e.y + 1.5, e.z, 5.0, (255, 240, 160), n=60)
            self._toast(f"{e.variant.replace('_', ' ').title()} defeated!")
        elif e.kind == "thorn_bush":
            self.flags["thorns_burned"] = int(self.flags["thorns_burned"]) + 1
            self._burst(e.x, e.y + 0.6, e.z, 1.6, (255, 150, 70), n=20)
            self._unlock_chest_of(e)
        elif e.kind == "anemoculus":
            self.flags["anemoculus_collected"] = int(self.flags["anemoculus_collected"]) + 1
            self.last_event = "Collected Anemoculus"
            self._toast("Anemoculus collected!")
            self._burst(e.x, e.y, e.z, 1.8, (150, 220, 255), n=24)
        elif e.kind == "anemograna":
            self.flags["anemograna_collected"] = int(self.flags["anemograna_collected"]) + 1
            self._toast("Wind Anemograna collected")
            for wc in self.alive_of("wind_current"):
                if not wc.data.get("active"):
                    wc.data["active"] = True
                    self.flags["wind_current_activated"] = int(self.flags["wind_current_activated"]) + 1
                    break
            self._burst(e.x, e.y, e.z, 1.2, (150, 240, 200), n=16)
        elif e.kind == "boulder":
            self.flags["boulders_broken"] = int(self.flags["boulders_broken"]) + 1
            self._burst(e.x, e.y + 0.6, e.z, 1.8, (190, 180, 160), n=22)
            self._unlock_chest_of(e)

    def _break_boulder(self, b: Entity) -> None:
        """Shatter a boulder with a charged attack and unlock whatever it guards."""
        if not b.alive:
            return
        b.hp = 0.0
        self.last_event = "Boulder broken"
        self._kill(b)

    def _unlock_chest_of(self, source: Entity) -> None:
        """Unlock the chest tied to ``source`` (a boulder or thorn bush)."""
        name = source.data.get("chest")
        if not name:
            return
        for c in self.entities:
            if c.kind == "chest" and c.name == name:
                c.data["locked"] = False
                c.data["key"] = "none"
                self._toast(f"{c.name} unlocked")

    def _unlock_guarded_chest(self) -> None:
        """Guarded chests pop open once their guard detail is dead."""
        guards = [e for e in self._combat_targets() if _dist2d(
            e.x, e.z, self.player.x, self.player.z) < 40.0]
        if guards:
            return
        for c in self.alive_of("chest"):
            if c.data.get("key") == "enemies" and c.data.get("locked"):
                c.data["locked"] = False
                c.data["key"] = "none"
                self._toast(f"{c.name} unlocked")

    def _burn_thorns(self, bush: Entity) -> None:
        """Ignite a thorn bush (pyro skill / burst / charged swing)."""
        if not bush.alive:
            return
        self.damage_entity(bush, 999.0, "pyro")

    def _freeze_surface(self, cx: float, cz: float, radius: float) -> None:
        """Freeze a water surface into walkable ice (cryo skill)."""
        created = 0
        for dx, dz in ((0.0, 0.0), (radius * 0.5, 0.0), (-radius * 0.5, 0.0), (0.0, radius * 0.5),
                       (0.0, -radius * 0.5), (radius * 0.4, radius * 0.4), (-radius * 0.4, -radius * 0.4)):
            x, z = cx + dx, cz + dz
            if self.height_at(x, z) < WATER_LEVEL - 0.05:
                self.ice_patches.append({"x": x, "z": z, "r": radius * 0.62, "age": 0.0, "ttl": 40.0})
                created += 1
        if created:
            self.flags["ice_created"] = int(self.flags["ice_created"]) + 1
            self._toast("Water frozen into ice")
            self._burst(cx, WATER_LEVEL + 0.2, cz, radius, (170, 230, 255), n=22, flat=True)

    # -- interaction --------------------------------------------------------------------

    def _interact_target(self) -> Entity | None:
        """Nearest interactable within :data:`INTERACT_RANGE`, or ``None``."""
        p = self.player
        best: tuple[float, Entity] | None = None
        for e in self.entities:
            if not e.alive or not e.interactable:
                continue
            if e.kind == "chest" and e.data.get("opened"):
                continue
            if e.kind == "waypoint" and e.data.get("activated"):
                continue
            d = _dist2d(e.x, e.z, p.x, p.z)
            if d > INTERACT_RANGE + e.radius * 0.8:
                continue
            if abs(e.y - p.y) > 2.5:
                continue
            if best is None or d < best[0]:
                best = (d, e)
        return best[1] if best is not None else None

    def _do_interact(self) -> None:
        """``F``: open chests, talk, activate waypoints, start trials, cook."""
        target = self._interact_target()
        if target is None:
            return
        if target.kind == "chest":
            self._open_chest(target)
        elif target.kind == "npc":
            self._open_dialogue(target)
        elif target.kind == "waypoint":
            target.data["activated"] = True
            if target.name not in self.flags["waypoints_activated"]:
                self.flags["waypoints_activated"].append(target.name)
            self.last_event = f"Activated waypoint {target.name}"
            self._toast(f"Waypoint {target.name} activated")
            self._burst(target.x, target.y + 1.5, target.z, 2.0, (140, 200, 255), n=18)
        elif target.kind == "cooking_pot":
            self.menu = "cook"
            self.menu_cursor = 0
            self._toast("Cooking pot: choose a recipe")
        elif target.kind == "time_trial":
            target.data["started"] = True
            self.trial_timer = TIME_TRIAL_SECONDS
            self.flags["time_trial_started"] = True
            self.last_event = "Time Trial started"
            self._toast("Time Trial started!")

    def _open_chest(self, chest: Entity) -> None:
        """Open a chest if it is unlocked, recording progress."""
        if chest.data.get("opened"):
            return
        if chest.data.get("locked"):
            key = str(chest.data.get("key", "none"))
            if key == "enemies" and not self._combat_targets():
                chest.data["locked"] = False
                chest.data["key"] = "none"
            else:
                self._toast("The chest is locked")
                return
        chest.data["opened"] = True
        self.flags["chests_opened"] = int(self.flags["chests_opened"]) + 1
        if chest.data.get("trial_target") and self.trial_timer > 0.0:
            self.flags["time_trial_cleared"] = True
            self._toast("Time Trial cleared!")
        self.last_event = f"Opened {chest.name}"
        self._toast(f"Opened {chest.name}!")
        self._burst(chest.x, chest.y + 0.8, chest.z, 1.8, (255, 225, 130), n=26)

    def _open_dialogue(self, npc: Entity) -> None:
        """Open the NPC's dialogue box."""
        script = DIALOGUES.get(str(npc.data.get("script", npc.name.lower())))
        if script is None:  # pragma: no cover - every spawned NPC has a script
            script = {
                "line": "Safe travels, traveler.",
                "options": ("Thanks.", "Goodbye."),
                "correct": 0,
                "reply": "May the wind guide you.",
            }
        self.dialogue = {
            "speaker": npc.name,
            "line": f"{npc.name}: {script['line']}",
            "options": tuple(script["options"]),
            "correct": int(script["correct"]),
            "reply": script["reply"],
            "cursor": 0,
            "npc": npc,
        }

    # -- physics ------------------------------------------------------------------------

    def _integrate(self, dt: float) -> None:
        """Apply gravity, wind lift, buoyancy, collisions and terrain following."""
        p = self.player
        if not p.alive:
            p.vx = p.vz = 0.0
        if not p.grounded and not p.climbing:
            p.vy -= GRAVITY * dt
        # Wind currents and updrafts.
        for wc in self.alive_of("wind_current"):
            if not wc.data.get("active"):
                continue
            if _dist2d(wc.x, wc.z, p.x, p.z) <= wc.radius + PLAYER_RADIUS:
                if wc.y - 0.5 <= p.y <= wc.y + wc.height:
                    p.vy = WIND_LIFT_SPEED
                    p.grounded = False
        if p.in_water and not p.climbing:
            target = WATER_LEVEL - 0.35
            p.vy += (target - p.y) * 12.0 * dt
            p.vy = float(np.clip(p.vy, -3.0, 3.0))
        p.x += p.vx * dt
        p.z += p.vz * dt
        p.y += p.vy * dt
        p.x = float(np.clip(p.x, -TERRAIN_HALF + 1.0, TERRAIN_HALF - 1.0))
        p.z = float(np.clip(p.z, -TERRAIN_HALF + 1.0, TERRAIN_HALF - 1.0))
        self._resolve_collisions()
        ground = self.ground_height(p.x, p.z)
        if p.y <= ground + 0.02:
            if p.vy <= 0.0:
                p.y = ground
                p.vy = 0.0
                p.grounded = True
                p.climbing = False
        else:
            p.grounded = False
        # Collectibles and barriers are touch-triggered.
        for e in list(self.entities):
            if not e.alive:
                continue
            if e.kind in ("anemoculus", "anemograna"):
                pickup = 2.6 if e.kind == "anemoculus" else 1.7
                if math.dist((e.x, e.y, e.z), p.position) < pickup:
                    self._kill(e)
            elif e.kind == "wind_barrier" and not e.data.get("broken"):
                if _dist2d(e.x, e.z, p.x, p.z) < e.radius * 0.9 and p.y < e.y + e.height:
                    if int(self.flags["anemograna_collected"]) > 0:
                        e.data["broken"] = True
                        self.flags["wind_barrier_broken"] = int(self.flags["wind_barrier_broken"]) + 1
                        self._toast("Wind Barrier dissipated")
                        self._burst(e.x, e.y + 1.0, e.z, e.radius, (150, 240, 200), n=30)
                        self._unlock_barrier_chest(e)
                    else:
                        # Blocked: push the player back out of the dome.
                        dx, dz = p.x - e.x, p.z - e.z
                        d = max(math.hypot(dx, dz), 1e-6)
                        p.x = e.x + dx / d * (e.radius * 0.9 + 0.05)
                        p.z = e.z + dz / d * (e.radius * 0.9 + 0.05)

    def _unlock_barrier_chest(self, barrier: Entity) -> None:
        """Unlock the chest that lives inside a broken wind barrier."""
        for c in self.alive_of("chest"):
            if c.data.get("barrier") == barrier.name or _dist2d(c.x, c.z, barrier.x, barrier.z) < barrier.radius + 0.5:
                c.data["locked"] = False
                c.data["key"] = "none"

    def _resolve_collisions(self) -> None:
        """Push the capsule out of solid entities (two relaxation passes)."""
        p = self.player
        for _ in range(2):
            for e in self.entities:
                if not e.solid or not e.alive:
                    continue
                if e.kind == "wind_barrier" and e.data.get("broken"):
                    continue
                if p.y > e.y + e.height - 0.05:
                    continue
                if p.y + PLAYER_RADIUS * 2.0 < e.y:
                    continue
                dx, dz = p.x - e.x, p.z - e.z
                d = math.hypot(dx, dz)
                min_d = e.horizontal_radius() + PLAYER_RADIUS
                if d >= min_d:
                    continue
                if d < 1e-5:
                    dx, dz, d = 1.0, 0.0, 1.0
                p.x = e.x + dx / d * min_d
                p.z = e.z + dz / d * min_d

    # -- world update -------------------------------------------------------------------

    def _update_world(self, dt: float, paused: bool) -> None:
        """Advance AI, telegraphs, the trial timer, particles and objective bookkeeping."""
        p = self.player
        if self.trial_timer > 0.0:
            self.trial_timer = max(0.0, self.trial_timer - dt)
        for e in self.entities:
            if e.kind == "enemy" and e.alive:
                self._update_enemy(e, dt, paused)
            elif e.kind == "boss" and e.alive:
                self._update_boss(e, dt, paused)
        self._update_telegraphs(dt, paused)
        for dn in self.damage_numbers:
            dn["age"] += dt
        self.damage_numbers = [d for d in self.damage_numbers if d["age"] < d["ttl"]]
        for pt in self.particles:
            pt["age"] += dt
        self.particles = [q for q in self.particles if q["age"] < q["ttl"]]
        for patch in self.ice_patches:
            patch["age"] += dt
        self.ice_patches = [q for q in self.ice_patches if q["age"] < q["ttl"]]
        if not p.alive:
            self.done_reason = self.done_reason or "player_defeated"

    def _update_enemy(self, e: Entity, dt: float, paused: bool) -> None:
        """Chase, telegraph and strike; enemies never enter deep water."""
        data = e.data
        data["flash"] = max(0.0, float(data.get("flash", 0.0)) - dt)
        cd = float(data.get("atk_cd", 1.0)) - dt
        data["atk_cd"] = cd
        tele = float(data.get("telegraph", 0.0))
        p = self.player
        if tele > 0.0:
            tele -= dt
            data["telegraph"] = max(0.0, tele)
            if tele <= 0.0 and p.alive and not paused:
                if _dist2d(e.x, e.z, p.x, p.z) <= float(data["reach"]) * 1.7:
                    self.damage_player(float(data["damage"]))
            return
        if paused or not p.alive:
            return
        d = _dist2d(e.x, e.z, p.x, p.z)
        if d > float(data["aggro"]):
            return
        if d <= float(data["reach"]) and cd <= 0.0:
            data["telegraph"] = 0.55
            data["atk_cd"] = 3.0
            self.telegraphs.append(
                {
                    "x": e.x,
                    "z": e.z,
                    "y": self.height_at(e.x, e.z),
                    "r": float(data["reach"]) * 1.5,
                    "age": 0.0,
                    "fuse": 0.55,
                    "damage": 0.0,
                    "kind": "melee",
                    "hit": False,
                }
            )
            return
        if cd > 0.0 and d < float(data["reach"]) * 1.4:
            return
        speed = float(data["speed"])
        step = speed * dt
        dx, dz = p.x - e.x, p.z - e.z
        n = max(math.hypot(dx, dz), 1e-6)
        nx, nz = e.x + dx / n * step, e.z + dz / n * step
        home = data.get("home", (e.x, e.z))
        if _dist2d(nx, nz, home[0], home[1]) > 42.0:
            return
        if not self.in_water_at(nx, nz) or e.data.get("floating"):
            e.x, e.z = nx, nz
            if not e.fly:
                e.y = self.height_at(e.x, e.z)
        else:
            # Sidestep along the shore instead of drowning.
            sx, sz = -dz / n * step, dx / n * step
            if not self.in_water_at(e.x + sx, e.z + sz):
                e.x += sx
                e.z += sz
                if not e.fly:
                    e.y = self.height_at(e.x, e.z)

    def _update_boss(self, e: Entity, dt: float, paused: bool) -> None:
        """Boss phase machine: shields, vulnerable cores, AoE rings and swoops."""
        data = e.data
        data["flash"] = max(0.0, float(data.get("flash", 0.0)) - dt)
        data["phase_t"] = float(data.get("phase_t", 0.0)) + dt
        p = self.player
        variant = str(data.get("variant", ""))
        if variant == "stormterror":
            self._update_stormterror(e, dt, paused)
            return
        # Hypostasis: shield -> vulnerable core, plus expanding AoE rings to dodge.
        if data["phase_t"] > (8.0 if data.get("phase") == "shield" else 5.0):
            data["phase_t"] = 0.0
            data["phase"] = "core" if data.get("phase") == "shield" else "shield"
        data["aoe_cd"] = float(data.get("aoe_cd", 0.0)) - dt
        if float(data["aoe_cd"]) <= 0.0 and not paused and p.alive:
            data["aoe_cd"] = 4.5
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            tx = p.x + math.cos(ang) * float(self.rng.uniform(0.0, 2.0))
            tz = p.z + math.sin(ang) * float(self.rng.uniform(0.0, 2.0))
            self.telegraphs.append(
                {
                    "x": tx,
                    "z": tz,
                    "y": self.height_at(tx, tz),
                    "r": 3.6,
                    "age": 0.0,
                    "fuse": 1.0,
                    "damage": float(data["damage"]),
                    "kind": "aoe",
                    "hit": False,
                }
            )
        # Slowly drift toward the player while vulnerable.
        if not paused and p.alive and data.get("phase") == "core":
            d = _dist2d(e.x, e.z, p.x, p.z)
            if d > 6.0:
                step = 1.6 * dt
                e.x += (p.x - e.x) / d * step
                e.z += (p.z - e.z) / d * step
                e.y = self.height_at(e.x, e.z)

    def _update_stormterror(self, e: Entity, dt: float, paused: bool) -> None:
        """Stormterror: circle overhead, telegraph a swoop, then dive and recover."""
        data = e.data
        p = self.player
        phase = str(data.get("phase", "fly"))
        if phase == "fly":
            ang = self.time * 0.7
            e.x = float(data["home"][0]) + math.cos(ang) * 9.0
            e.z = float(data["home"][1]) + math.sin(ang) * 9.0
            e.y = self.height_at(e.x, e.z) + 9.0
            data["phase_t"] = float(data.get("phase_t", 0.0)) + dt
            if float(data["phase_t"]) > 5.0 and not paused and p.alive:
                data["phase"] = "telegraph"
                data["phase_t"] = 0.0
                self.telegraphs.append(
                    {
                        "x": p.x,
                        "z": p.z,
                        "y": self.height_at(p.x, p.z),
                        "r": 4.5,
                        "age": 0.0,
                        "fuse": 1.2,
                        "damage": float(data["damage"]),
                        "kind": "swoop",
                        "hit": False,
                    }
                )
        elif phase == "telegraph":
            data["phase_t"] = float(data.get("phase_t", 0.0)) + dt
            if self.telegraphs:
                tx, tz = float(self.telegraphs[-1]["x"]), float(self.telegraphs[-1]["z"])
            else:
                tx, tz = p.x, p.z
            target_y = self.height_at(tx, tz) + 1.2
            e.x += (tx - e.x) * min(1.0, dt * 1.6)
            e.z += (tz - e.z) * min(1.0, dt * 1.6)
            e.y += (target_y - e.y) * min(1.0, dt * 1.6)
            if float(data["phase_t"]) > 1.2:
                data["phase"] = "swoop"
                data["phase_t"] = 0.0
        elif phase == "swoop":
            data["phase_t"] = float(data.get("phase_t", 0.0)) + dt
            if float(data["phase_t"]) > 3.0:
                data["phase"] = "recover"
                data["phase_t"] = 0.0
        else:  # recover: climb back to altitude
            data["phase_t"] = float(data.get("phase_t", 0.0)) + dt
            e.y += (self.height_at(e.x, e.z) + 9.0 - e.y) * min(1.0, dt * 1.2)
            if float(data["phase_t"]) > 3.0:
                data["phase"] = "fly"
                data["phase_t"] = 0.0

    def _update_telegraphs(self, dt: float, paused: bool) -> None:
        """Tick ground telegraphs and apply their damage when the fuse burns out."""
        p = self.player
        alive: list[dict[str, Any]] = []
        for tg in self.telegraphs:
            tg["age"] += dt
            if not tg["hit"] and tg["age"] >= tg["fuse"]:
                tg["hit"] = True
                if not paused and p.alive and float(tg["damage"]) > 0.0:
                    if _dist2d(p.x, p.z, float(tg["x"]), float(tg["z"])) <= float(tg["r"]) + PLAYER_RADIUS:
                        self.damage_player(float(tg["damage"]))
                        self._burst(p.x, p.y + 0.6, p.z, 1.6, (255, 90, 70), n=18)
            if tg["age"] <= tg["fuse"] + 0.3:
                alive.append(tg)
        self.telegraphs = alive

    def damage_player(self, amount: float) -> None:
        """Apply damage to the player; at 0 HP the episode ends."""
        p = self.player
        if not p.alive:
            return
        p.hp = max(0.0, p.hp - float(amount))
        self.hit_flash = 0.28
        self.damage_numbers.append(
            {
                "value": float(amount),
                "text": f"-{float(amount):.0f}",
                "x": p.x,
                "y": p.y + 2.0,
                "z": p.z,
                "age": 0.0,
                "ttl": 1.0,
                "color": (255, 120, 120),
                "drift": 0.0,
            }
        )
        if p.hp <= 0.0:
            p.alive = False
            p.hp = 0.0
            self.done_reason = "player_defeated"
            self.last_event = "Player defeated"
            self._toast("You have been defeated")

    # -- presentation helpers -----------------------------------------------------------

    def _burst(self, x: float, y: float, z: float, radius: float, rgb: tuple[int, int, int],
               n: int = 12, flat: bool = False) -> None:
        """Spawn an expanding translucent particle burst."""
        for _ in range(n):
            ang = float(self.rng.uniform(0.0, 2.0 * math.pi))
            up = 0.0 if flat else float(self.rng.uniform(0.2, 1.4))
            dist = float(self.rng.uniform(0.1, radius))
            self.particles.append(
                {
                    "x": x + math.cos(ang) * dist,
                    "y": y + up,
                    "z": z + math.sin(ang) * dist,
                    "r": float(self.rng.uniform(0.16, 0.42)),
                    "age": 0.0,
                    "ttl": float(self.rng.uniform(0.35, 0.8)),
                    "color": rgb,
                }
            )

    def _add_damage_overlay(self, fx: float, fz: float) -> None:
        """Small slash spark in front of the player for readability."""
        p = self.player
        self._burst(p.x + fx * 1.2, p.y + 1.0, p.z + fz * 1.2, 0.6, (255, 250, 220), n=5)

    def _toast(self, text: str, ttl: float = 2.4) -> None:
        """Show a short centred status message on the HUD."""
        self.toast = text
        self.toast_ttl = ttl

    def _decay(self, dt: float) -> None:
        """Tick every presentation and cooldown timer."""
        self.normal_cd = max(0.0, self.normal_cd - dt)
        self.charged_cd = max(0.0, self.charged_cd - dt)
        for k in self.skill_cd:
            self.skill_cd[k] = max(0.0, self.skill_cd[k] - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt)
        self.attack_anim = max(0.0, self.attack_anim - dt)
        self.skill_anim = max(0.0, self.skill_anim - dt)
        self.toast_ttl = max(0.0, self.toast_ttl - dt)
        if self.toast_ttl <= 0.0:
            self.toast = ""

    # -- objectives ---------------------------------------------------------------------

    def _refresh_objectives(self) -> None:
        """Recompute the quest tracker text and progress counter from the flags."""
        if self.task_id in _tasks.MISSION_CHAINS:
            chain = _tasks.MISSION_CHAINS[self.task_id]
            stage = 0
            for _, crit in chain:
                if _tasks.criterion_met(crit, self.flags):
                    stage += 1
                else:
                    break
            self.flags["quest_stage"] = stage
            self.objective_progress = (stage, len(chain))
            if stage < len(chain):
                self.objective_text = f"Act I - {chain[stage][0]}"
            else:
                self.objective_text = "Act I complete"
            return
        if self.task is not None and self.task.success_criteria:
            self.objective_progress = _tasks.criterion_progress(self.task.success_criteria, self.flags)
            self.objective_text = _tasks.objective_text(self.task.success_criteria, self.flags)
        else:
            self.objective_progress = (0, 0)
            self.objective_text = "Explore the open world"

    # -- HUD ----------------------------------------------------------------------------

    def hud_state(self) -> HUDState:
        """Fully populate the structured HUD read-out for this tick."""
        p = self.player
        dialogue = self.dialogue or {}
        names = tuple(name for name, _ in PARTY)
        return HUDState(
            hp=float(p.hp),
            max_hp=float(p.max_hp),
            stamina=float(p.stamina),
            max_stamina=float(p.max_stamina),
            position=(float(p.x), float(p.y), float(p.z)),
            yaw=float(p.yaw),
            region=self.region,
            quest=self.quest_title,
            objective=self.instruction or self.objective_text,
            objective_progress=(int(self.objective_progress[0]), int(self.objective_progress[1])),
            interact_prompt=self.interact_prompt(),
            dialogue_open=self.dialogue is not None,
            dialogue_speaker=str(dialogue.get("speaker", "")),
            dialogue_options=tuple(dialogue.get("options", ())),
            menu=self.menu,
            menu_cursor=int(self.menu_cursor),
            party=names,
            active_party_index=int(p.party_index),
            element=PARTY[p.party_index][1],
            nearby_entities=self.nearby_entities(30.0),
            alive=bool(p.alive),
            time_of_day=float(self.time_of_day),
        )

    # -- misc ---------------------------------------------------------------------------

    def sun_direction(self) -> tuple[float, float, float]:
        """Unit vector pointing from the world toward the sun (or moon at night)."""
        hour = self.time_of_day
        az = math.pi * (hour - 6.0) / 12.0
        alt = math.sin(math.pi * (hour - 6.0) / 12.0)
        if alt < 0.02:
            # Night: use the moon, mirrored in the sky.
            az = math.pi * (hour - 18.0) / 12.0
            alt = max(0.12, -alt)
        v = np.array([math.cos(az), max(alt, 0.06), 0.35], dtype=np.float64)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        return (float(v[0]), float(v[1]), float(v[2]))

    def entity_by_name(self, name: str) -> Entity | None:
        """First entity whose name matches exactly, or ``None``."""
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def summary(self) -> str:
        """One-line human summary, handy in logs and the preview demo."""
        p = self.player
        return (
            f"region={self.region} task={self.task_id} t={self.time:.1f}s "
            f"pos=({p.x:.1f},{p.y:.1f},{p.z:.1f}) hp={p.hp:.0f} "
            f"chests={self.flags['chests_opened']} kills={self.flags['enemies_defeated']} "
            f"progress={self.objective_progress}"
        )
