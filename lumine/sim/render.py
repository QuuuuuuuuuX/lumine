"""Software 3D renderer and HUD compositor for the sandbox.

No OpenGL, no game engine: a perspective projection, a painter's-algorithm polygon queue and
``cv2.fillPoly``.  That is enough to produce a frame a vision-language model can genuinely
read, and it runs anywhere Python runs, which matters more here than fill rate.

Design notes
------------
* **Painter's algorithm, not a z-buffer.** Everything becomes ``(depth, polygon, colour)``,
  sorted far-to-near.  With a few thousand small polygons this is faster in OpenCV's C than a
  per-pixel depth test would be in Python, and the visual artefacts are confined to
  interpenetrating geometry, which the sandbox does not have.
* **Rendering is decoupled from stepping.** :class:`~lumine.sim.env.SimEnv` calls
  :meth:`Renderer.render` at the perception rate (5 Hz), not the control rate (30 Hz).  A
  render costs a few milliseconds; paying it six times more often would be waste.
* **Ground truth for free.** Because the renderer knows where everything is, it emits
  :attr:`Renderer.entity_boxes` on every frame.  The agent's controller treats that exactly
  like an object detector's output, so the same control code works against a real game, where
  :class:`~lumine.perception.ColourDetector` takes over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from ..types import HUDState
from .world import ELEMENT_RGB, PARTY, TERRAIN_HALF

__all__ = ["Renderer", "REGION_PALETTE"]

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------

#: Per-region colours.  ``liyue`` is deliberately a different world to look at, not a
#: recoloured ``mondstadt``: warmer haze, rockier ground, darker and sparser vegetation.
REGION_PALETTE: dict[str, dict[str, Any]] = {
    "mondstadt": {
        "sky_top": (78, 140, 226),
        "sky_bottom": (196, 226, 246),
        "sun": (255, 246, 214),
        "fog": (188, 214, 238),
        "grass_low": (86, 148, 72),
        "grass_high": (124, 172, 88),
        "rock": (128, 124, 112),
        "rock_high": (176, 176, 168),
        "water": (58, 122, 176),
        "tree_trunk": (96, 70, 48),
        "tree_canopy": (60, 122, 62),
        "tree_canopy_alt": (86, 148, 70),
        "cloud": (246, 250, 255),
    },
    "liyue": {
        "sky_top": (150, 130, 190),
        "sky_bottom": (240, 208, 176),
        "sun": (255, 232, 180),
        "fog": (232, 200, 172),
        "grass_low": (150, 122, 78),
        "grass_high": (196, 160, 96),
        "rock": (140, 112, 92),
        "rock_high": (192, 166, 132),
        "water": (70, 128, 150),
        "tree_trunk": (86, 62, 44),
        "tree_canopy": (74, 104, 58),
        "tree_canopy_alt": (104, 122, 62),
        "cloud": (250, 240, 232),
    },
}

#: Entity body colours by kind.
KIND_RGB: dict[str, tuple[int, int, int]] = {
    "tree": (70, 130, 66),
    "rock": (132, 128, 118),
    "boulder": (118, 116, 108),
    "chest": (208, 160, 48),
    "enemy": (198, 70, 70),
    "boss": (150, 80, 190),
    "npc": (96, 176, 208),
    "monument": (170, 168, 190),
    "waypoint": (86, 150, 236),
    "cooking_pot": (110, 96, 88),
    "anemoculus": (120, 220, 250),
    "anemograna": (140, 240, 200),
    "wind_current": (176, 230, 240),
    "wind_barrier": (120, 210, 230),
    "thorn_bush": (74, 86, 52),
    "stone_pillar": (156, 152, 144),
    "time_trial": (232, 200, 96),
    "slime": (110, 200, 150),
    "hilichurl": (170, 96, 72),
    "whopperflower": (196, 120, 190),
}

NEAR = 0.22
SKY_HORIZON = 0.5


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------


@dataclass
class Camera:
    """Third-person camera derived from the player's yaw and pitch."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        forward = np.array([sy * cp, sp, cy * cp], dtype=np.float64)
        right = np.array([cy, 0.0, -sy], dtype=np.float64)
        up = np.cross(right, forward)
        return forward, right, up


class Renderer:
    """Rasterises a :class:`~lumine.sim.world.World` into an RGB frame plus HUD."""

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fog: bool = True,
        fov_y_deg: float = 55.0,
        camera_distance: float = 4.2,
        camera_height: float = 1.5,
        terrain_step: float = 2.5,
        terrain_span: float = 46.0,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fog = fog
        self.camera_distance = camera_distance
        self.camera_height = camera_height
        self.terrain_step = terrain_step
        self.terrain_span = terrain_span

        self.focal = (self.height * 0.5) / math.tan(math.radians(fov_y_deg) * 0.5)
        self.camera = Camera()
        self.palette: dict[str, Any] = REGION_PALETTE["mondstadt"]

        #: Ground-truth detections for this frame: name, kind, bbox, distance, screen_center.
        self.entity_boxes: list[dict[str, Any]] = []
        self.last_frame: np.ndarray | None = None
        self._queue: list[tuple[float, np.ndarray, tuple[int, int, int], int | None]] = []

    # -- projection ----------------------------------------------------------------------

    def project(self, point: Sequence[float]) -> tuple[float, float, float] | None:
        """World point -> ``(u, v, depth)`` in pixels, or ``None`` when behind the camera."""
        f, r, u = self._basis
        d = np.asarray(point, dtype=np.float64) - self._eye
        zc = float(d @ f)
        if zc <= NEAR:
            return None
        xc = float(d @ r)
        yc = float(d @ u)
        px = self.width * 0.5 + self.focal * xc / zc
        py = self.height * 0.5 - self.focal * yc / zc
        return px, py, zc

    # -- main entry point ------------------------------------------------------------------

    def render(self, world) -> np.ndarray:  # noqa: ANN001 - avoids importing World here
        """Render one frame of ``world`` and return it as an RGB ``uint8`` array."""
        self.palette = REGION_PALETTE.get(getattr(world, "region", "mondstadt"),
                                          REGION_PALETTE["mondstadt"])
        self.entity_boxes = []
        self._update_camera(world)

        frame = self._draw_sky(world)
        self._queue = []

        self._queue_terrain(world)
        self._queue_water(world)
        self._queue_ice(world)
        self._queue_ground_decals(world)
        self._queue_entities(world)

        # Painter's algorithm: far to near.
        self._queue.sort(key=lambda item: item[0], reverse=True)
        for _depth, pts, rgb, outline in self._queue:
            cv2.fillPoly(frame, [pts], rgb)
            if outline is not None:
                cv2.polylines(frame, [pts], True, outline, 1, cv2.LINE_AA)

        self._draw_particles(frame, world)
        self._draw_damage_numbers(frame, world)
        self._draw_hud(frame, world)

        self.last_frame = frame
        return frame

    # -- camera ---------------------------------------------------------------------------

    def _update_camera(self, world) -> None:  # noqa: ANN001
        p = world.player
        yaw, pitch = float(p.yaw), float(p.pitch)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        forward = np.array([sy * cp, sp, cy * cp])
        ground = float(world.ground_height(p.x, p.z))

        cand = np.array([p.x, p.y + 1.55, p.z]) - forward * self.camera_distance
        cand[1] += self.camera_height - 0.6
        # Never let the camera sink under the terrain it is standing behind.
        floor = float(world.ground_height(float(cand[0]), float(cand[2]))) + 0.6
        if cand[1] < floor:
            cand[1] = floor
        if world.in_water_at(float(cand[0]), float(cand[2])):
            cand[1] = max(cand[1], 0.5)

        self.camera = Camera(x=float(cand[0]), y=float(cand[1]), z=float(cand[2]),
                             yaw=yaw, pitch=pitch)
        self._eye = cand
        self._basis = self.camera.basis()
        self._forward = forward
        del ground

    # -- scene layers ----------------------------------------------------------------------

    def _draw_sky(self, world) -> np.ndarray:  # noqa: ANN001
        pal = self.palette
        t = float(getattr(world, "time_of_day", 12.0)) % 24.0
        # Dawn / day / dusk / night blend factor, 0 at midnight and 1 at noon.
        day = 0.5 - 0.5 * math.cos(2.0 * math.pi * (t - 6.0) / 24.0)
        dusk = max(0.0, 1.0 - abs(day - 0.22) * 4.0)

        top = np.array(pal["sky_top"], np.float64) * (0.35 + 0.65 * day)
        bottom = np.array(pal["sky_bottom"], np.float64) * (0.40 + 0.60 * day)
        # Warm the horizon at dusk so time of day is legible from a single frame.
        bottom = bottom * (1.0 - 0.45 * dusk) + np.array([246, 150, 92], np.float64) * (0.45 * dusk)
        top = np.clip(top, 0, 255).astype(np.uint8)
        bottom = np.clip(bottom, 0, 255).astype(np.uint8)

        ramp = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        img = (top[None, None, :] * (1 - ramp[:, :, None])
               + bottom[None, None, :] * ramp[:, :, None])
        frame = np.repeat(img.astype(np.uint8), self.width, axis=1)

        # Sun / moon: a disc plus a soft halo, placed from the hour angle.
        ang = math.pi * (t - 6.0) / 12.0
        sx = int(self.width * (0.5 + 0.42 * math.cos(ang)))
        sy = int(self.height * (0.42 - 0.34 * math.sin(ang)))
        is_day = 6.0 <= t <= 19.0
        col = pal["sun"] if is_day else (226, 232, 244)
        if -200 < sx < self.width + 200:
            for rad, alpha in ((46, 0.18), (26, 0.35), (14, 1.0)):
                overlay = frame.copy()
                cv2.circle(overlay, (sx, sy), rad, col, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # A few clouds, fixed in world space so they parallax with the camera.
        for i in range(7):
            cx = (i * 137.0 + 40.0) % 900.0 - 450.0
            cz = 220.0 + (i * 61.0) % 260.0
            cy = 46.0 + (i * 17.0) % 22.0
            self._cloud(frame, cx, cy, cz, 26.0 + (i % 3) * 9.0, pal["cloud"], day)
        return frame

    def _cloud(self, frame, x, y, z, size, rgb, day: float) -> None:  # noqa: ANN001
        proj = self.project((x, y, z))
        if proj is None:
            return
        u, v, depth = proj
        if not (-200 < u < self.width + 200 and -200 < v < self.height + 200):
            return
        r = max(3, int(self.focal * size / depth))
        colour = tuple(int(c * (0.55 + 0.45 * day)) for c in rgb)
        cv2.circle(frame, (int(u), int(v)), r, colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(u - r * 0.7), int(v + r * 0.25)), int(r * 0.7), colour, -1, cv2.LINE_AA)
        cv2.circle(frame, (int(u + r * 0.75), int(v + r * 0.2)), int(r * 0.62), colour, -1, cv2.LINE_AA)

    def _queue_terrain(self, world) -> None:  # noqa: ANN001
        pal = self.palette
        step = self.terrain_step
        span = self.terrain_span
        px, pz = world.player.x, world.player.z
        # Snap the grid to the step so quads do not shimmer as the player moves.
        x0 = math.floor((px - span) / step) * step
        z0 = math.floor((pz - span) / step) * step
        n = int(2 * span / step)
        sun = self._sun_dir(world)

        xs = [x0 + i * step for i in range(n + 1)]
        zs = [z0 + j * step for j in range(n + 1)]
        # One vectorised bilinear sample for the whole mesh instead of n^2 scalar lookups.
        height_grid = world.sample_heights(np.array(xs), np.array(zs)[:, None])

        for i in range(n):
            for j in range(n):
                xa, xb = xs[i], xs[i + 1]
                za, zb = zs[j], zs[j + 1]
                ha = float(height_grid[j, i])
                hb = float(height_grid[j, i + 1])
                hc = float(height_grid[j + 1, i + 1])
                hd = float(height_grid[j + 1, i])
                if max(ha, hb, hc, hd) < 0.0:
                    continue                                   # fully submerged -> water draws it
                corners = ((xa, ha, za), (xb, hb, za), (xb, hc, zb), (xa, hd, zb))
                pts, depth = self._project_poly(corners)
                if pts is None:
                    continue
                mean_h = 0.25 * (ha + hb + hc + hd)
                slope = max(abs(hb - ha), abs(hd - ha)) / step
                base = np.array(pal["grass_low"] if mean_h < 6.0 else pal["grass_high"], np.float64)
                if slope > 0.55:
                    base = np.array(pal["rock"], np.float64)
                elif slope > 0.30:
                    base = 0.5 * (base + np.array(pal["rock_high"], np.float64))

                # Slope already drives the colour choice; lighting comes from the quad's own
                # geometric normal, which costs nothing and is what makes hillsides read.
                nx = (ha - hb) / step
                nz = (ha - hd) / step
                inv = 1.0 / math.sqrt(nx * nx + nz * nz + 1.0)
                lam = max(0.0, (nx * sun[0] + sun[1] - nz * sun[2]) * inv)
                shade = 0.52 + 0.55 * lam
                rgb = self._fog(base * shade, depth)
                self._queue.append((depth, pts, rgb, None))

    def _queue_water(self, world) -> None:  # noqa: ANN001
        pal = self.palette
        step = self.terrain_step * 2.2
        span = self.terrain_span
        px, pz = world.player.x, world.player.z
        x0 = math.floor((px - span) / step) * step
        z0 = math.floor((pz - span) / step) * step
        n = int(2 * span / step)
        wobble = math.sin(world.time * 1.6) * 0.03
        for i in range(n):
            for j in range(n):
                xa, xb = x0 + i * step, x0 + (i + 1) * step
                za, zb = z0 + j * step, z0 + (j + 1) * step
                corners = ((xa, wobble, za), (xb, wobble, za), (xb, wobble, zb), (xa, wobble, zb))
                pts, depth = self._project_poly(corners)
                if pts is None:
                    continue
                if world.height_at(0.5 * (xa + xb), 0.5 * (za + zb)) > 0.05:
                    continue
                rgb = self._fog(np.array(pal["water"], np.float64), depth * 0.85)
                self._queue.append((depth, pts, rgb, None))

    def _queue_ice(self, world) -> None:  # noqa: ANN001
        for patch in getattr(world, "ice_patches", ()):
            pts, depth = self._ground_ring(float(patch["x"]), float(patch["z"]), float(patch["r"]),
                                           12, float(patch.get("y", 0.06)))
            if pts is None:
                continue
            self._queue.append((depth, pts, (198, 232, 246), None))

    def _queue_ground_decals(self, world) -> None:  # noqa: ANN001
        for tg in getattr(world, "telegraphs", ()):
            fuse = max(1e-3, float(tg.get("fuse", 0.6)))
            frac = min(1.0, float(tg.get("age", 0.0)) / fuse)
            r = float(tg["r"]) * (0.35 + 0.65 * frac)
            pts, depth = self._ground_ring(float(tg["x"]), float(tg["z"]), r, 20,
                                           float(tg.get("y", 0.0)) + 0.07)
            if pts is None:
                continue
            # Filling as it charges: the redder and more solid it looks, the sooner it lands.
            colour = (int(120 + 135 * frac), int(40 + 20 * frac), int(40 + 10 * frac))
            self._queue.append((depth, pts, colour, (255, 90, 90)))

    # -- entities --------------------------------------------------------------------------

    def _queue_entities(self, world) -> None:  # noqa: ANN001
        p = world.player
        for e in world.entities:
            if not e.alive and e.kind not in ("chest",):
                continue
            dx, dz = e.x - p.x, e.z - p.z
            dist = math.sqrt(dx * dx + dz * dz + (e.y - p.y) ** 2)
            if dist > self.terrain_span * 1.35:
                continue
            groups = self._entity_polygons(e, dist, world)
            if not groups:
                continue
            all_faces: list[tuple[np.ndarray, float]] = []
            for faces, rgb in groups:
                for pts, depth in faces:
                    self._queue.append((depth, pts, rgb, None))
                all_faces.extend(faces)
            if not all_faces:
                continue
            screen = self._screen_bounds([pts for pts, _d in all_faces])
            if screen is not None:
                x0, y0, x1, y1 = screen
                self.entity_boxes.append({
                    "name": e.name or e.kind,
                    "kind": e.kind,
                    "bbox": (int(x0), int(y0), int(x1), int(y1)),
                    "distance": round(float(dist), 2),
                    "screen_center": (float((x0 + x1) * 0.5), float((y0 + y1) * 0.5)),
                    "visible": bool(x1 > 0 and y1 > 0 and x0 < self.width and y0 < self.height),
                })

        # The player's own body, so the third-person view reads as a character.
        self._queue_player(world)

    def _entity_polygons(self, e, dist: float, world):  # noqa: ANN001, ANN201
        """Project one entity into ``[(faces, rgb), ...]``.

        Returning colour groups rather than a single fill is what lets a boss be grey with a
        white core, and a monument be stone with a glowing gem, without the primitive builders
        having to know anything about materials.

        (The previous version returned ``(faces, None)`` and relied on a colour argument that
        ``_box`` silently discarded. Every tree, rock and enemy in the world rendered solid
        black -- which a vision-language model reads as "nothing is here".)
        """
        kind = e.kind
        elevate = float(e.data.get("bob", 0.0)) if isinstance(e.data, dict) else 0.0
        y0 = float(e.y) + elevate
        fg = lambda rgb: self._fog(np.array(rgb, np.float64), dist)  # noqa: E731

        if kind == "tree":
            return [
                (self._box(e.x, y0, y0 + e.height * 0.45, e.z, 0.42, 0.42),
                 fg(self.palette["tree_trunk"])),
                (self._cone(e.x, y0 + e.height * 0.35, e.z, e.radius * 1.5, e.height * 0.75),
                 fg(self.palette["tree_canopy"])),
            ]

        if kind in ("rock", "boulder", "stone_pillar"):
            colour = KIND_RGB["stone_pillar" if kind == "stone_pillar" else kind]
            return [(self._box(e.x, y0, y0 + e.height, e.z, e.radius * 1.15, e.radius * 1.15),
                     fg(colour))]

        if kind == "wind_current":
            faces = []
            for k in range(3):
                yy = y0 + 0.6 + k * 2.4 + (world.time * 2.2 + k) % 2.4
                ring, depth = self._ground_ring(e.x, e.z, e.radius * (0.6 + 0.2 * k), 14, yy)
                if ring is not None:
                    faces.append((ring, depth))
            return [(faces, KIND_RGB["wind_current"])]

        if kind == "thorn_bush":
            spikes = []
            for k in range(6):
                ang = k * math.pi / 3.0 + e.x
                tx = e.x + math.cos(ang) * e.radius * 0.7
                tz = e.z + math.sin(ang) * e.radius * 0.7
                tri = ((e.x, y0, e.z), (tx, y0 + e.height, tz),
                       (e.x + math.cos(ang + 0.5) * 0.3, y0, e.z + math.sin(ang + 0.5) * 0.3))
                pts, depth = self._project_poly(tri)
                if pts is not None:
                    spikes.append((pts, depth))
            return [(spikes, KIND_RGB["thorn_bush"])]

        if kind in ("anemoculus", "anemograna"):
            r = e.radius * (1.6 if kind == "anemoculus" else 1.2)
            bob = math.sin(world.time * 2.0 + e.x) * 0.25
            base = KIND_RGB["anemoculus" if kind == "anemoculus" else "anemograna"]
            groups = [(self._box(e.x, y0 + bob, y0 + bob + r * 2, e.z, r, r),
                       tuple(min(255, int(c * 1.25)) for c in base))]
            halo, depth = self._ground_ring(e.x, e.z, r * 1.8, 12, y0 + bob + r)
            if halo is not None:
                groups.append(([(halo, depth)], base))
            return groups

        if kind == "chest":
            opened = bool(e.data.get("opened")) if isinstance(e.data, dict) else False
            locked = bool(e.data.get("locked")) if isinstance(e.data, dict) else False
            h = e.height * (0.55 if opened else 1.0)
            body = (150, 120, 60) if opened else ((150, 130, 90) if locked else KIND_RGB["chest"])
            return [
                (self._box(e.x, y0, y0 + h, e.z, e.radius * 1.5, e.radius * 1.1, ), body),
                # A bright lid band: the strongest visual cue that this is lootable.
                (self._box(e.x, y0 + h * 0.82, y0 + h, e.z, e.radius * 1.55, e.radius * 1.15),
                 (255, 236, 170)),
            ]

        if kind == "npc":
            return [(self._humanoid(e.x, y0, e.z, e.radius, e.height), fg(KIND_RGB["npc"]))]

        if kind == "boss":
            core = (250, 240, 255) if e.variant != "stormterror" else (200, 170, 250)
            body = fg(KIND_RGB["boss"] if e.variant != "stormterror" else (110, 90, 160))
            return [
                (self._box(e.x, y0, y0 + e.height, e.z, e.radius * 1.6, e.radius * 1.6), body),
                (self._box(e.x, y0 + e.height * 0.45, y0 + e.height * 0.72, e.z,
                           e.radius * 0.7, e.radius * 0.7), core),
            ]

        if kind == "enemy":
            return [(self._humanoid(e.x, y0, e.z, e.radius, e.height),
                     fg(KIND_RGB.get(e.variant, KIND_RGB["enemy"])))]

        if kind == "wind_barrier":
            faces = []
            for k in range(3):
                ring, depth = self._ground_ring(e.x, e.z, e.radius, 16, y0 + k * (e.height / 3.0))
                if ring is not None:
                    faces.append((ring, depth))
            return [(faces, KIND_RGB["wind_barrier"])]

        if kind == "time_trial":
            return [
                (self._box(e.x, y0, y0 + 0.3, e.z, e.radius * 1.6, e.radius * 1.6), (90, 74, 30)),
                (self._box(e.x, y0, y0 + e.height, e.z, 0.28, 0.28), KIND_RGB["time_trial"]),
            ]

        # monument, waypoint, cooking_pot and anything else: a simple block.
        groups = [(self._box(e.x, y0, y0 + e.height, e.z, e.radius * 1.4, e.radius * 1.4),
                   fg(KIND_RGB.get(kind, (170, 170, 170))))]
        if kind == "monument":
            groups.append((self._box(e.x, y0 + e.height, y0 + e.height + 0.55, e.z, 0.6, 0.6),
                           ELEMENT_RGB.get(e.element or "anemo", (200, 240, 220))))
        if kind == "waypoint":
            groups.append((self._box(e.x, y0 + e.height, y0 + e.height + 1.1, e.z, 0.7, 0.7),
                           (170, 220, 255)))
        return groups

    def _queue_player(self, world) -> None:  # noqa: ANN001
        p = world.player
        if not p.alive:
            return
        element = PARTY[p.party_index][1] if p.party_index < len(PARTY) else "anemo"
        accent = ELEMENT_RGB.get(element, (200, 220, 240))
        body_rgb = (236, 232, 244)
        for pts, depth in self._humanoid(p.x, p.y, p.z, 0.42, 1.75):
            self._queue.append((depth, pts, body_rgb, None))
        # A coloured ring under the feet marks the active element at a glance.
        ring, depth = self._ground_ring(p.x, p.z, 0.75, 14, p.y + 0.05)
        if ring is not None:
            self._queue.append((depth, ring, accent, None))

    # -- primitive construction ------------------------------------------------------------

    def _box(self, cx: float, y0: float, y1: float, cz: float, w: float, d: float
             ) -> list[tuple[np.ndarray, float]]:
        """Axis-aligned box: its four vertical faces plus a top, each with a view depth."""
        hw, hd = w * 0.5, d * 0.5
        bl = (cx - hw, y0, cz - hd)
        br = (cx + hw, y0, cz - hd)
        fr = (cx + hw, y0, cz + hd)
        fl = (cx - hw, y0, cz + hd)
        tl, tr = (bl[0], y1, bl[2]), (br[0], y1, br[2])
        trr, tll = (fr[0], y1, fr[2]), (fl[0], y1, fl[2])
        out: list[tuple[np.ndarray, float]] = []
        for quad in ((tll, trr, fr, fl), (tl, tr, trr, tll),
                     (bl, br, tr, tl), (fr, br, tr, trr), (fl, bl, tl, tll)):
            pts, depth = self._project_poly(quad)
            if pts is not None:
                out.append((pts, depth))
        return out

    def _cone(self, cx: float, y0: float, cz: float, radius: float, height: float
              ) -> list[tuple[np.ndarray, float]]:
        out: list[tuple[np.ndarray, float]] = []
        apex = (cx, y0 + height, cz)
        segments = 7
        for k in range(segments):
            a0 = 2 * math.pi * k / segments
            a1 = 2 * math.pi * (k + 1) / segments
            p0 = (cx + math.cos(a0) * radius, y0, cz + math.sin(a0) * radius)
            p1 = (cx + math.cos(a1) * radius, y0, cz + math.sin(a1) * radius)
            pts, depth = self._project_poly((p0, p1, apex))
            if pts is not None:
                out.append((pts, depth))
        return out

    def _humanoid(self, cx: float, y0: float, cz: float, radius: float, height: float
                  ) -> list[tuple[np.ndarray, float]]:
        body = self._box(cx, y0 + height * 0.28, y0 + height * 0.82, cz,
                         radius * 1.4, radius * 1.1)
        head = self._box(cx, y0 + height * 0.82, y0 + height * 1.02, cz,
                         radius * 1.0, radius * 0.9)
        legs = self._box(cx, y0, y0 + height * 0.28, cz, radius * 1.0, radius * 0.9)
        return body + head + legs

    def _ground_ring(self, cx: float, cz: float, radius: float, segments: int,
                     y: float) -> tuple[np.ndarray | None, float]:
        """A flat polygon lying on the ground, used for rings, decals and wind currents."""
        pts = []
        for k in range(segments):
            a = 2 * math.pi * k / segments
            proj = self.project((cx + math.cos(a) * radius, y, cz + math.sin(a) * radius))
            if proj is None:
                return None, 0.0
            pts.append((proj[0], proj[1]))
        arr = np.array(pts, np.int32)
        return arr, max(1.0, math.hypot(cx - self._eye[0], cz - self._eye[2]))

    # -- helpers ---------------------------------------------------------------------------

    def _project_poly(self, corners: Iterable[Sequence[float]]
                      ) -> tuple[np.ndarray | None, float]:
        pts = []
        depth = 0.0
        for c in corners:
            proj = self.project(c)
            if proj is None:
                return None, 0.0
            pts.append((proj[0], proj[1]))
            depth += proj[2]
        arr = np.array(pts, np.int32)
        if arr[:, 0].max() < -64 or arr[:, 0].min() > self.width + 64:
            return None, 0.0
        if arr[:, 1].max() < -64 or arr[:, 1].min() > self.height + 64:
            return None, 0.0
        return arr, depth / len(pts)

    def _screen_bounds(self, polys: Sequence[np.ndarray]
                       ) -> tuple[float, float, float, float] | None:
        if not polys:
            return None
        xs0 = min(int(p[:, 0].min()) for p in polys)
        ys0 = min(int(p[:, 1].min()) for p in polys)
        xs1 = max(int(p[:, 0].max()) for p in polys)
        ys1 = max(int(p[:, 1].max()) for p in polys)
        return float(xs0), float(ys0), float(xs1), float(ys1)

    def _sun_dir(self, world) -> np.ndarray:  # noqa: ANN001
        t = float(getattr(world, "time_of_day", 12.0)) % 24.0
        ang = math.pi * (t - 6.0) / 12.0
        d = np.array([math.cos(ang) * 0.6, max(0.25, math.sin(ang)), 0.45])
        return d / np.linalg.norm(d)

    def _fog(self, rgb: np.ndarray, depth: float) -> tuple[int, int, int]:
        if not self.fog:
            return tuple(int(np.clip(c, 0, 255)) for c in rgb)  # type: ignore[return-value]
        k = 1.0 - math.exp(-max(0.0, depth) / 58.0)
        fog = np.array(self.palette["fog"], np.float64)
        out = rgb * (1.0 - k) + fog * k
        return tuple(int(np.clip(c, 0, 255)) for c in out)  # type: ignore[return-value]

    # -- overlays --------------------------------------------------------------------------

    def _draw_particles(self, frame, world) -> None:  # noqa: ANN001
        for part in getattr(world, "particles", ()):
            ttl = max(1e-3, float(part.get("ttl", 0.5)))
            life = max(0.0, 1.0 - float(part.get("age", 0.0)) / ttl)
            proj = self.project((part["x"], part["y"], part["z"]))
            if proj is None:
                continue
            u, v, depth = proj
            r = max(1, int(self.focal * float(part.get("r", 0.25)) / max(0.5, depth)))
            col = tuple(int(c * (0.4 + 0.6 * life)) for c in part.get("color", (255, 255, 255)))
            cv2.circle(frame, (int(u), int(v)), r, col, -1, cv2.LINE_AA)

    def _draw_damage_numbers(self, frame, world) -> None:  # noqa: ANN001
        for dn in getattr(world, "damage_numbers", ()):
            ttl = max(1e-3, float(dn.get("ttl", 1.0)))
            life = max(0.0, 1.0 - float(dn.get("age", 0.0)) / ttl)
            proj = self.project((dn["x"], dn["y"] + (1.0 - life) * 0.8, dn["z"]))
            if proj is None:
                continue
            u, v, _d = proj
            col = tuple(int(c * (0.35 + 0.65 * life)) for c in dn.get("color", (255, 255, 255)))
            cv2.putText(frame, str(dn.get("text", "")), (int(u) - 12, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    # -- HUD -------------------------------------------------------------------------------

    def _draw_hud(self, frame, world) -> None:  # noqa: ANN001
        W, H = self.width, self.height
        hud: HUDState = world.hud_state()

        # -- quest tracker, top-left
        self._panel(frame, 8, 8, 214, 56)
        cv2.putText(frame, self._clip(str(world.quest_title), 30), (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 236, 168), 1, cv2.LINE_AA)
        cv2.putText(frame, self._clip(str(world.objective_text), 30), (16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (238, 242, 250), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{hud.objective_progress[0]}/{max(1, hud.objective_progress[1])}",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 226, 255), 1, cv2.LINE_AA)

        # -- standing order, top-centre
        order = self._clip(str(getattr(world, "instruction", "") or ""), 46)
        if order:
            (tw, _th), _ = cv2.getTextSize(order, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            x = max(8, (W - tw) // 2)
            self._panel(frame, x - 8, 8, tw + 16, 22)
            cv2.putText(frame, order, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # -- minimap, top-right
        self._draw_minimap(frame, world)

        # -- HP + stamina, bottom-left
        self._bar(frame, 16, H - 46, 170, 12, hud.hp / max(1.0, hud.max_hp), (70, 70, 220))
        cv2.putText(frame, f"HP {int(hud.hp)}/{int(hud.max_hp)}", (18, H - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        self._bar(frame, 16, H - 26, 140, 8, hud.stamina / max(1.0, hud.max_stamina),
                  (90, 210, 110))

        # -- party chips + skill icons, bottom-right
        # HUDState.party carries names only; the element comes from the world's party table.
        for k, name in enumerate(hud.party):
            elem = PARTY[k][1] if k < len(PARTY) else "anemo"
            cw, ch = 74, 20
            x = W - 16 - cw
            y = H - 20 - k * (ch + 4)
            active = k == hud.active_party_index
            self._panel(frame, x, y, cw, ch, alpha=0.62,
                        tint=(58, 88, 128) if active else (34, 34, 40))
            col = ELEMENT_RGB.get(elem, (230, 230, 230)) if active else (170, 170, 170)
            cv2.putText(frame, f"{k + 1} {name[:6]}", (x + 5, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
            if active:
                cv2.rectangle(frame, (x, y), (x + cw, y + ch), tuple(int(c) for c in col), 1)

        for k, key in enumerate(("E", "Q")):
            cx0, cy0 = W - 118 + k * 34, H - 34
            cd = float(world.skill_cd.get(key, 0.0))
            ready = cd <= 0.01 and (key != "Q" or world.energy >= 100.0)
            cv2.circle(frame, (cx0, cy0), 13, (30, 30, 36), -1)
            cv2.circle(frame, (cx0, cy0), 13, (235, 235, 235) if ready else (110, 110, 110), 1)
            cv2.putText(frame, key, (cx0 - 5, cy0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255) if ready else (140, 140, 140), 1, cv2.LINE_AA)

        # -- crosshair
        cx, cy = W // 2, H // 2
        cv2.line(frame, (cx - 6, cy), (cx + 6, cy), (245, 245, 245), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 6), (cx, cy + 6), (245, 245, 245), 1, cv2.LINE_AA)

        # -- interaction prompt
        if hud.interact_prompt:
            text = f"[F] {self._clip(hud.interact_prompt, 40)}"
            (tw, _th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            x = (W - tw) // 2
            self._panel(frame, x - 10, H // 2 + 34, tw + 20, 24)
            cv2.putText(frame, text, (x, H // 2 + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                        (255, 246, 210), 1, cv2.LINE_AA)

        if hud.dialogue_open:
            self._draw_dialogue(frame, world, hud)
        if hud.menu:
            self._draw_menu(frame, world, hud)
        if getattr(world, "toast_ttl", 0.0) > 0 and world.toast:
            (tw, _th), _ = cv2.getTextSize(world.toast, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            x = (W - tw) // 2
            self._panel(frame, x - 10, 76, tw + 20, 26)
            cv2.putText(frame, world.toast, (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 238, 170), 1, cv2.LINE_AA)

        if hud.hp / max(1.0, hud.max_hp) < 0.25:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (60, 60, 220), 3)

    def _draw_minimap(self, frame, world) -> None:  # noqa: ANN001
        W = self.width
        cx, cy, R = W - 56, 56, 40
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), R, (26, 30, 38), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.circle(frame, (cx, cy), R, (188, 196, 210), 1, cv2.LINE_AA)

        p = world.player
        scale = R / 45.0
        for e in world.entities:
            if not e.alive and e.kind != "chest":
                continue
            dx, dz = e.x - p.x, e.z - p.z
            if abs(dx) > 45 or abs(dz) > 45:
                continue
            # Rotate so the minimap is player-relative and up is forward.
            ca, sa = math.cos(-p.yaw), math.sin(-p.yaw)
            mx = dx * ca - dz * sa
            mz = dx * sa + dz * ca
            u, v = int(cx + mx * scale), int(cy - mz * scale)
            if (u - cx) ** 2 + (v - cy) ** 2 > R * R:
                continue
            col = self._minimap_colour(e)
            if col is None:
                continue
            cv2.circle(frame, (u, v), 2, col, -1, cv2.LINE_AA)

        # Player arrow.
        cv2.fillPoly(frame, [np.array([(cx, cy - 6), (cx - 4, cy + 4), (cx + 4, cy + 4)], np.int32)],
                     (255, 255, 255))
        cv2.putText(frame, str(getattr(world, "region", ""))[:10], (cx - 28, cy + R + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 234, 244), 1, cv2.LINE_AA)

    @staticmethod
    def _minimap_colour(e) -> tuple[int, int, int] | None:  # noqa: ANN001
        if e.kind in ("enemy", "boss"):
            return (80, 80, 240)
        if e.kind == "chest":
            return (60, 210, 240)
        if e.kind == "npc":
            return (230, 220, 90)
        if e.kind == "waypoint":
            return (240, 150, 80)
        if e.kind in ("anemoculus", "anemograna"):
            return (250, 250, 250)
        return None

    def _draw_dialogue(self, frame, world, hud: HUDState) -> None:  # noqa: ANN001
        W, H = self.width, self.height
        self._panel(frame, 40, H - 130, W - 80, 100, alpha=0.78, tint=(18, 20, 28))
        cv2.putText(frame, self._clip(hud.dialogue_speaker, 30), (56, H - 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 226, 150), 1, cv2.LINE_AA)
        line = self._clip(str(world.dialogue.get("line", "")) if world.dialogue else "", 74)
        cv2.putText(frame, line, (56, H - 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (232, 236, 244), 1, cv2.LINE_AA)
        for k, opt in enumerate(hud.dialogue_options[:3]):
            sel = k == world.dialogue_cursor if hasattr(world, "dialogue_cursor") else False
            col = (255, 236, 170) if sel else (198, 204, 216)
            cv2.putText(frame, f"{k + 1}. {self._clip(opt, 62)}", (56, H - 62 + k * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)

    def _draw_menu(self, frame, world, hud: HUDState) -> None:  # noqa: ANN001
        W, H = self.width, self.height
        title = str(hud.menu).upper()
        self._panel(frame, 70, 44, W - 140, H - 96, alpha=0.84, tint=(16, 18, 26))
        cv2.putText(frame, title, (92, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (255, 232, 160), 2, cv2.LINE_AA)

        entries = []
        try:
            entries = list(world.menu_entries())
        except Exception:  # noqa: BLE001
            entries = []

        if hud.menu == "cook":
            for k, (name, qty) in enumerate(world.ingredients_for_cooking()):
                cv2.putText(frame, f"{name}  x{qty}", (100, 100 + k * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 216, 228), 1, cv2.LINE_AA)

        for k, entry in enumerate(entries[:14]):
            sel = k == hud.menu_cursor
            if sel:
                cv2.rectangle(frame, (92, 84 + k * 18), (W - 92, 100 + k * 18), (60, 92, 140), -1)
            cv2.putText(frame, self._clip(str(entry), 52), (100, 97 + k * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255) if sel else (198, 204, 216), 1, cv2.LINE_AA)

        if hud.menu == "map":
            self._draw_world_map(frame, world)
        cv2.putText(frame, "ENTER select    M / ESC close", (92, H - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 178, 190), 1, cv2.LINE_AA)

    def _draw_world_map(self, frame, world) -> None:  # noqa: ANN001
        W, H = self.width, self.height
        x0, y0, x1, y1 = W - 300, 92, W - 100, H - 90
        cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 40, 34), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (150, 160, 170), 1)
        sx = (x1 - x0) / (2 * TERRAIN_HALF)
        sy = (y1 - y0) / (2 * TERRAIN_HALF)
        for e in world.entities:
            if e.kind not in ("waypoint", "chest", "boss", "monument", "npc"):
                continue
            u = int(x0 + (e.x + TERRAIN_HALF) * sx)
            v = int(y1 - (e.z + TERRAIN_HALF) * sy)
            col = self._minimap_colour(e) or (200, 200, 200)
            cv2.circle(frame, (u, v), 3, col, -1, cv2.LINE_AA)
        p = world.player
        u = int(x0 + (p.x + TERRAIN_HALF) * sx)
        v = int(y1 - (p.z + TERRAIN_HALF) * sy)
        cv2.circle(frame, (u, v), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (u, v), 9, (255, 255, 255), 1, cv2.LINE_AA)

    # -- small drawing helpers --------------------------------------------------------------

    def _panel(self, frame, x: int, y: int, w: int, h: int, alpha: float = 0.55,
               tint: tuple[int, int, int] = (18, 20, 26)) -> None:  # noqa: ANN001
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), tint, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _bar(self, frame, x: int, y: int, w: int, h: int, frac: float,
             rgb: tuple[int, int, int]) -> None:  # noqa: ANN001
        frac = float(np.clip(frac, 0.0, 1.0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (26, 26, 30), -1)
        cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), tuple(int(c) for c in rgb), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (210, 210, 216), 1)

    @staticmethod
    def _clip(text: str, n: int) -> str:
        text = str(text).replace("\n", " ")
        return text if len(text) <= n else text[: n - 1] + "\u2026"
