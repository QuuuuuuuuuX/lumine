"""Perception: getting 5 Hz raw pixels off whatever is actually rendering.

Three sources behind one interface:

``sim``     the built-in sandbox (default; runs anywhere)
``screen``  a real game window on this machine (X11 via ``mss``, or ffmpeg/x11grab)
``video``   a recorded gameplay file, for replay and offline data mining

The detector below is deliberately classical (colour + contour, ~10 ms) rather than a
learned object detector.  It runs at 30 Hz on the control thread, which is exactly the job
a big model cannot do; the VLM handles the semantic questions instead.  When the sandbox is
the source it already knows where everything is, so the detector defers to the renderer's
ground-truth boxes and the pipeline stays honest either way.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import cv2
import numpy as np

from .config import PerceptionConfig
from .types import Frame, HUDState, Observation


# --------------------------------------------------------------------------------------
# Entity boxes: the shared currency between pixels and control
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class EntityBox:
    name: str
    kind: str
    bbox: tuple[int, int, int, int]          # x0, y0, x1, y1
    distance: float = 0.0
    visible: bool = True

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


def boxes_from_info(info: dict[str, Any]) -> list[EntityBox]:
    out: list[EntityBox] = []
    for b in info.get("entity_boxes") or ():
        try:
            bb = tuple(int(v) for v in b["bbox"])  # type: ignore[index]
            out.append(EntityBox(
                name=str(b.get("name", b.get("kind", "entity"))),
                kind=str(b.get("kind", "entity")),
                bbox=bb,                                  # type: ignore[arg-type]
                distance=float(b.get("distance", 0.0)),
                visible=bool(b.get("visible", True)),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------------------
# Colour detector for the real-screen path
# --------------------------------------------------------------------------------------

#: Rough BGR ranges for the things that matter when there is no ground truth available.
_COLOURS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "chest": ((0, 150, 190), (90, 235, 255)),       # gold
    "enemy": ((0, 0, 150), (90, 90, 255)),          # red
    "npc": ((150, 120, 0), (255, 220, 90)),         # cyan-ish label
    "waypoint": ((200, 90, 0), (255, 200, 90)),     # blue spire
    "collectible": ((180, 120, 0), (255, 235, 120)),  # warm glow
}


class ColourDetector:
    """Fast classical detector used when the frame source has no ground-truth boxes."""

    def __init__(self, min_area: int = 80, max_boxes: int = 12) -> None:
        self.min_area = min_area
        self.max_boxes = max_boxes

    def detect(self, image: np.ndarray) -> list[EntityBox]:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        out: list[EntityBox] = []
        for kind, (lo, hi) in _COLOURS.items():
            mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if cv2.contourArea(c) < self.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                out.append(EntityBox(name=kind, kind=kind, bbox=(x, y, x + w, y + h)))
        out.sort(key=lambda b: b.area, reverse=True)
        return out[: self.max_boxes]


def detect_telegraph(image: np.ndarray) -> tuple[float, float] | None:
    """Locate an incoming-attack telegraph (a saturated red ring on the ground).

    Returns the screen centroid to flee from, or ``None``.  This is a reflex, so it has to
    run in single-digit milliseconds and must not depend on the sandbox.
    """
    h, w = image.shape[:2]
    roi = image[int(h * 0.35):, :]
    r = roi[:, :, 0].astype(np.int16)
    g = roi[:, :, 1].astype(np.int16)
    b = roi[:, :, 2].astype(np.int16)
    mask = (r > 150) & (g < 95) & (b < 95) & (r - g > 70)
    if int(mask.sum()) < 140:
        return None
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean()) + h * 0.35


# --------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------


class FrameSource:
    """Base class.  ``grab`` must be cheap enough to call at the perception rate."""

    name = "base"

    def grab(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ScreenSource(FrameSource):
    """Capture a region of the real display.

    Prefers ``mss`` (pure Python, fast, X11 and Wayland-with-XWayland).  Falls back to
    ``ffmpeg -f x11grab`` so that a machine without mss still works, since ffmpeg is
    usually present wherever a game is.
    """

    name = "screen"

    def __init__(self, cfg: PerceptionConfig) -> None:
        self.cfg = cfg
        self._sct = None
        self._ff: subprocess.Popen | None = None
        self._monitor: dict[str, int] = {}

        try:
            import mss  # type: ignore

            self._sct = mss.mss()
            mons = self._sct.monitors
            idx = min(cfg.monitor, len(mons) - 1)
            m = mons[idx] if idx >= 0 else mons[0]
            if cfg.capture_region:
                left, top, w, h = cfg.capture_region
                self._monitor = {"left": left, "top": top, "width": w, "height": h}
            else:
                self._monitor = {"left": m["left"], "top": m["top"],
                                 "width": m["width"], "height": m["height"]}
            return
        except Exception:
            self._sct = None

        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "screen capture needs either the 'mss' package or ffmpeg on PATH. "
                "Install one: pip install mss"
            )
        left, top, w, h = cfg.capture_region or (0, 0, cfg.width, cfg.height)
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "x11grab", "-video_size", f"{w}x{h}",
               "-i", f"{':0.0+' if not str(top).startswith('-') else ':0.0'}{left},{top}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self._ff = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._size = (h, w)

    def grab(self) -> np.ndarray:
        if self._sct is not None:
            shot = self._sct.grab(self._monitor)
            img = np.asarray(shot)[:, :, :3][:, :, ::-1]     # BGRA -> RGB
        else:
            assert self._ff and self._ff.stdout
            h, w = self._size
            buf = self._ff.stdout.read(h * w * 3)
            if len(buf) < h * w * 3:
                raise RuntimeError("ffmpeg capture stream ended")
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        return cv2.resize(img, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        if self._ff is not None:
            self._ff.kill()
            self._ff = None


class VideoSource(FrameSource):
    """Replay a recorded gameplay file at its native frame rate."""

    name = "video"

    def __init__(self, path: str, cfg: PerceptionConfig) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path!r}")
        self.cfg = cfg

    def grab(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        self.cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class SimSource(FrameSource):
    """Pulls frames straight out of the sandbox renderer."""

    name = "sim"

    def __init__(self, env) -> None:  # noqa: ANN001 - avoids a circular import
        self.env = env

    def grab(self) -> np.ndarray:
        return self.env.render_frame()


def build_source(cfg: PerceptionConfig, env=None) -> FrameSource:  # noqa: ANN001
    if cfg.source == "sim":
        if env is None:
            raise RuntimeError("perception.source='sim' requires an environment")
        return SimSource(env)
    if cfg.source == "screen":
        return ScreenSource(cfg)
    if cfg.source == "video":
        raise RuntimeError("perception.source='video' requires a path; use VideoSource directly")
    raise ValueError(f"unknown perception.source {cfg.source!r}")


# --------------------------------------------------------------------------------------
# The 5 Hz perception stage
# --------------------------------------------------------------------------------------


@dataclass
class Perceiver:
    """Turns a :class:`FrameSource` into the observation the agent reasons over.

    Rate limiting lives here rather than in the source, so that a 144 Hz game and a 30 Hz
    sim both present exactly ``cfg.hz`` frames per second to the brain.
    """

    cfg: PerceptionConfig
    source: FrameSource | None = None
    env: Any = None
    detector: ColourDetector = field(default_factory=ColourDetector)

    _last_frame: Frame | None = None
    _last_grab: float = 0.0
    _boxes: list[EntityBox] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Only the sandbox gets its frames from the object it steps; a LiveEnv drives its own
        # capture and hands us already-paced frames.
        if self.source is None and self.env is not None and self.cfg.source == "sim":
            self.source = SimSource(self.env)

    @property
    def period(self) -> float:
        return 1.0 / max(1e-6, self.cfg.hz)

    def observe(self, env_obs: Observation | None, force: bool = False) -> Observation:
        """Return an observation, refreshing the image only when the 5 Hz clock allows."""
        now = time.monotonic()
        if env_obs is not None:
            # The sandbox already paces its own rendering and reports ``frame_is_new``.
            if env_obs.frame_is_new:
                self._last_frame = env_obs.frame
                self._boxes = boxes_from_info(env_obs.info) or self.detector.detect(env_obs.frame.image)
            elif self._last_frame is not None:
                env_obs.frame = self._last_frame
                env_obs.frame_is_new = False
            return env_obs

        assert self.source is not None
        fresh = force or (now - self._last_grab) >= self.period
        if fresh or self._last_frame is None:
            img = self.source.grab()
            self._last_frame = Frame(image=img, t=now)
            self._last_grab = now
            self._boxes = self.detector.detect(img)
        return Observation(frame=self._last_frame, hud=HUDState(), t=now,
                           frame_is_new=fresh,
                           info={"entity_boxes": [b.__dict__ for b in self._boxes]})

    def boxes(self) -> list[EntityBox]:
        return list(self._boxes)

    def close(self) -> None:
        if self.source is not None:
            self.source.close()
