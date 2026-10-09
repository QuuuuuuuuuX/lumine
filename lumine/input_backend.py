"""Action emission back-ends: where a 30 Hz :class:`Action` actually goes.

``sim``     nothing.  The sandbox receives the action directly through ``env.step``.
``dryrun``  nothing, but logged.  Use it to measure the agent's decision rate without a game.
``uinput``  a real virtual keyboard and mouse, via ``/dev/uinput``.

The uinput back-end is written against the kernel ABI with ``struct`` rather than pulling in
``evdev`` or ``python-uinput``, because it is the one component that has to work on a gaming
box that is otherwise a stranger's machine, and a 60-line struct packer has no install step.

Requirements for ``uinput``: write access to ``/dev/uinput`` (usually the ``input`` group).
On Wayland, injected events from ``/dev/uinput`` are seen by games running under XWayland
and by native Wayland clients that use libinput; if a specific game ignores them, that is a
game-level anti-cheat or raw-input decision, not a bug here.
"""

from __future__ import annotations

import fcntl
import logging
import os
import struct
import time
from dataclasses import dataclass, field

from .types import Action

log = logging.getLogger("lumine.input")

# --------------------------------------------------------------------------------------
# Linux input constants
# --------------------------------------------------------------------------------------

EV_SYN, EV_KEY, EV_REL = 0x00, 0x01, 0x02
SYN_REPORT = 0x00

REL_X, REL_Y, REL_WHEEL = 0x00, 0x01, 0x08

BTN_LEFT, BTN_RIGHT, BTN_MIDDLE = 0x110, 0x111, 0x112

UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_DEV_SETUP = 0x405C5503

BUS_USB = 0x03

#: Our canonical key name -> Linux input event code.
KEYCODE: dict[str, int] = {
    "ESC": 1, "1": 2, "2": 3, "3": 4, "4": 5,
    "Q": 16, "W": 17, "E": 18, "R": 19, "T": 20,
    "ENTER": 28, "CTRL": 29, "A": 30, "S": 31, "D": 32, "F": 33,
    "J": 36, "K": 37, "L": 38, "Z": 44, "X": 45, "C": 46, "B": 48,
    "M": 50, "SPACE": 57, "TAB": 15, "SHIFT": 42,
    "UP": 103, "LEFT": 105, "RIGHT": 106, "DOWN": 108,
}

BUTTONCODE: dict[str, int] = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}

#: int(3) timeval + u16 type + u16 code + int(4) value
_INPUT_EVENT = struct.Struct("llHHi")
#: legacy uinput_user_dev
_USER_DEV = struct.Struct("80sHHHHi" + "i" * 256)


class ActionBackend:
    """Where actions go.  ``send`` is called once per control tick."""

    name = "base"

    def send(self, action: Action) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "ActionBackend":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SimBackend(ActionBackend):
    """No-op: ``SimEnv.step`` already receives the action object directly."""

    name = "sim"

    def send(self, action: Action) -> None:  # noqa: ARG002
        return


class DryRunBackend(ActionBackend):
    """Log every non-trivial action.  Useful for rate measurements off the sandbox."""

    name = "dryrun"

    def __init__(self, sample_every: int = 15) -> None:
        self.sample_every = sample_every
        self.ticks = 0
        self.emitted = 0

    def send(self, action: Action) -> None:
        self.ticks += 1
        if not action.is_noop():
            self.emitted += 1
            if self.ticks % self.sample_every == 0:
                log.info("tick %6d  %s", self.ticks, action.describe())


@dataclass
class _Device:
    fd: int
    name: str


class UInputBackend(ActionBackend):
    """A virtual keyboard + mouse created through ``/dev/uinput``.

    Keys are edge-triggered: the backend remembers what is currently held and only emits
    press/release events for the difference.  That is what makes holding ``W`` across 30
    ticks behave like a human holding a key instead of 30 rapid taps (which most games
    treat completely differently).
    """

    name = "uinput"

    def __init__(self, path: str = "/dev/uinput", mouse_scale: float = 1.0) -> None:
        self.path = path
        self.mouse_scale = mouse_scale
        self._kbd: _Device | None = None
        self._mouse: _Device | None = None
        self._held_keys: set[str] = set()
        self._held_buttons: set[str] = set()
        self._residual_x = 0.0
        self._residual_y = 0.0
        self._open()

    # -- setup ---------------------------------------------------------------------------

    def _open(self) -> None:
        if not os.path.exists(self.path):
            raise RuntimeError(
                f"{self.path} does not exist. The uinput kernel module is not loaded. "
                f"Try: sudo modprobe uinput"
            )
        try:
            self._kbd = self._create_device("lumine-keyboard", buttons=False)
            self._mouse = self._create_device("lumine-mouse", buttons=True)
        except PermissionError as exc:
            raise RuntimeError(
                f"permission denied opening {self.path}. Add yourself to the 'input' group "
                f"and re-log in:  sudo usermod -aG input $USER"
            ) from exc

    def _create_device(self, name: str, buttons: bool) -> _Device:
        fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
        fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
        fcntl.ioctl(fd, UI_SET_EVBIT, EV_SYN)

        if buttons:
            fcntl.ioctl(fd, UI_SET_EVBIT, EV_REL)
            for rel in (REL_X, REL_Y, REL_WHEEL):
                fcntl.ioctl(fd, UI_SET_RELBIT, rel)
            for code in BUTTONCODE.values():
                fcntl.ioctl(fd, UI_SET_KEYBIT, code)
        else:
            for code in KEYCODE.values():
                fcntl.ioctl(fd, UI_SET_KEYBIT, code)

        # Legacy uinput_user_dev: name, input_id, ff effects, absmax/min/fuzz/flat tables.
        dev = _USER_DEV.pack(
            name.encode()[:79], BUS_USB, 0x1234, 0x5678, 1, 0,
            *([0] * 256),
        )
        os.write(fd, dev)
        fcntl.ioctl(fd, UI_DEV_CREATE)
        # The kernel needs a moment before the node accepts events.
        time.sleep(0.12)
        return _Device(fd=fd, name=name)

    # -- emission ------------------------------------------------------------------------

    def _emit(self, dev: _Device, etype: int, code: int, value: int) -> None:
        os.write(dev.fd, _INPUT_EVENT.pack(0, 0, etype, code, value))

    def _sync(self, dev: _Device) -> None:
        self._emit(dev, EV_SYN, SYN_REPORT, 0)

    def _key(self, name: str, down: bool) -> None:
        code = KEYCODE.get(name)
        if code is None or self._kbd is None:
            return
        self._emit(self._kbd, EV_KEY, code, 1 if down else 0)
        self._sync(self._kbd)

    def _button(self, name: str, down: bool) -> None:
        code = BUTTONCODE.get(name)
        if code is None or self._mouse is None:
            return
        self._emit(self._mouse, EV_KEY, code, 1 if down else 0)

    def send(self, action: Action) -> None:
        want_keys = set(action.keys)
        for key in want_keys - self._held_keys:
            self._key(key, True)
        for key in self._held_keys - want_keys:
            self._key(key, False)
        self._held_keys = want_keys

        want_buttons = set(action.buttons)
        for btn in want_buttons - self._held_buttons:
            self._button(btn, True)
        for btn in self._held_buttons - want_buttons:
            self._button(btn, False)
        self._held_buttons = want_buttons

        # Accumulate sub-pixel mouse motion instead of truncating it away: at 30 Hz a
        # sensitivity of 0.4 would otherwise round to zero and the camera would never turn.
        if action.mouse_dx or action.mouse_dy:
            self._residual_x += action.mouse_dx * self.mouse_scale
            self._residual_y += action.mouse_dy * self.mouse_scale
            dx, dy = int(self._residual_x), int(self._residual_y)
            self._residual_x -= dx
            self._residual_y -= dy
            if self._mouse is not None and (dx or dy):
                if dx:
                    self._emit(self._mouse, EV_REL, REL_X, dx)
                if dy:
                    self._emit(self._mouse, EV_REL, REL_Y, dy)

        if self._mouse is not None:
            self._sync(self._mouse)

    def close(self) -> None:
        for key in list(self._held_keys):
            self._key(key, False)
        for btn in list(self._held_buttons):
            self._button(btn, False)
        for dev in (self._kbd, self._mouse):
            if dev is None:
                continue
            try:
                fcntl.ioctl(dev.fd, UI_DEV_DESTROY)
            except OSError:
                pass
            os.close(dev.fd)
        self._kbd = self._mouse = None


class XDoToolBackend(ActionBackend):
    """Fallback: shell out to ``xdotool``.  Much slower, but needs no kernel access."""

    name = "xdotool"

    _XDOTOOL_KEY = {
        "SPACE": "space", "SHIFT": "shift", "CTRL": "ctrl", "ENTER": "Return",
        "ESC": "Escape", "UP": "Up", "DOWN": "Down", "LEFT": "Left", "RIGHT": "Right",
        "TAB": "Tab",
    }

    def __init__(self) -> None:
        import shutil

        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool not found on PATH")
        self._held: set[str] = set()
        self.residual_x = 0.0
        self.residual_y = 0.0

    def send(self, action: Action) -> None:
        import subprocess

        def name(k: str) -> str:
            return self._XDOTOOL_KEY.get(k, k.lower())

        for key in set(action.keys) - self._held:
            subprocess.run(["xdotool", "keydown", name(key)], check=False)
        for key in self._held - set(action.keys):
            subprocess.run(["xdotool", "keyup", name(key)], check=False)
        self._held = set(action.keys)

        self.residual_x += action.mouse_dx
        self.residual_y += action.mouse_dy
        dx, dy = int(self.residual_x), int(self.residual_y)
        self.residual_x -= dx
        self.residual_y -= dy
        if dx or dy:
            subprocess.run(["xdotool", "mousemove_relative", "--", str(dx), str(dy)], check=False)

    def close(self) -> None:
        import subprocess

        for key in self._held:
            subprocess.run(["xdotool", "keyup", self._XDOTOOL_KEY.get(key, key.lower())],
                           check=False)


def build_backend(kind: str, mouse_scale: float = 1.0) -> ActionBackend:
    """Create the configured back-end, with the least-surprising available fallback."""
    if kind == "sim":
        return SimBackend()
    if kind == "dryrun":
        return DryRunBackend()
    if kind == "uinput":
        try:
            return UInputBackend(mouse_scale=mouse_scale)
        except RuntimeError as exc:
            log.warning("uinput unavailable (%s); trying xdotool", exc)
            try:
                return XDoToolBackend()
            except RuntimeError:
                raise
    raise ValueError(f"unknown action backend {kind!r}")
