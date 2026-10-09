"""The action head: the learned replacement for the hand-written controller.

Scope, stated plainly: this is **not** the brain.  It is the 30 Hz primitive layer that
Lumine's stage-0 pretraining produces -- "master action primitives" -- and it is small on
purpose (a few hundred thousand parameters, a few milliseconds per frame).  The VLM decides
*what*; this decides *how*, and it is what lets a 2-second brain drive a 30 Hz game.

Two backends behind one interface:

``TorchActionHead``  a small CNN with a key multi-label head, a mouse regression head and a
                     button head.  Trained with behaviour cloning on recorded play.
``NumpyActionHead``  logistic + ridge regression over downsampled pixels.  Weaker, but it
                     runs on a machine with no torch, which is the difference between a
                     recipe that reproduces somewhere else and one that only runs here.

Both save to the same ``.npz``-plus-JSON directory so a checkpoint is portable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import Action, BUTTONS, KEYS, NUM_KEYS


@dataclass
class PolicyOutput:
    action: Action
    confidence: float
    key_probs: np.ndarray
    mouse: np.ndarray

    def describe(self) -> str:
        return f"{self.action.describe()} (conf={self.confidence:.2f})"


def _prep(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return gray


# --------------------------------------------------------------------------------------
# Torch backend
# --------------------------------------------------------------------------------------


class TorchActionHead:
    backend = "torch"

    def __init__(self, width: int = 160, height: int = 90, channels: tuple[int, ...] = (32, 64, 64),
                 device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.width, self.height = width, height
        self.device = torch.device(device)
        self.mouse_scale = 20.0

        c1, c2, c3 = channels

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.body = nn.Sequential(
                    nn.Conv2d(1, c1, 5, stride=2, padding=2), nn.ReLU(),
                    nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((3, 5)),
                )
                feat = c3 * 15
                self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(feat, 256), nn.ReLU())
                self.key_head = nn.Linear(256, NUM_KEYS)
                self.mouse_head = nn.Linear(256, 2)
                self.button_head = nn.Linear(256, len(BUTTONS))

            def forward(self, x):  # noqa: ANN001, ANN201
                h = self.trunk(self.body(x))
                return self.key_head(h), self.mouse_head(h), self.button_head(h)

        self.net = Net().to(self.device)

    # -- inference --------------------------------------------------------------------

    def predict(self, image: np.ndarray) -> PolicyOutput:
        torch = self.torch
        gray = _prep(image, self.width, self.height)
        x = torch.from_numpy(gray)[None, None].to(self.device)
        with torch.no_grad():
            klog, mreg, blog = self.net(x)
            kprob = torch.sigmoid(klog)[0].cpu().numpy()
            mouse = mreg[0].cpu().numpy() * self.mouse_scale
            bprob = torch.sigmoid(blog)[0].cpu().numpy()

        keys = {KEYS[i] for i in range(NUM_KEYS) if kprob[i] > 0.5}
        buttons = {BUTTONS[i] for i in range(len(BUTTONS)) if bprob[i] > 0.5}
        conf = float(np.mean(np.abs(kprob - 0.5) * 2.0))
        return PolicyOutput(Action(keys=keys, buttons=buttons,
                                   mouse_dx=float(mouse[0]), mouse_dy=float(mouse[1])),
                            conf, kprob, mouse)

    # -- persistence ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.net.state_dict(), p / "action_head.pt")
        (p / "action_head.json").write_text(json.dumps({
            "backend": "torch", "width": self.width, "height": self.height,
            "mouse_scale": self.mouse_scale, "keys": list(KEYS), "buttons": list(BUTTONS),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "TorchActionHead":
        p = Path(path)
        meta = json.loads((p / "action_head.json").read_text())
        head = cls(width=meta["width"], height=meta["height"], device=device)
        head.mouse_scale = meta.get("mouse_scale", 20.0)
        head.net.load_state_dict(head.torch.load(p / "action_head.pt", map_location=head.device))
        head.net.eval()
        return head


# --------------------------------------------------------------------------------------
# NumPy backend
# --------------------------------------------------------------------------------------


class NumpyActionHead:
    """Logistic regression for keys, ridge regression for the mouse -- and nothing else.

    A deliberately weak model.  Its job is to prove the pipeline (collect -> train ->
    replay) end to end on a machine without torch, not to compete with the CNN.
    """

    backend = "numpy"

    def __init__(self, width: int = 32, height: int = 18) -> None:
        self.width, self.height = width, height
        self.dim = width * height + 1
        self.w_key = np.zeros((self.dim, NUM_KEYS), np.float32)
        self.w_mouse = np.zeros((self.dim, 2), np.float32)
        self.mouse_scale = 20.0

    def _features(self, image: np.ndarray) -> np.ndarray:
        gray = _prep(image, self.width, self.height).reshape(-1)
        return np.concatenate([gray, [1.0]]).astype(np.float32)

    def predict(self, image: np.ndarray) -> PolicyOutput:
        f = self._features(image)
        logits = f @ self.w_key
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        mouse = (f @ self.w_mouse) * self.mouse_scale
        keys = {KEYS[i] for i in range(NUM_KEYS) if probs[i] > 0.5}
        conf = float(np.mean(np.abs(probs - 0.5) * 2.0))
        return PolicyOutput(Action(keys=keys, mouse_dx=float(mouse[0]), mouse_dy=float(mouse[1])),
                            conf, probs, mouse)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p / "action_head.npz", w_key=self.w_key, w_mouse=self.w_mouse,
                            mouse_scale=self.mouse_scale)
        (p / "action_head.json").write_text(json.dumps({
            "backend": "numpy", "width": self.width, "height": self.height,
            "keys": list(KEYS), "buttons": list(BUTTONS),
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "NumpyActionHead":
        p = Path(path)
        z = np.load(p / "action_head.npz")
        head = cls()
        head.w_key = z["w_key"]
        head.w_mouse = z["w_mouse"]
        head.mouse_scale = float(z["mouse_scale"])
        return head


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_action_head(path: str | Path, device: str = "auto") -> Any:
    """Load whichever backend the checkpoint declares, preferring torch when available."""
    p = Path(path)
    meta = json.loads((p / "action_head.json").read_text())
    backend = meta.get("backend", "numpy")
    if backend == "torch":
        dev = device
        if dev == "auto":
            try:
                import torch

                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        return TorchActionHead.load(p, device=dev)
    return NumpyActionHead.load(p)


def best_backend(prefer_torch: bool = True) -> str:
    if not prefer_torch:
        return "numpy"
    try:
        import torch  # noqa: F401
    except Exception:
        return "numpy"
    return "torch"
