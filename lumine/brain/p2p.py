"""Bridge to Open Pixel2Play (P2P): an open keyboard-and-mouse game policy.

Why this file exists
--------------------
P2P (Player2 / elefant-ai, MIT, arXiv 2601.04575) is an 8,000-hour behaviour-cloning game
policy that takes **frames + text and emits keyboard and mouse** — the same interface this
project consumes. Its smallest checkpoint runs at roughly 80 Hz on a consumer GPU, against
the ~0.5 Hz we get from a cloud vision-language model. If it works, it slots in where the
hand-written controller is weakest.

Where it plugs in
-----------------
:class:`P2PPolicy` implements the same ``predict(image) -> PolicyOutput`` shape that
:meth:`lumine.controller.SkillController._blend` already calls for a learned action head, so
the agent does not change. :class:`P2PActionCodec` is the pure-logic half — P2P's 8-token
structured action to and from :class:`~lumine.types.Action` — and it is deliberately free of
torch, so it can be tested without the 2.2 GB checkpoint.

The action space, read from ``elefant/data/action_mapping.py`` upstream
-----------------------------------------------------------------------
Eight autoregressive tokens per control step, in order::

    [ key_0, key_1, key_2, key_3, mouse_button_0, mouse_button_1, delta_x, delta_y ]

* keyboard: 20 choices (0 = ``_no_key``), sorted ascending, zero-padded to 4 slots
* mouse buttons: 4 choices (0 = none, 1 = left, 2 = right, 3 = middle), padded to 2 slots
* ``delta_x``: 23 bins, centres from -501 to +501 px
* ``delta_y``: 17 bins, centres from -151 to +151 px (coarser; vertical is used less)

The honest limitation is in :func:`coverage_report`, and it is significant: P2P's keymap has
18 usable keys and this project's sandbox uses 31. See the module docstring of that function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..policy import PolicyOutput
from ..types import BUTTONS, KEYS, Action

__all__ = [
    "P2P_KEY_TO_DSH",
    "DSH_TO_P2P_KEY",
    "P2P_BUTTON_TO_DSH",
    "P2PActionCodec",
    "P2PPolicy",
    "coverage_report",
]

# --------------------------------------------------------------------------------------
# Key and button vocabularies, transcribed from upstream `action_mapping.py`
# --------------------------------------------------------------------------------------

#: P2P token id -> this project's key name.  Id 0 is the ``_no_key`` pad.
P2P_KEY_TO_DSH: dict[int, str] = {
    0: "",                 # _no_key
    1: "SPACE",
    2: "1",
    3: "2",
    4: "3",
    5: "4",
    6: "A",
    7: "D",
    8: "E",
    9: "F",
    10: "Q",
    11: "W",
    12: "S",
    13: "Z",
    14: "DOWN",
    15: "UP",
    16: "LEFT",
    17: "RIGHT",
    18: "SHIFT",
    19: "SHIFT",           # RightShift collapses onto our single SHIFT
}

DSH_TO_P2P_KEY: dict[str, int] = {
    "SPACE": 1, "1": 2, "2": 3, "3": 4, "4": 5,
    "A": 6, "D": 7, "E": 8, "F": 9, "Q": 10, "W": 11, "S": 12, "Z": 13,
    "DOWN": 14, "UP": 15, "LEFT": 16, "RIGHT": 17, "SHIFT": 18,
}

P2P_BUTTON_TO_DSH: dict[int, str] = {0: "", 1: "left", 2: "right", 3: "middle"}
DSH_TO_P2P_BUTTON: dict[str, int] = {v: k for k, v in P2P_BUTTON_TO_DSH.items() if v}

#: Bin edges and centres, copied verbatim from upstream so decoded magnitudes match theirs.
MOUSE_X_BIN_EDGES = [-5.001e02, -1.441e02, -7.610e01, -4.610e01, -2.910e01, -1.810e01,
                     -1.110e01, -6.100e00, -3.100e00, -1.100e00, -1.000e-01, 0.000e00,
                     1.000e00, 3.000e00, 6.000e00, 1.100e01, 1.800e01, 2.900e01,
                     4.600e01, 7.600e01, 1.440e02, 5.000e02]
MOUSE_X_BIN_CENTERS = [-501, -322, -110, -61, -38, -24, -15, -9, -5, -2, -1, 0,
                       1, 2, 5, 9, 15, 24, 38, 61, 110, 322, 501]
MOUSE_X_BIN_CONSERVATIVE = [-501, -145, -77, -47, -30, -19, -12, -7, -4, -2, -1, 0,
                            1, 2, 4, 7, 12, 19, 30, 47, 77, 145, 501]

MOUSE_Y_BIN_EDGES = [-1.501e02, -2.410e01, -1.210e01, -7.100e00, -4.100e00, -3.100e00,
                     -1.100e00, -1.000e-01, 0.000e00, 1.000e00, 3.000e00, 4.000e00,
                     7.000e00, 1.200e01, 2.400e01, 1.500e02]
MOUSE_Y_BIN_CENTERS = [-151, -87, -18, -10, -6, -4, -2, -1, 0, 1, 2, 4, 6, 10, 18, 87, 151]
MOUSE_Y_BIN_CONSERVATIVE = [-151, -25, -13, -8, -5, -4, -2, -1, 0, 1, 2, 4, 5, 8, 13, 25, 151]

N_KEY_SLOTS = 4
N_BUTTON_SLOTS = 2
SEQ_LEN = N_KEY_SLOTS + N_BUTTON_SLOTS + 2          # == 8
N_KEY_CHOICES = 20
N_BUTTON_CHOICES = 4
N_X_BINS = len(MOUSE_X_BIN_EDGES) + 1               # == 23
N_Y_BINS = len(MOUSE_Y_BIN_EDGES) + 1               # == 17


# --------------------------------------------------------------------------------------
# Codec: P2P tokens <-> this project's Action
# --------------------------------------------------------------------------------------


@dataclass
class P2PActionCodec:
    """Stateless translation between P2P's 8 tokens and :class:`~lumine.types.Action`.

    ``mouse_scale`` exists because the two systems measure camera motion differently: P2P
    emits integer pixel deltas, this project's controller emits abstract counts that the
    sandbox multiplies by ``MOUSE_YAW_GAIN``. The default of 1.0 treats a P2P pixel as one
    count; raise it if the camera turns too slowly, lower it if the view snaps around.
    """

    mouse_scale: float = 1.0
    mouse_sampling: str = "mean"          # "mean" | "conservative" | "truncated_normal"

    # -- decode ---------------------------------------------------------------------------

    def decode(self, keys: Sequence[int], buttons: Sequence[int],
               mouse_x: int, mouse_y: int) -> Action:
        """P2P's 8 tokens -> an :class:`Action`.

        Unknown or out-of-range tokens are dropped rather than raising: this decoder sits on
        the control path, and a model emitting a stray index must cost a tick, not the run.
        """
        names: set[str] = set()
        for k in list(keys)[:N_KEY_SLOTS]:
            name = P2P_KEY_TO_DSH.get(int(k), "")
            if name:
                names.add(name)

        btns: set[str] = set()
        for b in list(buttons)[:N_BUTTON_SLOTS]:
            name = P2P_BUTTON_TO_DSH.get(int(b), "")
            if name:
                btns.add(name)

        dx = self._center(MOUSE_X_BIN_CENTERS, MOUSE_X_BIN_CONSERVATIVE, mouse_x) * self.mouse_scale
        dy = self._center(MOUSE_Y_BIN_CENTERS, MOUSE_Y_BIN_CONSERVATIVE, mouse_y) * self.mouse_scale
        return Action(keys=names, buttons=btns, mouse_dx=float(dx), mouse_dy=float(dy))

    def decode_sequence(self, tokens: Sequence[int]) -> Action:
        """Decode one flat 8-token step: ``[k0..k3, b0, b1, dx, dy]``."""
        t = list(tokens) + [0] * SEQ_LEN
        return self.decode(t[0:4], t[4:6], int(t[6]), int(t[7]))

    def _center(self, mean: Sequence[int], conservative: Sequence[int], idx: int) -> float:
        i = int(idx)
        if i < 0 or i >= len(mean):
            return 0.0
        if self.mouse_sampling == "conservative":
            return float(conservative[i])
        if self.mouse_sampling == "truncated_normal":
            # Upstream samples inside the bin instead of taking its centre; the spread is
            # what makes motion look human rather than stepped. Same idea, same std.
            std = 96.0 if len(mean) == N_X_BINS else 22.0
            edges = MOUSE_X_BIN_EDGES if len(mean) == N_X_BINS else MOUSE_Y_BIN_EDGES
            if i == 0:
                lo, hi = edges[0] - 100.0, edges[0]
            elif i >= len(edges):
                lo, hi = edges[-1], edges[-1] + 100.0
            else:
                lo, hi = edges[i - 1], edges[i]
            if hi <= lo:
                return float(mean[i])
            return float(np.clip(np.random.normal(0.0, std), lo, hi))
        return float(mean[i])

    # -- encode ---------------------------------------------------------------------------

    def encode(self, action: Action) -> tuple[list[int], list[int], int, int]:
        """An :class:`Action` -> P2P's tokens. Used for recording and for A/B comparison.

        Keys this project uses but P2P cannot express are dropped; call
        :func:`coverage_report` to see which, and how much of the vocabulary that is.
        """
        ids = sorted({DSH_TO_P2P_KEY[k] for k in action.keys if k in DSH_TO_P2P_KEY})
        ids = ids[:N_KEY_SLOTS]
        keys = ids + [0] * (N_KEY_SLOTS - len(ids))

        bids = sorted({DSH_TO_P2P_BUTTON[b] for b in action.buttons if b in DSH_TO_P2P_BUTTON})
        bids = bids[:N_BUTTON_SLOTS]
        buttons = bids + [0] * (N_BUTTON_SLOTS - len(bids))

        return keys, buttons, self._bin(MOUSE_X_BIN_EDGES, action.mouse_dx / self.mouse_scale), \
            self._bin(MOUSE_Y_BIN_EDGES, action.mouse_dy / self.mouse_scale)

    @staticmethod
    def _bin(edges: Sequence[float], value: float) -> int:
        """Bin a pixel delta exactly as upstream does.

        Upstream uses ``torch.bucketize(x, edges)`` with the default ``right=False``, whose
        contract is ``edges[i-1] < x <= edges[i]``. That is numpy's ``side="left"``. Using
        ``"right"`` shifts every non-edge value one bin too far, which on a control path
        means the camera turns slightly wrong forever -- a bug that reads as "the model is
        bad" rather than as an off-by-one.
        """
        return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="left"))


# --------------------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------------------


class P2PPolicy:
    """Runs a P2P checkpoint and exposes it as a drop-in action head.

    Implements ``predict(image) -> PolicyOutput``, which is the contract
    :meth:`SkillController._blend` already expects, so wiring it up is::

        cfg.control.controller = "learned"
        agent.controller.policy = P2PPolicy.from_checkpoint("checkpoints/150M")

    The heavy import happens inside :meth:`from_checkpoint`, not at module import, so that
    the codec and the coverage report stay usable on a machine with no torch.

    Status: **the codec is tested, the model path is not.** Upstream's checkpoint is 2.2 GB
    and its inference stack additionally pulls ``lightning``, protobuf and a Gemma tokenizer.
    None of that is unavailable here — see the feasibility notes in the project README — but
    it has not been run end to end on this machine, and this class does not pretend otherwise.
    """

    def __init__(self, model: Any, codec: P2PActionCodec | None = None,
                 temperature: float = 1.0) -> None:
        self.model = model
        self.codec = codec or P2PActionCodec()
        self.temperature = temperature
        self.frames_seen = 0
        self.last_tokens: list[int] = []

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: str = "auto",
                        codec: P2PActionCodec | None = None) -> "P2PPolicy":
        """Load a downloaded P2P checkpoint.

        Raises a message that says exactly what is missing rather than an ImportError from
        six frames deep, because the failure mode here is usually an undownloaded 2.2 GB
        file or a HuggingFace endpoint that is unreachable from this network.
        """
        ckpt_dir = Path(checkpoint_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"no P2P checkpoint at {ckpt_dir}. Download one with:\n"
                f"  HF_ENDPOINT=https://hf-mirror.com \\\n"
                f"    huggingface-cli download elefantai/open-p2p "
                f"150M/checkpoint-step=00500000.ckpt --local-dir {ckpt_dir.parent}"
            )
        try:
            import lightning  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "P2P inference needs PyTorch Lightning: pip install --user lightning"
            ) from exc

        raise NotImplementedError(
            "P2PPolicy.from_checkpoint is not wired up yet. The action codec and the coverage "
            "analysis in this module are complete and tested; loading upstream's Lightning "
            "module also requires their `elefant` package tree, a Gemma tokenizer and the "
            "protobuf stubs. See README section 'Open Pixel2Play: feasibility' for the exact "
            "remaining steps and the 2.2 GB download."
        )

    # -- action-head interface ------------------------------------------------------------

    def predict(self, image: np.ndarray) -> PolicyOutput:
        """Frame -> Action. Resizes to the model's 192x192 input and decodes 8 tokens."""
        raise NotImplementedError("predict() requires a loaded checkpoint; see from_checkpoint")

    @staticmethod
    def preprocess(image: np.ndarray, size: int = 192) -> np.ndarray:
        """Match upstream's 192x192 RGB input.

        Upstream resizes via ``resize_image_for_model``; this is the equivalent nearest-
        neighbour-free path using OpenCV, which is already a dependency here.
        """
        import cv2

        if image.shape[0] == size and image.shape[1] == size:
            return image
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------------------
# Coverage: the finding that actually decides feasibility
# --------------------------------------------------------------------------------------


def coverage_report() -> dict[str, Any]:
    """How much of this project's action vocabulary P2P can actually express.

    This is the number that decides whether the idea is viable, so it is computed rather
    than asserted. The answer is uncomfortable: P2P's keymap was built for shooters and
    platformers (``wasd``, ``space``, ``shift``, arrows, ``e``/``q``/``f``/``z``) and this
    sandbox's keymap was built for an action-RPG. In particular P2P has **no ``J``**, which
    is this sandbox's normal attack.

    That is not fatal — the natural fix is to remap the sandbox's attack onto the left mouse
    button, which P2P does emit and which is what Genshin Impact uses on PC anyway. But it
    does mean P2P cannot be dropped in without also deciding what each missing key means.
    """
    supported = sorted(set(DSH_TO_P2P_KEY))
    unsupported = [k for k in KEYS if k not in DSH_TO_P2P_KEY]
    buttons_supported = sorted(DSH_TO_P2P_BUTTON)
    return {
        "keys_total": len(KEYS),
        "keys_supported": len(supported),
        "keys_unsupported": unsupported,
        "key_coverage": round(len(supported) / len(KEYS), 4),
        "buttons_total": len(BUTTONS),
        "buttons_supported": len(buttons_supported),
        "button_coverage": round(len(buttons_supported) / len(BUTTONS), 4),
        "mouse_x_bins": N_X_BINS,
        "mouse_y_bins": N_Y_BINS,
        "tokens_per_step": SEQ_LEN,
        "max_simultaneous_keys": N_KEY_SLOTS,
        # The mapping is many-to-one for SHIFT: LeftShift and RightShift collapse to one.
        "collapsed_keys": {"SHIFT": "LeftShift + RightShift"},
    }


def format_coverage(report: dict[str, Any] | None = None) -> str:
    r = report or coverage_report()
    lines = [
        "P2P <-> lumine action-space coverage",
        "",
        f"  keys     : {r['keys_supported']}/{r['keys_total']} supported "
        f"({r['key_coverage'] * 100:.0f}%)",
        f"  buttons  : {r['buttons_supported']}/{r['buttons_total']} "
        f"({r['button_coverage'] * 100:.0f}%)",
        f"  mouse    : {r['mouse_x_bins']} x-bins, {r['mouse_y_bins']} y-bins",
        f"  per step : {r['tokens_per_step']} tokens, "
        f"up to {r['max_simultaneous_keys']} keys at once",
        "",
        "  NOT expressible by P2P:",
    ]
    for k in r["keys_unsupported"]:
        lines.append(f"    - {k}")
    lines.append("")
    lines.append("  Attack note: this sandbox binds normal attack to J, which P2P has no token")
    lines.append("  for. P2P does emit the left mouse button, which is what Genshin Impact uses")
    lines.append("  on PC, so the fix is a sandbox rebind rather than an adapter hack.")
    return "\n".join(lines)
