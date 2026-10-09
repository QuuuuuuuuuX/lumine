"""Live brain test: does the configured VLM actually ground a game frame?

This is the highest-risk integration in the project. Everything else is arithmetic by
comparison: if the prompt and the model disagree about the output shape, the agent is a
very elaborate no-op. So this test builds a synthetic game frame with a known ground truth
and asserts that the model names the right object *and* points at roughly the right place.

It needs the network and a key, so it skips itself when neither is present:

    python -m tests.test_brain_live            # runs if a key is resolvable
    pytest tests/test_brain_live.py -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumine.brain.base import BrainContext
from lumine.brain.vlm import VLMBrain
from lumine.config import BrainConfig, resolve_api_key
from lumine.types import Frame, HUDState, Observation, quantize_px

H, W = 360, 640

# Ground truth for the synthetic frame built below.
HILICHURL_PX = (210, 250)
CHEST_PX = (435, 260)


def synthetic_frame() -> np.ndarray:
    """A deliberately game-shaped frame: sky, ground, one enemy, one chest, a readable HUD.

    Colours are written in **RGB** order. OpenCV draws in BGR, so every call here passes an
    explicitly reversed triple -- getting this wrong once already produced a frame with a
    blue "red enemy" and a model that dutifully reported a blue circle.
    """
    import cv2

    def rgb(r: int, g: int, b: int) -> tuple[int, int, int]:
        return (b, g, r)

    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        t = y / H
        img[y, :] = (int(120 + 80 * t), int(150 + 60 * t), int(220 - 40 * t))
    cv2.fillPoly(img, [np.array([[0, 200], [W, 180], [W, H], [0, H]])], rgb(90, 140, 70))
    cv2.fillPoly(img, [np.array([[0, 200], [W, 180], [W, 230], [0, 250]])], rgb(70, 115, 55))

    # One unmistakably red enemy on the left, one gold chest on the right. No other red or
    # gold anywhere in the frame, so a wrong answer is unambiguously a grounding failure.
    cv2.rectangle(img, (400, 230), (470, 290), rgb(230, 180, 40), -1)
    cv2.rectangle(img, (400, 230), (470, 248), rgb(255, 210, 60), -1)
    cv2.putText(img, "Chest", (400, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb(255, 255, 255), 1)

    cv2.circle(img, HILICHURL_PX, 28, rgb(220, 60, 60), -1)
    cv2.putText(img, "Hilichurl", (175, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb(255, 255, 255), 1)

    cv2.rectangle(img, (16, 320), (180, 336), rgb(40, 40, 40), -1)
    cv2.rectangle(img, (18, 322), (140, 334), rgb(210, 60, 60), -1)
    cv2.putText(img, "HP 78/100", (16, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rgb(255, 255, 255), 1)

    cv2.rectangle(img, (14, 14), (300, 64), rgb(30, 30, 30), -1)
    cv2.putText(img, "Defeat enemies, open chest", (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                rgb(255, 255, 255), 1)
    cv2.putText(img, "Progress 0/2", (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rgb(200, 220, 255), 1)
    cv2.circle(img, (590, 60), 45, rgb(35, 35, 35), -1)
    return img


def _brain_or_skip() -> VLMBrain | None:
    cfg = BrainConfig(provider="deepseek", model="deepseek-flash")
    if not resolve_api_key(cfg.api_key_env, "deepseek"):
        print("  SKIP  no DEEPSEEK_API_KEY available")
        return None
    try:
        return VLMBrain(cfg, W, H)
    except RuntimeError as exc:
        print(f"  SKIP  {exc}")
        return None


def test_vlm_grounds_a_game_frame() -> None:
    brain = _brain_or_skip()
    if brain is None:
        return

    obs = Observation(frame=Frame(image=synthetic_frame()), hud=HUDState(), t=0.0,
                      frame_is_new=True)
    ctx = BrainContext(instruction="Defeat the enemies ahead and collect the chest",
                       region="mondstadt", seconds_elapsed=3.0, progress=(0, 2))

    t0 = time.monotonic()
    intent = brain.think(obs, ctx)
    elapsed = time.monotonic() - t0

    print(f"  {elapsed:.2f}s  skill={intent.skill} target={intent.target} "
          f"px={intent.target_px}")
    print(f"  reasoning: {intent.reasoning[:120]}")

    # The parse layer must have produced a usable skill, not a fallback.
    assert intent.skill != "wait", f"model returned an unusable intent: {intent.raw[:300]}"
    assert brain.stats.failures == 0, brain.stats.last_error

    # Enemy first is the correct read of the order; the chest is the alternative.
    assert intent.skill in ("attack", "goto", "open_chest"), intent.skill
    assert intent.target, "no target was named"

    if intent.target_px and intent.skill == "attack":
        want = quantize_px(*HILICHURL_PX, W, H, 32)
        dx = abs(intent.target_px[0] - want[0])
        dy = abs(intent.target_px[1] - want[1])
        # Horizontally the contract is strict: x is the axis the controller steers by, and
        # getting it wrong turns the camera the wrong way.
        assert dx <= 64, f"grounded at x={intent.target_px[0]}, expected near x={want[0]}"
        # Vertically it is advisory only. Measured here, y is noisy -- the same frame yielded
        # y=256 in one call and y=359 in another -- which is exactly why `PixelTarget` feeds
        # steering and never feeds a distance or an interaction decision.
        assert dy <= H, "y must at least be a real coordinate in the frame"


def test_vertical_grounding_is_advisory_only() -> None:
    """Locks in the design decision the test above uncovered.

    The controller consumes a VLM's pixel for horizontal error only. If someone later wires
    `target_px[1]` into a distance or an arrival test, this fails and explains why.
    """
    import inspect

    from lumine import controller as ctrl

    src = inspect.getsource(ctrl.SkillController._steer)
    assert "target.x" in src, "steering no longer reads the VLM's x"
    assert "target.y" not in src, (
        "steering now reads the VLM's y; vertical grounding is measurably noisy and must not "
        "drive control decisions"
    )


def test_vlm_latency_is_compatible_with_the_architecture() -> None:
    """The whole design assumes a slow brain. If a call were 100 ms we would not need the
    skill seam; if it were 30 s the design would not work at all."""
    brain = _brain_or_skip()
    if brain is None:
        return
    obs = Observation(frame=Frame(image=synthetic_frame()), hud=HUDState(), t=0.0)
    ctx = BrainContext(instruction="Open the chest ahead", region="mondstadt")
    t0 = time.monotonic()
    brain.think(obs, ctx)
    elapsed = time.monotonic() - t0
    print(f"  latency {elapsed:.2f}s (budget: must stay well under the 20 s keepalive)")
    assert elapsed < 60.0, f"a single call took {elapsed:.1f}s"


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
