"""Tests for the Open Pixel2Play action-space bridge.

    python -m tests.test_p2p

The codec is pure logic, so it is fully testable without the 2.2 GB checkpoint. The test
that matters most is :func:`test_binning_matches_torch_bucketize`: upstream encodes pixel
deltas with ``torch.bucketize(x, edges)`` and any mismatch shifts every non-edge value by
one bin, which would present as a subtly wrong camera rather than as an obvious bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumine.brain.p2p import (DSH_TO_P2P_KEY, MOUSE_X_BIN_CENTERS, MOUSE_X_BIN_EDGES,
                              MOUSE_Y_BIN_CENTERS, MOUSE_Y_BIN_EDGES, N_BUTTON_CHOICES,
                              N_KEY_CHOICES, N_X_BINS, N_Y_BINS, P2P_KEY_TO_DSH, P2PActionCodec,
                              SEQ_LEN, coverage_report, format_coverage)
from lumine.types import BUTTONS, KEYS, Action


# --------------------------------------------------------------------------------------


def test_vocabulary_shape_matches_upstream() -> None:
    """Numbers read off upstream `elefant/data/action_mapping.py`."""
    assert N_KEY_CHOICES == 20
    assert N_BUTTON_CHOICES == 4
    assert N_X_BINS == 23          # 22 edges + 1
    assert N_Y_BINS == 17          # 16 edges + 1
    assert SEQ_LEN == 8            # 4 keys + 2 buttons + dx + dy
    assert len(MOUSE_X_BIN_CENTERS) == N_X_BINS
    assert len(MOUSE_Y_BIN_CENTERS) == N_Y_BINS
    assert len(MOUSE_X_BIN_EDGES) == N_X_BINS - 1
    assert len(MOUSE_Y_BIN_EDGES) == N_Y_BINS - 1


def test_key_ids_are_non_negative_and_round_trip() -> None:
    for kid, name in P2P_KEY_TO_DSH.items():
        if not name:
            continue
        # RightShift (19) collapses onto SHIFT (18), so it maps back to 18 rather than 19.
        back = DSH_TO_P2P_KEY[name]
        assert P2P_KEY_TO_DSH[back] == name, f"{kid} -> {name} -> {back} is not stable"
    assert P2P_KEY_TO_DSH[0] == "" and P2P_KEY_TO_DSH[1] == "SPACE"


def test_decode_a_known_token_vector() -> None:
    codec = P2PActionCodec()
    # w (11) + LeftShift (18) held, left mouse down, mouse pushed right and slightly up.
    x_right = MOUSE_X_BIN_CENTERS.index(15)
    y_up = MOUSE_Y_BIN_CENTERS.index(-2)
    a = codec.decode_sequence([11, 18, 0, 0, 1, 0, x_right, y_up])
    assert a.keys == {"W", "SHIFT"}
    assert a.buttons == {"left"}
    assert a.mouse_dx == 15.0
    assert a.mouse_dy == -2.0


def test_padding_and_no_key_are_ignored() -> None:
    codec = P2PActionCodec()
    a = codec.decode_sequence([0, 0, 0, 0, 0, 0, 11, 8])       # 11 and 8 are the *centre* bins
    assert a.keys == set() and a.buttons == set()
    # Centre bins decode to zero motion, which is what "no movement" must look like.
    assert a.mouse_dx == 0.0 and a.mouse_dy == 0.0


def test_out_of_range_tokens_are_dropped_not_raised() -> None:
    """A stray index from a model must cost a tick, not the run."""
    codec = P2PActionCodec()
    a = codec.decode_sequence([999, -5, 11, 0, 77, 0, 999, -3])
    assert a.keys == {"W"}                  # only the valid token survives
    assert a.buttons == set()
    assert a.mouse_dx == 0.0 and a.mouse_dy == 0.0


def test_at_most_four_keys_and_two_buttons() -> None:
    codec = P2PActionCodec()
    a = codec.decode_sequence([11, 6, 12, 7, 1, 2, 11, 8])
    assert len(a.keys) == 4 and a.keys == {"W", "A", "S", "D"}
    assert a.buttons == {"left", "right"}


def test_encode_decode_round_trip_for_supported_actions() -> None:
    codec = P2PActionCodec()
    original = Action(keys={"W", "SHIFT", "E"}, buttons={"left"}, mouse_dx=24, mouse_dy=-6)
    keys, buttons, mx, my = codec.encode(original)
    back = codec.decode(keys, buttons, mx, my)
    assert back.keys == original.keys
    assert back.buttons == original.buttons
    assert back.mouse_dx == 24.0 and back.mouse_dy == -6.0


def test_unsupported_keys_are_dropped_by_encode() -> None:
    """J is this sandbox's normal attack and P2P has no token for it."""
    codec = P2PActionCodec()
    keys, _b, _x, _y = codec.encode(Action(keys={"J", "W"}))
    back = codec.decode(keys, [0, 0], 11, 8)
    assert back.keys == {"W"}, "an unsupported key leaked into the P2P token stream"


def test_binning_matches_torch_bucketize() -> None:
    """The one test that would have caught a real off-by-one.

    Upstream bins with ``torch.bucketize(x, edges)`` using the default ``right=False``,
    whose contract is ``edges[i-1] < x <= edges[i]``. numpy needs ``side="left"`` for that;
    ``"right"`` agrees everywhere except exactly on an edge, which is precisely where a
    silently-wrong camera would hide.
    """
    torch = __import__("torch")
    rng = np.random.default_rng(0)

    for edges, name in ((MOUSE_X_BIN_EDGES, "x"), (MOUSE_Y_BIN_EDGES, "y")):
        e = torch.tensor(edges)
        probes = list(edges) + [edges[0] - 50, edges[-1] + 50] + \
            list(rng.uniform(edges[0] - 60, edges[-1] + 60, 400))
        for v in probes:
            assert P2PActionCodec._bin(edges, float(v)) == int(torch.bucketize(torch.tensor(float(v)), e)), \
                f"{name}-bin disagrees with torch.bucketize at {v}"


def test_bin_centres_are_monotonic_and_symmetric() -> None:
    for centres in (MOUSE_X_BIN_CENTERS, MOUSE_Y_BIN_CENTERS):
        assert centres == sorted(centres), "centres must be monotonic for steering to be sane"
        assert centres[len(centres) // 2] == 0, "there must be a zero-motion centre bin"
        for lo, hi in zip(centres, reversed(centres)):
            assert lo == -hi, f"bin centres are not symmetric: {lo} vs {hi}"


def test_mouse_scale_is_applied() -> None:
    fast = P2PActionCodec(mouse_scale=2.0)
    slow = P2PActionCodec(mouse_scale=0.5)
    idx = MOUSE_X_BIN_CENTERS.index(38)
    assert fast.decode_sequence([0, 0, 0, 0, 0, 0, idx, 8]).mouse_dx == 76.0
    assert slow.decode_sequence([0, 0, 0, 0, 0, 0, idx, 8]).mouse_dx == 19.0


def test_conservative_sampling_reduces_magnitude_but_keeps_sign() -> None:
    mean = P2PActionCodec(mouse_sampling="mean")
    cons = P2PActionCodec(mouse_sampling="conservative")
    for idx in range(1, N_X_BINS - 1):
        m = mean.decode_sequence([0, 0, 0, 0, 0, 0, idx, 8]).mouse_dx
        c = cons.decode_sequence([0, 0, 0, 0, 0, 0, idx, 8]).mouse_dx
        assert (m >= 0) == (c >= 0), f"sign flipped at bin {idx}"
        assert abs(c) <= abs(m) + 1e-6, f"conservative exceeded mean at bin {idx}"


# --------------------------------------------------------------------------------------


def test_coverage_report_is_honest_about_the_gap() -> None:
    r = coverage_report()
    assert r["keys_total"] == len(KEYS)
    assert r["keys_supported"] < r["keys_total"], "P2P cannot cover this keymap; the report lies"
    assert r["button_coverage"] == 1.0
    # The specific finding that drives the integration plan.
    assert "J" in r["keys_unsupported"], "J must be reported as unsupported (attack rebind needed)"
    for k in ("M", "C", "B", "ESC", "ENTER", "TAB"):
        assert k in r["keys_unsupported"], f"{k} should be reported as unsupported"


def test_coverage_formatting_mentions_the_attack_rebind() -> None:
    text = format_coverage()
    assert "coverage" in text.lower()
    assert "attack" in text.lower(), "the report must flag the J -> left-mouse rebind"


def test_unsupported_keys_are_a_known_bounded_set() -> None:
    """Pin the gap so it cannot grow silently when someone adds a key to KEYS."""
    unsupported = set(coverage_report()["keys_unsupported"])
    assert unsupported == {"J", "K", "L", "CTRL", "R", "T", "M", "B", "C", "ESC",
                           "ENTER", "TAB", "X"}, sorted(unsupported)


# --------------------------------------------------------------------------------------


def test_model_path_fails_loudly_and_actionably() -> None:
    """Without weights the loader must say what to download, not raise an ImportError."""
    from lumine.brain.p2p import P2PPolicy

    try:
        P2PPolicy.from_checkpoint("/nonexistent/path/150M")
    except FileNotFoundError as exc:
        msg = str(exc)
        assert "hf-mirror" in msg, "the error must mention the reachable mirror"
        assert "open-p2p" in msg
    else:
        raise AssertionError("expected FileNotFoundError for a missing checkpoint")


def test_preprocess_targets_the_model_resolution() -> None:
    from lumine.brain.p2p import P2PPolicy

    out = P2PPolicy.preprocess(np.zeros((360, 640, 3), np.uint8))
    assert out.shape == (192, 192, 3)


def _main() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
