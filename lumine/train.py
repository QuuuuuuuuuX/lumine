"""Training the action head.

Not the brain.  This trains the 30 Hz primitive layer by behaviour cloning, which is
Lumine's stage-0 objective, and it is what makes ``control.controller = "learned"`` mean
anything.

Two backends, chosen automatically:

* **torch** -- proper CNN, BCE for the key and button heads, MSE for the mouse head.  Uses
  the GPU when there is one.
* **numpy** -- logistic regression for keys and ridge regression for the mouse.  Weaker by
  a wide margin, but it means the recipe still reproduces on a machine with no torch
  installed, which is the difference between a paper and a script.

Class imbalance is the real problem here, and it is why the loss is weighted.  A player
holds no keys most ticks, so an unweighted key head learns to predict "nothing" and scores
98% while being useless.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrainConfig
from .data import ShardReader, STAGE_BY_NAME
from .policy import NumpyActionHead, TorchActionHead, best_backend
from .types import NUM_KEYS


@dataclass
class TrainReport:
    backend: str
    stage: str
    samples: int
    epochs: int
    seconds: float
    final_loss: float
    key_accuracy: float
    key_f1: float
    mouse_mae: float
    checkpoint: str
    history: list[dict[str, float]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Torch path
# --------------------------------------------------------------------------------------


def _train_torch(cfg: TrainConfig, stage: str, reader: ShardReader,
                 device: str) -> TrainReport:
    import torch
    import torch.nn as nn

    data = reader.load_all()
    if len(data["frames"]) == 0:
        raise RuntimeError(
            f"no training data in stage {stage!r}. Run 'lumine collect --stage {stage}' first."
        )

    frames = torch.from_numpy(data["frames"]).float().unsqueeze(1) / 255.0
    actions = torch.from_numpy(data["actions"])
    keys = actions[:, :NUM_KEYS]
    mouse = actions[:, NUM_KEYS:NUM_KEYS + 2]
    buttons = actions[:, NUM_KEYS + 2:]

    n = len(frames)
    head = TorchActionHead(channels=cfg.channels, device=device)
    head.mouse_scale = float(max(1.0, mouse.abs().max().item()))

    opt = torch.optim.AdamW(head.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Key presses are sparse; weight the positive class up so "press nothing" is not optimal.
    pos = keys.mean(dim=0).clamp(min=1e-4)
    key_weight = ((1 - pos) / pos).clamp(max=50.0).to(head.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=key_weight)
    bce_btn = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.monotonic()
    history: list[dict[str, float]] = []
    last_loss = float("nan")

    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=g)
        total = 0.0
        batches = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = frames[idx].to(head.device)
            kb = keys[idx].to(head.device)
            mb = mouse[idx].to(head.device) / head.mouse_scale
            bb = buttons[idx].to(head.device)

            klog, mreg, blog = head.net(xb)
            loss = bce(klog, kb) + bce_btn(blog, bb) + 2.0 * mse(mreg, mb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.net.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            batches += 1

        last_loss = total / max(1, batches)
        metrics = _evaluate_head(head, frames, keys, mouse, head.device)
        history.append({"epoch": epoch + 1, "loss": round(last_loss, 5), **metrics})
        print(f"  epoch {epoch + 1:2d}/{cfg.epochs}  loss={last_loss:.5f}  "
              f"key_acc={metrics['key_accuracy']:.4f}  f1={metrics['key_f1']:.4f}  "
              f"mouse_mae={metrics['mouse_mae']:.3f}")

    out = Path(cfg.out_dir) / f"action_head_{stage}"
    head.save(out)
    return TrainReport(
        backend="torch", stage=stage, samples=n, epochs=cfg.epochs,
        seconds=round(time.monotonic() - t0, 2), final_loss=round(last_loss, 6),
        key_accuracy=history[-1]["key_accuracy"], key_f1=history[-1]["key_f1"],
        mouse_mae=history[-1]["mouse_mae"], checkpoint=str(out), history=history,
    )


def _evaluate_head(head: TorchActionHead, frames, keys, mouse, device) -> dict[str, float]:
    import torch

    with torch.no_grad():
        n = min(2048, len(frames))
        xb = frames[:n].to(device)
        klog, mreg, _ = head.net(xb)
        pred = (torch.sigmoid(klog) > 0.5).float().cpu()
        truth = keys[:n]
        tp = float(((pred == 1) & (truth == 1)).sum())
        fp = float(((pred == 1) & (truth == 0)).sum())
        fn = float(((pred == 0) & (truth == 1)).sum())
        acc = float((pred == truth).float().mean())
        f1 = 2 * tp / max(1e-6, 2 * tp + fp + fn)
        mae = float((mreg.cpu() * head.mouse_scale - mouse[:n]).abs().mean())
    return {"key_accuracy": round(acc, 5), "key_f1": round(f1, 5), "mouse_mae": round(mae, 4)}


# --------------------------------------------------------------------------------------
# NumPy path
# --------------------------------------------------------------------------------------


def _train_numpy(cfg: TrainConfig, stage: str, reader: ShardReader) -> TrainReport:
    data = reader.load_all()
    if len(data["frames"]) == 0:
        raise RuntimeError(
            f"no training data in stage {stage!r}. Run 'lumine collect --stage {stage}' first."
        )

    import cv2

    head = NumpyActionHead()
    n = len(data["frames"])
    x = np.stack([head._features(cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)) for f in data["frames"]])
    y_key = data["actions"][:, :NUM_KEYS]
    y_mouse = data["actions"][:, NUM_KEYS:NUM_KEYS + 2]
    head.mouse_scale = float(max(1.0, np.abs(y_mouse).max()))

    t0 = time.monotonic()
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(cfg.seed)

    # Weighted logistic regression, SGD with momentum.
    pos = np.clip(y_key.mean(axis=0), 1e-4, None)
    weight = np.minimum((1 - pos) / pos, 50.0).astype(np.float32)
    w = head.w_key
    vel = np.zeros_like(w)
    for epoch in range(cfg.epochs):
        perm = rng.permutation(n)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = x[idx], y_key[idx]
            p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
            grad = (xb.T @ ((p - yb) * weight)) / len(idx)
            vel = 0.9 * vel - cfg.lr * grad
            w += vel
        head.w_key = w
        pred = (1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))) > 0.5
        tp = float(((pred == 1) & (y_key == 1)).sum())
        fp = float(((pred == 1) & (y_key == 0)).sum())
        fn = float(((pred == 0) & (y_key == 1)).sum())
        metrics = {"key_accuracy": round(float((pred == y_key).mean()), 5),
                   "key_f1": round(2 * tp / max(1e-6, 2 * tp + fp + fn), 5)}
        history.append({"epoch": epoch + 1, **metrics})
        print(f"  epoch {epoch + 1:2d}/{cfg.epochs}  key_acc={metrics['key_accuracy']:.4f}  "
              f"f1={metrics['key_f1']:.4f}")

    # Ridge regression in closed form for the mouse head.
    lam = 1.0
    a = x.T @ x + lam * np.eye(x.shape[1], dtype=np.float32)
    head.w_mouse = np.linalg.solve(a, x.T @ y_mouse).astype(np.float32)
    mae = float(np.abs((x @ head.w_mouse) * head.mouse_scale - y_mouse).mean())

    out = Path(cfg.out_dir) / f"action_head_{stage}"
    head.save(out)
    return TrainReport(
        backend="numpy", stage=stage, samples=n, epochs=cfg.epochs,
        seconds=round(time.monotonic() - t0, 2), final_loss=0.0,
        key_accuracy=history[-1]["key_accuracy"], key_f1=history[-1]["key_f1"],
        mouse_mae=round(mae, 4), checkpoint=str(out), history=history,
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def train_action_head(cfg: TrainConfig, data_cfg=None) -> TrainReport:  # noqa: ANN001
    if cfg.stage not in STAGE_BY_NAME:
        raise ValueError(f"unknown stage {cfg.stage!r}; expected one of {list(STAGE_BY_NAME)}")

    reader = ShardReader.discover(cfg.data_root, cfg.stage)
    print(f"stage={cfg.stage}  shards={len(reader.paths)}  samples={len(reader) if reader.paths else 0}")

    device = cfg.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    backend = best_backend(prefer_torch=(device != "none"))
    if backend == "numpy" and not cfg.allow_numpy_fallback:
        raise RuntimeError("torch is not installed and allow_numpy_fallback is False")
    print(f"backend={backend}  device={device}")

    report = (_train_torch(cfg, cfg.stage, reader, device) if backend == "torch"
              else _train_numpy(cfg, cfg.stage, reader))

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"train_report_{cfg.stage}.json").write_text(
        json.dumps(report.as_dict(), indent=2))
    print(f"checkpoint -> {report.checkpoint}")
    return report
