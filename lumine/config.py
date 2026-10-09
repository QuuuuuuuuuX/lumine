"""Configuration: one dataclass tree, YAML/JSON loadable, env-var aware.

Every rate that defines Lumine lives here so an ablation is a one-line change::

    perception_hz: 5     control_hz: 30     reason_hz: 0.5

Secrets never live in the config file.  ``api_key_env`` names an environment variable and
:func:`resolve_api_key` reads it, with a fallback to the DSH credential store so the
harness's own key can be reused without ever being copied to disk.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------

_DSH_CREDENTIALS = Path.home() / ".dsh" / ".credentials.yaml"


def resolve_api_key(api_key_env: str | None, provider: str = "deepseek") -> str | None:
    """Resolve a key from the environment, then from the DSH credential store.

    The DSH store is a small YAML file holding ``refs:`` entries.  We read it directly
    rather than parsing YAML so we do not take a dependency on PyYAML for one lookup.
    """
    if api_key_env:
        v = os.environ.get(api_key_env)
        if v:
            return v.strip()
    for name in (api_key_env, f"{provider.upper()}_API_KEY"):
        if not name:
            continue
        try:
            txt = _DSH_CREDENTIALS.read_text()
        except OSError:
            break
        m = re.search(rf"^\s*{re.escape(name)}:\s*(\S+)\s*$", txt, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return None


# --------------------------------------------------------------------------------------
# Config tree
# --------------------------------------------------------------------------------------


@dataclass
class PerceptionConfig:
    #: Raw-pixel capture rate.  Lumine feeds the model at 5 Hz.
    hz: float = 5.0
    width: int = 640
    height: int = 360
    #: "sim" | "screen" | "video".  "screen" is the real-machine path.
    source: str = "sim"
    monitor: int = 0
    capture_region: tuple[int, int, int, int] | None = None   # left, top, w, h
    #: Frames are downscaled to this before being handed to a learned controller.
    policy_width: int = 160
    policy_height: int = 90


@dataclass
class ControlConfig:
    #: Key/mouse emission rate.  Lumine outputs 30 Hz keyboard-mouse actions.
    hz: float = 30.0
    #: "rules" | "learned" | "hybrid".  "learned" loads an action-head checkpoint.
    controller: str = "rules"
    checkpoint: str | None = None
    #: "sim" | "uinput" | "dryrun".  "uinput" drives a real game on this machine.
    backend: str = "sim"
    mouse_sensitivity: float = 12.0
    #: Blend factor when controller == "hybrid": learned output overrides rules when the
    #: learned head is confident, otherwise the hand-written controller keeps the agent safe.
    hybrid_threshold: float = 0.55


@dataclass
class BrainConfig:
    #: Provider key from ``lumine.registry``: deepseek | qwen | glm | kimi | doubao | stepfun
    #: | openai | anthropic | gemini | ollama | none.
    provider: str = "deepseek"
    model: str = "deepseek-flash"
    base_url: str | None = None
    api_key_env: str | None = "DEEPSEEK_API_KEY"
    #: Grounding is a measurement, not a creative act. Sampling only adds noise to the
    #: coordinates the controller steers by.
    temperature: float = 0.0
    max_tokens: int = 1500
    #: Passed straight through to providers that support it.  Measured effect on
    #: deepseek-flash: "low" == 2.0 s/call vs 2.4 s default (reasoning tokens 342 vs 407).
    reasoning_effort: str = "low"
    timeout: float = 90.0
    max_retries: int = 2
    #: Coordinate quantization for grounding, in pixels.
    ground_grid: int = 32
    #: When the API is unreachable, fall back to the scripted brain instead of dying.
    fallback_to_scripted: bool = True


@dataclass
class ReasonConfig:
    """When to spend 2 seconds of VLM time.  This is Lumine's "adaptive thinking"."""

    #: Hard ceiling on VLM calls per second, averaged.
    hz: float = 0.5
    #: Minimum game-seconds between two reasoning calls, whatever triggered them.
    #:
    #: Not an optimisation -- a correctness requirement. Several triggers describe a *state*
    #: (being stuck, a dialogue being open) rather than an event, so without a floor they
    #: re-fire on every one of the 30 ticks per second. Measured before this existed: 200
    #: model calls in 14 seconds of game time, all reaching the same conclusion. At 2.0 s
    #: this also caps the worst case at exactly the 0.5 Hz the architecture is designed around.
    min_interval_s: float = 2.0
    #: Always re-plan if this long has passed with no call (safety net).
    keepalive_s: float = 20.0
    #: Re-plan when the task/instruction changes.
    on_instruction_change: bool = True
    #: Re-plan when the current skill reports success or failure.
    on_skill_end: bool = True
    #: Stuck detector: if the agent's position has barely moved for this long, think.
    stuck_seconds: float = 6.0
    stuck_distance: float = 1.5
    #: Re-plan when the HUD objective counter changes (a chest opened, an enemy died).
    on_progress_change: bool = True
    #: Re-plan on a large visual surprise (frame novelty above this z-score).
    surprise_z: float = 3.0
    #: Re-plan when HP drops by more than this fraction in one window.
    hp_drop_frac: float = 0.25
    #: Cap on how long a single skill may run before the reasoner checks in.
    max_skill_seconds: float = 45.0


@dataclass
class SimConfig:
    region: str = "mondstadt"
    seed: int = 0
    #: Offscreen sandbox rendering, in sim units per second.
    day_length_s: float = 600.0
    fog: bool = True
    #: Render scale.  1.0 == the perception resolution.
    render_scale: float = 1.0
    #: When True the sim accepts actions faster than real time (used for data collection).
    realtime: bool = False
    #: Multiplier for data collection; the recorded actions are still 30 Hz logical ticks.
    speedup: float = 1.0


@dataclass
class DataConfig:
    root: str = "artifacts/data"
    #: Stage 0: raw gameplay, no language.  Stage 1: + instructions.  Stage 2: + reasoning.
    stages: tuple[str, ...] = ("pretrain", "instruct", "reason")
    record_every_n_ticks: int = 1
    shard_size: int = 2000


@dataclass
class TrainConfig:
    data_root: str = "artifacts/data"
    out_dir: str = "artifacts/checkpoints"
    stage: str = "pretrain"
    epochs: int = 8
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "auto"          # auto | cpu | cuda
    #: Small on purpose: the action head grounds intent into keystrokes, it is not the brain.
    width: int = 32
    channels: tuple[int, ...] = (32, 64, 64)
    num_workers: int = 0
    seed: int = 0
    #: Pure-numpy fallback so the recipe runs on a machine without torch.
    allow_numpy_fallback: bool = True


@dataclass
class EvalConfig:
    out_dir: str = "artifacts/eval"
    tasks: tuple[str, ...] = ("combat", "boss", "puzzle", "npc", "gui", "icl")
    episodes_per_task: int = 2
    seeds: tuple[int, ...] = (0, 1)
    #: Evaluate on a held-out region the agent never saw -- Lumine's Liyue test.
    ood_regions: tuple[str, ...] = ("liyue",)
    report_json: str = "artifacts/eval/report.json"
    report_md: str = "artifacts/eval/REPORT.md"


@dataclass
class LumineConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    reason: ReasonConfig = field(default_factory=ReasonConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    #: "pixels" (faithful to Lumine) or "pixels+hud" (ablation with privileged state).
    observation_mode: str = "pixels"
    log_level: str = "INFO"
    seed: int = 0

    # -- (de)serialisation -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LumineConfig":
        cfg = cls()
        _apply(cfg, d)
        return cfg

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None) -> "LumineConfig":
        if not path:
            return cls()
        p = Path(path)
        text = p.read_text()
        data = json.loads(text) if p.suffix in (".json",) else _mini_yaml(text)
        return cls.from_dict(data)

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


def _apply(obj: Any, data: Mapping[str, Any]) -> None:
    """Recursively overlay a plain mapping onto a dataclass instance."""
    valid = {f.name: f for f in fields(obj)}
    for k, v in data.items():
        if k not in valid:
            raise KeyError(f"unknown config field {k!r} for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, Mapping):
            _apply(cur, v)
        elif isinstance(v, list) and isinstance(cur, tuple):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)


def _mini_yaml(text: str) -> dict[str, Any]:
    """A deliberately tiny YAML subset: nested maps + scalars, two-space indent, ``#`` comments.

    Enough for the shipped config files, and it keeps the project dependency-free.  If
    PyYAML happens to be installed we defer to it, because silently mis-parsing a user's
    real YAML would be worse than a hard dependency.
    """
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        val = val.strip()
        if not val:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(val)
    return root


def _scalar(v: str) -> Any:
    if v in ("null", "~", ""):
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.startswith("[") and v.endswith("]"):
        return [_scalar(x.strip()) for x in v[1:-1].split(",") if x.strip()]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v.strip("'\"")
