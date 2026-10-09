# Lumine — a generalist agent for 3D open worlds

A working reimplementation of the architecture in **[*Lumine: Building Generalist Agents in
3D Open Worlds*](https://arxiv.org/abs/2511.08892)** (ByteDance Seed & NTU), with a
built-in 3D sandbox standing in for the game so the entire system runs on one laptop.

> **The one idea.** Lumine is not "a model that plays a game". It is a *rate hierarchy*:
> raw pixels at **5 Hz**, keyboard and mouse at **30 Hz**, and a vision-language model
> invoked at roughly **0.5 Hz, only when necessary**. Everything in this repository exists
> to keep those three rates from contaminating each other.

```
                    ┌──────────────────────────────────────────────┐
   ~0.5 Hz          │  SLOW BRAIN — vision-language model          │
   ~2 s/call        │  reads the raw frame, answers in JSON:       │
                    │  {skill, target, target_px, reasoning}       │
                    └───────────────────┬──────────────────────────┘
                                        │  Intent  (language: WHAT)
                    ┌───────────────────▼──────────────────────────┐
   30 Hz            │  CONTROLLER — action primitives              │
   every 33 ms      │  visual servoing + skill FSMs + reflexes     │
                    │  emits {W, SHIFT, J, mouse_dx, ...}          │
                    └───────────────────┬──────────────────────────┘
                                        │  Action  (keys: HOW)
                    ┌───────────────────▼──────────────────────────┐
   5 Hz             │  WORLD — sandbox, or a real game via uinput  │
   raw pixels       └──────────────────────────────────────────────┘
```

The seam between the top two boxes is the whole design. A two-second brain cannot emit 30 Hz
control, and a 30 Hz controller cannot read a quest log. So the model speaks in **skills and
targets** and a fast controller grounds them into keystrokes — which is exactly the two-stage
curriculum Lumine trains (action primitives first, then instruction following).

---

## Quick start

```bash
cd lumine
python3 -m lumine tasks                    # what can it be asked to do?
python3 -m lumine providers                # which brains can I plug in?
python3 -m lumine rates                    # measure the 5 / 30 / 0.5 Hz design

# No API key needed — the scripted brain drives the hand-written controller.
python3 -m lumine play --config configs/scripted.yaml --task combat_defeat_and_chest

# With a vision-language model actually looking at the screen.
export DEEPSEEK_API_KEY=sk-...             # https://platform.deepseek.com/api_keys
python3 -m lumine probe                    # prove the key really sees images
python3 -m lumine play --task boss_hypostasis_electro --video artifacts/boss.mp4
```

The full pipeline:

```bash
python3 -m lumine collect --minutes 10 --stage pretrain    # scripted teacher plays, data recorded
python3 -m lumine corpus                                   # how much did we get?
python3 -m lumine train   --stage pretrain                 # action head (GPU if torch is present)
python3 -m lumine eval    --regions mondstadt liyue --per-category 1   # + held-out region
```

---

## Which brain should I plug in?

`deepseek-flash` is verified working **on this machine, with the harness's own credential**,
and it is the default. It is also the only DeepSeek model with vision — `deepseek-v4-pro`
rejects image parts outright, which this repo tests for.

| provider | default model | base_url | vision | $ per 1M in/out | notes |
|---|---|---|---|---|---|
| `deepseek` | `deepseek-flash` | `https://api.deepseek.com` | yes | 0.15 / 0.60 | **Default.** ~1.4-2.0 s per frame; 1024-token image cap makes cost predictable |
| `qwen` | `qwen3-vl-flash` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | yes | **0.022 / 0.215** | ~$0.0001 per frame. The family Lumine fine-tunes |
| `zai` | `glm-5.3-flash` | `https://api.z.ai/api/paas/v4` | yes | 0.15 / 0.50 | `glm-4.6v-flash` is **free**; `glm-5.3-flashx` has a published ~200 tok/s |
| `ollama` | `qwen3-vl:8b-instruct-q4_K_M` | `http://localhost:11434/v1` | yes | free | **Fully local**, 6.1 GB, no network in the latency budget |
| `doubao` | `doubao-seed-2-1-lite-260915` | `https://ark.cn-beijing.volces.com/api/v3` | yes | ¥0.80 / ¥2.70 | The Lumine authors' own platform; 1280-token floor per image |
| `stepfun` | `step-3.7-flash` | `https://api.stepfun.com/v1` | yes | 0.20 / 1.15 | The celebrated grounding scores belong to *open-weight* Step-GUI-4B/8B, not these |
| `kimi` | `kimi-k3` | `https://api.moonshot.ai/v1` | yes | 3.00 / 15.00 | **Refuses public image URLs** — base64 only |
| `gemini` | `gemini-3.5-flash-lite` | `.../v1beta/openai` | yes | 0.30 / 2.50 | Best latency/cost outside China |
| `openai` | `gpt-5.6-luna` | `https://api.openai.com/v1` | yes | 0.20 / 1.20 | Strongest reasoning, worst latency |
| `anthropic` | `claude-haiku-5-5` | `https://api.anthropic.com/v1` | yes | 0.10 / 0.50 | Its OpenAI shim **silently ignores `response_format`** — dangerous for strict JSON |
| `none` | — | — | no | free | Scripted brain. Always available |

`python3 -m lumine providers` prints the full registry with per-provider quirks.
Prices marked with a currency other than USD, and every OpenAI/Anthropic/Gemini figure, are
third-party — those vendors' pricing pages block automated fetches, so re-verify before you
commit spend. Everything else was read from the vendor's own documentation.

**Top three, and why**

1. **`deepseek`** — cheapest credible option, and the only one verified end to end here.
   1.4-2.0 s per 640x360 frame at `reasoning_effort: low`. Start here.
2. **`qwen`** — best grounding per dollar by an order of magnitude, and the family Lumine
   builds on (`Qwen2-VL-7B-Base`). Switch here when the agent must click precise GUI targets.
3. **`ollama`** — for no key, no network, and no round trip in the latency budget.
   `qwen3-vl:8b` at q4_K_M is 6.1 GB and fits an 8 GB GPU.

Switching providers is a config change, never a code change — see `lumine/brain/providers.py`.

---

## What is actually reproduced, and what is not

Being precise about this matters more than the demo.

**Reproduced faithfully**

- The three-rate hierarchy (5 Hz perception / 30 Hz control / ~0.5 Hz adaptive reasoning), and
  the skill-level seam that makes it possible. Measured, not asserted: `lumine rates`.
- Adaptive reasoning with nine named triggers (`lumine/brain/adaptive.py`): instruction change,
  skill completion, objective progress, HP loss, visual surprise, being stuck, skill timeout,
  call-rate budget, keepalive. The VLM is *not* called on a timer.
- The three-stage data curriculum and its 1731 / 200 / 15 hour shape, with the ratio made
  explicit because the ratio is the point (`lumine/data.py`).
- Behaviour cloning for the 30 Hz primitive layer, trained on GPU, with a numpy fallback so the
  pipeline reproduces without torch (`lumine/train.py`).
- Evaluation split into Lumine's own categories — combat, boss, puzzle, NPC, GUI, in-context
  learning — plus a **held-out region** for the generalisation question that Liyue raises.

**Not reproduced, and not claimed**

- **Scale.** Lumine trains on 1731 hours of human gameplay. We generate minutes, from a
  scripted teacher. The action head here demonstrates that the pipeline learns; it is not a
  competitive controller, and the README says so in the results section too.
- **The model.** We *call* a VLM; Lumine *fine-tunes* one (Qwen2-VL-7B). No training of a
  vision-language model happens anywhere in this repository.
- **Genshin itself.** Genshin does not run on the Linux box this was built on. The sandbox
  stands in for it. Live mode against a real game is implemented (`lumine/live.py`,
  `lumine/input_backend.py`) and can inject real keyboard and mouse events through
  `/dev/uinput`, but it has not been validated against a real game here — see
  [Live mode](#live-mode-driving-a-real-game).

---

## The sandbox

`lumine/sim/` is a software 3D renderer (numpy + OpenCV, painter's algorithm, z-ordered, with
fog and a full game HUD) plus a game-logic layer. It exists because a generalist-agent
architecture you cannot run is a diagram, not a system.

It renders, at 640×360 and ≥ 200 logical steps/second:

- terrain with height, slope shading and distance fog, sky gradient driven by time of day,
  water, trees, rocks, clouds
- enemies (slime, hilichurl, whopperflower), three bosses (Electro Hypostasis, Anemo
  Hypostasis, Stormterror) with telegraphed AoE attacks that must be dodged
- chests with guards, thorn-locked chests needing Pyro, boulders needing a charged attack,
  elemental monuments, time trials, wind currents, anemograna, anemoculus
- NPCs with dialogue trees, teleport waypoints, a cooking pot, weapon menus
- a HUD a VLM can actually read: quest tracker, minimap with coloured entity dots, HP and
  stamina bars, party chips with skill cooldowns, interaction prompts, damage numbers

Two regions: `mondstadt` (in-distribution) and `liyue` (a **held-out** region with different
terrain, palette, landmark layout and NPC names). The second one is the interesting one.

---

## Live mode: driving a real game

```bash
python3 -m lumine play --config configs/live.yaml \
  --task "Defeat the enemies ahead and collect the chest"
```

`LiveEnv` presents a real screen and a real keyboard/mouse behind the same
`reset`/`step`/`success` interface as the sandbox, so `Agent` runs unmodified. Needs:

1. Write access to `/dev/uinput` — `sudo usermod -aG input $USER`, then re-login.
2. `pip install mss` for fast capture (falls back to `ffmpeg -f x11grab`).
3. A **windowed** game, not exclusive fullscreen.

> **Status on this machine, verified rather than assumed.** `/dev/uinput` exists, but the
> current user is not in the `input` group, so the backend raises a precise, actionable error
> instead of failing silently; `mss` is not installed either. The injection code is real —
> it packs Linux `input_event` structs directly and needs no extra dependency — but it has
> **not** been validated against a live game here, and this README will not claim otherwise.

Two honesty notes: a real game gives no success signal, so `success()` is always `False` and
termination comes from the time budget or your own `checker` callback; and real frame pacing
means `step` blocks to hold a true 30 Hz loop, unlike the sandbox.

---

## Repository layout

```
lumine/
  types.py           Action / Observation / Intent / TaskSpec — the shared vocabulary
  config.py          one dataclass tree; every rate and provider lives here
  skills.py          the skill catalogue handed to the VLM as its tool list
  perception.py      5 Hz capture (sim | screen | video) + reflex detectors
  controller.py      30 Hz action primitives, the language→keystroke seam
  input_backend.py   real key/mouse injection via /dev/uinput (no extra dependency)
  agent.py           the loop; sync and async reasoning modes
  live.py            a real game wearing the sandbox's interface
  memory.py          working memory + the long-horizon failure ledger
  policy.py          the action head (torch CNN, numpy logistic/ridge fallback)
  data.py            recorder, shard reader, three-stage curriculum
  train.py           behaviour cloning, GPU
  evaluate.py        task suite, category + region breakdown, markdown report
  cli.py             python -m lumine ...
  brain/
    vlm.py           urllib-based OpenAI-compatible client; never raises
    providers.py     verified endpoints, model ids and quirks
    prompt.py        prompt + the tolerant JSON parser
    adaptive.py      when to spend two seconds of model time
    scripted.py      the no-key fallback brain and the data-generating teacher
  sim/               3D renderer, world, tasks, gym-style env
```

---

## Design notes worth reading

**A brain failure must never be fatal.** Every layer degrades instead of raising: transport
errors, HTTP errors, truncated completions and unparseable JSON all become a safe `wait`
intent. One malformed JSON blob cannot end a five-hour episode. See
`tests/test_brain_failure_does_not_kill_the_episode`.

**Reasoning models eat their own output.** `deepseek-flash` is a reasoning model: with a tight
`max_tokens` it returns an empty `content` and puts everything in `reasoning_content`. The
client salvages the JSON from there rather than losing the call, and the measured latency knob
(`reasoning_effort: low` → 2.0 s vs 2.4 s default) is pinned in the config.

**Coordinates are coarse on purpose.** VLMs ground regions well and pixels badly, so the
prompt asks for a point on a 32 px grid and the controller refines it with classical CV.

**Async reasoning is not an optimisation, it is a requirement.** In sync mode the sandbox waits
while the agent thinks — deterministic and reproducible, which is what evaluation needs. In
async mode a worker thread reasons while the current skill keeps executing, because freezing
input for two seconds in a live game means dying. Both modes share every other line of code.

**The failure ledger.** A 2-second-latency brain with no memory of its own failures will loop
until the timer expires. `MissionMemory` records `(skill, target) → failure count` and injects
"these approaches already failed, do something different" into the next prompt.

---

## Measured on this machine

Everything below was run here, not estimated. Full results in
[`artifacts/eval/REPORT.md`](artifacts/eval/REPORT.md).

| what | result | how to reproduce |
|---|---|---|
| control rate | **30.0 Hz** on target | `python -m lumine rates` |
| perception rate | **5.0 Hz** on target | `python -m lumine rates` |
| sandbox speed | **4.0x** real time, 45 ms per 640x360 frame | `python -m lumine rates` |
| vision model | `deepseek-flash` reads a game frame; **1.4-2.0 s** per call | `python -m lumine probe` |
| adaptive reasoning | **0.41 Hz** observed over a live episode against a 0.5 Hz design | `python -m lumine play --task boss_hypostasis_electro` |
| GPU | torch **2.14.1+cu130**, CUDA on the RTX 5060 (sm_120) | `python -c "import torch;print(torch.cuda.is_available())"` |
| tests | **53 passing** (32 core, 18 sandbox, 3 live-model) | `make test && make test-live` |

**Grounding is asymmetric, and that shaped the design.** Given a frame with one enemy at
(210, 250), `deepseek-flash` returned x=224 — within a single 32 px cell — but put y at the
bottom of the frame on one call and at y=256 on another. So the controller consumes the
model's pixel for **horizontal** error only and takes distance from the detector, never from
the model. `tests/test_brain_live.py::test_vertical_grounding_is_advisory_only` fails if
anyone ever wires `target_px[1]` into a control decision.

**Four bugs worth knowing about**, all found by running the thing rather than reading it,
and all now pinned by regression tests:

- *The idle path bypassed every trigger.* `_maybe_reason` computed the adaptive decision when
  no skill was active and then ignored it, so the agent called the model on every 30 Hz tick.
  Measured before: **1.67 Hz average, 30 Hz burst**. After: 0.48 Hz. This was the single most
  expensive bug in the project.
- *Persistent conditions re-fired every tick.* "Stuck" is a state, not an event; so is "a
  dialogue is open". Edge-triggered and rate-floored now.
- *Every tree, rock and enemy rendered solid black.* The primitive builders took a colour
  argument and discarded it. The frame still looked like a scene and passed a "not flat"
  check — but a vision-language model reads a black silhouette as an empty room. The sandbox
  test suite now carries an explicit black-pixel budget.
- *A model's phrasing could end the whole run.* The guard against brain failures was built
  into the JSON parser and nowhere else, so when `deepseek-flash` returned a dialogue
  **option's text** where an index was expected, an unguarded `int()` raised out of
  `run_episode` and killed the episode. Parameters are now coerced tolerantly, and the
  skill FSM has a blanket handler so a bad parameter costs one *skill*, never the *run*.
  Every absorbed fault is attributed to `skill:ExceptionType` and surfaced in the evaluation
  report — a bare count cannot tell "the guard worked once" from "this skill is broken".

---

## Results

Full report: [`artifacts/eval/REPORT.md`](artifacts/eval/REPORT.md), regenerated by
`python -m lumine eval --per-category 1 --regions mondstadt liyue`.

**The scripted brain, 14 episodes (one task per category, both regions):**

| | result |
|---|---|
| overall | **14.3%** (2/14) |
| npc interaction | **100%** (2/2) |
| combat / boss / puzzle / GUI / ICL / mission | 0% each |
| mondstadt (in-distribution) | 14.3% |
| liyue (held out) | 14.3% |
| reasoning rate | 0.22-0.49 Hz against a 0.5 Hz design ceiling |

The identical in-distribution and held-out numbers are the expected result *and* a useful
check. The scripted controller is purely geometric — it steers by detector boxes and world
positions — so it has no region-specific knowledge that could overfit. A learned controller
would be expected to show a gap here; that gap is what the Liyue hold-out exists to measure.

**The VLM agent, `boss_hypostasis_electro`, one episode end to end:**

- found the boss from the standing order, having never seen one
- tracked it across visual states: *"the cloudy form"*, *"grey cubes on the platform"*,
  *"the red cube cluster at top-centre"*
- dodged when HP collapsed, then spent twelve consecutive decisions trying to eat a healing
  dish, correctly noticing that *"eight Sweet Madame menu attempts in a row have failed"*
- died at 1 HP
- **34 model calls over 86.9 s at 0.41 Hz**, with the controller emitting keys throughout

**Read the success numbers honestly.** The hand-written controller is competent at movement,
approach, interaction and simple puzzles, and weak at sustained combat and multi-step GUI
flows — which is why combat, boss and GUI all sit at 0%. That is the architecture working and
the controller losing. It is also exactly the gap the trained action head exists to close, and
closing it needs the 1731 hours this machine does not have:

| stage | frames collected here | hours | Lumine target | coverage |
|---|---|---|---|---|
| pretrain | 4,510 | 0.125 | 1,731 h | **0.0072%** |
| instruct | 0 | 0 | 200 h | 0% |
| reason | 0 | 0 | 15 h | 0% |

The action head nevertheless trains and converges on what exists: 96% key accuracy, F1 0.62,
mouse MAE 4.5 on the 4,510-frame corpus, in 6 epochs on the GPU.

Two things this report deliberately does **not** claim: that the agent plays Genshin Impact,
and that any of these numbers would survive contact with a real 3D game at real frame rates.

## Open Pixel2Play: feasibility (investigated, not integrated)

[Open Pixel2Play](https://github.com/elefant-ai/open-p2p) (MIT, arXiv 2601.04575) is the most
relevant open project to this one: **8,000+ hours** of human-annotated gameplay, frames +
text in, **keyboard and mouse** out, and a 150M checkpoint that runs at roughly 80 Hz on a
consumer GPU — against the ~0.5 Hz a cloud VLM gives us. It is the natural fix for the
architecture's main weakness (84% of wall-clock spent waiting on the model).

Findings, from reading upstream's actual `elefant/data/action_mapping.py`:

**The action space matches ours in shape.** Eight tokens per control step —
`[k0, k1, k2, k3, button0, button1, delta_x, delta_y]` — with 4 keys max, 2 buttons max,
23 mouse-x bins (±501 px) and 17 mouse-y bins (±151 px). `lumine/brain/p2p.py` implements the
full codec and it is tested (17 tests), including a check that our binning agrees with
`torch.bucketize` at the exact bin edges, where an off-by-one would hide as a subtly wrong
camera rather than as an obvious bug.

**But the keymap is only 58% compatible.** `python -m tests.test_p2p` computes this rather
than asserting it:

```
keys     : 18/31 supported (58%)
buttons  : 3/3 (100%)
NOT expressible by P2P:
  CTRL  J  K  L  R  T  M  B  C  ESC  ENTER  TAB  X
```

P2P's keymap was built for shooters and platformers (`wasd`, space, shift, arrows,
`e`/`q`/`f`/`z`). This sandbox was built for an action-RPG, so P2P has **no `J`** — which is
the normal attack — and none of the GUI keys. The fix is a sandbox rebind (attack → left
mouse button, which P2P *does* emit and which Genshin Impact uses on PC anyway), not an
adapter hack. That rebind is not done yet.

**Blockers to actually running it here**

1. **HuggingFace is unreachable from this machine** — connections are reset on both IPv4 and
   IPv6 (not a DNS problem; `github.com` works fine). The fix is the mirror:
   `export HF_ENDPOINT=https://hf-mirror.com`, which is verified reachable and does serve
   both `elefantai/open-p2p` and `nvidia/NitroGen`.
2. **The 150M checkpoint is 2.2 GB**, not the ~600 MB the parameter count suggests.
3. Inference pulls `lightning`, protobuf and a **Gemma tokenizer**, plus upstream's
   `elefant` package tree (~92 files). No Rust or FFmpeg needed — those are training-side.
4. Model input is **192x192 RGB** with a 200-frame attention history, not our 640x360.

**Verdict:** the interface is compatible and the codec is done and tested. What remains is a
2.2 GB download, a dependency install, and — the part that actually needs a decision — a
rebind of the sandbox keymap so that `J` becomes the left mouse button. That last step changes
the sandbox's action space, so it should be a config option rather than a silent edit.

## References

- Lumine: Building Generalist Agents in 3D Open Worlds — https://arxiv.org/abs/2511.08892
- Lumine project page — https://www.lumine-ai.org/
- Open Pixel2Play — https://github.com/elefant-ai/open-p2p (MIT)
- NitroGen, NVIDIA — https://nitrogen.minedojo.org (arXiv 2601.02427)
