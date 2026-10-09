# Lumine Agent — archive manifest

A consolidated copy of the project, saved to `A:\lumine` (`/mnt/a/lumine` in WSL).

## What this is

A reimplementation of the architecture in *Lumine: Building Generalist Agents in 3D Open
Worlds* (ByteDance Seed & NTU, [arXiv 2511.08892](https://arxiv.org/abs/2511.08892)), with a
built-in 3D sandbox standing in for the game so the whole system runs on one machine.

The defining idea is a **rate hierarchy**, not a model:

```
  ~0.5 Hz   vision-language model   reads the frame, answers with a skill + target
   30 Hz    controller              turns that into keystrokes
    5 Hz    world                   raw pixels out (sandbox, or a real game via /dev/uinput)
```

Measured on the build machine: control **30.0 Hz** and perception **5.0 Hz**, both exactly
on target; **75/75** tests passing.

## Layout

```
lumine/
├── README.md            start here: architecture, providers, measured results, limitations
├── Makefile             make test / rates / play / collect / train / eval
├── requirements.txt
├── configs/             default (deepseek) · scripted (no key) · local (ollama) · live (real game)
├── lumine/              the package — 35 modules
│   ├── types.py         Action / Observation / Intent — the shared vocabulary
│   ├── config.py        every rate and provider lives here
│   ├── skills.py        the skill catalogue handed to the VLM as its tool list
│   ├── perception.py    5 Hz capture + reflex detectors
│   ├── controller.py    30 Hz action primitives
│   ├── input_backend.py real key/mouse injection via /dev/uinput (no extra dependency)
│   ├── agent.py         the loop; sync and async reasoning modes
│   ├── live.py          a real game behind the sandbox's interface
│   ├── memory.py        working memory + the long-horizon failure ledger
│   ├── policy.py        the action head (torch CNN, numpy fallback)
│   ├── data.py          recorder + the three-stage curriculum
│   ├── train.py         behaviour cloning, GPU
│   ├── evaluate.py      task suite, category + region breakdown, report
│   ├── cli.py           python -m lumine ...
│   ├── brain/           vlm · providers · prompt · adaptive · scripted · p2p
│   └── sim/             render · world · tasks · env  (the built-in 3D sandbox)
├── tests/               75 tests
│   ├── test_core.py     40 — types, config, adaptive triggers, memory, data, regressions
│   ├── test_sim.py      18 — sandbox contract, determinism, held-out region
│   ├── test_p2p.py      17 — the Pixel2Play action-space bridge
│   └── test_brain_live.py  3 — calls a real VLM (skips without a key)
├── docs/
│   ├── research/        four research reports: VLM provider survey, pricing, benchmarks
│   └── upstream/        reference source from elefant-ai/open-p2p, with provenance
└── artifacts/           products (14 MB)
    ├── boss_electro.mp4              an end-to-end VLM episode
    ├── eval/REPORT.md + report.json  the evaluation
    ├── checkpoints/                  the trained action head
    ├── data/pretrain/                4,510 recorded frames
    └── sim_previews/                 23 rendered sandbox frames
```

## Running it

```bash
cd lumine
python3 -m tests.test_core && python3 -m tests.test_sim && python3 -m tests.test_p2p
python3 -m lumine rates        # verify the 5 / 30 / 0.5 Hz design on this machine
python3 -m lumine tasks        # the 23-task suite
python3 -m lumine play --config configs/scripted.yaml --task combat_defeat_and_chest
```

Runtime dependencies: numpy, OpenCV, Pillow. Optional: torch (action-head training; the
trainer falls back to numpy), mss (real-screen capture).

For a live vision-language brain, `~/.dsh/.credentials.yaml` supplies `DEEPSEEK_API_KEY` on
the build machine. Elsewhere, export your own key — `python -m lumine providers` lists twelve
providers and how they compare.

## What is finished, and what is not

Stated plainly, because the difference matters:

**Finished and verified** — the rate hierarchy (measured, not asserted); the sandbox and its
23 tasks; end-to-end VLM play with an episode recording; the data → train → checkpoint
pipeline on GPU; the evaluation harness with an in-distribution / held-out split; 75 tests.

**Written but never verified against a real game** — `live.py` and the `/dev/uinput`
injection path. The code is complete and dependency-free, but this machine's user is not in
the `input` group, so it has never driven an actual game.

**Not attempted** — no vision-language model was fine-tuned; the corpus is 0.125 hours
against Lumine's 1,731 (0.0072%); boss, combat and GUI tasks all score 0%.

**Investigated, not integrated** — Open Pixel2Play, an open keyboard-and-mouse game policy
whose action space is 58% compatible with this one. See the README section and
`docs/upstream/open-p2p/`.

## Provenance

- Original location: `~/Deep Seek Harness/lumine/` (WSL)
- Copied: 2026-10-09, `rsync -a`, excluding `__pycache__`
- Verified: all 90 files md5-identical, and the full test suite re-run from this copy
- Upstream reference commit: `a329d98cbe62119679a254d71bea6446773541bc`
