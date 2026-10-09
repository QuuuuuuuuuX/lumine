# GUI Grounding / Screen-Understanding Benchmarks — VLM Comparison (retrieved Oct 2026)

**Snapshot date of leaderboard data:** 2026-10-09 (LLM-Stats "data checked Oct 9").
**Purpose:** advise a builder of a real-time game-playing VLM agent reading 640x360 screenshots.

## Provenance legend (read this first)

| Tag | Meaning |
|---|---|
| **[O]** | Officially reported / run by the model vendor or the benchmark owner (primary source). |
| **[A]** | Third-party aggregator or leaderboard. LLM-Stats explicitly marks all of these **"Self-reported", Verified = 0, Status: Unverified** — it aggregates vendor scorecards/blogs. BenchLM is a second aggregator. |
| **[P]** | Third-party academic paper, numbers reproduced by those authors under their own harness. |
| **n/f** | Not found / unverified. No primary or credible source located. |

**Do not treat [A] numbers as apples-to-apples.** Scaffold, step budget, screenshot resolution and retry policy are not controlled. Where an aggregator and the benchmark owner disagree, both are shown below.

## Main table

Model | ScreenSpot-Pro | OSWorld (v1 / Verified) | AndroidWorld | Other | Source
---|---|---|---|---|---
**Anthropic Claude** | | | |
Claude 4 Sonnet | n/f | **43.9%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
Claude Sonnet 4.5 | n/f | 61.4 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Claude Haiku 4.5 | n/f | 50.7 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Claude Opus 4.5 | n/f | 66.3 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld · https://www.anthropic.com/news/claude-opus-4-5
Claude Sonnet 4.6 | n/f | 72.5 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Claude Opus 4.6 | n/f | 72.7 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Claude Opus 4.7 | n/f | Verified 78.0 [A] | n/f | OSWorld 2.0: 318 avg tool calls [O] | https://llm-stats.com/benchmarks/osworld-verified · https://osworld-v2.xlang.ai/
Claude Opus 4.8 | 87.9 [A] | Verified 83.4 [A] | n/f | **OSWorld 2.0: 20.6% binary / 54.8% partial (best model)** [O] | https://llm-stats.com/benchmarks/screenspot-pro · https://osworld-v2.xlang.ai/
Claude Opus 5 | n/f | OSWorld 2.0: 70.6 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
Claude Opus 5.5 | n/f | OSWorld 2.0: 81.8 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
Claude Sonnet 5 | n/f | Verified 81.2 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Claude Fable 5 | n/f | Verified 85.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Claude Fable 5.1 | n/f | OSWorld 2.0: 77.9 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
**OpenAI GPT** | | | |
GPT-4o | n/f | **5%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
GPT-5.2 | 86.3 [A] | n/f | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
GPT-5.3 Codex | n/f | Verified 64.7 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
GPT-5.4 | n/f | Verified 75.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
GPT-5.4 mini | n/f | Verified 72.1 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
GPT-5.4 nano | n/f | Verified 39.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
GPT-5.5 | n/f | Verified 78.7 [A] | n/f | **OSWorld 2.0: ~13–14% [O] — see discrepancy below** | https://llm-stats.com/benchmarks/osworld-verified · https://osworld-v2.xlang.ai/
GPT-6 Astra | **92.7** [A] | OSWorld 2.0: 72.6 [A] | n/f | 30-benchmark grounding avg 71.35 [P] | https://llm-stats.com/benchmarks/screenspot-pro · https://arxiv.org/html/2609.39600
GPT-6.1 Sol | n/f | OSWorld 2.0: 71.4 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
**Google Gemini** | | | |
Gemini 3 Pro | 72.7 [A] | n/f | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Gemini 3 Flash | 69.1 [A] | n/f | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Gemini 2.5 Flash | n/f | n/f | n/f | — | n/f on all four leaderboards checked
Gemini 3.5 Flash | n/f | Verified 78.4 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Gemini 3.5 Flash-Lite | n/f | Verified 74.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Gemini 3.6 Flash | n/f | Verified 83.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Gemini 3.8 Flash | n/f | OSWorld 2.0: 59.0 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
Gemini 4 Argon | n/f | OSWorld 2.0: 69.2 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-2.0
**Alibaba Qwen3-VL** | | | |
Qwen2.5-VL-32B-Instruct | n/f | 5.9 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Qwen2.5-VL-72B-Instruct | 43.6 [A] | 8.8 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld
Qwen3-VL-4B-Instruct | 59.5 [A] | 26.2 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-4B-Thinking | 49.2 [A] | 31.4 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-8B-Instruct | 54.6 [A] | 33.9 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-8B-Thinking | 46.6 [A] | 33.9 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-30B-A3B-Instruct | 60.5 [A] | 30.3 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-30B-A3B-Thinking | 57.3 [A] | 30.6 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-32B-Instruct | 57.9 [A] | 32.6 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
Qwen3-VL-32B-Thinking | 57.1 [A] | 41.0 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro
**Qwen3-VL-235B-A22B-Instruct** | **62.0** [A] | **66.7** [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld
Qwen3-VL-235B-A22B-Thinking | 61.8 [A] | 38.1 [A] | n/f | OSWorld-G 68.3 [A] | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-g
Qwen3.5-27B | 70.3 [A] | Verified 56.2 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-verified
Qwen3.5-35B-A3B | 68.6 [A] | Verified 54.5 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-verified
Qwen3.5-122B-A10B | 70.4 [A] | Verified 58.0 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-verified
Qwen3.6 Plus | 68.2 [A] | Verified 62.5 [A] | n/f | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-verified
Qwen3.7-Plus | 79.0 [A] | Verified 73.3 [A] | 81.0 [A] | — | https://llm-stats.com/benchmarks/screenspot-pro · /androidworld
Qwen3.8 Max | 84.5 [A] | Verified **86.1** [A] | **85.3** [A] | — | https://llm-stats.com/benchmarks/screenspot-pro · /osworld-verified · /androidworld
Qwen3.8 Flash / Flash-Next | n/f | OSWorld 2.0: 19.4 [A] | 84.5 [A] | — | https://llm-stats.com/benchmarks/androidworld
Qwen3.8-27B | n/f | Verified 84.3 [A] | 81.9 [A] | — | https://llm-stats.com/benchmarks/androidworld
**Zhipu GLM** | | | |
GLM-4.5V | **n/f** | n/f | n/f | — | n/f
GLM-4.6V | **n/f** | 37.2% [secondary, citing vendor model card] | n/f | MMMU 76 | https://aimlapi.com/models/z-ai-glm-4-6v (cites huggingface.co/zai-org/GLM-4.6V)
GLM-4.6V-Flash | n/f | n/f | n/f | — | n/f
GLM-5V-Turbo | n/f | 62.3 [A] | 75.7 [A] | **WebVoyager 88.5** [A] | https://llm-stats.com/benchmarks/osworld · /androidworld · /webvoyager
GLM-OCR | n/f (not a GUI-grounding model) | n/f | n/f | OmniDocBench V1.5 **94.62** [O] | https://docs.z.ai/guides/vlm/glm-ocr
GLM-5.3-Flash | n/f | n/f | n/f | 320B total / 18B active; docs claim CUA+BUA but publish no GUI score [O] | https://docs.z.ai/guides/vlm/glm-5.3-flash
**ByteDance Doubao / Seed** | | | |
Doubao-Seed-1.6-vision | **n/f** | n/f | n/f | — | n/f
Seed 1.8 | n/f | 61.9 [A] | 70.7 [A] | — | https://llm-stats.com/benchmarks/osworld · /androidworld
Seed 2.1 Turbo | n/f | 76.4 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
Seed 2.1 Pro | n/f | **78.8** [A] | n/f | — | https://llm-stats.com/benchmarks/osworld
**StepFun** | | | |
Step-3 / Step-1o-vision | **n/f** | n/f | n/f | — | n/f
Step3-VL-10B | evaluated but numbers not retrievable this session → **n/f** | n/f | n/f | MMMU 80.11, MathVision 75.95, AIME2025 94.43 [O] | https://arxiv.org/abs/2601.09668 · https://stepfun-ai.github.io/Step3-VL-10B
**Moonshot Kimi** | | | |
Kimi K2.6 | n/f | Verified 73.1 [A] | n/f | — | https://llm-stats.com/benchmarks/osworld-verified
Kimi K3 | n/f | n/f | n/f | Exists (priced on AIMLAPI) but no grounding score found | https://aimlapi.com/models/kimi-k2-6
**DeepSeek** | | | |
DeepSeek-V4.1-Flash | **n/f** | n/f | n/f | No published grounding benchmark located | n/f (only a Chinese forum thread, unverified)
**Agent systems / references** | | | |
UI-TARS | n/f | **40.0%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
CoACT-1 | n/f | **60.76%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
Agent S2.5 w/ o3 | n/f | **56.0%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
GTA1 w/ o3 | n/f | **53.1%** [O] | n/f | — | https://xlang.ai/blog/osworld-verified
o3 | n/f | 9.1–23.0% (step-budget dependent) [O] | n/f | — | https://xlang.ai/blog/osworld-verified
Human baseline | n/f | **~72%** [O] | n/f | WAA human 74.5% | https://xlang.ai/blog/osworld-verified
UI-Venus-1.5 (Ant Group, Qwen3-VL-based) | **69.6%** [O] | n/f | **77.6%** [O] | WebVoyager 76.0% [O]; VenusBench-GD 75.0% [O] | https://arxiv.org/abs/2602.09082
LLaDA-UI (16.7B diffusion GUI agent) | 65.2 [P] | 41.8 [P] | 57.8 [P] | SS-v2 94.0; MobileWorld 25.6; WebVoyager 56.9 [P] | https://arxiv.org/html/2609.13287v1
GroundAnything-VLM (4B) | see paper | n/f | n/f | 30-grounding-benchmark avg **72.42%**, vs GPT-6 Astra 71.35 [P] | https://arxiv.org/html/2609.39600
Navi (WindowsAgentArena) | n/f | n/f | n/f | **WAA 19.5%** vs human 74.5% (2024) [O] | https://microsoft.github.io/WindowsAgentArena/

## Benchmarks with no usable current numbers

- **ScreenSpot-v2:** only two data points located. LLaDA-UI **94.0** [P] (https://arxiv.org/html/2609.13287v1) and a Hugging Face commit message claiming Muse-Glimmer-30B-GUI-Grounding-Fast at **88.1%** (unverified, no accessible model card). No frontier vendor publishes ScreenSpot-v2 on the pages reachable here. Treat as **largely n/f**.
- **WindowsAgentArena:** the Microsoft project page publishes only the original 2024 result (Navi 19.5%, human 74.5%, 154 tasks). **No 2026 leaderboard or current-model numbers were found.**
- **VisualWebArena:** **n/f** — no leaderboard numbers retrieved.
- **GUI-Odyssey:** **n/f**. (Note: UI-Venus-2 evaluates a different benchmark spelled "Odysseys", average rubric score over 200 tasks — do not conflate.)
- **WebVoyager:** only 3 points — GLM-5V-Turbo 88.5 [A], UI-Venus-1.5 76.0 [O], LLaDA-UI 56.9 [P]. The 88.5 vs 76.0 gap is a split/harness difference, not a capability difference.
- **Game-specific agent benchmarks:** only the Lumine paper (below) and StepFun's **GEBench** dataset (https://huggingface.co/datasets/stepfun-ai/GEBench — exists, details not retrievable here). No numeric leaderboard for Cradle, Jarvis-1, or Voyager-style agents was located.

## Conflicts and discrepancies (important)

1. **OSWorld 2.0 is the biggest conflict.** The benchmark owner reports best-model **binary completion 20.6%** (Claude Opus 4.8, 500 steps, max thinking). LLM-Stats lists **Claude Opus 5.5 at 81.8** and GPT-6 Astra at 72.6 for "OSWorld 2.0". These cannot be the same metric. The owner's abstract explicitly describes a *binary-completion* metric; the aggregator values are almost certainly partial-credit or a different variant. **Use the owner's numbers (20.6% / 54.8% partial / GPT-5.5 ~13%).**
2. **GPT-5.5 on OSWorld 2.0:** the same official page says "~13%" in the abstract and "near 14%" in the body. Minor internal inconsistency on the owner's own site.
3. **OSWorld v1 vs OSWorld-Verified vs OSWorld 2.0 are three different things.** Vendor PR often quotes whichever is highest. The official Verified re-run (July 2025) had CoACT-1 at 60.76% and Claude 4 Sonnet at 43.9% — note that Claude Opus 4.5's 66.3 on the aggregator's plain "OSWorld" is not comparable to the Verified numbers.
4. **ScreenSpot-Pro:** two independent aggregators agree on the top number (GPT-6 Astra 92.7 — BenchLM's own headline says "GPT-6 Astra Leads at 92.7%", matching LLM-Stats). That cross-check raises confidence in the ranking, but both ultimately trace to vendor self-reports.
5. **"Verified = 0" everywhere.** Every LLM-Stats grounding/agent leaderboard used above reports 0 independently verified results and labels itself Unverified.

## Reader-relevant takeaways for a 640x360 real-time game agent

- ScreenSpot-Pro is a **high-resolution desktop screenshot** benchmark. A 640x360 game frame is far lower resolution than its intended regime, so small-model ScreenSpot-Pro ranking is a weak proxy for game-screenshot grounding.
- The Qwen3-VL family shows a clean, monotone size ladder on ScreenSpot-Pro (4B 49–60 → 8B 47–55 → 32B 57 → 235B 62), which is useful for picking a local model if you self-host.
- **Instruction-vs-thinking variants invert on agentic benchmarks:** Qwen3-VL-235B **Instruct** scores 66.7 on OSWorld vs **38.1** for **Thinking** — a 28-point gap. Worth testing both on your game loop.
- Real-time constraint is the binding one. Lumine's solution (5 Hz perception, 30 Hz actions, adaptive reasoning) is the only published game-agent recipe found; OSWorld-style agents take seconds per action, which is unusable at 640x360 game frame rates.

## Lumine paper (arXiv 2511.08892) — what it actually reports

Title: **"Lumine: An Open Recipe for Building Generalist Agents in 3D Open Worlds"**. Byline: **ByteDance Seed** (with NTU authors). Submitted **12 Nov 2025**, CC BY 4.0. Project page: https://www.lumine-ai.org/
Sources: https://arxiv.org/abs/2511.08892 and https://arxiv.org/html/2511.08892v1 (both retrieved successfully).

- **It is not a benchmark suite.** It is an "open recipe"/agent. There is no leaderboard.
- **Reasoning brain:** Lumine is a **7B-parameter model built upon Qwen2-VL-7B-Base**.
- **Frequency/resolution:** consumes **raw pixels at 5 Hz**, emits textual keyboard-mouse actions at **30 Hz** via action chunking. It uses a **hybrid-thinking** strategy: at each step the model first emits either `<|thought_start|>` or `<|action_start|>`, so reasoning is invoked adaptively, not every frame.
- **Resolution:** the abstract, intro and model section do NOT state an input resolution. **Resolution = n/f** (do not assume one).
- **Memory:** up to **20 recent steps** kept in context as short-term memory; reasoning steps retained as long-term memory.
- **Training:** three stages — **1731 h** human gameplay pretraining, **200 h** instruction-following, **15 h** reasoning. End-to-end optimisation gives a **25.3x** latency reduction.
- **Its own benchmark:** a custom suite with four task categories — Collection, Combat, NPC Interaction, Puzzle. No ScreenSpot-Pro/OSWorld-style numbers.
- **Reported results:** >80% success on 10 s–several-minute instruction tasks; completes Act I of the Genshin Impact Mondstadt storyline at expert-human efficiency (~1 h); Acts II+III (~4 h) comparable; zero-shot Liyue one-hour mission; **zero-shot cross-game** — 100-minute mission in Wuthering Waves and the full five-hour first chapter of Honkai: Star Rail.
- **Sec. 2 Table 1** positions Lumine as the only row with Open-World + 5-hour task horizon + adaptive reasoning + real-time + keyboard/mouse, vs Voyager, Cradle, SIMA, CombatVLA, Jarvis-VLA.
- **Directly relevant warning in the paper:** GUI agents' absolute-click "teleport the cursor" abstraction **fails in games**; 3D games need **relative mouse movement** (camera control) and **key down/up/hold** semantics.

## Source URLs

Primary (vendor / benchmark owner):
- https://arxiv.org/abs/2511.08892 — Lumine abstract
- https://arxiv.org/html/2511.08892v1 — Lumine full text (backbone, 5 Hz/30 Hz, training)
- https://arxiv.org/abs/2601.09668 · https://arxiv.org/html/2601.09668v2 · https://stepfun-ai.github.io/Step3-VL-10B — StepFun Step3-VL-10B
- https://arxiv.org/abs/2602.09082 — UI-Venus-1.5 (Ant Group): ScreenSpot-Pro 69.6, AndroidWorld 77.6, WebVoyager 76.0
- https://arxiv.org/html/2609.00028v1 — UI-Venus-2
- https://arxiv.org/abs/2606.29537 · https://osworld-v2.xlang.ai/ — OSWorld 2.0 official
- https://osworld-v1.xlang.ai/ — OSWorld v1 official
- https://xlang.ai/blog/osworld-verified — official OSWorld-Verified re-run results
- https://microsoft.github.io/WindowsAgentArena/ — WindowsAgentArena official
- https://docs.z.ai/guides/vlm/glm-ocr — GLM-OCR official (OmniDocBench V1.5 94.62)
- https://docs.z.ai/guides/vlm/glm-5.3-flash — GLM-5.3-Flash official
- https://docs.z.ai/guides/vlm/glm-5v-turbo — GLM-5V-Turbo official
- https://www.anthropic.com/news/claude-opus-4-5 — Claude Opus 4.5 announcement
- https://seed.bytedance.com/en/seed1_8 — ByteDance Seed model index
- https://huggingface.co/datasets/stepfun-ai/GEBench — StepFun game benchmark (metadata only; HF unreachable here)

Third-party aggregators (treat as unverified/self-reported):
- https://llm-stats.com/benchmarks/screenspot-pro
- https://llm-stats.com/benchmarks/osworld
- https://llm-stats.com/benchmarks/osworld-verified
- https://llm-stats.com/benchmarks/osworld-2.0
- https://llm-stats.com/benchmarks/osworld-g
- https://llm-stats.com/benchmarks/androidworld
- https://llm-stats.com/benchmarks/webvoyager
- https://benchlm.ai/benchmarks/screenspot-pro
- https://benchlm.ai/benchmarks/osworld-verified
- https://benchlm.ai/benchmarks/androidworld
- https://aimlapi.com/models/z-ai-glm-4-6v — GLM-4.6V OSWorld 37.2% (secondary, cites vendor HF)

Third-party papers used as cross-references:
- https://arxiv.org/html/2609.13287v1 — LLaDA-UI
- https://arxiv.org/html/2609.39600 — GroundAnything

Not retrievable / blocked in this environment: Hugging Face model cards (all fetches failed), arXiv PDFs (unsupported content type), qwen.ai blog (JS-rendered empty), benchlm.ai model pages (JS-rendered), z.ai blog (empty).
