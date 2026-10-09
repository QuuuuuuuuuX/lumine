# VLM APIs as the "reasoning brain" for a real-time game agent

**Compiled: ~Oct 2026.** Target use case: 640x360 game screenshots -> VLM -> structured JSON
(subgoal / key presses / mouse movement) at ~1-2 Hz, with a local policy at 30 Hz.

All prices are **USD per 1M tokens** unless marked ¥ (CNY). Sources are linked per claim.
Labels: **[OFFICIAL]** = vendor docs/pricing page, **[3P]** = third-party database/aggregator,
**[UNVERIFIED]** = could not confirm.

---

## 0. Three corrections to the brief (read these first)

**0.1 DeepSeek's public API *does* have vision now.** Your belief that it is text-only is out of date.
The docs page `Vision` exists, `deepseek-flash` accepts images, and the legacy id
`deepseek-v4-flash-vision-exp` is explicitly documented (retired; requests now served by
DeepSeek-V4.1-Flash). `deepseek-v4-pro` does **not** support vision. **[OFFICIAL]**
Sources: <https://api-docs.deepseek.com/guides/vision> , <https://api-docs.deepseek.com/quick_start/pricing>

**0.2 Lumine is a ByteDance Seed paper, not Alibaba/Taobao.** The HTML v1 byline reads
"ByteDance Seed" with correspondence `shiguang.sg@bytedance.com` / `weihao001@ntu.edu.sg`.
So "Qwen-VL is the natural fit because Lumine is Alibaba" is a false premise. The real Qwen
argument is stronger and simpler: **Lumine is literally built on `Qwen2-VL-7B-Base`**.
Also: Lumine perceives at **5 Hz** with 30 Hz keyboard/mouse output (not 1-2 Hz), uses a
hybrid-thinking scheme (emit `<|thought_start|>` only when needed, else `<|action_start|>`),
keeps ~20 recent steps in context, and reports a **25.3x** end-to-end latency reduction.
**[OFFICIAL]** Sources: <https://arxiv.org/abs/2511.08892> , <https://arxiv.org/html/2511.08892v1>

**0.3 There is no `qwen3.8-vl` / `qwen3.7-vl` / `qwen3.6-vl`.** The dedicated Qwen3-VL line stopped at
`qwen3-vl-plus` / `qwen3-vl-flash` and was reclassified as **legacy** while remaining served.
Vision moved *into* the general flagship: `qwen3.8-max`, `qwen3.8-flash`, `qwen3.7-plus` are
natively multimodal. **[OFFICIAL]**
Sources: <https://www.alibabacloud.com/help/en/model-studio/model-list-visual-understanding> ,
<https://www.alibabacloud.com/help/en/model-studio/models>

---

## 1. Master table

`OV` = OpenAI-compatible `/chat/completions` endpoint available.

| Provider / Family | Exact model id | Images? how passed | OV? exact base_url | In $/1M | Out $/1M | Latency tier | GUI grounding |
|---|---|---|---|---|---|---|---|
| **Alibaba Qwen-VL** (Singapore/intl) | `qwen3.8-max` | yes — `image_url` public URL **or** base64 data URL; `video_url`; 2048 imgs, 16M px/img, ≤2 GB video | yes — `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | 2.00 | 6.00 | no (flagship) | SS-Pro **84.5** [A], OSWorld-Verified **86.1** [A], AndroidWorld **85.3** [A] |
| Alibaba Qwen (intl) | `qwen3.8-flash` | same as above | same | 0.15 | 0.47 | **yes (flash)** | OSWorld 2.0: 19.4 [A], AndroidWorld 84.5 [A] |
| Alibaba Qwen (intl) | `qwen3.7-plus` | same | same | 0.40 (≤256K) / 1.20 | 1.60 / 4.80 | no | ScreenSpot-Pro **79.0**, OSWorld-Verified **73.3**, AndroidWorld **81.0** (thinking off) [O/A] |
| Alibaba Qwen (intl) | `qwen3-vl-plus` *(legacy, still served)* | `image_url` URL/base64; `video_url` | same | 0.20 (≤32K) / 0.30 / 0.60 | 1.60 / 2.40 / 4.80 | no | "world-leading … on OS World" (no numbers) |
| Alibaba Qwen (intl) | `qwen3-vl-flash` *(legacy, still served)* | same | same | **0.05** (≤32K) / 0.075 / 0.12 | **0.40** / 0.60 / 0.96 | **yes (flash)** | 2D/3D localization; no numbers |
| Alibaba Qwen (mainland) | `qwen3.8-max` | same | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 1.65 | 4.951 | no | unverified |
| Alibaba Qwen (mainland) | `qwen3-vl-flash` | same | same | **0.022** | **0.215** | **yes** | — |
| Alibaba Qwen (mainland) | `qwen3-vl-plus` | same | same | 0.144 | 1.434 | no | — |
| Alibaba (open-weight, served) | `qwen3-vl-235b-a22b-instruct` | same | same | 0.40 (SG) / 0.287 (BJ) | 1.60 / 1.147 | no | ScreenSpot **95.4**, ScreenSpot-Pro **62.0**, OSWorldG **66.7**, AndroidWorld **63.7** |
| Alibaba (open-weight, served) | `qwen3-vl-8b-instruct` | same | same | 0.18 (SG) / 0.072 (BJ) | 0.70 / 0.287 | — | not published |
| **Zhipu GLM** (z.ai intl) | `glm-5.3-flash` | `type: image_url` — URL *recommended*, or base64 data URL; multi-image = multiple blocks | yes — `https://api.z.ai/api/paas/v4` | 0.15 | 0.50 | **yes (Flash)** | visual coding / CUA-BUA; no benchmark numbers |
| Zhipu GLM (z.ai intl) | `glm-5.3-flashx` | same | same | 0.37 | 1.25 | **yes — 200 tok/s stated** | same |
| Zhipu GLM (z.ai intl) | `glm-4.6v` | same | same | 0.30 | 0.90 | no | OSWorld 37.2% [3P, secondary]; 106B-A12B; claimed ≈ Qwen3-VL-235B |
| Zhipu GLM (z.ai intl) | `glm-5v-turbo` | same | same | 5.00 [3P] | 22.00 [3P] | no | OSWorld 62.3 [A], AndroidWorld 75.7 [A], WebVoyager 88.5 [A] |
| Zhipu GLM (z.ai intl) | `glm-4.6v-flashx` | same | same | **0.04** | 0.40 | **yes** | — |
| Zhipu GLM (z.ai intl) | `glm-4.6v-flash` | same | same | **free** | free | yes | 9B open-weight; no grounding benchmark found |
| Zhipu GLM (z.ai intl) | `glm-4.5v` | same | same | 0.60 | 1.80 | no | — |
| Zhipu GLM (z.ai intl) | `glm-ocr` | PDF/JPG/PNG, ≤10 MB img, ≤50 MB PDF, ≤100 pages | same | 0.03 | 0.03 | yes (0.9B, vLLM/SGLang) | OCR only, not agentic |
| Zhipu GLM (mainland) | same ids `glm-4.6v` etc. | same | `https://open.bigmodel.cn/api/paas/v4` | ¥1 / 1M [3P+press] | ¥3 / 1M [3P+press] | — | — |
| **Moonshot Kimi** | `kimi-k3` | base64 data URL **or** Files-API `ms://<file-id>` — **public image URLs NOT supported** | yes — `https://api.moonshot.ai/v1` | 3.00 (cached 0.30) | 15.00 | no | unverified |
| Moonshot Kimi | `kimi-k2.6` | same | same | 0.95 (cached 0.16) | 4.00 | no | unverified |
| Moonshot Kimi | `kimi-k2.7-code` | same | same | 0.95 (cached 0.19) | 4.00 | no | unverified |
| Moonshot Kimi | `kimi-k2.7-code-highspeed` | same | same | 1.90 (cached 0.38) | 8.00 | **yes (highspeed)** | unverified |
| **ByteDance Doubao / Ark** (CN) | `doubao-seed-2-1-lite-260915` | Files API `file_id` (≤512 MB, recommended) / base64 data URL (<10 MB, body <64 MB) / public https URL; default `high` = **fixed 1280 tokens/image** | yes — `https://ark.cn-beijing.volces.com/api/v3` | ¥0.80 [O] | ¥2.70 [O] | **yes — `service_tier: "fast"`** | unverified (see §2.9) |
| Doubao / Ark (CN) | `doubao-seed-2-1-turbo-260628` | same | same | ¥3.00 [O] | ¥15.00 [O] | **yes — `service_tier: "fast"`** | unverified |
| Doubao / Ark (CN) | `doubao-seed-2-1-pro-260915` | same | same | ¥6.00 [O] | ¥30.00 [O] | no | unverified |
| Doubao / Ark (CN) | `doubao-seed-2-0-mini-260428` | same | same | ¥0.2 / ¥0.4 / ¥0.8 tiered [O] | ¥2.0 / ¥4.0 / ¥8.0 [O] | **yes — `service_tier: "fast"`** | unverified |
| Doubao / Ark (CN) | `doubao-seed-2-0-lite-260428` | same | same | ¥0.6 / ¥0.9 / ¥1.8 [O] | ¥3.6 / ¥5.4 / ¥10.8 [O] | **yes — `service_tier: "fast"`** | unverified |
| Doubao / Ark (CN) | `doubao-seed-evolving` (rolling, no version suffix) | same | same | ¥6.00 [O] | ¥30.00 [O] | no | unverified |
| Doubao / Ark (intl) | same ids, BytePlus ModelArk | same | `https://ark.ap-southeast.bytepluses.com/api/v3` (ap-southeast-1), `https://ark.eu-west.bytepluses.com/api/v3` (eu-west-1) | CNY list only — **no official USD list** | — | same | unverified |
| ~~Doubao~~ | ~~`doubao-seed-1-6-vision-250815`~~ | **RETIRED** — replaced by `doubao-seed-2-0-lite-260428` | — | — | — | — | — |
| **StepFun** (mainland) | `step-5-preview` | URL **or** base64; ≤60 imgs/req; detail `low`/`high` | yes — `https://api.stepfun.com/v1` | 1.00 (¥7) | 2.70 (¥20) | no (flagship, 1M ctx) | **not published for API models** — see §2.5 |
| StepFun (mainland) | `step-3.7-flash` | same; 256K ctx; `reasoning_effort` low/medium/high | same | **0.20 (¥1.35)** | 1.15 (¥8.1) | **yes (flash)** | **not published for API models** — see §2.5 |
| StepFun (mainland) | `step-1o-turbo-vision` | via `detail`; **169 tokens/image by default** | same | 0.36 (¥2.5) | 1.15 (¥8) | yes | **not published for API models** — see §2.5 |
| StepFun (open-weight) | `Step-GUI-4B` / `Step-GUI-8B` | local / self-host (Qwen3-VL backbone) | self-hosted | $0 | $0 | GPU-bound | SS-Pro **60.0 / 62.6**, SS-v2 **93.6 / 95.1**, OSWorld-G 66.9 / 70.0 [O] |
| StepFun (intl) | `step-5-preview` / `step-3.7-flash` / `step-1o-turbo-vision` | same | `https://api.stepfun.ai/v1` | same | same | same | same |
| ~~StepFun~~ | ~~`step-1o-vision-32k` / `-highres`~~ | **RETIRED** → `step-1o-turbo-vision` | — | — | — | — | — |
| ~~StepFun~~ | ~~`step-3.5-flash`~~ | **text/reasoning ONLY — not a vision model** | `https://api.stepfun.com/v1` | 0.10 (¥0.7) | 0.30 (¥2.1) | yes | n/a |
| **OpenAI** | `gpt-5.6-luna` | base64 data URL **or** public URL; `detail` param | yes (native) — `https://api.openai.com/v1` | 0.20 [3P] | 1.20 [3P] | **yes (Luna = small tier)** | unverified |
| OpenAI | `gpt-5.4-nano` | same | same | 0.20 [3P] | 1.25 [3P] | **yes (nano)** | unverified |
| OpenAI | `gpt-5.6-terra` | same | same | 2.00 [3P] | 12.00 [3P] | mid | unverified |
| OpenAI | `gpt-5.6-sol` | same | same | 4.00 [3P] | 20.00 [3P] | no | unverified |
| OpenAI | `gpt-5.1` | same | same | 1.25 [3P] | 10.00 [3P] | no | unverified |
| OpenAI | `gpt-5.2` | same | same | — | — | no | ScreenSpot-Pro **86.3** [A] |
| OpenAI | `gpt-6-astra` | same | same | 10.00 [3P] | 50.00 [3P] | no | ScreenSpot-Pro **92.7** [A] (top of leaderboard); OSWorld 2.0 72.6 [A] |
| **Anthropic Claude** | `claude-haiku-5-5` | base64 / URL / Files API; native `image` block | **shim only** — `https://api.anthropic.com/v1/` (OpenAI SDK layer; images OK, but Anthropic's doc says *not for production* and silently ignores `response_format`) | 0.10 [3P] | 0.50 [3P] | **yes (Haiku)** | unverified |
| Anthropic Claude | `claude-sonnet-5` | same | same | 2.00 [3P] | 10.00 [3P] | mid | OSWorld-Verified 81.2 [A] |
| Anthropic Claude | `claude-opus-5` | same | same | 5.00 [3P] | 25.00 [3P] | no | OSWorld 2.0 70.6 [A] |
| Anthropic Claude | `claude-opus-4-8` | same | same | 5.00 [3P] | 25.00 [3P] | no | SS-Pro 87.9 [A]; OSWorld-Verified 83.4 [A]; **OSWorld 2.0 20.6% binary / 54.8% partial [O] = best** |
| **Google Gemini** | `gemini-3.8-flash` | inline base64 **or** Files API; OpenAI-compat shim takes `image_url` | `https://generativelanguage.googleapis.com/v1beta/openai` (see §2.6) | 0.75 [3P] | 3.75 [3P] | **yes (Flash)** | OSWorld 2.0 59.0 [A] |
| Google Gemini | `gemini-3.5-flash-lite` | same | same | 0.30 [3P] | 2.50 [3P] | **yes (Flash-Lite)** | OSWorld-Verified 74.0 [A] |
| Google Gemini | `gemini-3.1-flash-lite` | same | same | 0.25 [3P] | 1.50 [3P] | **yes (Flash-Lite)** | unverified |
| Google Gemini | `gemini-3-flash-preview` | same | same | 0.50 [3P] | 3.00 [3P] | **yes (Flash)** | ScreenSpot-Pro 69.1 [A] |
| Google Gemini | `gemini-3.1-pro-preview` | same | same | 2.00 [3P] | 12.00 [3P] | no | ScreenSpot-Pro 72.7 [A] (Gemini 3 Pro) |
| Google Gemini | `gemini-2.5-computer-use-preview-10-2025` | screenshot in, UI actions out (purpose-built CUA) | `.../v1beta/openai` | 1.25 [3P] | 10.00 [3P] | no | purpose-built for screen control |
| **DeepSeek** | `deepseek-flash` | **yes** — base64 data URL, public https URL (≤8192 chars, ≤32 MiB, 60 s), or Files-API `file_id`; max **1024 tokens/image**; `detail: low` → 512x512 | yes — `https://api.deepseek.com` | **0.15 off-peak / 0.30 peak** | **0.60 off-peak / 1.20 peak** | no formal flash tier, but V4.1-Flash is the cheap/fast SKU | unverified |
| **Local (Ollama)** | `qwen3-vl:4b` | local image path / base64 | yes — `http://localhost:11434/v1` | $0 (electricity) | $0 | GPU-bound | inherits Qwen3-VL |
| Local (Ollama) | `qwen3-vl:8b` | same | same | $0 | $0 | ~6.1 GB, tight on 8 GB | better grounding |
| Local (Ollama) | `qwen3-vl:2b` | same | same | $0 | $0 | ~1.9 GB | weakest |
| Local (Ollama) | `qwen2.5vl:3b` / `:7b` | same | same | $0 | $0 | older line | Qwen2.5-VL |

---

## 2. Detail per family

### 2.1 Alibaba Qwen-VL (DashScope / Model Studio)

**Base URLs — both legacy shared and workspace-dedicated forms are officially "still available".**
[OFFICIAL] <https://www.alibabacloud.com/help/en/model-studio/base-url>

| Region | OpenAI-compatible | Anthropic-compatible | Native |
|---|---|---|---|
| Singapore / international | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `.../api/v1` |
| Chinese mainland (Beijing) | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `.../api/v1` |
| US (Virginia) | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `.../api/v1` |
| China (Hong Kong) | `https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `.../api/v1` |

Workspace-dedicated (recommended for production, higher throughput / lower latency):
`https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1` with region in
`cn-beijing | ap-southeast-1 | ap-northeast-1 | eu-central-1 | us-east-1 | cn-hongkong`.
Trial: `https://trial.{region}.maas.aliyuncs.com/compatible-mode/v1`.
**API keys are region-specific — a key/domain mismatch returns 401.**

**Image token accounting [OFFICIAL]:** `tokens = int(h_bar * w_bar / (32*32)) + 2` (32x32 patch for
Qwen3-VL and Qwen3.5-3.8; 28x28 for QVQ/Qwen2.5-VL). Max **16,384 tokens** and max **16,777,216 px**
(16M) per image; min 10x10 px; aspect ≤ 200:1. Default `max_pixels` 2,621,440 for Qwen3-VL.
For a **640x360** screenshot: 640*360 = 230,400 px → ~225 tokens + 2 ≈ **227 tokens/image** — extremely cheap.

**Function calling caveat:** supported in Beijing, **explicitly unsupported in Singapore** for
`qwen3-vl-plus` / `qwen3-vl-flash`. If you plan to use tool-calling, use the mainland endpoint or a
current-generation model.

**GUI grounding [OFFICIAL]:**
- `qwen3.7-plus`: ScreenSpot-Pro **79.0**, OSWorld-Verified **73.3**, AndroidWorld **81.0** (thinking off).
- `qwen3-vl-235b-a22b-instruct`: ScreenSpot **95.4**, ScreenSpot-Pro **62.0**, OSWorldG **66.7**, AndroidWorld **63.7**, RefCOCO_avg 91.9.
  (Note `OSWorldG` is the grounding variant, *not* full OSWorld.)
- Benchmark tables are published as **images** in the Qwen blog/README, which is why they don't appear in text search.
  Sources: <https://www.alibabacloud.com/blog/603206> , <https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md> ,
  <https://arxiv.org/pdf/2511.21631>
- `gui-plus` (Beijing only) is a purpose-built cross-app GUI model but publishes **no benchmark numbers**.

### 2.2 Zhipu GLM (z.ai + bigmodel.cn)

- OpenAI-compatible base_url: `https://api.z.ai/api/paas/v4` (z.ai) and `https://open.bigmodel.cn/api/paas/v4` (mainland). [OFFICIAL / 3P]
- Images: add a `type: image_url` content block; **URL recommended**, base64 data URL also works. [OFFICIAL]
  <https://docs.z.ai/guides/vlm/glm-5.3-flash>
- `glm-5.3-flashx` is documented at **200 tokens/s** — the only vendor-published throughput number I found for any model in this report. [OFFICIAL]
- GLM-4.6V is 106B-A12B and GLM-4.6V-Flash is 9B; both open-weighted at <https://github.com/zai-org/GLM-V>. Mainland price cut to ¥1 in / ¥3 out per 1M, Flash free. [3P + press]
- A third-party entry `glm-5v-turbo` ($5.00/$22.00) appears in models.dev but is **not** on z.ai's own pricing page — **[UNVERIFIED]**.

### 2.3 Moonshot Kimi

- **Critical constraint: public image URLs are not supported.** Only base64 data URLs and
  Files-API uploads (`ms://<file-id>`). If your agent posts a screenshot as a URL, Kimi will fail.
  [OFFICIAL] <https://platform.kimi.ai/docs/guide/use-kimi-vision-model.md>
- SVG is rejected. Request body ≤ 100 MB. Recommended max resolution 4K.
- Video is also supported (`video_url` with `ms://`), which is unusual and potentially useful for
  a game agent that wants to look at a short clip rather than single frames.
- Context: `kimi-k3` 1,048,576 tokens; K2-series 262,144.

### 2.4 ByteDance Doubao / Volcengine Ark

**Base URLs — both confirmed official.** [OFFICIAL]
- Mainland: `https://ark.cn-beijing.volces.com/api/v3` (OpenAI-compatible `POST /chat/completions`, plus `/responses`).
  Ark's docs state it is compatible with both the OpenAI and Anthropic protocols.
- International (BytePlus ModelArk): `https://ark.ap-southeast.bytepluses.com/api/v3` (ap-southeast-1) and
  `https://ark.eu-west.bytepluses.com/api/v3` (eu-west-1).

**The dedicated vision SKU is gone.** `doubao-seed-1-6-vision-250815` is **retired** — it is absent from
the live model list and appears in the official deprecation/migration guide with replacement
`doubao-seed-2-0-lite-260428`. Every current Doubao flagship is natively multimodal, so **there is no
`*-vision` successor to look for**. Current vision roster: `doubao-seed-evolving` (unified rolling id,
no version suffix), `doubao-seed-2-1-pro-260915`, `doubao-seed-2-1-turbo-260628`,
`doubao-seed-2-1-lite-260915`, `doubao-seed-2-0-lite-260428`, `doubao-seed-2-0-mini-260428`,
`doubao-seed-2-0-pro-260215`, `doubao-seed-1-8-251228`. `doubao-seed-2.0-lite/mini` are marketed as
"轻量均衡型全模态模型" (text+image+audio+video, GUI closed-loop ops). [OFFICIAL]

**Image input (official):** (a) **Files API `file_id` — recommended**, images up to **512 MB**, poll until
status `active`; (b) base64 data URL `data:{mime_type};base64,{data}` — single image < 10 MB, body < 64 MB;
(c) public https URL via `image_url`. Pixel range [196, 36,000,000] px or hard error.
`detail` = `low` / `high` / `xhigh` (Evolving); `image_pixel_limit` overrides `detail`.

**Image token formula (documented):** `tokens = w × h ÷ 1764`, capped by mode. For Seed 2.0+ the default
`high` mode is a **fixed 1280 tokens/image** regardless of size. Docs' own worked examples: 1280×960 →
697 tokens; 2560×1440 → 2090 → capped at 1280. Seed 1.8 cut ≤4 MP image tokens to 44.4% of the prior
encoder. [OFFICIAL] *(For your 640x360 frames: 640×360÷1764 ≈ 131 tokens, but the 1280-token floor
applies on Seed 2.0+, so Ark is the most expensive per-frame of the cheap options.)*

**Pricing — CNY only; Ark publishes no official USD list.** [OFFICIAL] Standard online inference
(¥ per 1M tokens), reconstructed from the pricing page's own table markup:

| Model | Input | Cache hit | Output |
|---|---|---|---|
| `doubao-seed-evolving` | ¥6.00 | ¥1.20 | ¥30.00 |
| `doubao-seed-2-1-pro` | ¥6.00 | ¥1.20 | ¥30.00 |
| `doubao-seed-2-1-turbo` | ¥3.00 | ¥0.60 | ¥15.00 |
| `doubao-seed-2-1-lite` | ¥0.80 | ¥0.16 | ¥2.70 |
| `doubao-seed-2-0-pro` (tiered) | ¥3.2 / ¥4.8 / ¥9.6 | — | ¥16.0 / ¥24.0 / ¥48.0 |
| `doubao-seed-2-0-lite` (tiered) | ¥0.6 / ¥0.9 / ¥1.8 | — | ¥3.6 / ¥5.4 / ¥10.8 |
| `doubao-seed-2-0-mini` (tiered) | ¥0.2 / ¥0.4 / ¥0.8 | — | ¥2.0 / ¥4.0 / ¥8.0 |
| `doubao-seed-1-8` (tiered) | ¥0.80 / ¥0.80 / ¥1.20 / ¥2.40 | ¥0.80 | ¥1.00 / ¥2.00 / ¥8.00 / ¥12.00 |

Tiers are by input length in k tokens: `[0,32] / (32,128] / (128,256]`. Cache storage ¥0.017 per 1M·token·hour.
Ark documents 分段计价 explicitly with a worked example: 200k in + 14k out lands in `(128,256]` → billed
¥2.4 in / ¥24 out per 1M.

**Latency tier — yes, and it is request-scoped.** Ark offers 在线推理（低延迟）, opted into per request with
`service_tier: "fast"` (Chat + Responses API) or as an endpoint default. It explicitly targets TPOT and
**auto-downgrades to standard inference on throttling / burst protection** — so do not build a hard
real-time guarantee on it. Supported ids: `doubao-seed-2-1-lite-260915`, `doubao-seed-2-1-turbo-260628`,
`doubao-seed-2-0-pro-260215`, `doubao-seed-2-0-lite-260428`, `doubao-seed-2-0-lite-260215`,
`doubao-seed-2-0-mini-260428` (no fine-tuned/branch models). **No absolute TPOT/TPS/ms is published.**
One qualitative official figure: Seed-Evolving image+video understanding task time reduced ~40% vs the
previous generation. Also available: a low-priority "Flex" tier and TPM guarantee packages. [OFFICIAL]

**GUI grounding: no officially published benchmark for any Doubao model.**
`GUI 任务处理能力` is a first-class model-list category with a GUI Agent tutorial, and Seed 2.0 Lite/Pro plus
Seed-Evolving carry `GUI 任务处理` and/or `视觉定位` (2D/3D grounding) tags — but **no ScreenSpot-Pro /
OSWorld / AndroidWorld number exists**. **Warning:** ByteDance's published GUI-grounding scores belong to
the *separate open-source UI-TARS / UI-TARS-2 family*; do not attribute those to Doubao API SKUs.

*Method note for anyone continuing this research:* the Volcengine doc pages are client-rendered, but their
backend JSON API works — `https://www.volcengine.com/api/doc/getDocDetail?LibraryCode=ark&DocumentCode=<code>&lang=zh`
returns Quill "ACE table" JSON.

### 2.5 StepFun

Clean official docs (append `.md` to Mintlify URLs to get the full page):
- Base URLs: `https://api.stepfun.com/v1` (mainland), `https://api.stepfun.ai/v1` (international). [OFFICIAL]
- **Live vision lineup is exactly three models:** `step-5-preview`, `step-3.7-flash`, `step-1o-turbo-vision`.
  `step-1o-vision-32k` and `-highres` are **retired**; **`step-3.5-flash` is text/reasoning only — it is
  not a vision model.** [OFFICIAL]
- `step-3.7-flash` at $0.20/$1.15 (or ¥1.35/¥8.1) is the standout value here: native image+video, 256K ctx,
  and **three reasoning-effort levels** (`low`/`medium`/`high`) — directly relevant to a 1-2 Hz budget,
  since you can run `low` for routine frames and `high` only on the adaptive-reasoning ticks.
- Official CNY prices to pair with the USD list: `step-5-preview` ¥7 in / ¥0.35 cache-hit / ¥20 out;
  `step-3.7-flash` ¥1.35 / ¥0.27 / ¥8.1; `step-1o-turbo-vision` ¥2.5 / ¥0.5 / ¥8 per 1M. [OFFICIAL]
- `step-1o-turbo-vision` bills **169 tokens per image by default** with `detail` off; with `detail`
  enabled the token count scales with image size. For 640x360 you almost certainly want `detail` off. [OFFICIAL]
- Limits: longest side < 4096 px, JPG/PNG/static GIF/WebP, combined upload < 20 MB,
  http/https URLs must be reachable **from mainland China**. [OFFICIAL]

**GUI grounding — real numbers exist, but there is a scope trap.** StepFun's strong GUI results belong to
the **open-source Step-GUI-4B / 8B specialists (Qwen3-VL backbone), NOT to the API-served
`step-5-preview` / `step-3.7-flash`.** Do not put these in a row for the API models. From the official
Step-GUI Technical Report (arXiv 2512.15431, GELab-Team/StepFun, Dec 2025) [O]:

| Model | ScreenSpot-Pro | ScreenSpot-v2 | OSWorld-G | MMBench-GUI-L2 | VisualWebBench |
|---|---|---|---|---|---|
| Step-GUI-8B | 62.6 | 95.1 | 70.0 | 85.6 | 89.7 |
| Step-GUI-4B | 60.0 | 93.6 | 66.9 | 84.0 | 90.7 |

End-to-end Pass@3: 8B OSWorld-Verified **48.5**, AndroidWorld **80.2**; 4B 40.4 / 75.8.
Pass@1: 8B 40.2 / 67.7; 4B 31.9 / 63.9. Proprietary AndroidDaily: 8B 89.91% static / 52.50% E2E.
*The paper is internally inconsistent on ScreenSpot-Pro — 65 in the Introduction vs 62.6 in the abstract,
contributions and Table 3. 62.6 is the consistent value.*

- Sources: <https://platform.stepfun.ai/docs/en/guides/pricing/details.md> ,
  <https://platform.stepfun.ai/docs/en/guides/models/vision.md> ,
  <https://platform.stepfun.ai/docs/en/guides/developer/openai.md> ,
  <https://platform.stepfun.com/docs/zh/guides/developer/openai.md> ,
  <https://arxiv.org/abs/2512.15431>

### 2.6 OpenAI / Anthropic / Google

**Important caveat on this whole section:** `platform.openai.com`, `developers.openai.com`,
`openai.com`, `ai.google.dev`, `cloud.google.com` and `huggingface.co` all returned **HTTP 403 or
connection failure** to the fetcher used for this report, and `platform.claude.com` cross-origin
redirects. Everything in these three rows is therefore **[3P]** (models.dev, an open-source model
database) unless noted, and should be re-confirmed against the vendor pricing page before you commit.

- **OpenAI**: native OpenAI-compatible by definition (`https://api.openai.com/v1`). OpenAI remains the
  reference implementation for `image_url` + `detail` handling. The small tiers (`gpt-5.6-luna`,
  `gpt-5.4-nano`, `gpt-6-luna`) are the relevant low-latency/cheap options.
- **Anthropic**: the native API is the **Messages** API at `https://api.anthropic.com/v1` — it is *not*
  a `/chat/completions` endpoint. Anthropic does publish an OpenAI-SDK compatibility layer at
  `https://api.anthropic.com/v1/` (confirmed via a mirror of Anthropic's own `openai-sdk` doc page,
  since `platform.claude.com` redirected cross-origin for us). Read the caveats carefully before using
  it in a game loop:
  - **Image inputs ARE supported** through the shim.
  - **"This compatibility layer is for testing and comparison, not production."**
  - **Unsupported fields are silently ignored** rather than erroring — including `response_format`.
    That is dangerous if you rely on strict JSON schema enforcement for your action output;
    scheme for it in the prompt instead, or use the native Messages API.
  - Other ignored params: `logprobs`, `presence_penalty`, `frequency_penalty`, `seed`, `audio`,
    `modalities`. Prompt caching is **not** available through the shim — use the native SDK.
  - System messages are hoisted and concatenated at the start.
  `claude-haiku-5-5` at $0.10/$0.50 is remarkably cheap if the [3P] price holds.
  *(The mirror we read is Claude-3 era, so the supported-model list on it is stale — the structural
  facts above are what matter; re-check the shim against a current model before relying on it.)*
- **Google Gemini**: `gemini-*-flash` / `-flash-lite` are the latency tiers. Gemini also ships
  `gemini-2.5-computer-use-preview-10-2025`, a purpose-built computer-use model (screenshot in,
  UI actions out) — the closest thing any US lab offers to a drop-in screen agent, though it is
  trained for desktop/web UI rather than 3D game control. The OpenAI-compatibility base_url
  `https://generativelanguage.googleapis.com/v1beta/openai` is confirmed by
  [Microsoft Learn](https://learn.microsoft.com/en-us/azure/api-management/openai-compatible-google-gemini-api),
  which cites Google's own OpenAI-compatibility doc and uses `POST /chat/completions` against it —
  Google's page itself was network-blocked here, so treat the base_url as well-corroborated but
  not first-party-verified in this session.
- Sources: <https://models.dev/providers/openai> , <https://models.dev/providers/anthropic> ,
  <https://models.dev/providers/google> , <https://claude.com/pricing#api> ,
  Anthropic OpenAI-SDK compat (mirror, Claude-3 era): <https://raw.githubusercontent.com/meirm/nano-agent/main/ai_docs/anthropic_openai_compat.md>

### 2.7 DeepSeek — the sleeper pick

Fully verified, and much better than the brief assumed. [OFFICIAL]

- Model id: **`deepseek-flash`** (one id; the legacy `deepseek-v4-flash-vision-exp` and
  `deepseek-v4-flash` names are still accepted but route to DeepSeek-V4.1-Flash). `deepseek-v4-pro` = no vision.
- base_url: `https://api.deepseek.com` (OpenAI-compatible), `https://api.deepseek.com/anthropic` (Anthropic-compatible).
- Images: base64 data URL / public https URL / Files-API `file_id`; JPEG, PNG, GIF, WebP.
  **Max 1024 tokens per image**, auto-resized to ~1300x1300-equivalent total pixels, so resolution
  beyond ~1300px is free of extra cost. `detail: low` downscales to 512x512.
  Images are allowed **only in `user` messages** (400 in system/assistant).
- Price: **$0.15 in / $0.60 out** off-peak, doubling at peak (peak = 01:00-04:00 and 06:00-10:00 UTC
  Mon-Fri, excluding Chinese holidays). Cache hit $0.003 off-peak.
- 1M context, 384K max output, JSON output + tool calls + Responses API all supported.
- Also available on Volcengine Ark as `deepseek-v4-flash-0731` ($0.45/$1.34 [3P]).

### 2.8 Local, single 8 GB VRAM laptop GPU

Verified live Ollama tags and sizes (all 256K context, text+image input):
[OFFICIAL] <https://ollama.com/library/qwen3-vl> , <https://ollama.com/library/qwen3-vl/tags> , <https://ollama.com/library/qwen2.5vl/tags>

| Ollama tag | Size | Note for 8 GB VRAM |
|---|---|---|
| `qwen3-vl:2b` | 1.9 GB | fits easily, leaves room for long context |
| `qwen3-vl:4b` | 3.3 GB | **sweet spot** — fits with a large KV cache |
| `qwen3-vl:8b` | 6.1 GB | fits the weights, but a 256K KV cache will spill to CPU; keep context short |
| `qwen3-vl:30b` | 20 GB | no |
| `qwen3-vl:32b` | 21 GB | no |
| `qwen3-vl:235b` | 143 GB | no (there is a `qwen3-vl:235b-cloud` offload tag) |
| `qwen2.5vl:3b` | — | older line |
| `qwen2.5vl:7b` | — | older line |

Explicit quantisation tags exist, e.g. `qwen3-vl:4b-instruct-q4_K_M`, `qwen3-vl:8b-thinking-q4_K_M`,
`qwen3-vl:8b-instruct-q8_0`, `qwen3-vl:2b-thinking-bf16`, and both `-instruct` and `-thinking`
variants for 2b/4b/8b/30b-a3b/32b/235b-a22b. Requires **Ollama >= 0.12.7**.

**License caveat [UNVERIFIED]:** `huggingface.co` was unreachable from this host, so I could not
re-confirm the license on the model cards. Note that `Qwen2.5-VL-3B-Instruct` in particular has a
publicly discussed **non-commercial / research** license, so if you need commercial use, prefer the
Qwen3-VL small models or the Apache-licensed Qwen2.5-VL-7B and verify on the model card first.

**Do not expect a 4B local model to be your reasoning brain.** For a 640x360 game feed, treat the
local VLM as a fallback/offline mode or an auxiliary (e.g. OCR of a quest log), and keep the cloud
VLM as the planner. This matches Lumine's own design: it fine-tuned a 7B, but with 1731 h of
gameplay pretraining plus instruction and reasoning stages — a raw 4B base model will not ground
game semantics out of the box.

**One local option is purpose-built for this and worth a look: `Step-GUI-4B` / `Step-GUI-8B`**
(open-weight, Qwen3-VL backbone, from StepFun — arXiv 2512.15431). They are GUI-grounding specialists:
Step-GUI-8B scores ScreenSpot-Pro **62.6**, ScreenSpot-v2 **95.1**, OSWorld-G **70.0**, and 4B scores
60.0 / 93.6 / 66.9 — i.e. a *4B local model matching the 235B Qwen3-VL on ScreenSpot-Pro*. Note these
are **not** the API-served StepFun models (§2.5); you run them yourself. On 8 GB VRAM the 4B is the
realistic one.

### 2.9 GUI-grounding benchmarks — what is actually known

**Read the provenance tags.** `[O]` = benchmark owner / vendor primary. `[A]` = third-party aggregator,
and note that **LLM-Stats marks every one of these entries "Self-reported, Verified = 0, Status:
Unverified"**. `[P]` = reproduced in a third-party paper. `n/f` = not found.

| Model | ScreenSpot-Pro | OSWorld (variant noted) | AndroidWorld | Source |
|---|---|---|---|---|
| GPT-6 Astra | **92.7** [A] | OSWorld 2.0: 72.6 [A] | — | llm-stats |
| Claude Opus 4.8 | 87.9 [A] | **OSWorld 2.0: 20.6% binary / 54.8% partial [O]** | — | osworld-v2.xlang.ai |
| GPT-5.2 | 86.3 [A] | — | — | llm-stats |
| **Qwen3.8 Max** | **84.5** [A] | OSWorld-Verified **86.1** [A] | **85.3** [A] | llm-stats |
| Qwen3.7-Plus | 79.0 [A] | OSWorld-Verified 73.3 [A] | 81.0 [A] | llm-stats / Qwen blog |
| Claude Sonnet 5 | n/f | OSWorld-Verified 81.2 [A] | — | llm-stats |
| Gemini 3 Pro | 72.7 [A] | — | — | llm-stats |
| Gemini 3 Flash | 69.1 [A] | — | — | llm-stats |
| Qwen3-VL-235B-A22B-Instruct | 62.0 [A] | 66.7 [A] | 63.7 [A] | llm-stats |
| Qwen3-VL-235B-A22B-**Thinking** | 61.8 [A] | **38.1** [A] | — | llm-stats |
| Qwen2.5-VL-72B *(reference)* | 43.6 [A] | 8.8 [A] | 8.8 [A] | llm-stats |
| UI-Venus-1.5 (Qwen3-VL-based, Ant Group) | **69.6%** [O] | — | 77.6% [O] | arXiv 2602.09082 |
| **Step-GUI-8B** (open-weight specialist) | 62.6 [O] | OSWorld-G 70.0 [O]; OSWorld-Verified 48.5 Pass@3 [O] | 80.2 Pass@3 [O] | arXiv 2512.15431 |
| **Step-GUI-4B** (open-weight specialist) | 60.0 [O] | OSWorld-G 66.9 [O]; OSWorld-Verified 40.4 Pass@3 [O] | 75.8 Pass@3 [O] | arXiv 2512.15431 |
| Claude 4 Sonnet | n/f | 43.9% [O] | — | xlang.ai |
| GPT-4o | n/f | 5% [O] | — | xlang.ai |
| Human baseline | — | ~72% [O] | — | xlang.ai |

**Not found (do not fill these in):** GLM-4.5V, GLM-4.6V-Flash, `doubao-seed-1-6-vision`,
Step-3 / Step-1o-vision, DeepSeek-V4.1-Flash, Gemini 2.5 Flash, Kimi K3 — **no grounding benchmark
located on any leaderboard or vendor page**. Step3-VL-10B's paper does evaluate ScreenSpot-Pro and
OSWorld-G but the numeric table was not retrievable. ScreenSpot-v2, VisualWebArena, GUI-Odyssey and
any 2026 WindowsAgentArena leaderboard are also unavailable.

**Two conflicts you must not paper over:**

1. **OSWorld 2.0 is reported with two different metrics.** The benchmark owner reports best *binary
   completion* **20.6%** (Claude Opus 4.8, 500 steps), while LLM-Stats lists Opus 5.5 at **81.8**.
   These are almost certainly binary vs partial credit. **Trust the owner's number.**
2. OSWorld v1, OSWorld-Verified and OSWorld 2.0 are three different benchmarks, and vendor PR quotes
   whichever is highest. Always record which one a number came from.

**The honest conclusion for a 640x360 game agent:** ScreenSpot-Pro is a **high-resolution desktop**
benchmark. 640x360 game frames sit far outside its regime, so it is a *weak* proxy for what you care
about. Use it only to rank candidates, then build your own eval on actual frames from your game.
Two concrete findings worth acting on: the Qwen3-VL self-hosted size ladder is clean and monotone
(4B ~49-60 → 235B 62 on ScreenSpot-Pro), and there is a **28-point Instruct-vs-Thinking inversion**
on OSWorld for Qwen3-VL-235B (66.7 Instruct vs 38.1 Thinking) — so test both modes rather than
assuming thinking is better.

**There is no usable game-agent leaderboard.** Beyond Lumine's own custom suite (Collection / Combat /
NPC Interaction / Puzzle) the only game-adjacent artifact found is StepFun's **GEBench** dataset
(<https://huggingface.co/datasets/stepfun-ai/GEBench>) — the dataset exists but no numeric leaderboard
was retrievable. No comparable numbers exist for Cradle, Jarvis-1 or Voyager. **This means you cannot
select a VLM for your use case from public benchmarks alone; a local eval harness is mandatory.**

---

## 3. Top 3 picks

### Pick 1 — `qwen3-vl-flash`, mainland endpoint, as the high-frequency default
**`https://dashscope.aliyuncs.com/compatible-mode/v1`** → $0.022 in / $0.215 out per 1M.
Why: a 640x360 screenshot is only ~227 image tokens, so a 1-2 Hz loop with a ~2K-token prompt and a
~200-token JSON reply costs roughly **$0.0001 per call, ~$0.3/hour**. It is the cheapest genuinely
capable GUI/spatial model in this report, it is the direct descendant of the Qwen2-VL that Lumine
actually trained on, and it ships `-instruct` and `-thinking` variants so you can run thinking-off for
routine ticks and thinking-on only when your adaptive-reasoning trigger fires.
Tradeoff: the id is **legacy** (still served, but not the strategic path), and **function calling is
not supported on the Singapore endpoint** for this model — use Beijing or move to `qwen3.8-flash`.
If you want the non-legacy version of the same idea, use **`qwen3.8-flash` at $0.15/$0.47 (intl)**.

### Pick 2 — `deepseek-flash` as the cost/latency-balanced all-rounder
**`https://api.deepseek.com`** → $0.15 in / $0.60 out off-peak.
Why: it is the only provider here whose vision pricing is explicitly **half-rate for most of the day**
(off-peak is everything except 01:00-04:00 and 06:00-10:00 UTC on weekdays), images are hard-capped at
**1024 tokens each** so your per-frame cost is fully predictable regardless of resolution, and it
supports JSON output + tool calls + the Responses API on one id. Same base_url also speaks the
Anthropic protocol, so you can A/B two SDKs without changing providers.
Tradeoff: no published GUI-grounding benchmark at all, and DeepSeek has historically been a
reasoning-first lab rather than a grounding-first one — **prototype it against your actual game
before trusting it**, and keep Qwen-VL as the fallback.

### Pick 3 — `glm-5.3-flash` (or `glm-4.6v-flashx`) for the lowest measured latency
**`https://api.z.ai/api/paas/v4`** → $0.15/$0.50 for `glm-5.3-flash`; `glm-5.3-flashx` $0.37/$1.25
(the only model in this report with a vendor-published **200 tokens/s**), and `glm-4.6v-flashx` at
**$0.04/$0.40** is the cheapest paid tier anywhere in this table.
Why: GLM-5.3-Flash is natively multimodal with a 1M context and is explicitly marketed for GUI/CUA/BUA
agent loops, taking screenshots as tool parameters without a text round-trip.
Tradeoff: no numeric ScreenSpot-Pro/OSWorld figures published, so grounding quality is unproven
relative to Qwen. `glm-4.6v-flash` is **free**, which makes it the obvious zero-cost A/B baseline.

### What I would actually do
Run a **two-tier** setup, mirroring Lumine's hybrid-thinking design:
1. **Every tick (1-2 Hz):** a cheap flash model with thinking **off** and `reasoning_effort` minimal,
   asking for a compact JSON action. Candidates in cost order: `qwen3-vl-flash` (Beijing, $0.022/$0.215
   — but legacy id, no published grounding number), `qwen3.8-flash` ($0.15/$0.47 — current id,
   AndroidWorld 84.5 [A]), or `deepseek-flash` ($0.15/$0.60 off-peak, images hard-capped at 1024 tokens).
2. **Adaptive-reasoning tick (every N seconds, or on a "stuck" trigger):** escalate to
   `qwen3.8-max` (OSWorld-Verified 86.1 [A], AndroidWorld 85.3 [A]), `qwen3.7-plus` (SS-Pro 79.0,
   OSWorld-Verified 73.3 [A]), `glm-5.3-flashx`, or `step-3.7-flash` at `reasoning_effort: high` to
   set the subgoal and re-plan. This is exactly the pattern Lumine reports (25.3x latency reduction
   from not thinking every step).
3. **Fallback:** `gemini-3.5-flash-lite` or `claude-haiku-5-5` if you need a non-Chinese vendor for
   compliance, and for offline development run `Step-GUI-4B` (GUI specialist, SS-Pro 60.0) rather than
   a generic 4B — or `qwen3-vl:4b` if you want the plain model.

**Do not assume "thinking" beats "not thinking" for grounding.** Qwen3-VL-235B scores 66.7 on OSWorld
in Instruct mode vs 38.1 in Thinking mode [A] — a 28-point inversion. Benchmark both modes on your
own frames before committing.

---

## 4. Practical gotchas for this specific architecture

1. **Image size is the wrong lever.** 640x360 is already tiny (~227 Qwen tokens, ~169 StepFun tokens,
   capped at 1024 DeepSeek tokens). Do not downscale further — GUI grounding degrades on small text
   and small buttons, and you are not saving meaningful money. **The exception is Doubao/Ark**, where
   Seed 2.0+ bills a **fixed floor of 1280 tokens/image** in default `high` mode — there, sending a
   smaller image genuinely does not help, and `detail: low` is the only lever.
2. **Kimi cannot take a URL.** If your pipeline uploads the frame to a bucket and passes a link,
   Kimi will reject it. Base64 or its Files API only.
3. **StepFun http/https image URLs must be reachable from mainland China.** A frame served from a
   non-CN bucket will add real first-token latency or fail.
4. **Don't send absolute-click-only actions to a 3D game.** Lumine's paper explicitly calls out that
   GUI agents' "teleport the cursor then click" abstraction fails when the mouse drives a camera;
   you need relative mouse deltas and explicit key down/up/hold events. Whatever VLM you pick, make
   sure its JSON schema can express relative movement and key state, not just `click(x, y)`.
5. **Structured output support varies.** Qwen3-VL supports structured outputs on both regions;
   `qwen3-vl-plus` supports function calling in Beijing but not Singapore; Kimi supports JSON Mode and
   Partial Mode; StepFun supports JSON Mode and JSON Schema. Validate against the specific id+region.
6. **Cache your prompt prefix.** Every provider here discounts cached input heavily
   (DeepSeek $0.003 vs $0.15; Qwen explicit cache read 10% of input; Kimi $0.30 vs $3.00;
   Anthropic/Zhipu similar). With a long fixed system prompt describing the game and the action
   schema, this is the single biggest cost lever after model choice.

---

## 5. Known gaps / unverified items

- No published numeric TTFT or tokens/s for any Qwen VL model; Alibaba's "Prime / fast mode" (1.5-2x TPS)
  does **not** cover any vision-understanding model. The only vendor-published throughput figure in
  this whole report is z.ai's **200 tokens/s** for `glm-5.3-flashx`.
- Doubao/Ark: **no official USD price list exists (CNY only)**, so any USD figure for Doubao is a
  conversion, not a list price. The exact `ep-...` endpoint-id string format is unverified (the
  mechanism is documented, but no literal example is printed).
- **StepFun GUI numbers belong to Step-GUI-4B/8B (open-weight), NOT to `step-5-preview` /
  `step-3.7-flash`.** No grounding benchmark is published for the API-served StepFun models.
- **OpenAI / Anthropic / Google: every price and model id in those rows is third-party sourced
  (models.dev).** `platform.openai.com`, `developers.openai.com`, `openai.com`, `ai.google.dev`,
  `cloud.google.com` all returned HTTP 403 / connection failure, and `platform.claude.com`
  cross-origin-redirects — so these could not be first-party verified. Re-check before committing.
- **GUI grounding — no number found at all for:** GLM-4.5V, GLM-4.6V-Flash, all `doubao-seed-*`,
  `step-5-preview`, `step-3.7-flash`, `step-1o-turbo-vision`, DeepSeek-V4.1-Flash, Gemini 2.5 Flash,
  Kimi K3. GLM-4.6V has only a secondary-cited OSWorld 37.2%. Step3-VL-10B's benchmark table was not
  retrievable.
- **Benchmarks that are effectively dead for current models:** ScreenSpot-v2, VisualWebArena,
  GUI-Odyssey, and WindowsAgentArena (its official page still shows only the 2024 result).
- Every grounding number tagged `[A]` traces to a self-report on an aggregator that marks it
  **Verified = 0**. Treat the whole `[A]` column as a ranking hint, not a measurement.
- Local open-weight licenses: unconfirmed (`huggingface.co` unreachable). `Qwen2.5-VL-3B-Instruct`
  in particular is publicly discussed as **non-commercial**, so verify before shipping.
- Lumine's input resolution is not stated anywhere in the paper (abstract/intro/model sections) — `n/f`.

