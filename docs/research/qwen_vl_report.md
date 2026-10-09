# Qwen-VL / vision-language models on Alibaba Cloud Model Studio (DashScope / Bailian 百炼)

Verified live on the web, October 2026. Every figure below is read off the cited page.
Anything not found is marked **UNVERIFIED**.

---

## 0. Headline correction to the premise

**There is no `qwen3.8-vl`, `qwen3.7-vl`, `qwen3.6-vl`, or `qwen3.5-vl`.** No such id appears in
the official Vision model list or in either language's pricing page.

The dedicated **Qwen3-VL** ids did **not** advance past `qwen3-vl-plus`. They are still served,
but the docs have reclassified the whole `qwen3-vl-*` family as **legacy (旧版)**.

What actually happened: vision moved **into the general text-family flagship**. The current
vision-capable flagships are `qwen3.8-max` / `qwen3.7-plus` / `qwen3.8-flash` — natively
multimodal (`Text + Image + Video` in, `Text` out), not "VL"-suffixed.

> Doc inconsistency worth knowing: as of the Sep 28, 2026 revision the **English** visual-understanding
> page still says *"Start with `qwen3.7-plus`, the flagship Qwen model"*, while the **Chinese** page
> says *"推荐从 `qwen3.8-max` 开始"*. The per-model page confirms `qwen3.8-max` is the flagship.

---

## 1. Exact API model ids (verbatim)

### 1a. Current recommended vision-capable models (the real "current current" VL route)

| model id (verbatim) | context | max px/image | max video | max images | max videos | function calling | built-in tools | structured output |
|---|---|---|---|---|---|---|---|---|
| `qwen3.8-max` | 1M | 16M | 2 hours / 2 GB | 2048 | 64 | Supported | Supported | Supported |
| `qwen3.8-max-0902` (alias `qwen3.8-max-2026-09-02`) | 1M | 16M | 2 hours / 2 GB | 2048 | 64 | Supported | Supported | Supported |
| `qwen3.8-flash` | 1M | 16M | 2 hours / 2 GB | 2048 | 64 | Supported | Supported | Supported |
| `qwen3.8-omni-flash` | 1M | 16M | 2 hours / 2 GB | 2048 | 64 | Supported | `web_search` | JSON Object |
| `qwen3.7-plus` | 1M | 16M | 2 hours / 2 GB | 2048 | 64 | Supported | Supported | Supported |
| `qwen3.7-flash` | 1M | 16M | 2 hours / 2 GB | 256 | 64 | Supported | Supported | Supported |
| `qwen3.5-omni-plus` | 256k | — | 1 hour / 2 GB | 2048 | 512 | Supported | — | JSON Object |

Other vision-capable ids listed in the docs: `qwen3.7-max-2026-06-08`, `qwen3.7-plus-2026-05-26`,
`qwen3.7-flash-2026-07-15`, `qwen3.6-plus`, `qwen3.6-plus-2026-04-02`, `qwen3.6-flash`,
`qwen3.6-flash-2026-04-16`, `qwen3.6-35b-a3b`, `qwen3.5-plus`, `qwen3.5-plus-2026-02-15`,
`qwen3.5-flash`, `qwen3.5-flash-2026-02-23`, `qwen3.5-397b-a17b`, `qwen3.5-122b-a10b`,
`qwen3.5-27b`, `qwen3.5-35b-a3b`.

### 1b. Legacy — but STILL SERVED — dedicated VL / OCR / QVQ / GUI ids

| model id (verbatim) | kind | notes |
|---|---|---|
| `qwen3-vl-plus` | VL flagship (legacy) | snapshot-equiv `qwen3-vl-plus-2025-12-19`; older snapshots `-2025-09-23` |
| `qwen3-vl-flash` | VL small/fast (legacy) | snapshot-equiv `qwen3-vl-flash-2026-01-22`; older `-2025-10-15` |
| `qwen-vl-max` | legacy VL | snapshot-equiv `qwen-vl-max-2025-08-13` |
| `qwen-vl-plus` | legacy VL | snapshot-equiv `qwen-vl-plus-2025-08-15` |
| `qwen-vl-ocr` | OCR | snapshot-equiv `qwen-vl-ocr-2025-11-20`; also `qwen-vl-ocr-latest`, `-2025-04-13`, `-1028` |
| `qwen3.5-ocr` | OCR | Beijing only in the tables I could read |
| `qvq-max` | visual reasoning | |
| `qvq-plus` | visual reasoning | Beijing only |
| `gui-plus` | GUI agent / screen understanding | **Beijing only**; snapshot `gui-plus-2026-02-26` |
| `qwen3-vl-embedding` | VL embedding | not a chat model |
| `qwen3-vl-rerank` | VL rerank | not a chat model |

### 1c. Open-weight Qwen3-VL sizes also served on Model Studio

`qwen3-vl-235b-a22b-instruct`, `qwen3-vl-235b-a22b-thinking`, `qwen3-vl-30b-a3b-instruct`,
`qwen3-vl-30b-a3b-thinking`, `qwen3-vl-32b-instruct`, `qwen3-vl-32b-thinking`,
`qwen3-vl-8b-instruct`, `qwen3-vl-8b-thinking`.

> **Not found / UNVERIFIED:** `qwen3-vl-max`, `qwen3-vl-flash-us` (the US scope is a *deployment
> scope* of `qwen3-vl-flash`, not a separate id), `qwen3.8-vl`, `qwen3.7-vl`, `qwen3.6-vl`, `qwen3.5-vl`.

---

## 2. Image / video input mechanisms and token accounting

### Input mechanisms (all documented on Model Studio)

| method | OpenAI-compatible form | Native DashScope form |
|---|---|---|
| Public HTTPS URL | `{"type":"image_url","image_url":{"url":"https://..."}}` | `{"image":"https://..."}` |
| Base64 data URL | `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` | `{"image":"data:image/png;base64,..."}` |
| Local file | **not** available over HTTP/OpenAI-compat | `{"image":"file:///path/to/x.jpg"}` — DashScope Python/Java SDK only |
| Video file | `{"type":"video_url","video_url":{"url":"..."},"fps":2}` | `{"video":"https://...","fps":2}` |
| Video as frames | `{"type":"video","video":["url1","url2",...],"fps":2}` | frame list |

Native endpoint path: `POST {base}/api/v1/services/aigc/multimodal-generation/generation`.

> **OSS note:** the old `getPolicy`/upload OSS flow no longer appears in the current
> "File input methods" documentation. Documented methods are Public URL / Base64 / Local file path.

### Documented token accounting (verbatim)

- English camera-ready formula: **`h x w / (32 x 32) + 2`**
- Doc prose form: `Image Tokens = h_bar * w_bar / token_pixels + 2`
- Code form: `token = int(h_bar * w_bar / (32 * 32)) + 2`

| quantity | value |
|---|---|
| pixels per token | 32×32 for Qwen3-VL / Qwen3.5–3.8 / `qwen-vl-max` / `qwen-vl-plus`; **28×28** for QVQ and other Qwen2.5-VL |
| max tokens per image | 16,384 |
| max pixels per image | 16,777,216 (16M) = 16384 × 32 × 32 |
| default `max_pixels` | 2,621,440 (Qwen3-VL et al.); 1,310,720 (`qwen-vl-max`/`plus`); 1,003,520 (28×28 models) |
| min resolution | 10×10 px; aspect ratio ≤ 200:1 |
| recommended max | ≤ 8K (7680×4320); 4K–8K only JPEG/JPG/PNG |
| `vl_high_resolution_images=true` | overrides `max_pixels`, fixes image at 16,384 tokens |

**Video:** `video_token = ceil(num_frames / 2) * h/32 * w/32 + 2`. Default `fps` = 2.0, range `[0.1, 10]`;
`FPS_MIN_FRAMES = 4`, `FPS_MAX_FRAMES = 2000`. `VIDEO_TOTAL_PIXELS` = `131072*32*32` (Plus) else `65536*32*32`.

**Size limits:** image URL ≤ 20 MB (Qwen3-VL) / ≤ 10 MB (others); base64 source and data URI ≤ 20 MB;
video URL ≤ 2 GB; base64 video < 10 MB; local path ≤ 100 MB; ≤ 64 videos; ≤ 256 images (Qwen3-VL) via
URL/local, ≤ 250 via base64.

---

## 3. OpenAI-compatible endpoint — exact base URLs

**Yes, it is offered, and the older non-workspace forms are still valid.** The Base URL overview page
states verbatim: *"DashScope domain (`dashscope.aliyuncs.com`): Legacy shared domain. **Still available**;
migration to a workspace-dedicated domain is recommended."*

| region | OpenAI-compatible | Anthropic-compatible | native |
|---|---|---|---|
| **Singapore / international** | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `https://dashscope-intl.aliyuncs.com/apps/anthropic` | `/api/v1` |
| **Chinese mainland (Beijing)** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `https://dashscope.aliyuncs.com/apps/anthropic` | `/api/v1` |
| US (Virginia) | `https://dashscope-us.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `/api/v1` |
| China (Hong Kong) | `https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1` | `.../apps/anthropic` | `/api/v1` |

**Workspace-dedicated (recommended for production):**

| region | OpenAI-compatible |
|---|---|
| Singapore | `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| Chinese mainland (Beijing) | `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| Japan (Tokyo) | `https://{WorkspaceId}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| Germany (Frankfurt) | `https://{WorkspaceId}.eu-central-1.maas.aliyuncs.com/compatible-mode/v1` |
| US (Virginia) | `https://{WorkspaceId}.us-east-1.maas.aliyuncs.com/compatible-mode/v1` |
| China (Hong Kong) | `https://{WorkspaceId}.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1` |

Trial: `https://trial.{region}.maas.aliyuncs.com/compatible-mode/v1`.
`dashscope-*` and `{WorkspaceId}.*` are **different hosts** — Singapore and mainland URLs explicitly differ.
API keys are region-specific; a key/domain mismatch returns 401. The current API docs' own code samples use
the workspace form, but the legacy `dashscope-intl` / `dashscope` forms remain documented and working, and
the official Qwen3.7-Plus blog still ships them as the default in its sample code.

---

## 4. Current pricing (USD per 1M tokens)

Only `qwen3-vl-plus` and `qwen3-vl-flash` are tiered. Everything else is flat.

| model id | region | tier (input tokens) | input | output |
|---|---|---|---|---|
| `qwen3-vl-plus` | Singapore | ≤32K | 0.20 | 1.60 |
| `qwen3-vl-plus` | Singapore | 32K–128K | 0.30 | 2.40 |
| `qwen3-vl-plus` | Singapore | 128K–256K | 0.60 | 4.80 |
| `qwen3-vl-plus` | Beijing | ≤32K | 0.144 (agg. page: 0.143) | 1.434 |
| `qwen3-vl-plus` | Beijing | 32K–128K | 0.216 (0.215) | 2.151 |
| `qwen3-vl-plus` | Beijing | 128K–256K | 0.431 (0.43) | 4.301 |
| `qwen3-vl-flash` | Singapore | ≤32K | 0.05 | 0.40 |
| `qwen3-vl-flash` | Singapore | 32K–128K | 0.075 | 0.60 |
| `qwen3-vl-flash` | Singapore | 128K–256K | 0.12 | 0.96 |
| `qwen3-vl-flash` | Beijing | ≤32K | 0.022 | 0.215 |
| `qwen3-vl-flash` | Beijing | 32K–128K | 0.043 | 0.43 |
| `qwen3-vl-flash` | Beijing | 128K–256K | 0.086 | 0.859 |
| `qwen3.8-max` | Singapore | flat | 2.00 | 6.00 |
| `qwen3.8-max` | Beijing | flat | 1.65 | 4.951 |
| `qwen3.8-flash` | Singapore | flat | 0.15 | 0.47 |
| `qwen3.8-flash` | Beijing | flat | 0.113 | 0.382 |
| `qwen3.7-plus` | Singapore | ≤256K | 0.40 | 1.60 |
| `qwen3.7-plus` | Singapore | 256K–1M | 1.20 | 4.80 |
| `qwen3.7-plus` | Beijing | ≤256K | 0.276 | 1.101 |
| `qwen3.7-plus` | Beijing | 256K–1M | 0.826 | 3.301 |
| `qwen-vl-max` | Singapore | flat | 0.80 | 3.20 |
| `qwen-vl-max` | Beijing | flat | 0.229 (0.23) | 0.573 (0.574) |
| `qwen-vl-plus` | Singapore | flat | 0.21 | 0.63 |
| `qwen-vl-plus` | Beijing | flat | 0.115 | 0.287 |
| `qwen-vl-ocr` | Singapore | flat | 0.07 | 0.16 |
| `qwen-vl-ocr` | Beijing | flat | 0.043 | 0.072 |
| `qwen3.5-ocr` | Beijing | flat | 0.069 | 0.275 |
| `qvq-max` | Singapore | flat | 1.20 | 4.80 |
| `qvq-max` | Beijing | flat | 1.147 | 4.588 |
| `qvq-plus` | Beijing | flat | 0.287 | 0.717 |
| `gui-plus` | Beijing | flat | 1.5 **CNY** (USD conv. UNVERIFIED) | 4.5 **CNY** |
| `qwen3-vl-8b-instruct` | Singapore / Beijing | flat | 0.18 / 0.072 | 0.70 / 0.287 |
| `qwen3-vl-8b-thinking` | Singapore / Beijing | flat | 0.18 / 0.072 | 2.10 / 0.717 |
| `qwen3-vl-30b-a3b-instruct` | Singapore / Beijing | flat | 0.20 / 0.108 | 0.80 / 0.43 |
| `qwen3-vl-30b-a3b-thinking` | Singapore / Beijing | flat | 0.20 / 0.108 | 2.40 / 1.075 |
| `qwen3-vl-32b-instruct` | Singapore / Beijing | flat | 0.16 / 0.287 | 0.64 / 1.147 |
| `qwen3-vl-32b-thinking` | Singapore / Beijing | flat | 0.16 / 0.287 | 0.64 / 2.868 |
| `qwen3-vl-235b-a22b-instruct` | Singapore / Beijing | flat | 0.40 / 0.287 | 1.60 / 1.147 |
| `qwen3-vl-235b-a22b-thinking` | Singapore / Beijing | flat | 0.40 / 0.287 | 4.00 / 2.867 |

**Discounts.** Implicit (automatic) cache ≈ 20% of input price. Explicit cache creation ≈ 125% of input,
explicit cache read ≈ 10% of input. **Batch:** Beijing only for most VL models — `qwen3-vl-plus` batch file
= 0.072/0.717 at the ≤32K tier (~50% off); `qwen3-vl-flash` batch file = 0.011/0.108. Singapore has no batch
inference for any `qwen3-vl-*` model (the capability table says `Batch Inference: Unsupported`).

**Region gaps:** `qvq-plus`, `qwen3.5-ocr`, `gui-plus` have **no Singapore listing found**. `gui-plus` is
CNY-only, and its English doc slug 404s on the international domain.

**Rounding caveat:** detail pages round to 3 decimals, the aggregate pricing page rounds differently.
Both figures are official; both are shown above where they differ.

> The aggregate pricing page **does** truncate under `web_fetch`. Fetching it with `curl` and parsing the
> HTML yields the complete Vision/QVQ/OCR/open-weight tables.

---

## 5. Latency

- **Yes, there is a "flash" tier:** `qwen3-vl-flash` (legacy dedicated VL), plus `qwen3.8-flash` and
  `qwen3.7-flash` in the current generation. `qwen3-vl-flash` is documented as delivering
  *"superior performance compared to the open-source Qwen3-VL-30B-A3B while maintaining fast response speeds"*.
  `qwen3-vl-plus-2025-12-19` is described as offering *"lower latency and faster response speeds"* than the
  `-2025-09-23` snapshot. `qwen-vl-ocr-2025-11-20` cites *"substantial reductions in end-to-end latency"*.
- **Fast mode (Prime)** raises TPS to **1.5–2× the standard API** — but its supported-model list is
  `glm-5.2-fast-preview` (text) and `wan3.0-video-prime` (video *generation*). **No vision-understanding
  model is in Prime's supported list.**
- **No numeric TTFT or tokens/sec figures are published for any VL model.** UNVERIFIED.
  As a proxy, published rate limits (RPM / TPM): `qwen3-vl-plus` Beijing 3,000 / 5,000,000 and
  Singapore 1,200 / 1,000,000; `qwen3-vl-flash` Beijing 3,000 / 5,000,000 and Singapore 1,200 / 1,000,000;
  `qwen3.8-max` and `qwen3.8-flash` use dynamic rate limiting (tiered by monthly spend) in Beijing and
  Singapore; `gui-plus` Beijing 80 RPM / 540,000 TPM.

---

## 6. GUI grounding / screen understanding / spatial reasoning

### `qwen3.7-plus` — official (Qwen/Alibaba Cloud blog, table image)

| benchmark | `qwen3.7-plus` | `qwen3.6-plus` |
|---|---|---|
| ScreenSpot Pro | **79.0** | 68.2 |
| OSWorld-Verified | **73.3** | 62.5 |
| AndroidWorld | **81.0** | 67.2 |
| QwenVision2Code | **1772.0** | 1522.0 |
| ClawEval-MM | **55.7** | 49.1 |
| MMMU-Pro | 79.0 | 78.8 |
| MathVision | 90.3 | 88.0 |
| BabyVision | 70.4 / 64.7 | 37.4 |
| CharXiv(RQ) | 85.9 / 84.4 | 80.4 |
| RealWorldQA | 86.9 | 85.4 |
| OmniDocBench 1.5 | 91.4 | 91.2 |
| OCR-Bench-V2 (EN / ZH) | 70.7 / 67.1 | 67.0 / 63.6 |
| VideoMME (w/ sub.) | 88.0 | 87.8 |
| VideoMMMU | 85.4 | 84.0 |
| MLVU (M-Avg) | 87.4 | 86.7 |

ScreenSpot Pro and OSWorld-Verified are reported with `enable_thinking=False`.

### `qwen3-vl-235b-a22b-instruct` — official (Qwen3-VL README table image, non-thinking)

| benchmark | score |
|---|---|
| ScreenSpot | **95.4** |
| ScreenSpot Pro | **62.0** |
| OSWorldG | **66.7** |
| AndroidWorld | **63.7** |
| RefCOCO (avg) | **91.9** |
| CountBench | 93.0 |
| ODinW13 | 48.6 |
| ARKitScenes (3D) | 56.9 |
| HyperSim (3D) | 13.0 |
| SUNRGBD (3D) | 39.4 |
| VSI-Bench (spatial) | 62.6 |
| EmbSpatialBench | 83.1 |
| RefSpatialBench | 65.5 |
| RoboSpatialHome | 69.5 |
| Design2Code | 92.0 |
| ChartMimic v2 Direct | 80.5 |
| UniSVG | 69.3 |

`OSWorldG` is the label used verbatim in the table; it is the GUI-grounding variant, **not** full OSWorld.
Thinking-mode tables and the 30B-A3B / 2B–32B tables exist as separate images on the same README.

### Qualitative official claims

- `qwen3-vl-plus` / `qwen3-vl-flash` model pages: *"achieving world-leading performance in visual agent
  capabilities on public benchmark datasets such as OS World"* — **no numbers given on the doc pages.**
- `qwen3-vl-flash`: *"Equipped with 2D/3D visual localization capabilities"*.
- `qwen3.7-plus`: *"perceive real-world scenes, read screens and interact with GUIs, generate code based on
  visual references, and perform end-to-end navigation within mobile apps."*
- `gui-plus`: purpose-built GUI model for phone and desktop *"图形界面理解与交互任务"*, cross-app multi-step
  planning, multi-agent collaboration. **No benchmark numbers published on its page.**
- Visual coding / grounding: HTML/CSS/JS from design screenshots, QwenVL HTML and QwenVL Markdown document
  parsing, 2D bbox/point output, 3D bbox output as
  `[{"bbox_3d": [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw], "label": "..."}]`.

> **UNVERIFIED (no official number found):** GUI-Odyssey, VisualWebArena / WebArena for Qwen VL models,
> RefCOCO/RefCOCO+/RefCOCOg per-split values beyond the `RefCOCO_avg` 91.9 above, and any GUI benchmark
> number for `qwen3.8-max`, `gui-plus`, or `qwen3-vl-plus` specifically. Where numbers exist for
> Qwen3.7-Plus and Qwen3-VL-235B-A22B they are official; blog **tables are published as images**, which is
> why they don't surface as text in search or plain fetches.

---

## 7. Model-list pages to re-check

| purpose | URL |
|---|---|
| **Vision model list (authoritative, updated Sep 23 2026)** | https://www.alibabacloud.com/help/en/model-studio/model-list-visual-understanding |
| Vision model list (ZH) | https://help.aliyun.com/zh/model-studio/model-list-visual-understanding/ |
| Recommended / latest models (updated Sep 28 2026) | https://www.alibabacloud.com/help/en/model-studio/models |
| Recommended / latest models (ZH) | https://help.aliyun.com/zh/model-studio/models |
| Visual understanding guide (updated Sep 28 2026) | https://www.alibabacloud.com/help/en/model-studio/vision-model |
| Image & video understanding how-to (updated Oct 08 2026) | https://www.alibabacloud.com/help/en/model-studio/vision |
| Model List index (updated Oct 01 2026) | https://www.alibabacloud.com/help/en/model-studio/model-studio-model-list/ |
| Pricing | https://www.alibabacloud.com/help/en/model-studio/model-pricing |
| Console model market (live availability) | https://modelstudio.console.alibabacloud.com/ap-southeast-1/model/market |

---

## Source URLs

**Base URLs / endpoints**
- https://www.alibabacloud.com/help/en/model-studio/base-url
- https://www.alibabacloud.com/blog/603206 (official sample code, legacy `dashscope-intl` / `dashscope` domains)
- https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md (official `dashscope.aliyuncs.com/compatible-mode/v1`)

**Model ids, capabilities, context limits, rate limits**
- https://www.alibabacloud.com/help/en/model-studio/model-list-visual-understanding
- https://www.alibabacloud.com/help/en/model-studio/vision-model
- https://www.alibabacloud.com/help/en/model-studio/models
- https://help.aliyun.com/zh/model-studio/vision-model
- https://help.aliyun.com/zh/model-studio/models
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-plus
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-flash
- https://www.alibabacloud.com/help/en/model-studio/qwen-vl-max
- https://www.alibabacloud.com/help/en/model-studio/qwen-vl-plus
- https://www.alibabacloud.com/help/en/model-studio/qwenvl-ocr
- https://www.alibabacloud.com/help/en/model-studio/qwen3-5-ocr
- https://help.aliyun.com/zh/model-studio/gui-plus
- https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max
- https://help.aliyun.com/zh/model-studio/qwen3-8-max
- https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus
- https://www.alibabacloud.com/help/en/model-studio/qwen3-8-flash

**Pricing**
- https://www.alibabacloud.com/help/en/model-studio/model-pricing  (curl only — `web_fetch` truncates before the Vision section)
- https://www.alibabacloud.com/help/zh/model-studio/model-pricing  (cross-check, same USD)
- per-model pages listed above; full 39-row matrix in `qwen_vl_pricing_research.md`

**Image/video input + token accounting**
- https://www.alibabacloud.com/help/en/model-studio/vision
- https://www.alibabacloud.com/help/en/model-studio/vision-model
- https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md (32× compression, processor pixel budgets)

**Benchmarks**
- https://www.alibabacloud.com/blog/603206  (Qwen3.7-Plus; table image `https://yqintl.alicdn.com/4bf3dfa024483ccbc02411ad9382c7544c3b0e21.png`)
- https://qwen.ai/blog?id=qwen3.7-plus
- https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/README.md  (tables: `https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/table_nothinking_vl.jpg`, `.../table_thinking_vl_.jpg`)
- https://arxiv.org/pdf/2511.21631  (Qwen3-VL Technical Report)
- https://github.com/QwenLM/Qwen3-VL

**Latency**
- https://www.alibabacloud.com/help/en/model-studio/prime-mode
- rate limits on each per-model page above
