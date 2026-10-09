# Alibaba Cloud Model Studio (DashScope/Bailian) — VL / vision model API pricing

Research date: 2026-10 (live web, pages stamped "Last Updated: Sep 28, 2026" for detail pages; aggregate pricing page stamped "Oct 08, 2026").

All prices USD per 1M tokens unless noted. Tiers are input-length tiers per single request.

| model id | region | tier | input USD/1M | output USD/1M | cache/batch notes |
|---|---|---|---|---|---|
| qwen3-vl-plus | Singapore (International) | 0<Token<=32K | 0.2 | 1.6 | implicit 0.04; explicit cache create 0.25; explicit cache read 0.02 |
| qwen3-vl-plus | Singapore (International) | 32K<Token<=128K | 0.3 | 2.4 | implicit 0.06; explicit create 0.375; explicit read 0.03 |
| qwen3-vl-plus | Singapore (International) | 128K<Token<=256K | 0.6 | 4.8 | implicit 0.12; explicit create 0.75; explicit read 0.06 |
| qwen3-vl-plus | China (Beijing) | 0<Token<=32K | 0.144 (detail page) / 0.143 (pricing page) | 1.434 | implicit 0.029; batch file in 0.072 / out 0.717; explicit create 0.18; explicit read 0.014; batch chat = same as standard |
| qwen3-vl-plus | China (Beijing) | 32K<Token<=128K | 0.216 (detail) / 0.215 (pricing) | 2.151 / 2.15 | implicit 0.044; batch file in 0.108 / out 1.075; explicit create 0.27; explicit read 0.022 |
| qwen3-vl-plus | China (Beijing) | 128K<Token<=256K | 0.431 (detail) / 0.43 (pricing) | 4.301 | implicit 0.087; batch file in 0.215 / out 2.15; explicit create 0.539; explicit read 0.043 |
| qwen3-vl-flash | Singapore (International) | 0<Token<=32K | 0.05 | 0.4 | implicit 0.01; explicit create 0.0625; explicit read 0.005 |
| qwen3-vl-flash | Singapore (International) | 32K<Token<=128K | 0.075 | 0.6 | implicit 0.015; explicit create 0.09375; explicit read 0.0075 |
| qwen3-vl-flash | Singapore (International) | 128K<Token<=256K | 0.12 | 0.96 | implicit 0.024; explicit create 0.15; explicit read 0.012 |
| qwen3-vl-flash | China (Beijing) | 0<Token<=32K | 0.022 | 0.215 | implicit 0.005; batch file in 0.011 / out 0.108; explicit create 0.028; explicit read 0.002; batch chat = same |
| qwen3-vl-flash | China (Beijing) | 32K<Token<=128K | 0.043 | 0.43 | implicit 0.009; batch file in 0.022 / out 0.215; explicit create 0.054; explicit read 0.004 |
| qwen3-vl-flash | China (Beijing) | 128K<Token<=256K | 0.086 | 0.859 | implicit 0.018; batch file in 0.043 / out 0.43; explicit create 0.108; explicit read 0.009 |
| qwen-vl-max | Singapore (International) | flat (no tier) | 0.8 | 3.2 | implicit cache 0.16; no batch (Batch Inference unsupported in SG) |
| qwen-vl-max | China (Beijing) | flat (no tier) | 0.229 (detail) / 0.23 (pricing) | 0.573 (detail) / 0.574 (pricing) | implicit 0.046; batch file in 0.115 / out 0.287 |
| qwen-vl-ocr | Singapore (International) | flat (no tier) | 0.07 | 0.16 | no cache/batch listed |
| qwen-vl-ocr | China (Beijing) | flat (no tier) | 0.043 | 0.072 | batch file in 0.022 / out 0.036 |
| qwen-vl-ocr-2025-08-28 / -2025-04-13 / -2024-10-28 (early versions) | China (Beijing) | flat | 0.717 | 0.717 | legacy |
| qwen-vl-plus | Singapore (International) | flat (no tier) | 0.21 | 0.63 | implicit cache 0.042 |
| qwen-vl-plus | China (Beijing) | flat (no tier) | 0.115 | 0.287 | implicit 0.023; batch file in 0.057 / out 0.143 |
| qwen3.5-ocr | China (Beijing) | flat (no tier) | 0.069 | 0.275 | no cache/batch listed |
| qwen3.5-ocr | Singapore (International) | — | unverified / not found | unverified / not found | no Singapore section on the model detail page; absent from SG pricing table |
| gui-plus | China (Beijing) | flat (no tier) | 1.5 CNY (USD conversion unverified) | 4.5 CNY (USD conversion unverified) | price quoted in CNY on the page ("Price (CNY) Per 1M tokens") |
| gui-plus | Singapore (International) | — | unverified / not found | unverified / not found | absent from all English/international pricing tables |
| qvq-max | Singapore (International) | flat (no tier) | 1.2 | 4.8 | no cache/batch |
| qvq-max | China (Beijing) | flat (no tier) | 1.147 | 4.588 | no cache/batch |
| qvq-plus | China (Beijing) | flat (no tier) | 0.287 | 0.717 | no cache/batch |
| qvq-plus | Singapore (International) | — | unverified / not found | unverified / not found | model detail page shows China (Beijing) only |
| qwen3-vl-235b-a22b-instruct | Singapore (International) | flat (no tier) | 0.4 | 1.6 | non-thinking only; no cache/batch |
| qwen3-vl-235b-a22b-instruct | China (Beijing) | flat (no tier) | 0.287 | 1.147 | no cache/batch |
| qwen3-vl-235b-a22b-thinking | Singapore (International) | flat (no tier) | 0.4 | 4 | thinking only; no cache/batch |
| qwen3-vl-235b-a22b-thinking | China (Beijing) | flat (no tier) | 0.287 | 2.867 | no cache/batch |
| qwen3-vl-30b-a3b-instruct | Singapore (International) | flat (no tier) | 0.2 | 0.8 | no cache/batch |
| qwen3-vl-30b-a3b-instruct | China (Beijing) | flat (no tier) | 0.108 | 0.43 (detail) / 0.431 (pricing) | no cache/batch |
| qwen3-vl-30b-a3b-thinking | Singapore (International) | flat (no tier) | 0.2 | 2.4 | no cache/batch |
| qwen3-vl-30b-a3b-thinking | China (Beijing) | flat (no tier) | 0.108 | 1.075 (detail) / 1.076 (pricing) | no cache/batch |
| qwen3-vl-32b-instruct | Singapore (International) | flat (no tier) | 0.16 | 0.64 | no cache/batch |
| qwen3-vl-32b-instruct | China (Beijing) | flat (no tier) | 0.287 | 1.147 | no cache/batch |
| qwen3-vl-32b-thinking | Singapore (International) | flat (no tier) | 0.16 | 0.64 | thinking only; no cache/batch |
| qwen3-vl-32b-thinking | China (Beijing) | flat (no tier) | 0.287 | 2.868 | no cache/batch |
| qwen3-vl-8b-instruct | Singapore (International) | flat (no tier) | 0.18 | 0.7 | no cache/batch |
| qwen3-vl-8b-instruct | China (Beijing) | flat (no tier) | 0.072 | 0.287 | no cache/batch |
| qwen3-vl-8b-thinking | Singapore (International) | flat (no tier) | 0.18 | 2.1 | no cache/batch |
| qwen3-vl-8b-thinking | China (Beijing) | flat (no tier) | 0.072 | 0.717 | no cache/batch |

## Newer VL flagship?

**None found.** The official Vision model list (Last Updated: Sep 23, 2026) contains exactly 16 entries — qwen3-vl-flash, qwen3-vl-plus, qwen-vl-max, qwen-vl-ocr, qwen-vl-plus, qwen3.5-ocr, qvq-max, qvq-plus, qwen3-vl-235b-a22b-instruct, qwen3-vl-235b-a22b-thinking, qwen3-vl-30b-a3b-instruct, qwen3-vl-30b-a3b-thinking, qwen3-vl-32b-instruct, qwen3-vl-32b-thinking, qwen3-vl-8b-instruct, qwen3-vl-8b-thinking. No `qwen3.5-vl`, `qwen3.6-vl`, `qwen3.7-vl`, `qwen3.8-vl`, or `qwen3.8-vl-plus` id appears anywhere in the official Vision list or in either language's pricing page.

Closest newer vision-capable flagship (NOT a "VL" id): **qwen3.7-plus** — detail page lists Input Modality Text/Image/Video.
- Singapore (International): 0<Token<=256K $0.4 in / $1.6 out; 256K<Token<=1M $1.2 / $4.8 (implicit cache 0.08 / 0.24)
- China (Beijing): 0<Token<=256K $0.276 / $1.101; 256K<Token<=1M $0.826 / $3.301 (implicit cache 0.056 / 0.166; batch file in 0.143 / out 0.574)

Also present on the pricing page but outside the asked list: `qwen3-vl-embedding`, `qwen3-vl-rerank`, and legacy `qwen2-vl-72b-instruct` (CN $2.294/$6.881), `qwen2-vl-7b-instruct` (limited-time free), `qwen2-vl-2b-instruct`.

## Source URL → coverage map

- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-plus — qwen3-vl-plus (SG/CN/DE/US/HK, 3 tiers, cache+batch)
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-flash — qwen3-vl-flash (SG/CN/DE/US, 3 tiers, cache+batch)
- https://www.alibabacloud.com/help/en/model-studio/qwen-vl-max — qwen-vl-max (CN + SG, implicit cache, batch file)
- https://www.alibabacloud.com/help/en/model-studio/qwen-vl-plus — qwen-vl-plus (CN + SG)
- https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr — landing/how-to page; pricing section not rendered (truncated)
- https://www.alibabacloud.com/help/en/model-studio/qwenvl-ocr — qwen-vl-ocr + all its snapshots (CN + SG)
- https://www.alibabacloud.com/help/en/model-studio/qwen3-5-ocr — qwen3.5-ocr (CN only, $0.069/$0.275)
- https://www.alibabacloud.com/help/en/model-studio/qvq-max — qvq-max (CN + SG)
- https://www.alibabacloud.com/help/en/model-studio/qvq-plus — qvq-plus (CN only)
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-235b-a22b-instruct
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-235b-a22b-thinking
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-30b-a3b-instruct
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-30b-a3b-thinking
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-32b-instruct
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-32b-thinking
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-8b-instruct
- https://www.alibabacloud.com/help/en/model-studio/qwen3-vl-8b-thinking
- https://www.alibabacloud.com/help/en/model-studio/model-list-visual-understanding/ — authoritative list of current VL model ids (Sep 23, 2026)
- https://www.alibabacloud.com/help/en/model-studio/model-pricing — QVQ / Qwen-VL / Qwen-OCR / open-source Qwen-VL sections (fetched with curl; web_fetch truncates before the vision section)
- https://www.alibabacloud.com/help/zh/model-studio/model-pricing — Chinese pricing page (cross-check; same USD figures)
- https://help.aliyun.com/en/model-studio/gui-plus and https://help.aliyun.com/zh/model-studio/gui-plus — gui-plus pricing (CNY, China Beijing only)
- https://help.aliyun.com/zh/model-studio/gui-automation — GUI Plus model intro + price table (CNY)
- https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus — qwen3.7-plus (vision-capable, non-VL id)

## Unverified / not found

- `qwen3.5-vl`, `qwen3.6-vl`, `qwen3.7-vl`, `qwen3.8-vl`, `qwen3.8-vl-plus` — **unverified / not found** (not in the official Vision list nor either pricing page).
- `gui-plus` — USD price **unverified**; only CNY 1.5 / 4.5 per 1M tokens, China (Beijing). No Singapore listing.
- `qwen3.5-ocr` Singapore — **unverified / not found**.
- `qvq-plus` Singapore — **unverified / not found**.
- https://www.qianwenai.com/pricing/api — HTTP 404.
- https://platform.qianwenai.com/pricing/api — loads prices client-side ("正在加载模型价格…"); no extractable data. **unverified.**
- https://docs.modelstudio.console.alibabacloud.com/en/model-studio/model-pricing — JS app shell; no pricing tables in fetched HTML. **unverified.**
- https://www.alibabacloud.com/help/en/model-studio/gui-plus and /gui-automation — HTTP 404 on the international English domain.
- https://help.aliyun.com/zh/model-studio/vision-model-list — returned a page with no VL model ids; the authoritative list used was the English Vision page above.

## Cross-page discrepancies (both figures are on official Alibaba pages)

Detail pages round to 3 decimals; the aggregate pricing page rounds some values. Affected: qwen3-vl-plus CN (0.144 vs 0.143; 0.216 vs 0.215; 0.431 vs 0.43), qwen-vl-max CN (0.229 vs 0.23; 0.573 vs 0.574), qwen3-vl-30b-a3b-instruct CN output (0.43 vs 0.431), qwen3-vl-30b-a3b-thinking CN output (1.075 vs 1.076). All values above are quoted exactly as printed on each page.
