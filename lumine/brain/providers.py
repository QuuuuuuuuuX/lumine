"""Provider registry: verified endpoints, model ids, and quirks for VLM backends.

Everything in :data:`PROVIDERS` was checked against provider documentation; entries marked
``verified=False`` carry a best-known value that must be confirmed before being trusted.
The registry exists so that swapping the agent's brain is a config change, not a code
change::

    cfg.brain.provider = "ollama"     # -> http://localhost:11434/v1
    cfg.brain.model = "qwen3-vl:8b-instruct-q4_K_M"

Notes that bite in practice
---------------------------
* **DeepSeek does have vision now**, on ``deepseek-flash`` only (``deepseek-v4-pro`` is
  text-only and will reject image parts).  It is the cheapest credible option here and it
  is what this repo measures against.
* **Moonshot Kimi refuses public image URLs** — base64 data URLs or uploaded file ids only.
* **Ollama** is OpenAI-compatible at ``/v1`` and ignores the API key, but the OpenAI SDK
  insists on a non-empty one, so we always send the placeholder ``"ollama"``.
* Image-bearing messages are only legal in the ``user`` role on DeepSeek.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    base_url: str
    default_model: str
    api_key_env: str
    vision: bool = True
    openai_compatible: bool = True
    #: Providers that cannot fetch a remote URL and need inline base64.
    requires_base64: bool = False
    #: Extra keys merged into the chat-completions request body.
    extra_body: dict = field(default_factory=dict)
    #: Extra options merged into the image_url part.
    image_detail: str | None = None
    #: Where a human should go to get a key.
    console: str = ""
    notes: str = ""
    verified: bool = True
    #: USD per 1M tokens (input, output) when known.
    price_in: float | None = None
    price_out: float | None = None


_MEDIA_URL = "https://platform.moonshot.cn/console/api-keys"

PROVIDERS: dict[str, ProviderSpec] = {
    # ------------------------------------------------------------------------------ #
    # Verified working on this machine, with the harness's own credential.
    # ------------------------------------------------------------------------------ #
    "deepseek": ProviderSpec(
        key="deepseek",
        label="DeepSeek (deepseek-flash)",
        base_url="https://api.deepseek.com",
        default_model="deepseek-flash",
        api_key_env="DEEPSEEK_API_KEY",
        console="https://platform.deepseek.com/api_keys",
        image_detail="low",
        price_in=0.15,
        price_out=0.60,
        notes=(
            "Vision lives on 'deepseek-flash' ONLY; 'deepseek-v4-pro' rejects image parts. "
            "Images may be a base64 data URL, a public https URL, or an uploaded file_id, and "
            "are capped at 1024 tokens each, which makes per-frame cost predictable. It is a "
            "reasoning model, so max_tokens must leave headroom or content comes back empty "
            "and everything lands in reasoning_content. Measured on this box: 1.4-2.0 s per "
            "640x360 call at reasoning_effort='low'. Off-peak pricing; doubles at peak."
        ),
    ),
    # ------------------------------------------------------------------------------ #
    # China-hosted VLMs with strong GUI/screen grounding.
    # ------------------------------------------------------------------------------ #
    "zai": ProviderSpec(
        key="zai",
        label="Zhipu / Z.AI",
        base_url="https://api.z.ai/api/paas/v4",
        default_model="glm-5.3-flash",
        api_key_env="ZAI_API_KEY",
        console="https://z.ai/manage-apikey/apikey-list",
        price_in=0.15,
        price_out=0.50,
        notes="Native multimodal, 1M context. 'glm-4.6v-flashx' is the cheap-and-fast tier "
              "($0.04/$0.40) and 'glm-4.6v-flash' is free -- a zero-cost A/B baseline. "
              "'glm-5.3-flashx' is the only model in this table with a vendor-published "
              "throughput figure (~200 tok/s).",
    ),
    "kimi": ProviderSpec(
        key="kimi",
        label="Moonshot Kimi",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-k3",
        api_key_env="MOONSHOT_API_KEY",
        requires_base64=True,
        console="https://platform.moonshot.ai/console/api-keys",
        price_in=3.00,
        price_out=15.00,
        notes="Public image URLs are NOT fetched. Base64 or an uploaded file id only. "
              "'kimi-k2.6' ($0.95/$4.00) is the cheaper sibling.",
    ),
    "qwen": ProviderSpec(
        key="qwen",
        label="Alibaba Qwen-VL (DashScope)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen3-vl-flash",
        api_key_env="DASHSCOPE_API_KEY",
        console="https://bailian.console.alibabacloud.com/",
        price_in=0.022,
        price_out=0.215,
        notes="The strongest grounding-per-dollar in this table. A 640x360 frame is about 227 "
              "image tokens, so 'qwen3-vl-flash' costs roughly $0.0001 per frame (~$0.30/hour "
              "at 0.5 Hz). This is the line Lumine itself builds on: the paper fine-tunes "
              "Qwen2-VL-7B-Base, and qwen3-vl is its direct descendant. Note the id is legacy "
              "but still served, and function calling is unsupported on the Singapore endpoint. "
              "Current flagships -- 'qwen3.8-flash', 'qwen3.8-max', 'qwen3.7-plus' -- are "
              "natively multimodal; there is no 'qwen3.8-vl'.",
    ),
    "doubao": ProviderSpec(
        key="doubao",
        label="ByteDance Doubao (Volcengine Ark)",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-seed-2-1-lite-260915",
        api_key_env="ARK_API_KEY",
        console="https://console.volcengine.com/ark",
        notes="The Lumine authors' own platform (ByteDance Seed), so this is the closest thing "
              "to reproducing the paper. All current Doubao flagships are natively multimodal "
              "and bill a fixed 1280-token floor per image -- cheap frames, expensive ones "
              "alike. Model ids on Ark are often per-account endpoint ids (ep-xxxx).",
    ),
    "stepfun": ProviderSpec(
        key="stepfun",
        label="StepFun Step",
        base_url="https://api.stepfun.com/v1",
        default_model="step-3.7-flash",
        api_key_env="STEPFUN_API_KEY",
        console="https://platform.stepfun.com/",
        price_in=0.20,
        price_out=1.15,
        notes="Best-in-class GUI grounding, but note *where* that reputation comes from: the "
              "strong ScreenSpot-Pro numbers belong to the open-weight Step-GUI-4B/8B, not to "
              "these API models. StepGUI-4B fits an 8 GB GPU and matches a 235B Qwen3-VL on "
              "screen grounding, which makes it the most interesting local option here -- see "
              "the Ollama entry for how to serve an open-weight model.",
    ),
    "glm-cn": ProviderSpec(
        key="glm-cn",
        label="Zhipu (mainland)",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4.6v",
        api_key_env="ZHIPUAI_API_KEY",
        console="https://open.bigmodel.cn/usercenter/apikeys",
    ),
    # ------------------------------------------------------------------------------ #
    # Western frontier models. Prices here are third-party (models.dev): the vendors'
    # own pricing pages block automated fetches, so re-verify before committing spend.
    # ------------------------------------------------------------------------------ #
    "openai": ProviderSpec(
        key="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-5.6-luna",
        api_key_env="OPENAI_API_KEY",
        console="https://platform.openai.com/api-keys",
        price_in=0.20,
        price_out=1.20,
        notes="Strongest general vision reasoning; highest latency and cost per call. Prices "
              "are third-party and unverified.",
        verified=False,
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        label="Anthropic Claude",
        base_url="https://api.anthropic.com/v1",
        default_model="claude-haiku-5-5",
        api_key_env="ANTHROPIC_API_KEY",
        console="https://console.anthropic.com/settings/keys",
        price_in=0.10,
        price_out=0.50,
        notes="WARNING for this project specifically: Anthropic's OpenAI-compatible shim is "
              "documented as not-for-production and silently ignores 'response_format'. This "
              "agent depends on strict JSON, so treat Claude as best-effort here and always "
              "keep the tolerant parser in the loop.",
        verified=False,
    ),
    "gemini": ProviderSpec(
        key="gemini",
        label="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        default_model="gemini-3.5-flash-lite",
        api_key_env="GEMINI_API_KEY",
        console="https://aistudio.google.com/apikey",
        price_in=0.30,
        price_out=2.50,
        notes="The Flash-Lite tier is the best latency/cost compromise among the frontier "
              "labs if you are outside China. Prices are third-party and unverified.",
        verified=False,
    ),
    # ------------------------------------------------------------------------------ #
    # Fully local, no key, no network: fits an 8 GB laptop GPU.
    # ------------------------------------------------------------------------------ #
    "ollama": ProviderSpec(
        key="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="qwen3-vl:8b-instruct-q4_K_M",
        api_key_env="OLLAMA_API_KEY",
        console="https://ollama.com/download",
        price_in=0.0,
        price_out=0.0,
        notes="Offline, free, and the only option with no per-call latency floor from a "
              "network round trip. Verified tags: qwen3-vl:{2b,4b,8b,30b-a3b,32b} at "
              "q4_K_M/q8_0/bf16, or the older qwen2.5vl:{3b,7b}. qwen3-vl:8b at q4_K_M is "
              "6.1 GB and fits an 8 GB card; qwen3-vl:4b is 3.3 GB and leaves room for the "
              "action head.",
    ),
    "none": ProviderSpec(
        key="none",
        label="No model (scripted)",
        base_url="",
        default_model="",
        api_key_env="",
        vision=False,
        notes="Zero-dependency fallback brain. Used when no key is available so the whole "
              "pipeline still runs, collects data, and evaluates.",
    ),
}

#: Ordered shortlist for a real-time game agent: cheap, fast, and visually grounded.
RECOMMENDED: tuple[tuple[str, str], ...] = (
    ("deepseek", "Verified end to end here and the cheapest credible option: ~$0.15/$0.60 per "
                 "1M, a hard 1024-token image cap that makes cost predictable, and 1.4-2.0 s "
                 "per 640x360 frame. Start here."),
    ("qwen", "Best grounding per dollar by an order of magnitude -- 'qwen3-vl-flash' is "
             "$0.022/$0.215, about $0.0001 per frame. It is also the family Lumine itself "
             "fine-tunes. Switch here when the agent must click precise GUI targets."),
    ("ollama", "Fully local on the 8 GB GPU: no key, no network, no per-call cost, and no "
               "network round trip in the latency budget. 'qwen3-vl:8b' at q4_K_M is 6.1 GB."),
    ("zai", "'glm-4.6v-flash' is free, which makes it the obvious A/B baseline; "
            "'glm-5.3-flashx' is the only model here with a published throughput (~200 tok/s)."),
    ("doubao", "The Lumine authors' own platform, if reproducing the paper matters more than "
               "cost. All current flagships are natively multimodal."),
    ("stepfun", "For GUI-heavy work, but read the notes: the celebrated grounding scores "
                "belong to the open-weight Step-GUI-4B/8B, not these API models."),
)


def get_provider(key: str) -> ProviderSpec:
    try:
        return PROVIDERS[key]
    except KeyError:
        raise KeyError(
            f"unknown brain provider {key!r}. available: {', '.join(sorted(PROVIDERS))}"
        ) from None


def render_table() -> str:
    """A markdown table of every provider, for the README and ``--list-providers``."""
    head = ("| provider | default model | base_url | vision | in/out $ per 1M | verified |\n"
            "|---|---|---|---|---|---|")
    rows = []
    for p in PROVIDERS.values():
        if not p.base_url:
            continue
        price = "free" if p.price_in == 0 else (
            f"{p.price_in}/{p.price_out}" if p.price_in is not None else "n/a"
        )
        rows.append(f"| `{p.key}` | `{p.default_model}` | `{p.base_url}` | "
                    f"{'yes' if p.vision else 'no'} | {price} | {'ok' if p.verified else 'check'} |")
    return head + "\n" + "\n".join(rows)
