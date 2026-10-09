# Upstream reference sources

These files are **not ours**. They are verbatim copies of source files from an upstream
project, kept here for one reason: `lumine/brain/p2p.py` claims to implement that project's
action space, and a claim like that is only checkable if the evidence is next to it.

## Provenance

| | |
|---|---|
| Repository | https://github.com/elefant-ai/open-p2p |
| Paper | *Scaling Behavior Cloning Improves Causal Reasoning: An Open Model for Real-Time Video Game Playing*, arXiv [2601.04575](https://arxiv.org/abs/2601.04575) |
| Licence | MIT (see `LICENSE` in this directory) |
| Commit | `a329d98cbe62119679a254d71bea6446773541bc` |
| Retrieved | 2026-10-09 |
| Method | GitHub contents API (see note below) |

## Why only these files

The full repository is ~92 files. A complete archive could not be downloaded from this
machine: `codeload.github.com` and `github.com/.../archive/` both truncate the tarball at
roughly 700-860 KB across repeated attempts, while the GitHub contents API is reliable but
rate-limited to 60 requests per hour unauthenticated.

So this directory holds the files that the adapter actually depends on, not the whole repo.
To get the rest:

```bash
git clone https://github.com/elefant-ai/open-p2p.git     # works via git, unlike codeload
```

## What each file is, and whether we rely on it

| file | upstream path | why it is here |
|---|---|---|
| `action_mapping.py` | `elefant/data/action_mapping.py` | **The authority.** All 20 key tokens, 4 mouse-button tokens, and the 23 x-bins / 17 y-bins with their exact edges and centres are transcribed from this file. |
| `action_decoder.py` | `elefant/policy_model/action_decoder.py` | Shows the 8 tokens are emitted autoregressively, one forward pass each. Confirms the per-step token count. |
| `elefant_policy_model_inference.py` | `elefant/policy_model/inference.py` | The inference entry point. Source of `MODEL_INPUT_HEIGHT = MODEL_INPUT_WIDTH = 192` and the import surface. Renamed with a prefix because it is not an importable module here. |
| `elefant_policy_model_config.py` | `elefant/policy_model/config.py` | Model shape: action decoder `embed_dim: 1024`, 200-frame attention history, `truncated_normal` mouse sampling. |
| `environment_mapping.py` | `elefant/data/environment_mapping.py` | Small, and it documents how upstream names game environments. |
| `UPSTREAM_README.md` | `README.md` | Install and inference instructions, checkpoint names and sizes, hardware notes. |
| `LICENSE` | `LICENSE` | MIT. Required for redistributing the above. |

## Files deliberately **not** copied

* `elefant/config.py` — the fetch returned an empty body; the file is a config loader and
  nothing in our adapter depends on it.
* Everything else — training pipeline, data loaders, tokenizers, protobuf stubs. Not needed
  to *understand* the action space, only to run the model, and that is documented as
  not-yet-done in the project README.

## The one thing to know if you edit `p2p.py`

Upstream bins mouse deltas with `torch.bucketize(x, edges)` using the default
`right=False`, whose contract is `edges[i-1] < x <= edges[i]`. That is numpy's
`side="left"`, **not** `"right"`. The two agree everywhere except exactly on an edge, so
getting it wrong produces a camera that is subtly off rather than an error.
`tests/test_p2p.py::test_binning_matches_torch_bucketize` pins this against real torch.
