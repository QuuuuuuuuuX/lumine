# Restoring this archive for development

This directory (`A:\lumine`) is an **archive**. It is complete and it runs — the full test
suite passes from here — but it should not be your working copy.

## Why not develop here

`A:` is mounted into WSL through **9p** (`drvfs`), which is a network-style protocol carrying
every filesystem call across the VM boundary. Measured on this machine:

| | native (`~/`) | `A:` (9p) | |
|---|---|---|---|
| sequential write | 2.1 GB/s | 154 MB/s | **13.6x slower** |
| `import lumine` | 0.22 s | 0.63 s | **2.9x slower** |

The import number is the one that matters: it is paid on every process start, and a test run
starts several. Python also writes `__pycache__` next to the source, so every test run does
hundreds of small 9p writes.

Two secondary reasons: 9p reports every file as `-rwxrwxrwx` regardless of its real mode, so
permission bugs are invisible here; and Windows tools (antivirus, indexer) may touch files
mid-run.

## Restore

```bash
mkdir -p ~/projects
rsync -a /mnt/a/lumine/ ~/projects/lumine/
cd ~/projects/lumine
```

`rsync -a` preserves timestamps and modes. Do not use `cp -r` without `-a` if you care about
the recorded times.

Verify the copy before trusting it:

```bash
diff <(cd /mnt/a/lumine && find . -type f -exec md5sum {} \; | sort -k2) \
     <(cd ~/projects/lumine && find . -type f -exec md5sum {} \; | sort -k2) \
  && echo "identical"
```

## Dependencies

**Required** (all three are needed for the sandbox and the tests):

```bash
pip install --user --break-system-packages numpy opencv-python Pillow
```

On this machine `--break-system-packages` is necessary: the system Python 3.14 is
externally managed (PEP 668), and `python3 -m venv` fails because `ensurepip` is not
installed. Installing to `--user` leaves the system interpreter alone. If you are on a
machine where `venv` works, prefer a virtualenv.

**Optional:**

```bash
pip install --user --break-system-packages torch       # only for training the action head;
                                                       # the trainer falls back to numpy
pip install --user --break-system-packages mss         # only for real-screen capture
```

There is no `pyproject.toml`; `pip install -e .` is not supported. Run everything from the
project root as `python3 -m lumine ...` and `python3 -m tests.test_...`.

## A vision-language brain

The default config uses DeepSeek and reads its key from the environment:

```bash
export DEEPSEEK_API_KEY=sk-...        # https://platform.deepseek.com/api_keys
python3 -m lumine probe               # confirm the key really accepts images
```

On the machine this was archived from, that key lives in `~/.dsh/.credentials.yaml` and is
picked up automatically — `lumine/config.py::resolve_api_key` checks the environment first,
then that file. It is not in this archive, and it should not be.

Any of twelve providers works, and switching is a config change: `python3 -m lumine
providers`. Without any key the scripted brain still runs everything:

```bash
python3 -m lumine play --config configs/scripted.yaml --task combat_defeat_and_chest
```

## Verify the restore

```bash
python3 -m tests.test_core      # 40
python3 -m tests.test_sim       # 18
python3 -m tests.test_p2p       # 17
python3 -m tests.test_brain_live # 3, needs network + a key, skips itself otherwise
python3 -m lumine rates         # should report control 30.0 Hz and perception 5.0 Hz
python3 -m lumine tasks         # 23 tasks in 7 categories
```

Expected: **75/75** on the first three, and `rates` reporting both rates exactly on target.

## What is *not* in this archive

| missing | size | how to get it |
|---|---|---|
| P2P model checkpoint | 2.2 GB | `export HF_ENDPOINT=https://hf-mirror.com` then download `elefantai/open-p2p` → `150M/checkpoint-step=00500000.ckpt`. HuggingFace is **unreachable** from the build machine (connection reset, not DNS), so the mirror is required. |
| NitroGen weights | large | same mirror, `nvidia/NitroGen` |
| PyTorch | ~1 GB | `pip install --user --break-system-packages torch` |
| Any API key | — | supply your own |
| Real-game integration testing | — | never done; see below |

## State of the project, honestly

Everything in `artifacts/` was produced here and is reproducible from the archive:

- `artifacts/eval/REPORT.md` — 14 episodes, 14.3% overall, npc 100%, everything else 0%
- `artifacts/boss_electro.mp4` — a real VLM episode, 34 model calls at 0.41 Hz
- `artifacts/checkpoints/action_head_pretrain/` — trained on 4,510 frames, 96% key accuracy
- `artifacts/data/pretrain/` — the corpus, from the **built-in sandbox**, not from any game

Not done, and not claimed anywhere in the docs: no VLM was fine-tuned; the corpus is 0.125
hours against Lumine's 1,731 (0.0072%); `live.py` and `/dev/uinput` injection are written but
have never driven a real game, because the build machine's user is not in the `input` group.

## This directory is also a git working copy

`A:\lumine` is not just an archive — it is a git repository with a GitHub remote:

```
remote  git@github.com:QuuuuuuuuuX/lumine.git
branch  main
```

So there are two things living here at once: the files, and `.git/`. Keep that in mind
before doing anything with `--delete` (see the warning below).

Check state, commit, push:

```bash
cd /mnt/a/lumine
git status
git add -A && git commit -m "..."
git push
```

## Re-archiving after you work on it

> **`--delete` can destroy the repository and the `.gitignore`.** Everything that lives
> *only* in this directory is "extraneous" as far as rsync is concerned, and `--delete`
> removes extraneous things. That means `.git/` — the entire history and the remote — and
> `.gitignore`, neither of which exists in `~/projects/lumine/`. The exclude below protects
> `.git/`; rsync protects excluded paths from `--delete` by default, so do **not** add
> `--delete-excluded`.

```bash
rsync -a --delete --dry-run -v \
      --exclude='.git/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      ~/projects/lumine/ /mnt/a/lumine/
```

**Always dry-run first, and always with `-v`.** Without `-v`, rsync does not print its
`deleting ...` lines, so a quiet dry-run looks identical whether it is about to delete
nothing or delete the repository. This exact false-negative happened while writing this
document: a `--dry-run` without `-v` was read as "0 deletions" when the real answer was
`.git/` and `.gitignore`. If the dry run prints any `deleting` line you did not expect, stop.

Once the dry run is clean, drop `--dry-run`:

```bash
rsync -a --delete \
      --exclude='.git/' \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      ~/projects/lumine/ /mnt/a/lumine/
```

Then confirm the repository survived:

```bash
cd /mnt/a/lumine && git status        # must still say "On branch main"
```

The excludes keep compiled bytecode out, since `.pyc` files are specific to one Python
version and are already gitignored. Run the tests once *before* syncing and not after — or
clean up afterwards, because running them from `A:` leaves `__pycache__` behind.
