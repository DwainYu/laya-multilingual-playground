# GitHub Publish Report

## Repository

<https://github.com/DwainYu/laya-multilingual-playground>

## Visibility

Public (`isPrivate: false`)

## Branch

`main` (default branch on GitHub)

## Commit

Released snapshot (`origin/main` at publish time):

```text
0e6d08662c618bebcd87f4519936e936409cbe12
0e6d086  docs: prepare project for GitHub release
```

4 commits at that point, all authored by `DwainYu <playmaker_ai@qq.com>`, working tree clean.
This report is added by the commit immediately after it, so `main` ends up one ahead of the
hash above; nothing but this file differs.

## Local Validation

Re-run immediately before pushing, on WSL2 + CPU, model already on disk
(weights were **not** re-downloaded):

| Target | Result | Evidence |
|---|---|---|
| `make demo` | PASS (exit 0) | `department = billing`, confidence `0.9822` |
| `make typed` | PASS (exit 0) | choice + score + noul in one pass, 322.80 ms |
| `make compare` | PASS (exit 0) | 10 Chinese scenarios, same labels/confidences as the experiment run |
| `make benchmark` | PASS (exit 0) | cold start 22.57 s; warm 1q avg 140.50 ms |
| `make test` | PASS (exit 0) | **13 passed in 23.21 s** |

Decisions and probabilities came back identical to the original experiment run; only
millisecond-level latency drifted (~±10 %), which is documented as expected on WSL2 CPU.

## Reproduction Check (clean clone)

The actual point of publishing was verified: the repo was cloned fresh into an empty
directory on a different path and driven end to end.

```text
git clone https://github.com/DwainYu/laya-multilingual-playground.git
uv sync --extra dev          → resolves and installs cleanly
make download                → 646.8 MiB from ModelScope into models/laya-multilingual
make demo                    → department = billing, confidence 0.9822   (identical)
make test                    → 13 passed
sha256sum model.safetensors  → identical to the original local download
```

The clone and the temp directory used for it were then deleted.

## Model

```text
convaiinnovations/laya-multilingual
```

322M parameters (mmBERT-base encoder + decision head), 1024-token context, 100+ languages.

## Model Source

ModelScope — <https://www.modelscope.cn/models/convaiinnovations/laya-multilingual>

Downloaded with `modelscope.snapshot_download(..., local_dir="models/laya-multilingual")` and
loaded straight from that directory via `laya.load(<local path>)`. No Hugging Face download is
involved; verified by loading under `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
The ModelScope and Hugging Face files were verified file-by-file using SHA-256 in this
experiment (weights: `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`).

## Weights

**Not committed to Git.**

```bash
$ git ls-tree -r HEAD --name-only | grep -E '\.(safetensors|bin|pt|pth|ckpt)$'
(no output)
```

Only `models/.gitkeep` is tracked; `.gitignore` has `models/*` + `!models/.gitkeep`.
Tracked tree: 20 files (19 + this report); largest is `uv.lock` at 264 KB. Total push is well
under a megabyte — the 646.8 MiB checkpoint stays on disk.

A full-history secret scan (`git grep` over every commit for API-key / token / private-key
patterns, incl. `ms-*`, `sk-*`, `ghp_*`, `AKIA*`, `BEGIN PRIVATE KEY`) found nothing; no
`.env` or credential file exists in the tree.

## Important Experiment Finding

Choice-based decisions worked reasonably well in the local Chinese test set
(8 of 10 clearly sensible).

**Noul showed unreliable zero-shot behavior and is documented as a limitation.** A pure refund
ticket scored `noul = 0.9409` for "is this related to AI model deployment?", and "the web page
won't open" scored `0.97`. This matches the checkpoint's own model card: zero-shot
typed-decisions 0.342 against a 0.318 random baseline, with uncalibrated
`temperature = [1.0, 1.0, 1.0]`, i.e. systematically over-confident.

This is written up as **"Known Limitation / Experiment Finding"** in the README — not as
"Laya Noul is broken", and not hidden behind a cherry-picked success case.

## Changes Made During Release Prep

No experiment code, model conclusions, or benchmark figures were changed.

- README restructured with an English front-matter (Overview / Experiment Results / Known
  Limitation / Installation / Model / Architecture / Project Structure); Chinese detail preserved
- Latency figures converted from single points to measured ranges across runs
- ModelScope↔HF sync statement softened to what was actually verified (this snapshot only)
- Added Apache-2.0 `LICENSE` (verbatim text) + copyright notice in README, so the repo's
  existing "Apache 2.0" claim is backed by a file
- `.gitignore`: added `.ruff_cache/` and `.cache/`
- Renamed `master` → `main`
- Rewrote the 3 pre-existing commits' author from the placeholder
  `playground <pg@local>` to `DwainYu <playmaker_ai@qq.com>` **before any push**, so no
  public history carries a throwback identity. Content hashes of the trees are unchanged;
  commit hashes therefore differ from the local pre-publish ones.
- Repo description + 10 topics set (`laya`, `decision-model`, `agent`, `ai`, `modelscope`,
  `machine-learning`, `transformers`, `python`, `uv`, `wsl`). Deliberately **not** tagged
  `ollama` / `mcp` / `rag` / `llm` — none of them is implemented here; the README lists them
  as future work only.

## Next Possible Experiments

1. **Laya + Ollama** — Laya gates/routes, Ollama plans and explains (System 1 / System 2 split,
   see `docs/architecture.md`)
2. **Laya + Agent Router** — `laya.Router(preload=True)` for mixed-language traffic
3. **Laya + MCP** — `choice` output selects the tool; Code executes the call
4. **Laya + Tool Selection** — confidence-gated tool dispatch, after fitting temperatures on
   your own labelled data

Prerequisite for 1 and 4: fine-tune and calibrate first. As shipped, this checkpoint's `noul`
and `score` probabilities are not trustworthy enough to gate on.
