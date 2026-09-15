# quant-degradation

Empirical research project: characterizing **non-uniform capability
degradation under LLM quantization**.

## Core claim

Aggregate accuracy hides that some capabilities degrade far more than
others under quantization, and the degradation pattern is predictable
across bit-width, quantization scheme, and model scale. A single "overall
accuracy" number is the wrong unit of analysis for this question — every
result must be reported per capability axis, never averaged across axes.

## Models

- Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct
- Qwen2.5-32B-Instruct

fp16 is the baseline for every model size. Quantization schemes to cover:

- bitsandbytes NF4 (on-the-fly, load-time quantization)
- GPTQ (8/4/3-bit)
- AWQ (8/4/3-bit)

Prefer **official Qwen pre-quantized checkpoints** where they exist (e.g.
`Qwen/Qwen2.5-7B-Instruct-AWQ`, `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`).
Fall back to on-the-fly bitsandbytes quantization of the fp16 checkpoint
when no official pre-quantized checkpoint exists for a given bit-width.

## Harness

`lm-eval-harness` is the evaluation engine. We add a **thin custom layer**
on top (`src/`) for:
- loading a model+tokenizer from a declarative config (fp16 / bnb-nf4 /
  gptq / awq) with identical generation settings across all of them
- calling into lm-eval-harness the same way regardless of quantization
  scheme
- writing raw results to `results/raw/` and later reducing them into
  per-capability degradation curves in `results/processed/`

Do not reimplement anything lm-eval-harness already does (prompt
formatting, few-shot sampling, metric computation). The custom layer only
handles model loading and orchestration/bookkeeping around it.

## Capability axes

Measured **separately**, never aggregated into one score:

| Axis | Task |
|---|---|
| Reasoning | gsm8k |
| Factual recall | triviaqa |
| Long-context retrieval | needle-style |
| Instruction-following | ifeval |
| Code | humaneval / mbpp |

Current milestone only wires up **reasoning (gsm8k)**. The other axes are
planned but not yet implemented — see "Current status" below before
assuming any of them exist.

## Hard constraints (paper validity depends on these)

1. **Identical prompts, tokenization, few-shot count, and generation
   config** between fp16 and every quantized config. Only the model
   weights/quantization scheme may vary.
2. **Route everything through the same lm-eval task call.** Do not hand-rig
   prompts per config; use lm-eval's task definitions uniformly so the fp16
   run and the quantized runs are the same eval, just a different model.
3. **Determinism is required.** Same seed must produce identical output.
   Greedy decoding, fixed seeds for `random`/`numpy`/`torch`, fixed batch
   size (batch size affects padding, which affects logits/generation, so
   it counts as a generation-config parameter — never let it drift between
   configs for the same task).
4. **Never silently vary generation params between configs.** If a
   parameter must differ for one config to even run (e.g. a quantization
   library forcing a specific dtype), that has to be a visible, documented
   exception — not a default that quietly diverges. `src/check_determinism.py`
   is the enforcement mechanism: if it fails, do not trust any comparison
   run since.

## Environment

Development happens on a Windows/WSL2 laptop. Code is pushed to GitHub and
pulled onto a **separate Linux box** where the actual GPU-heavy evals run.
Consequences:
- No hardcoded paths (Windows or otherwise). Everything configurable via
  YAML configs or CLI args.
- No assumption of a specific CUDA version, GPU count, or local model
  cache location — read from env vars / config, never hardcode.
- The laptop side never imports `torch`-dependent modules in a way that
  would break editing/reviewing code without a GPU (i.e. don't do
  expensive imports at module load time beyond what's needed).

## Current status / milestone

First milestone (implemented): Qwen2.5-7B-Instruct on gsm8k in three
configs — fp16, bnb-nf4, official AWQ
(`Qwen/Qwen2.5-7B-Instruct-AWQ`) — via `src/run_milestone.py`, printing the
three accuracy numbers side by side.

**Not yet built** (intentionally, to keep the first loop small):
- 14B/32B model configs
- GPTQ configs
- 3-bit configs
- Any capability axis other than gsm8k (triviaqa, needle, ifeval,
  humaneval/mbpp)
- The degradation analysis layer that reduces `results/raw/` into
  `results/processed/` and figures

When expanding: extend `configs/models.yaml` / `configs/tasks.yaml` rather
than adding new code paths — `src/run_eval.py` and `src/load_model.py`
are meant to be config-driven and should not need per-model or per-task
special-casing except where a quantization scheme genuinely requires
different loading code.

## Repo layout

```
configs/     declarative YAML: which models, which quant schemes, which
             tasks, generation/seed settings. Editing the eval matrix
             should never require touching src/.
src/         load_model.py   — model+tokenizer loading per config
             run_eval.py     — thin lm-eval-harness wrapper, writes JSON
             check_determinism.py — fp16 run twice, assert identical
             run_milestone.py — the current 3-config x gsm8k loop
results/raw/       raw JSON output per (model, task) run. Tracked in git
                   (synced manually between laptop and GPU box) — do not
                   put anything here that shouldn't be committed.
results/processed/ derived tables (degradation deltas, per-axis summaries)
                   built from results/raw/. Never hand-edited.
figures/     exported plots for the paper.
notebooks/   exploratory analysis against results/processed/, not raw.
```
