# quant-degradation

Characterizing non-uniform capability degradation under LLM quantization.
See `CLAUDE.md` for the full project brief (hypothesis, constraints,
model/task matrix, current status).

## GPU box setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install a CUDA-matched torch build FIRST, pinned to the range
# requirements.txt expects (2.4.x-2.5.x) -- check your driver's CUDA
# version with `nvidia-smi`, then pick the matching index, e.g. for CUDA 12.1:
pip install "torch>=2.4.0,<2.6.0" --index-url https://download.pytorch.org/whl/cu121

# Now the rest (this will see torch already satisfied and skip it):
pip install -r requirements.txt
```

`requirements.txt` is pinned with upper bounds to a set known to work
together for Qwen2.5 + AWQ + bitsandbytes (torch 2.4-2.5, transformers
4.44-4.46, matching accelerate/datasets/lm-eval/bitsandbytes/autoawq).
Don't loosen these casually -- an unpinned resolver has previously pulled
bleeding-edge torch that broke native-extension builds.

`autoawq` builds against the installed torch/CUDA version, so it must be
installed after torch, not before -- installing via `requirements.txt` in
the order above is the safe path. If it fails to build from source, check
for a prebuilt wheel matching your Python/CUDA/torch versions on the
project's GitHub releases page.

**GPTQ is not installed by `requirements.txt`.** Milestone 1 doesn't need
it, and `auto-gptq` builds a native extension that's fragile against
whatever torch version happens to be resolved. Once the GPTQ scheme is
actually added to `configs/models.yaml`, install it separately, after
`requirements.txt` so it builds against the already-pinned torch:

```bash
pip install -r requirements-gptq.txt
```

## Running the milestone

```bash
# Sanity check first: proves the pipeline is actually deterministic
# before trusting any comparison between configs.
python src/check_determinism.py

# Quick smoke test (20 examples per config, ~minutes not hours):
python src/run_milestone.py --limit 20

# Full milestone run:
python src/run_milestone.py
```

This runs Qwen2.5-7B-Instruct on gsm8k in three configs (bf16 baseline,
bnb-nf4, official AWQ) and prints the three accuracy numbers side by
side. Raw JSON output for each run lands in `results/raw/`.

Determinism is enforced strictly (`torch.use_deterministic_algorithms(True,
warn_only=False)`, `CUBLAS_WORKSPACE_CONFIG` set from within
`src/load_model.py`). If `check_determinism.py` or a milestone run raises
a nondeterministic-operation error for a specific quantization backend,
that's a real result to record (that backend can't guarantee reproducible
output) -- don't work around it by relaxing determinism back to warn-only.

## Analysis layer

Once raw results exist (real, in `results/raw/`, or synthetic, in
`fixtures/synthetic_raw/` -- see `fixtures/README.md` for generating fake
data to test against when the GPU box is unavailable):

```bash
# Reduce results/raw/*.json into one tidy CSV.
python src/aggregate.py

# Compute per-capability-axis degradation vs. the fp16 baseline and plot it.
python src/analyze.py
```

Both scripts point at `results/raw/` / `results/processed/` / `figures/`
by default; pass `--raw-dir fixtures/synthetic_raw` (and matching
`--output`/`--input`/`--figure` paths) to run against synthetic data
instead without touching real results. gsm8k reports both strict-match and
flexible-extract; `src/aggregate.py --metric-filter` picks which one feeds
the whole pipeline (default: `strict-match`) -- never mixed across a run.

## Repo layout

- `configs/` -- which models, quant schemes, tasks, and generation/seed
  settings to run. Expand this to grow the eval matrix; avoid touching
  `src/` for that.
- `src/` -- `load_model.py` (model+tokenizer loading per config),
  `run_eval.py` (thin lm-eval-harness wrapper), `check_determinism.py`,
  `run_milestone.py` (the current milestone loop), `aggregate.py` (raw
  JSON -> tidy CSV), `analyze.py` (degradation table + figure).
- `results/raw/` -- raw per-(model, task) JSON, tracked in git.
- `results/processed/` -- derived tables (`aggregate.py`/`analyze.py`
  output). Never hand-edited.
- `figures/` -- exported plots from `analyze.py`.
- `fixtures/` -- synthetic, fake results/raw/-shaped data for testing the
  analysis layer without the GPU box. See `fixtures/README.md`.
- `notebooks/` -- exploratory analysis, not yet implemented.
- `requirements.txt` -- pinned core + AWQ/bnb deps.
- `requirements-gptq.txt` -- `auto-gptq`, install separately, later.

## Workflow

Develop and review on the laptop; the actual eval runs happen on a
separate Linux GPU box that pulls this repo. Push from the laptop, pull +
run on the GPU box, commit `results/raw/*.json` there and push back.
