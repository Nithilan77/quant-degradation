# quant-degradation

Characterizing non-uniform capability degradation under LLM quantization.
See `CLAUDE.md` for the full project brief (hypothesis, constraints,
model/task matrix, current status).

## GPU box setup (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate

# Install a CUDA-matched torch build FIRST -- check your driver's CUDA
# version with `nvidia-smi`, then pick the matching index, e.g. for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Now the rest (this will see torch already satisfied and skip it):
pip install -r requirements.txt
```

`autoawq` and `auto-gptq` both build against the installed torch/CUDA
version, so they must be installed after torch, not before -- installing
via `requirements.txt` in the order above is the safe path. If either
fails to build from source, check for a prebuilt wheel matching your
Python/CUDA/torch versions on the project's GitHub releases page.

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

This runs Qwen2.5-7B-Instruct on gsm8k in three configs (fp16, bnb-nf4,
official AWQ) and prints the three accuracy numbers side by side. Raw
JSON output for each run lands in `results/raw/`.

## Repo layout

- `configs/` -- which models, quant schemes, tasks, and generation/seed
  settings to run. Expand this to grow the eval matrix; avoid touching
  `src/` for that.
- `src/` -- `load_model.py` (model+tokenizer loading per config),
  `run_eval.py` (thin lm-eval-harness wrapper), `check_determinism.py`,
  `run_milestone.py` (the current milestone loop).
- `results/raw/` -- raw per-(model, task) JSON, tracked in git.
- `results/processed/` -- derived tables, not yet implemented.
- `figures/` -- exported plots, not yet implemented.
- `notebooks/` -- exploratory analysis, not yet implemented.

## Workflow

Develop and review on the laptop; the actual eval runs happen on a
separate Linux GPU box that pulls this repo. Push from the laptop, pull +
run on the GPU box, commit `results/raw/*.json` there and push back.
