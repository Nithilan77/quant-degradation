"""Generate synthetic results/raw/-shaped JSON fixtures.

The GPU box is down; this lets src/aggregate.py and src/analyze.py be built
and exercised end to end without it. Output is NOT real eval data -- plausible
fake gsm8k numbers for the three milestone configs (fp16 baseline, bnb-nf4,
official AWQ), shaped exactly like the payload src/run_eval.py writes (same
keys, same nesting -- model_cfg/task_cfg/generation_cfg pulled from the real
configs/*.yaml so a fixture never drifts out of sync with the actual eval
matrix). Output goes to fixtures/synthetic_raw/, never results/raw/, so it
can't be mistaken for a real run.

Fake exact_match values are picked to be plausible relative to published
Qwen2.5-7B-Instruct gsm8k numbers: fp16 highest, bnb-nf4 (plain round-to-
nearest 4-bit) taking the largest hit, official AWQ (activation-aware)
a smaller hit than bnb-nf4 -- a believable, not arbitrary, degradation
ordering to sanity-check the analysis code against.
"""

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "synthetic_raw"

TASK_KEY = "gsm8k"

FAKE_RESULTS = {
    "qwen2.5-7b-instruct-fp16": {
        "exact_match,strict-match": 0.8506,
        "exact_match_stderr,strict-match": 0.0098,
        "exact_match,flexible-extract": 0.8597,
        "exact_match_stderr,flexible-extract": 0.0096,
    },
    "qwen2.5-7b-instruct-bnb-nf4": {
        "exact_match,strict-match": 0.7901,
        "exact_match_stderr,strict-match": 0.0112,
        "exact_match,flexible-extract": 0.8052,
        "exact_match_stderr,flexible-extract": 0.0109,
    },
    "qwen2.5-7b-instruct-awq": {
        "exact_match,strict-match": 0.8286,
        "exact_match_stderr,strict-match": 0.0104,
        "exact_match,flexible-extract": 0.8362,
        "exact_match_stderr,flexible-extract": 0.0102,
    },
}


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    models_cfg = load_yaml(ROOT / "configs" / "models.yaml")
    tasks_cfg = load_yaml(ROOT / "configs" / "tasks.yaml")
    task_cfg = tasks_cfg["tasks"][TASK_KEY]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, fake_metrics in FAKE_RESULTS.items():
        model_cfg = models_cfg["models"][model_key]

        payload = {
            "model_key": model_key,
            "task_key": TASK_KEY,
            "model_cfg": model_cfg,
            "task_cfg": task_cfg,
            "generation_cfg": models_cfg["generation"],
            "seed": tasks_cfg["seed"],
            "limit": None,
            "timestamp": time.time(),
            "results": {"alias": TASK_KEY, **fake_metrics},
        }

        out_path = OUTPUT_DIR / f"{model_key}__{TASK_KEY}.json"
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
