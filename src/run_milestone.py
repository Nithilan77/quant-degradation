"""Milestone 1: Qwen2.5-7B-Instruct on gsm8k in three configs (fp16,
bnb-nf4, official AWQ), printing the three accuracy numbers side by side.

This is intentionally the smallest possible loop, not the full sweep --
it exists to prove load_model -> run_eval -> compare works end to end
before configs/models.yaml and configs/tasks.yaml grow to cover the full
model x scheme x bit-width x capability-axis matrix from CLAUDE.md.
"""

import argparse

from run_eval import run_eval

MILESTONE_MODELS = [
    "qwen2.5-7b-instruct-fp16",
    "qwen2.5-7b-instruct-bnb-nf4",
    "qwen2.5-7b-instruct-awq",
]
TASK_KEY = "gsm8k"


def extract_accuracy(task_results: dict):
    """gsm8k's primary metric key has varied across lm-eval-harness
    versions (e.g. exact_match,strict-match vs exact_match,flexible-extract).
    Grab whichever exact_match-style key is present rather than hardcoding
    one, and fall back to the raw dict if nothing matches.
    """
    for key, value in task_results.items():
        if key.startswith("exact_match") and "stderr" not in key:
            return key, value
    return None, task_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap examples per config for a quick smoke test"
    )
    args = parser.parse_args()

    rows = []
    for model_key in MILESTONE_MODELS:
        print(f"Running {model_key} on {TASK_KEY} ...")
        out_path, results = run_eval(model_key, TASK_KEY, limit=args.limit)
        metric_key, value = extract_accuracy(results[TASK_KEY])
        rows.append((model_key, metric_key, value, out_path))

    print(f"\n{'model':38s} {'metric':28s} {'value':>10s}")
    print("-" * 80)
    for model_key, metric_key, value, _ in rows:
        print(f"{model_key:38s} {str(metric_key):28s} {str(value):>10s}")


if __name__ == "__main__":
    main()
