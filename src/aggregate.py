"""Aggregate raw lm-eval-harness result JSONs into one tidy dataframe.

Reads every payload src/run_eval.py writes (one JSON per (model, task) run)
and produces one row per run with columns:

    model, size, scheme, bit_width, task, capability_axis, metric, value

No `torch` import here (or anywhere else in this module) -- this runs on
the laptop side against results synced from the GPU box, not on the GPU
box itself.

Metric selection is explicit, not inferred per-run: gsm8k reports several
metric variants (e.g. strict-match vs flexible-extract), and CLAUDE.md's
determinism/no-silent-divergence rule applies here too -- comparisons must
never quietly mix metric variants across configs. This script picks one
metric filter for the whole run (--metric-filter, default strict-match)
and raises if any result is missing that exact key, rather than falling
back to whatever key happens to be present.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import yaml

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?B)", re.IGNORECASE)

DATAFRAME_COLUMNS = [
    "model",
    "size",
    "scheme",
    "bit_width",
    "task",
    "capability_axis",
    "metric",
    "value",
]


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_results(raw_dir: Path) -> list[dict]:
    payloads = []
    for path in sorted(raw_dir.glob("*.json")):
        with open(path) as f:
            payloads.append(json.load(f))
    return payloads


def extract_size(base_model: str) -> str:
    """Parse e.g. '7B' out of 'Qwen/Qwen2.5-7B-Instruct-AWQ'. Derived from
    the checkpoint id rather than read from a config field, so it stays
    consistent across an fp16 repo and a separately-uploaded quantized
    repo for the same model size.
    """
    match = SIZE_RE.search(base_model)
    if not match:
        raise ValueError(f"Could not parse model size out of base_model={base_model!r}")
    return match.group(1).upper()


def metric_key_for(task_cfg: dict, metric_filter: str | None) -> str:
    base_metric = task_cfg["metric"]
    if metric_filter:
        return f"{base_metric},{metric_filter}"
    return base_metric


def build_dataframe(
    payloads: list[dict], tasks_cfg: dict, metric_filter: str = "strict-match"
) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        model_cfg = payload["model_cfg"]
        task_key = payload["task_key"]
        task_cfg = tasks_cfg["tasks"][task_key]
        results = payload["results"]

        key = metric_key_for(task_cfg, metric_filter)
        if key not in results:
            available = sorted(k for k in results if "stderr" not in k and k != "alias")
            raise KeyError(
                f"{payload['model_key']}__{task_key}: metric key {key!r} not found in "
                f"results (available: {available}). Pass --metric-filter to select a "
                f"different variant."
            )

        rows.append(
            {
                "model": model_cfg["base_model"],
                "size": extract_size(model_cfg["base_model"]),
                "scheme": model_cfg["scheme"],
                "bit_width": model_cfg.get("bit_width"),
                "task": task_key,
                "capability_axis": task_cfg["capability_axis"],
                "metric": key,
                "value": results[key],
            }
        )

    df = pd.DataFrame(rows, columns=DATAFRAME_COLUMNS)
    # Nullable integer dtype: bit_width is genuinely absent (not 0 or NaN-as-
    # float) for the unquantized fp16 baseline.
    df["bit_width"] = df["bit_width"].astype("Int64")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="results/raw",
        help="Directory of run_eval.py-shaped JSON files to read (default: results/raw). "
        "Point this at fixtures/synthetic_raw for a dry run without the GPU box.",
    )
    parser.add_argument("--tasks-config", default="configs/tasks.yaml")
    parser.add_argument("--output", default="results/processed/aggregated.csv")
    parser.add_argument(
        "--metric-filter",
        default="strict-match",
        help="lm-eval metric filter variant to use for every run, e.g. "
        "strict-match or flexible-extract (default: strict-match). Applied "
        "uniformly across all configs -- never mixed per-run.",
    )
    args = parser.parse_args()

    tasks_cfg = load_yaml(args.tasks_config)
    raw_dir = Path(args.raw_dir)
    payloads = load_raw_results(raw_dir)
    if not payloads:
        raise SystemExit(f"No raw result JSONs found in {raw_dir}")

    df = build_dataframe(payloads, tasks_cfg, args.metric_filter)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path} (metric filter: {args.metric_filter!r})")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
