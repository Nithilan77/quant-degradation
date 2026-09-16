"""Thin wrapper around lm-eval-harness.

Deliberately does as little as possible: load a model per configs/models.yaml,
hand it to lm-eval-harness's own HFLM wrapper, call simple_evaluate with the
task settings from configs/tasks.yaml, and write the raw result JSON to
results/raw/. All prompt formatting, few-shot sampling, and metric
computation is lm-eval's job, not ours -- that's what makes the fp16 and
quantized runs directly comparable (same task call, only the model differs).

Every config for a given task shares num_fewshot / batch_size / seed from
configs/tasks.yaml, so nothing here should read per-model overrides for
those fields. If a future quantization backend genuinely requires a
different batch_size to run at all, that must become a visible, documented
exception in configs/models.yaml -- not a default threaded through here.
"""

import argparse
import json
import time
from pathlib import Path

import yaml
import lm_eval
from lm_eval.models.huggingface import HFLM

from load_model import load_model, set_deterministic_seed


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_eval(
    model_key: str,
    task_key: str,
    models_config_path: str = "configs/models.yaml",
    tasks_config_path: str = "configs/tasks.yaml",
    output_dir: str = "results/raw",
    limit: int | None = None,
):
    models_cfg = load_yaml(models_config_path)
    tasks_cfg = load_yaml(tasks_config_path)

    model_cfg = models_cfg["models"][model_key]
    task_cfg = tasks_cfg["tasks"][task_key]
    generation_cfg = models_cfg["generation"]
    seed = tasks_cfg["seed"]

    model, tokenizer = load_model(model_cfg, seed, generation_cfg)

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=task_cfg.get("batch_size", 1),
    )

    # Re-seed immediately before evaluation: load_model seeds before
    # weight loading, but lm-eval's own few-shot sampling / generation
    # happens after this point and must start from the same state every
    # time for check_determinism.py to hold.
    set_deterministic_seed(seed)

    effective_limit = limit if limit is not None else task_cfg.get("limit")

    # Belt-and-suspenders with apply_generation_config in load_model.py:
    # gsm8k's own lm-eval task YAML sets its own generation_kwargs
    # (e.g. do_sample), and CLI/API-level gen_kwargs takes precedence
    # over a task's built-in ones. Passed explicitly here so the shared
    # `generation` block from configs/models.yaml is what actually reaches
    # generate(), not whatever a given lm-eval task happens to default to.
    gen_kwargs = ",".join(f"{k}={v}" for k, v in generation_cfg.items())

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task_cfg["lm_eval_task"]],
        num_fewshot=task_cfg["num_fewshot"],
        limit=effective_limit,
        gen_kwargs=gen_kwargs,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
    )

    task_results = results["results"][task_cfg["lm_eval_task"]]

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = output_dir_path / f"{model_key}__{task_key}.json"

    payload = {
        "model_key": model_key,
        "task_key": task_key,
        "model_cfg": model_cfg,
        "task_cfg": task_cfg,
        "generation_cfg": generation_cfg,
        "seed": seed,
        "limit": effective_limit,
        "timestamp": time.time(),
        "results": task_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    return out_path, {task_cfg["lm_eval_task"]: task_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Key from configs/models.yaml")
    parser.add_argument("--task", required=True, help="Key from configs/tasks.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--tasks-config", default="configs/tasks.yaml")
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument("--limit", type=int, default=None, help="Override task limit for a quick run")
    args = parser.parse_args()

    out_path, results = run_eval(
        args.model, args.task, args.models_config, args.tasks_config, args.output_dir, args.limit
    )
    print(f"Wrote {out_path}")
    print(json.dumps(results, indent=2, default=str))
