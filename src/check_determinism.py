"""Determinism gate: run the fp16 config twice on a small gsm8k slice
with the fixed seed from configs/tasks.yaml and assert the results are
identical.

This must pass before trusting any fp16-vs-quantized comparison -- if the
pipeline isn't deterministic on its own, differences between configs
could just be run-to-run noise rather than a real quantization effect.
"""

import json
import sys

from run_eval import run_eval

MODEL_KEY = "qwen2.5-7b-instruct-fp16"
TASK_KEY = "gsm8k"
N_EXAMPLES = 10


def main() -> None:
    _, results_a = run_eval(
        MODEL_KEY, TASK_KEY, output_dir="results/raw/_determinism_check", limit=N_EXAMPLES
    )
    _, results_b = run_eval(
        MODEL_KEY, TASK_KEY, output_dir="results/raw/_determinism_check", limit=N_EXAMPLES
    )

    serialized_a = json.dumps(results_a, sort_keys=True, default=str)
    serialized_b = json.dumps(results_b, sort_keys=True, default=str)

    if serialized_a != serialized_b:
        print("DETERMINISM CHECK FAILED: two runs of the same fp16 config on the")
        print(f"same {N_EXAMPLES} gsm8k examples with the same seed produced different results.")
        print("Run A:", serialized_a)
        print("Run B:", serialized_b)
        sys.exit(1)

    print(f"Determinism check passed: {N_EXAMPLES} gsm8k examples, identical results across two runs.")


if __name__ == "__main__":
    main()
