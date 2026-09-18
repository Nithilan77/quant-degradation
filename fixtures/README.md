# fixtures/

Synthetic, `results/raw/`-shaped JSON for exercising `src/aggregate.py` and
`src/analyze.py` without a GPU. **Not real eval output** -- no number in
here is a research result. Never copy these files into `results/raw/`.

Regenerate with:

```bash
python fixtures/generate_synthetic_results.py
```

Output lands in `fixtures/synthetic_raw/`, one file per (model, task),
named the same way `src/run_eval.py` names real output
(`<model_key>__<task_key>.json`) so it exercises the exact same
aggregate/analyze code path as real data -- just pointed at a different
directory via `--raw-dir`.
