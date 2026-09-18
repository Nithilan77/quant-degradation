"""Compute per-capability-axis degradation relative to the fp16 baseline,
and plot it.

Reads the tidy dataframe src/aggregate.py writes (model, size, scheme,
bit_width, task, capability_axis, metric, value). Within each (size, task)
group, the fp16 row is the baseline; every other scheme+bit_width row in
that group gets:

    abs_drop      = baseline_value - value
    rel_drop_pct  = abs_drop / baseline_value * 100

Degradation is never averaged across capability axes -- CLAUDE.md's core
claim is that aggregate accuracy is the wrong unit of analysis here, so
the figure keeps capability axis on the x-axis and one bar group per
scheme+bit-width, not a single collapsed number.

No `torch` import -- laptop-side, like aggregate.py.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASELINE_SCHEME = "fp16"

DEGRADATION_COLUMNS = [
    "size",
    "task",
    "capability_axis",
    "scheme",
    "bit_width",
    "baseline_value",
    "value",
    "abs_drop",
    "rel_drop_pct",
]

# Validated categorical palette (dataviz skill, references/palette.md),
# light mode. Assigned in fixed order by scheme -- never cycled/generated --
# so a given scheme keeps the same color across figures. "fp16" itself never
# appears here since it's the baseline every bar is measured against, not a
# plotted series.
SCHEME_COLORS = {
    "bnb-nf4": "#2a78d6",  # slot 1 blue
    "awq": "#eb6834",  # slot 2 orange
    "gptq": "#1baf7a",  # slot 3 aqua -- not used yet (GPTQ not added), reserved
}
FALLBACK_COLORS = ["#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE_AXIS = "#c3c2b7"


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def series_label(scheme: str, bit_width) -> str:
    if pd.isna(bit_width):
        return scheme
    return f"{scheme} ({int(bit_width)}-bit)"


def compute_degradation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size, task), group in df.groupby(["size", "task"]):
        baseline_rows = group[group["scheme"] == BASELINE_SCHEME]
        if baseline_rows.empty:
            raise ValueError(
                f"No {BASELINE_SCHEME!r} baseline for size={size!r} task={task!r}; "
                "cannot compute degradation without one."
            )
        if len(baseline_rows) > 1:
            raise ValueError(
                f"Multiple {BASELINE_SCHEME!r} baselines for size={size!r} task={task!r}; "
                "expected exactly one."
            )
        baseline_row = baseline_rows.iloc[0]
        baseline_value = baseline_row["value"]
        capability_axis = baseline_row["capability_axis"]

        for _, row in group.iterrows():
            if row["scheme"] == BASELINE_SCHEME:
                continue
            abs_drop = baseline_value - row["value"]
            rel_drop_pct = (abs_drop / baseline_value * 100) if baseline_value else float("nan")
            rows.append(
                {
                    "size": size,
                    "task": task,
                    "capability_axis": capability_axis,
                    "scheme": row["scheme"],
                    "bit_width": row["bit_width"],
                    "baseline_value": baseline_value,
                    "value": row["value"],
                    "abs_drop": abs_drop,
                    "rel_drop_pct": rel_drop_pct,
                }
            )

    return pd.DataFrame(rows, columns=DEGRADATION_COLUMNS)


def color_for_scheme(scheme: str, seen: dict) -> str:
    if scheme in SCHEME_COLORS:
        return SCHEME_COLORS[scheme]
    if scheme not in seen:
        seen[scheme] = FALLBACK_COLORS[len(seen) % len(FALLBACK_COLORS)]
    return seen[scheme]


def plot_degradation(degradation_df: pd.DataFrame, metric_label: str, value_col: str, ylabel: str):
    axes = sorted(degradation_df["capability_axis"].unique())
    # (scheme, bit_width) pairs, in first-seen order, fixed for the whole
    # figure -- same series gets the same bar position/color everywhere.
    series = list(
        degradation_df[["scheme", "bit_width"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=CHART_SURFACE)
    ax.set_facecolor(CHART_SURFACE)

    n_series = max(len(series), 1)
    group_width = 0.8
    bar_width = group_width / n_series
    x = range(len(axes))
    fallback_seen: dict = {}

    for i, (scheme, bit_width) in enumerate(series):
        label = series_label(scheme, bit_width)
        color = color_for_scheme(scheme, fallback_seen)
        offsets = [xi - group_width / 2 + bar_width * i + bar_width / 2 for xi in x]
        values = []
        for axis in axes:
            match = degradation_df[
                (degradation_df["capability_axis"] == axis)
                & (degradation_df["scheme"] == scheme)
                & (degradation_df["bit_width"] == bit_width)
            ]
            values.append(match.iloc[0][value_col] if not match.empty else float("nan"))
        ax.bar(offsets, values, width=bar_width * 0.9, label=label, color=color, zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(axes, color=PRIMARY_INK)
    ax.set_ylabel(ylabel, color=PRIMARY_INK)
    ax.set_title(
        f"Capability degradation vs. fp16 baseline (metric: {metric_label})",
        color=PRIMARY_INK,
        loc="left",
    )
    ax.axhline(0, color=BASELINE_AXIS, linewidth=1, zorder=2)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE_AXIS)
    ax.tick_params(colors=MUTED_INK)
    ax.legend(frameon=False, labelcolor=SECONDARY_INK)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/processed/aggregated.csv",
        help="Tidy dataframe CSV from src/aggregate.py.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/processed/degradation.csv",
        help="Where to write the per-axis degradation table.",
    )
    parser.add_argument(
        "--figure",
        default="figures/degradation.png",
        help="Where to save the degradation figure.",
    )
    parser.add_argument(
        "--value",
        choices=["abs_drop", "rel_drop_pct"],
        default="rel_drop_pct",
        help="Which degradation quantity to plot (default: rel_drop_pct).",
    )
    args = parser.parse_args()

    df = load_dataframe(args.input)
    if df.empty:
        raise SystemExit(f"{args.input} has no rows; run src/aggregate.py first.")

    metric_labels = df["metric"].unique()
    if len(metric_labels) > 1:
        raise ValueError(
            f"Input dataframe mixes metric variants {sorted(metric_labels)}; "
            "re-run src/aggregate.py with a single --metric-filter so degradation "
            "is computed on a consistent metric."
        )
    metric_label = metric_labels[0]

    degradation_df = compute_degradation(df)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    degradation_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(degradation_df)} rows to {out_csv}")
    print(degradation_df.to_string(index=False))

    ylabel = (
        "% relative drop vs. fp16" if args.value == "rel_drop_pct" else "absolute drop vs. fp16"
    )
    fig = plot_degradation(degradation_df, metric_label, args.value, ylabel)

    out_fig = Path(args.figure)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote figure to {out_fig}")


if __name__ == "__main__":
    main()
