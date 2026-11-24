"""Calibrate the agent's switching thresholds on saved benchmark results."""

import argparse

import pandas as pd

from tsagent.agent.decision import DEFAULT_PREFERENCE
from tsagent.analysis.calibration import evaluate_rule, grid_search
from tsagent.analysis.compare import bootstrap_mean_ci, pivot_mase


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="results/m5_n500_w8")
    parser.add_argument("--start", type=int, default=4)
    args = parser.parse_args()

    mase = pivot_mase(pd.read_csv(f"{args.prefix}_per_series.csv"))
    last = mase.index.get_level_values("window").max()

    base = evaluate_rule(mase, args.start)
    print(f"{len(base)} (window, series) pairs, windows {args.start} to {last}")
    print(f"always {DEFAULT_PREFERENCE[0]}: MASE {base['default_mase'].mean():.3f}")

    # Reference: always take the model with the lowest mean past MASE, no thresholds at all.
    diff = (base["free_mase"] - base["default_mase"]).groupby(base["unique_id"]).mean()
    mean, low, high = bootstrap_mean_ci(diff.to_numpy())
    print(f"no thresholds, best past model: {mean:+.4f} (95% CI {low:+.4f} to {high:+.4f})")

    grid = grid_search(
        mase,
        shares=[0.5, 0.625, 0.75, 0.875, 1.0],
        gains=[0.0, 0.02, 0.05, 0.1],
        start=args.start,
    )
    pd.set_option("display.width", 140)
    print("\nRule minus always-default MASE (negative = rule better), bootstrap over series:")
    print(grid.round(4).to_string(index=False))

    better = int((grid["ci_high"] < 0).sum())
    print(f"\ncells where the whole interval is below zero: {better} of {len(grid)}")
    grid.to_csv(f"{args.prefix}_rule_grid.csv", index=False)


if __name__ == "__main__":
    main()