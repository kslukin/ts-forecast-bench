"""Evaluate the agent's switching rule against always using the default model."""

from __future__ import annotations

import pandas as pd

from tsagent.agent.decision import decide
from tsagent.analysis.compare import bootstrap_mean_ci


def evaluate_rule(mase: pd.DataFrame, start: int = 4, **rule_kwargs: float) -> pd.DataFrame:
    """Apply `decide` to every series and every window j >= start, using only windows before j.

    Window j starts where the test period of window j-1 ends, so everything the rule sees
    was already known at the cutoff of window j: no look-ahead.

    Args:
        mase: rows (window, unique_id), one column per model, no missing values.
        start: first window that is scored; also the number of past windows the rule needs.
        rule_kwargs: thresholds for `decide` (min_win_share, min_gain).

    Returns:
        One row per (window, series): MASE of the default model, of the rule's choice and of
        the unrestricted choice (lowest mean past MASE), and whether the rule switched.
    """
    rows = []
    for uid, table in mase.groupby(level="unique_id"):
        table = table.droplevel("unique_id")  # rows are windows now
        for j in table.index[table.index >= start]:
            history = table.loc[table.index < j]
            if history.empty:
                continue  # the series had no earlier window to learn from
            decision = decide(history, min_windows=start, **rule_kwargs)
            free = min(decision.mean_mase, key=decision.mean_mase.get)
            rows.append(
                {
                    "window": j,
                    "unique_id": uid,
                    "default_mase": table.loc[j, decision.default],
                    "rule_mase": table.loc[j, decision.chosen],
                    "free_mase": table.loc[j, free],
                    "switched": decision.chosen != decision.default,
                }
            )
    return pd.DataFrame(rows)


def grid_search(
    mase: pd.DataFrame,
    shares: list[float],
    gains: list[float],
    start: int = 4,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Rule minus default MASE for every threshold pair, with a bootstrap over series.

    A negative difference means the rule is better than always using the default. The
    series is the unit that gets resampled, because its windows are not independent.
    """
    rows = []
    for share in shares:
        for gain in gains:
            res = evaluate_rule(mase, start, min_win_share=share, min_gain=gain)
            diff = (res["rule_mase"] - res["default_mase"]).groupby(res["unique_id"]).mean()
            mean, low, high = bootstrap_mean_ci(diff.to_numpy(), n_boot, seed)
            rows.append(
                {
                    "min_win_share": share,
                    "min_gain": gain,
                    "switch_rate": float(res["switched"].mean()),
                    "rule_minus_default": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)