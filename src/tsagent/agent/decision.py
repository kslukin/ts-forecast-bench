"""Conservative, deterministic choice of a model from one series' backtest results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Benchmark order: the model that was marginally best on average comes first.
DEFAULT_PREFERENCE = ("timesfm3", "chronos2", "lightgbm", "seasonal_naive")


@dataclass(frozen=True)
class Decision:
    chosen: str
    default: str
    reason: str  # default_kept | challenger_clearly_better | too_few_windows | only_default
    n_windows: int
    mean_mase: dict[str, float]
    min_windows: int
    min_win_share: float
    min_gain: float
    challenger: str | None = None
    challenger_win_share: float | None = None
    challenger_gain: float | None = None  # default minus challenger mean MASE (> 0: better)


def decide(
    window_mase: pd.DataFrame,
    preference: tuple[str, ...] = DEFAULT_PREFERENCE,
    min_windows: int = 4,
    min_win_share: float = 0.75,
    min_gain: float = 0.05,
) -> Decision:
    """Keep the default model unless a challenger is clearly and consistently better.

    The thresholds are uncalibrated starting values: they still have to be tuned
    on the saved benchmark results.

    Args:
        window_mase: rows are backtest windows, columns are models (MASE of ONE series).
        preference: the first model of this list that has results is the default.
    """
    models = [m for m in preference if m in window_mase.columns]
    if not models:
        raise ValueError("none of the preferred models has backtest results")
    table = window_mase[models].dropna()  # windows scored by every model
    if table.empty:
        raise ValueError("no backtest window is scored by all models")

    default = models[0]
    means = {m: float(table[m].mean()) for m in models}
    base = {
        "default": default,
        "n_windows": len(table),
        "mean_mase": means,
        "min_windows": min_windows,
        "min_win_share": min_win_share,
        "min_gain": min_gain,
    }
    if len(models) == 1:
        return Decision(chosen=default, reason="only_default", **base)

    challenger = min((m for m in models if m != default), key=means.get)
    gain = means[default] - means[challenger]
    share = float((table[challenger] < table[default]).mean())
    details = {
        "challenger": challenger,
        "challenger_win_share": share,
        "challenger_gain": gain,
    }
    if len(table) < min_windows:
        return Decision(chosen=default, reason="too_few_windows", **base, **details)
    if gain >= min_gain and share >= min_win_share:
        return Decision(chosen=challenger, reason="challenger_clearly_better", **base, **details)
    return Decision(chosen=default, reason="default_kept", **base, **details)