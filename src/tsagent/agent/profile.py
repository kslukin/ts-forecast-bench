"""Describe one series with simple, deterministic statistics."""

from __future__ import annotations

import pandas as pd

RECENT_DAYS = 56  # window for the recent level and the share of zero days
SHORT_BELOW = 365  # fewer observations than this: "short"
INTERMITTENT_FROM = 0.5  # at least this share of zero days: "intermittent"


def profile_series(df: pd.DataFrame, recent: int = RECENT_DAYS) -> dict:
    """Profile a single series.

    Args:
        df: columns ds and y of ONE series.
        recent: number of last observations used for the recent level and zero share.
    """
    ordered = df.sort_values("ds")
    tail = ordered["y"].tail(recent)
    n_obs = len(ordered)
    zero_share = float((tail == 0).mean())
    return {
        "n_obs": n_obs,
        "recent_days": min(recent, n_obs),
        "zero_share_recent": zero_share,
        "level_recent": float(tail.mean()),
        "length": "short" if n_obs < SHORT_BELOW else "long",
        "intermittency": "intermittent" if zero_share >= INTERMITTENT_FROM else "regular",
        "last_date": ordered["ds"].iloc[-1].date().isoformat(),
    }