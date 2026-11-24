"""Where the agent gets a series and its backtest results from."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class BacktestProvider(Protocol):
    """Source of series and per-window backtest scores.

    CachedProvider reads saved benchmark results. A live provider that runs the models
    on demand can implement the same four methods without any change to the agent graph.
    """

    def series_ids(self) -> list[str]:
        ...

    def series(self, series_id: str) -> pd.DataFrame:
        ...

    def models(self) -> list[str]:
        ...

    def window_mase(self, series_id: str, models: list[str]) -> pd.DataFrame:
        ...


class CachedProvider:
    """Reads the per-series results saved by scripts/run_compare.py."""

    def __init__(self, results_csv: str, long: pd.DataFrame) -> None:
        self._results = pd.read_csv(results_csv)
        self._long = long  # columns unique_id, ds, y

    def series_ids(self) -> list[str]:
        return sorted(self._results["unique_id"].unique())

    def series(self, series_id: str) -> pd.DataFrame:
        rows = self._long[self._long["unique_id"] == series_id]
        if rows.empty:
            raise KeyError(f"unknown series {series_id}")
        return rows[["ds", "y"]].reset_index(drop=True)

    def models(self) -> list[str]:
        return sorted(self._results["model"].unique())

    def window_mase(self, series_id: str, models: list[str]) -> pd.DataFrame:
        rows = self._results[
            (self._results["unique_id"] == series_id) & self._results["model"].isin(models)
        ]
        table = rows.pivot(index="window", columns="model", values="mase")
        return table.dropna()  # keep windows scored by every model
    