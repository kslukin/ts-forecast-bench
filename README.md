# Time-series forecasting benchmark

Honest comparison of modern time-series forecasting models, with a LangGraph agent on top (work in progress).

The goal is a reproducible project that shows (1) how to compare forecasting models fairly, (2) how to use time-series foundation models in zero-shot mode, and (3) how to build an agent whose decisions and numbers come from deterministic code, with a validator that rejects any number in the explanation that code did not compute (an LLM narrator can be plugged in later).

## Status

Work in progress.

- [x] M5 loader, rolling-origin backtest without look-ahead, metrics (MASE, scaled quantile loss, interval coverage) with unit tests
- [x] Models behind one interface: seasonal naive, LightGBM (direct multi-horizon quantiles), Chronos-2, TimesFM 3.0
- [x] Bootstrap confidence intervals, per-segment analysis, model-selection analysis
- [x] Docker image for the foundation models (CPU)
- [x] LangGraph agent, first version (deterministic nodes, template explanations checked by a number validator)
- [ ] Calibrate the agent's switching rule on the saved results
- [ ] Second dataset (German hourly electricity load, SMARD)
- [ ] Streamlit demo
- [ ] LLM narrator (optional)
- [ ] German README

## Preliminary results

M5 (Walmart), random sample of 500 series, 8 rolling windows of 28 days. Lower MASE is better; the nominal coverage of the 10–90% interval is 0.80.

| Model | MASE | Scaled quantile loss | Interval coverage |
|---|---|---|---|
| TimesFM 3.0 (zero-shot) | 0.760 | 0.260 | 0.831 |
| Chronos-2 (zero-shot) | 0.763 | 0.262 | 0.786 |
| LightGBM | 0.766 | 0.262 | 0.891 |
| Seasonal naive | 1.062 | 0.368 | 0.870 |

- All three modern models beat the seasonal-naive baseline by about 0.3 MASE; the paired bootstrap intervals (resampling series) are far from zero.
- TimesFM, Chronos-2 and LightGBM are practically tied: the mean differences are at most 0.006 MASE, while the MASE of any single model varies between about 0.72 and 0.83 from window to window.
- Averaged over windows, TimesFM has a slightly lower mean MASE than LightGBM (difference 0.006, 95% bootstrap CI 0.001 to 0.011), yet LightGBM has the lower MASE on 63% of the series: it loses on average because of a minority of series where its error is larger.
- Interval calibration: Chronos-2 is closest to the nominal 0.80 (0.786), TimesFM is at 0.831 and LightGBM's intervals are too wide (0.891).
- Choosing a model per series from past backtest windows did **not** beat always using TimesFM: the MASE was higher by 0.022, 0.012 and 0.009 when 1, 3 or 7 past windows were used (the 95% intervals exclude zero). The best fixed model was picked on the same data, which favours the fixed policy.
- Cost on a laptop CPU (Intel i9, inside Docker), 8 windows x 500 series: 10 s for seasonal naive, 155 s for LightGBM (including retraining), 546 s for Chronos-2 and 748 s for TimesFM 3.0.

Caveats: one dataset and one random sample (seed 0); the foundation models may have seen M5 during pretraining, which cannot be verified; the intervals reflect variation across series, not across time; the segment analysis for short series rests on about 15 series and is not reported.

## Agent (first version)

```
profile_series -> plan_models -> run_backtests -> decide_model -> explain -> validate -> end
                                                                                 \-> fallback -> end
```

- `decide_model` is deterministic and conservative: the default model (TimesFM 3.0) is kept unless a challenger is better in at least 75% of the windows, lowers the mean MASE by at least 0.05, and at least 4 windows exist. These thresholds are starting values and are not calibrated yet.
- `validate` checks that every number in the explanation matches a number computed by code; otherwise the agent falls back to a template text. It checks that a number exists in the facts, not that the sentence uses it correctly.
- The agent currently reads the saved benchmark results; running the backtests live is not implemented yet. There is no LLM in the loop yet.

Example output for one M5 series:

> Series FOODS_1_002_WI_1: 1940 days of history (long), 43% zero days in the last 56 days (regular). I compared 4 models over 8 rolling windows of 28 days. Mean MASE, lower is better: timesfm3 0.852, chronos2 0.933, lightgbm 0.915, seasonal_naive 1.153. I kept the default timesfm3: no challenger cleared the bar (better in at least 75% of the windows and a lower mean MASE by at least 0.05). The closest, lightgbm, was better in 50% of the windows, and its mean MASE was higher by 0.063. Caveat: differences between the leading models are often within noise.

## Method

- Rolling-origin backtest: models only see data up to the cutoff (checked by tests).
- Metrics: MASE (scaled by the in-sample seasonal-naive error), quantile loss scaled the same way, and coverage of the 10–90% interval. Series with a zero scale are excluded and counted.
- Paired comparison with a bootstrap over series, plus a per-segment breakdown (length, intermittency, volume).

## Reproduce

```bash
uv sync
uv run pytest
```

The M5 data is not included. Accept the competition rules on Kaggle, then:

```bash
uv run kaggle competitions download -c m5-forecasting-accuracy -p data/raw
unzip data/raw/m5-forecasting-accuracy.zip -d data/raw/m5
```

Foundation models run inside Docker, because PyTorch has no wheels for Intel Macs:

```bash
docker build -t hail-mary-fm .
docker run --rm -v hail-mary-hf:/models \
  -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" \
  -v "$PWD/src:/app/src" -v "$PWD/scripts:/app/scripts" \
  hail-mary-fm python scripts/run_compare.py --n-series 500 --n-windows 8 --out results/m5_n500_w8
```

Analysis and agent (no Docker needed; the saved results are in `results/`):

```bash
uv run python scripts/analyze_results.py --prefix results/m5_n500_w8 --n-series 500 --n-windows 8
uv run python scripts/analyze_selection.py --prefix results/m5_n500_w8 --lookbacks 1 3 7
uv run python scripts/run_agent.py --series FOODS_1_002_WI_1
```

TimesFM 3.0 weights are released under a non-commercial license; check the model card before reusing them.

## Stack

Python, pandas, LightGBM, PyTorch (CPU), Chronos-2, TimesFM 3.0, LangGraph, Docker, pytest, ruff, uv. Planned: Streamlit.