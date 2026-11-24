import pandas as pd
import pytest

from tsagent.analysis.calibration import evaluate_rule, grid_search


def make_mase(per_series):
    frames = []
    for uid, models in per_series.items():
        frame = pd.DataFrame(models, index=pd.RangeIndex(8, name="window"))
        frame["unique_id"] = uid
        frames.append(frame.reset_index().set_index(["window", "unique_id"]))
    return pd.concat(frames)


def test_rule_does_not_see_the_window_it_is_scored_on():
    # lightgbm is much better only in window 4. The rule sees windows 0-3 at that point,
    # so it must still use the default there.
    mase = make_mase({"s1": {"timesfm3": [1.0] * 8, "lightgbm": [1.1] * 4 + [0.1] + [1.1] * 3}})
    out = evaluate_rule(mase, start=4).set_index("window")
    assert out.loc[4, "rule_mase"] == pytest.approx(1.0)
    assert not out.loc[4, "switched"]


def test_rule_switches_when_the_challenger_is_consistently_better():
    mase = make_mase({"s1": {"timesfm3": [1.0] * 8, "lightgbm": [0.5] * 8}})
    out = evaluate_rule(mase, start=4)
    assert len(out) == 4  # windows 4 to 7
    assert out["switched"].all()
    assert out["rule_mase"].tolist() == [0.5] * 4


def test_series_without_history_are_skipped_and_short_history_keeps_the_default():
    mase = make_mase(
        {
            "s1": {"timesfm3": [1.0] * 8, "lightgbm": [1.1] * 8},
            "s2": {"timesfm3": [1.0] * 8, "lightgbm": [0.5] * 8},
        }
    )
    mase = mase.drop(index=[(w, "s2") for w in range(6)])  # s2 only has windows 6 and 7
    out = evaluate_rule(mase, start=4)
    s2 = out[out["unique_id"] == "s2"]
    assert s2["window"].tolist() == [7]  # window 6 has no earlier window to learn from
    assert not s2["switched"].any()  # one past window is too few to justify a switch


def test_grid_has_one_row_per_threshold_pair():
    mase = make_mase(
        {
            "s1": {"timesfm3": [1.0] * 8, "lightgbm": [0.5] * 8},
            "s2": {"timesfm3": [1.0] * 8, "lightgbm": [1.1] * 8},
        }
    )
    grid = grid_search(mase, shares=[0.5, 0.75], gains=[0.0, 0.05], start=4, n_boot=50)
    assert len(grid) == 4
    assert {"switch_rate", "rule_minus_default", "ci_low", "ci_high"} <= set(grid.columns)