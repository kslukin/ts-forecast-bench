import pandas as pd

from tsagent.agent.decision import decide
from tsagent.agent.graph import build_agent
from tsagent.agent.narrate import TemplateNarrator, find_unsupported_numbers
from tsagent.agent.profile import profile_series


def make_table(**columns):
    n = len(next(iter(columns.values())))
    return pd.DataFrame(columns, index=pd.RangeIndex(n, name="window"))


def make_series(values):
    return pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=len(values)), "y": values})


def test_default_is_kept_when_the_gain_is_small():
    table = make_table(timesfm3=[1.0] * 8, lightgbm=[0.98] * 8)
    decision = decide(table)
    assert decision.chosen == "timesfm3"
    assert decision.reason == "default_kept"


def test_switch_when_a_challenger_is_clearly_better():
    table = make_table(timesfm3=[1.0] * 8, lightgbm=[0.8] * 8, seasonal_naive=[1.2] * 8)
    decision = decide(table)
    assert decision.chosen == "lightgbm"
    assert decision.reason == "challenger_clearly_better"
    assert abs(decision.challenger_gain - 0.2) < 1e-9


def test_no_switch_when_the_challenger_wins_too_rarely():
    # Better by a lot in 3 of 8 windows: the mean gain is large, the win share is not.
    table = make_table(timesfm3=[1.0] * 8, lightgbm=[0.2] * 3 + [1.1] * 5)
    assert decide(table).chosen == "timesfm3"


def test_too_few_windows_keeps_the_default():
    decision = decide(make_table(timesfm3=[1.0] * 3, lightgbm=[0.5] * 3))
    assert decision.chosen == "timesfm3"
    assert decision.reason == "too_few_windows"


def test_default_follows_the_preference_when_the_top_model_is_missing():
    decision = decide(make_table(chronos2=[1.0] * 8, lightgbm=[1.0] * 8))
    assert decision.default == "chronos2"


def test_profile_labels():
    short = profile_series(make_series([0.0] * 100))
    assert short["length"] == "short"
    assert short["intermittency"] == "intermittent"
    long = profile_series(make_series([1.0] * 400))
    assert long["length"] == "long"
    assert long["intermittency"] == "regular"
    assert long["last_date"] == "2021-02-03"


def test_validator_accepts_numbers_from_facts_and_ignores_names():
    facts = {"gain": 0.123, "n": 8, "mean_mase": {"timesfm3": 0.8}}
    text = "Series A_1_CA_1 gain 0.123 over 8 windows, timesfm3 0.800 on 2020-01-02."
    assert find_unsupported_numbers(text, facts, mask=["A_1_CA_1", "timesfm3"]) == []


def test_validator_flags_an_invented_number():
    facts = {"gain": 0.123, "mean_mase": {"timesfm3": 0.8}}
    bad = find_unsupported_numbers("Accuracy improves by 37.5%.", facts, mask=[])
    assert bad == ["37.5"]


class FakeProvider:
    def __init__(self, table):
        self.table = table

    def series_ids(self):
        return ["S1"]

    def series(self, series_id):
        return make_series([float(i % 5) for i in range(400)])

    def models(self):
        return list(self.table.columns)

    def window_mase(self, series_id, models):
        return self.table[models]


def run(table, narrator=None):
    agent = build_agent(FakeProvider(table), narrator=narrator)
    return agent.invoke({"series_id": "S1", "used_fallback": False})


def test_agent_switches_only_when_the_challenger_is_clear():
    result = run(make_table(timesfm3=[1.0] * 8, lightgbm=[0.8] * 8))
    assert result["decision"].chosen == "lightgbm"
    assert "lightgbm" in result["explanation"]
    assert result["used_fallback"] is False


def test_agent_keeps_the_default_otherwise():
    result = run(make_table(timesfm3=[1.0] * 8, lightgbm=[0.98] * 8))
    assert result["decision"].chosen == "timesfm3"
    assert "kept" in result["explanation"]


def test_an_invented_number_triggers_the_template_fallback():
    class LyingNarrator:
        def explain(self, facts):
            return "The chosen model improves accuracy by 37.5% overall."

    table = make_table(timesfm3=[1.0] * 8, lightgbm=[0.8] * 8)
    result = run(table, narrator=LyingNarrator())
    assert result["used_fallback"] is True
    assert result["explanation"] == TemplateNarrator().explain(result["facts"])
    names = ["S1", *result["facts"]["mean_mase"]]
    assert find_unsupported_numbers(result["explanation"], result["facts"], mask=names) == []