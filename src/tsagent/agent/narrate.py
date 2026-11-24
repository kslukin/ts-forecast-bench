"""Turn a decision into plain-language text, and check that the text invents no numbers."""

from __future__ import annotations

import re
from typing import Protocol

from tsagent.agent.decision import Decision

NUMBER = re.compile(r"\d+(?:\.\d+)?")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CAVEAT = "Caveat: differences between the leading models are often within noise."


class Narrator(Protocol):
    """Anything that turns a facts dict into text; an LLM-backed narrator fits here too."""

    def explain(self, facts: dict) -> str:
        ...


def build_facts(series_id: str, profile: dict, decision: Decision, horizon: int) -> dict:
    """Collect every number the explanation may mention. The validator allows only these."""
    share = decision.challenger_win_share
    return {
        "series_id": series_id,
        "horizon": horizon,
        "n_obs": profile["n_obs"],
        "recent_days": profile["recent_days"],
        "zero_pct": profile["zero_share_recent"] * 100,
        "level_recent": profile["level_recent"],
        "length": profile["length"],
        "intermittency": profile["intermittency"],
        "n_models": len(decision.mean_mase),
        "n_windows": decision.n_windows,
        "mean_mase": decision.mean_mase,
        "default": decision.default,
        "chosen": decision.chosen,
        "reason": decision.reason,
        "challenger": decision.challenger,
        "win_pct": None if share is None else share * 100,
        "gain": decision.challenger_gain,
        "min_windows": decision.min_windows,
        "min_win_share_pct": decision.min_win_share * 100,
        "min_gain": decision.min_gain,
    }


class TemplateNarrator:
    """Rule-based explanation. Every number in the text is taken from `facts`."""

    def explain(self, facts: dict) -> str:
        f = facts
        means = ", ".join(f"{name} {value:.3f}" for name, value in f["mean_mase"].items())
        history = (
            f"Series {f['series_id']}: {f['n_obs']} days of history ({f['length']}), "
            f"{f['zero_pct']:.0f}% zero days in the last {f['recent_days']} days "
            f"({f['intermittency']})."
        )
        backtest = (
            f"I compared {f['n_models']} models over {f['n_windows']} rolling windows of "
            f"{f['horizon']} days. Mean MASE, lower is better: {means}."
        )
        return " ".join([history, backtest, self._decision(f), CAVEAT])

    @staticmethod
    def _decision(f: dict) -> str:
        reason = f["reason"]
        if reason == "challenger_clearly_better":
            return (
                f"I chose {f['chosen']} instead of the default {f['default']}: it was better "
                f"in {f['win_pct']:.0f}% of the windows and its mean MASE was lower by "
                f"{f['gain']:.3f}."
            )
        if reason == "too_few_windows":
            return (
                f"Only {f['n_windows']} windows are available, fewer than the "
                f"{f['min_windows']} needed to justify a switch, so I kept the default "
                f"{f['default']}."
            )
        if reason == "only_default":
            return (
                f"Only {f['default']} has results for this series, "
                "so there is nothing to compare."
            )
        direction = "lower" if f["gain"] >= 0 else "higher"
        return (
            f"I kept the default {f['default']}: no challenger cleared the bar (better in at "
            f"least {f['min_win_share_pct']:.0f}% of the windows and a lower mean MASE by at "
            f"least {f['min_gain']:.2f}). The closest, {f['challenger']}, was better in "
            f"{f['win_pct']:.0f}% of the windows, and its mean MASE was {direction} by "
            f"{abs(f['gain']):.3f}."
        )


def _numbers(value: object) -> list[float]:
    """All numbers stored anywhere in `value` (nested dicts and lists included)."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _numbers(v)]
    if isinstance(value, list | tuple):
        return [n for v in value for n in _numbers(v)]
    return []


def find_unsupported_numbers(text: str, facts: dict, mask: list[str]) -> list[str]:
    """Numbers in `text` that match no number in `facts` (compared at the text's precision).

    Names listed in `mask` (series id, model names) and ISO dates are removed first, because
    their digits are not claims. Limit: this checks that a number exists in the facts, not
    that the sentence uses it correctly.
    """
    for name in sorted(mask, key=len, reverse=True):
        text = text.replace(name, " ")
    text = ISO_DATE.sub(" ", text)
    allowed = [abs(x) for x in _numbers(facts)]
    unsupported = []
    for token in NUMBER.findall(text):
        decimals = len(token.split(".")[1]) if "." in token else 0
        value = float(token)
        if not any(abs(round(a, decimals) - value) < 1e-9 for a in allowed):
            unsupported.append(token)
    return unsupported