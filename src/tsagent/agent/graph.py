"""LangGraph agent: profile a series, read its backtests, decide conservatively, explain."""

from __future__ import annotations

from typing import TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from tsagent.agent.decision import DEFAULT_PREFERENCE, Decision, decide
from tsagent.agent.narrate import (
    Narrator,
    TemplateNarrator,
    build_facts,
    find_unsupported_numbers,
)
from tsagent.agent.profile import profile_series
from tsagent.agent.providers import BacktestProvider


class AgentState(TypedDict, total=False):
    """Shared state. Node names below differ from these keys on purpose: as far as I
    know, a node and a state key with the same name conflict in LangGraph."""

    series_id: str
    profile: dict
    candidates: list[str]
    window_mase: pd.DataFrame
    decision: Decision
    facts: dict
    explanation: str
    problems: list[str]
    used_fallback: bool


def build_agent(
    provider: BacktestProvider,
    narrator: Narrator | None = None,
    horizon: int = 28,
    preference: tuple[str, ...] = DEFAULT_PREFERENCE,
    **rule_kwargs: float,
):
    """Build and compile the agent graph.

    Args:
        provider: where series and backtest scores come from.
        narrator: writes the explanation; defaults to the rule-based TemplateNarrator.
        horizon: forecast horizon of the backtests (only used in the explanation text).
        preference: model order for the default choice.
        rule_kwargs: thresholds passed to `decide` (min_windows, min_win_share, min_gain).
    """
    narrator = narrator or TemplateNarrator()
    fallback = TemplateNarrator()

    def profile_node(state: AgentState) -> dict:
        return {"profile": profile_series(provider.series(state["series_id"]))}

    def plan_node(state: AgentState) -> dict:
        # Deterministic for now: every model that has results. An LLM planner could
        # narrow this list; the rest of the graph would not change.
        return {"candidates": provider.models()}

    def backtest_node(state: AgentState) -> dict:
        table = provider.window_mase(state["series_id"], state["candidates"])
        return {"window_mase": table}

    def decide_node(state: AgentState) -> dict:
        return {"decision": decide(state["window_mase"], preference, **rule_kwargs)}

    def explain_node(state: AgentState) -> dict:
        facts = build_facts(state["series_id"], state["profile"], state["decision"], horizon)
        return {"facts": facts, "explanation": narrator.explain(facts)}

    def validate_node(state: AgentState) -> dict:
        names = [state["series_id"], *state["facts"]["mean_mase"]]
        problems = find_unsupported_numbers(state["explanation"], state["facts"], mask=names)
        return {"problems": problems}

    def route(state: AgentState) -> str:
        return "fallback" if state["problems"] else END

    def fallback_node(state: AgentState) -> dict:
        return {"explanation": fallback.explain(state["facts"]), "used_fallback": True}

    graph = StateGraph(AgentState)
    graph.add_node("profile_series", profile_node)
    graph.add_node("plan_models", plan_node)
    graph.add_node("run_backtests", backtest_node)
    graph.add_node("decide_model", decide_node)
    graph.add_node("explain", explain_node)
    graph.add_node("validate", validate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "profile_series")
    graph.add_edge("profile_series", "plan_models")
    graph.add_edge("plan_models", "run_backtests")
    graph.add_edge("run_backtests", "decide_model")
    graph.add_edge("decide_model", "explain")
    graph.add_edge("explain", "validate")
    graph.add_conditional_edges("validate", route, ["fallback", END])
    graph.add_edge("fallback", END)
    return graph.compile()