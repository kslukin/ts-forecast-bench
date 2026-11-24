"""Run the forecasting agent on one series of the M5 benchmark sample."""

import argparse

from tsagent.agent.graph import build_agent
from tsagent.agent.providers import CachedProvider
from tsagent.data.m5 import load_m5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", default=None, help="unique_id; default: the first one")
    parser.add_argument("--list", action="store_true", help="print some series ids and exit")
    parser.add_argument("--prefix", default="results/m5_n500_w8")
    parser.add_argument("--n-series", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The same sample as in the comparison run: same --n-series and --seed.
    long, _ = load_m5("data/raw/m5", n_series=args.n_series, seed=args.seed)
    provider = CachedProvider(f"{args.prefix}_per_series.csv", long)
    if args.list:
        print("\n".join(provider.series_ids()[:10]))
        return
    series_id = args.series or provider.series_ids()[0]

    agent = build_agent(provider)
    state = agent.invoke({"series_id": series_id, "used_fallback": False})

    decision = state["decision"]
    print(f"series: {series_id}")
    print(f"profile: {state['profile']}")
    print(f"chosen: {decision.chosen} (default {decision.default}, reason: {decision.reason})")
    print(f"\n{state['explanation']}")
    print(f"\nfallback used: {state['used_fallback']}")


if __name__ == "__main__":
    main()