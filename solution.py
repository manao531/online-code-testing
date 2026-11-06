"""OBV bullish-divergence event study.

Signal (no future data): over the 7 trading days ending on day t, the linear
regression slope of close is negative and the slope of OBV is positive.
Performance is the 5th subsequent trading day of the same osid.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

WINDOW = 7
HORIZON = 5
TIME_INDEX = np.arange(WINDOW, dtype=float)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """First day is 0. Later days add volume on an up close and subtract it on a down close."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0.0
    return (direction * volume).cumsum()


def rolling_slope(values: np.ndarray) -> float:
    """Slope of a degree-1 fit of the window against time index 0..6."""
    return float(np.polyfit(TIME_INDEX, values, 1)[0])


def attach_signals(prices: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in prices.groupby("osid", sort=True):
        group = group.sort_values("trade_date").copy()
        if group["trade_date"].duplicated().any():
            raise ValueError(f"duplicate trade_date for osid {group['osid'].iloc[0]}")
        group["obv"] = compute_obv(group["priceclose"], group["volume"])
        group["price_slope"] = (
            group["priceclose"].rolling(WINDOW).apply(rolling_slope, raw=True)
        )
        group["obv_slope"] = group["obv"].rolling(WINDOW).apply(rolling_slope, raw=True)
        group["exit_date"] = group["trade_date"].shift(-HORIZON)
        group["exit_close"] = group["priceclose"].shift(-HORIZON)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def build_events(prices: pd.DataFrame) -> pd.DataFrame:
    signal = (prices["price_slope"] < 0) & (prices["obv_slope"] > 0)
    events = prices.loc[signal, ["trade_date", "osid"]].copy()
    events = events.sort_values(["trade_date", "osid"]).reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1))
    return events.rename(columns={"trade_date": "event_date"})


def build_returns(
    events: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    lookup = prices.set_index(["osid", "trade_date"])
    bench_close = benchmark.set_index("trade_date")["bench_close"]

    rows = []
    skipped = 0
    for event in events.itertuples(index=False):
        row = lookup.loc[(event.osid, event.event_date)]
        if pd.isna(row["exit_date"]):
            skipped += 1
            continue
        exit_date = row["exit_date"]
        if event.event_date not in bench_close.index or exit_date not in bench_close.index:
            raise ValueError(
                f"benchmark missing close for event {event.event_id} "
                f"({event.event_date.date()} -> {pd.Timestamp(exit_date).date()})"
            )
        fwd_return = row["exit_close"] / row["priceclose"] - 1.0
        bench_return = bench_close.loc[exit_date] / bench_close.loc[event.event_date] - 1.0
        rows.append(
            {
                "event_id": int(event.event_id),
                "event_date": event.event_date,
                "osid": int(event.osid),
                "exit_date": exit_date,
                "fwd_return_5d": float(fwd_return),
                "bench_return_5d": float(bench_return),
                "alpha_5d": float(fwd_return - bench_return),
            }
        )

    returns = pd.DataFrame(rows, columns=[
        "event_id",
        "event_date",
        "osid",
        "exit_date",
        "fwd_return_5d",
        "bench_return_5d",
        "alpha_5d",
    ])
    return returns, skipped


def _stat_block(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "positive_share": float((series > 0).mean()),
    }


def build_summary(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    skipped: int,
    metadata: pd.DataFrame,
) -> dict:
    n_events = int(len(events))
    n_evaluated = int(len(returns))
    if n_events != n_evaluated + skipped:
        raise ValueError("event counts do not add up")

    evaluated = returns.merge(metadata[["osid", "sector", "sector_group"]], on="osid", how="left")
    if evaluated["sector_group"].isna().any():
        raise ValueError("event osid missing from stock_metadata")

    sector_rows = []
    for sector_group, group in evaluated.groupby("sector_group", sort=True):
        sectors = sorted(group["sector"].unique())
        sector_rows.append(
            {
                "sector_group": int(sector_group),
                "sectors": sectors,
                "n_evaluated": int(len(group)),
                "mean_fwd_return_5d": float(group["fwd_return_5d"].mean()),
                "median_fwd_return_5d": float(group["fwd_return_5d"].median()),
                "mean_alpha_5d": float(group["alpha_5d"].mean()),
                "median_alpha_5d": float(group["alpha_5d"].median()),
                "effectiveness": float((group["alpha_5d"] > 0).mean()),
            }
        )

    alpha = returns["alpha_5d"]
    effectiveness = float((alpha > 0).mean()) if n_evaluated else float("nan")
    return {
        "n_events": n_events,
        "n_evaluated": n_evaluated,
        "n_skipped": int(skipped),
        "fwd_return_5d": _stat_block(returns["fwd_return_5d"]),
        "bench_return_5d": _stat_block(returns["bench_return_5d"]),
        "alpha_5d": _stat_block(alpha),
        "event_effectiveness": effectiveness,
        "by_sector_group": sector_rows,
        "definitions": {
            "event": "7-day price slope < 0 and 7-day OBV slope > 0; every such day is an event",
            "fwd_return_5d": "close[t+5] / close[t] - 1 on the same osid trading calendar",
            "bench_return_5d": "bench_close[t+5] / bench_close[t] - 1 on the same dates",
            "alpha_5d": "fwd_return_5d - bench_return_5d",
            "event_effectiveness": "share of evaluated events with alpha_5d > 0",
            "skipped": "events with fewer than 5 subsequent trading days; omitted from event_returns",
        },
    }


def _format_dates(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
    return out


def main() -> None:
    prices = pd.read_csv(DATA_DIR / "daily_prices.csv", parse_dates=["trade_date"])
    benchmark = pd.read_csv(DATA_DIR / "benchmark_returns.csv", parse_dates=["trade_date"])
    metadata = pd.read_csv(DATA_DIR / "stock_metadata.csv")

    prices = attach_signals(prices)
    events = build_events(prices)
    returns, skipped = build_returns(events, prices, benchmark)
    summary = build_summary(events, returns, skipped, metadata)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _format_dates(events, ["event_date"]).to_csv(OUTPUT_DIR / "events.csv", index=False)
    _format_dates(returns, ["event_date", "exit_date"]).to_csv(
        OUTPUT_DIR / "event_returns.csv", index=False
    )
    with (OUTPUT_DIR / "performance_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"events={summary['n_events']} evaluated={summary['n_evaluated']} skipped={summary['n_skipped']}")
    print(f"effectiveness={summary['event_effectiveness']:.6f}")
    print(f"mean_alpha={summary['alpha_5d']['mean']:.6f}")


if __name__ == "__main__":
    main()
