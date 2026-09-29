#!/usr/bin/env python3
"""
Stage 2 - Features and candidate entries for strategies A and B.

  A. VWAP mean reversion: 5m bar touches VWAP -k sigma band, closes back inside,
     RSI(14) on 5m oversold and turning up (short side mirrored).
  B. 15-minute opening range: 5m close outside the range, then a retest of the
     broken edge within N minutes, with no 5m close back inside (short mirrored).

No look-ahead: 5m/15m values are used only from bar_end onward, entries are filled
on the first 1-second bar at/after the decision time.
Outcomes (win/loss, R) are Stage 3 - this stage only produces events + features.

Input : Parquet folder from build_dataset.py (naive ET, RTH, 1-second OHLCV)
Output: bars_5m / bars_15m / orb_daily / events_A / events_B (.parquet) in --out

Usage:
    python features_ab.py --data ~/nq_data/clean --out ~/nq_data/features
    python features_ab.py --data ... --out ... --a-k 2.5 --b-retest-ticks 8
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, fields
from datetime import time
from pathlib import Path

import numpy as np
import polars as pl

# join_asof(by="day") cannot verify sortedness; inputs are sorted explicitly below
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")


@dataclass
class Params:
    tick: float = 0.25
    rsi_len: int = 14
    # --- A: VWAP mean reversion
    a_k: float = 2.0              # band touched: VWAP -/+ k*sigma
    a_rsi_long: float = 35.0      # 5m RSI below this (long)
    a_rsi_short: float = 65.0     # 5m RSI above this (short)
    a_start: str = "09:45"
    a_end: str = "11:30"
    # --- B: opening range break + retest
    or_minutes: int = 15
    b_retest_ticks: int = 4       # retest zone: within N ticks of the broken edge
    b_min_ext_ticks: int = 8      # price must first extend N ticks beyond the edge
    b_window_min: int = 30        # retest must happen within N minutes of the break
    b_last_break: str = "11:30"   # ignore breaks that close after this time
    or_pctl_days: int = 60        # lookback for opening-range width percentile


def _t(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


# --------------------------------------------------------------------------- #
# 1-second layer
# --------------------------------------------------------------------------- #
def load(data: Path) -> pl.DataFrame:
    df = pl.read_parquet(data / "*.parquet").sort("ts")
    return df.with_columns(day=pl.col("ts").dt.date())


def add_vwap(df: pl.DataFrame, tick: float) -> pl.DataFrame:
    """Session VWAP + volume-weighted sigma (typical price), reset daily."""
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    df = df.with_columns(
        _cv=pl.col("volume").cum_sum().over("day"),
        _cpv=(tp * pl.col("volume")).cum_sum().over("day"),
        _cpv2=(tp * tp * pl.col("volume")).cum_sum().over("day"),
    )
    vwap = pl.when(pl.col("_cv") > 0).then(pl.col("_cpv") / pl.col("_cv"))
    df = df.with_columns(vwap=vwap).with_columns(
        sd=(pl.col("_cpv2") / pl.col("_cv") - pl.col("vwap") ** 2).clip(lower_bound=0).sqrt())
    sd_ok = pl.when(pl.col("sd") >= tick).then(pl.col("sd"))  # avoid /0 in the first seconds
    return df.with_columns(
        lo_dev=(pl.col("low") - pl.col("vwap")) / sd_ok,
        hi_dev=(pl.col("high") - pl.col("vwap")) / sd_ok,
    ).drop("_cv", "_cpv", "_cpv2")


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def rsi(close: pl.Expr, n: int) -> pl.Expr:
    d = close.diff()
    au = d.clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
    ad = (-d).clip(lower_bound=0).ewm_mean(alpha=1 / n, adjust=False)
    val = pl.when(ad == 0).then(100.0).otherwise(100 - 100 / (1 + au / ad))
    return pl.when(pl.int_range(pl.len()) >= n).then(val)  # warm-up = null


def resample(df: pl.DataFrame, minutes: int, p: Params) -> pl.DataFrame:
    bars = (df.group_by_dynamic("ts", every=f"{minutes}m", closed="left", label="left", group_by="day")
              .agg(open=pl.col("open").first(), high=pl.col("high").max(),
                   low=pl.col("low").min(), close=pl.col("close").last(),
                   volume=pl.col("volume").sum(),
                   vwap=pl.col("vwap").last(), sd=pl.col("sd").last(),
                   lo_dev_min=pl.col("lo_dev").min(), hi_dev_max=pl.col("hi_dev").max())
              .rename({"ts": "bar_start"}).sort("bar_start"))
    sd_ok = pl.when(pl.col("sd") >= p.tick).then(pl.col("sd"))
    return bars.with_columns(
        bar_end=pl.col("bar_start") + pl.duration(minutes=minutes),
        close_dev=(pl.col("close") - pl.col("vwap")) / sd_ok,
        rsi=rsi(pl.col("close"), p.rsi_len),   # continuous across RTH days (like an RTH chart)
    ).with_columns(rsi_prev=pl.col("rsi").shift(1))


def opening_range(df: pl.DataFrame, p: Params) -> pl.DataFrame:
    t = pl.col("ts").dt.time()
    end = time(9, 30 + p.or_minutes) if p.or_minutes < 30 else time(10, p.or_minutes - 30)
    orb = (df.filter((t >= time(9, 30)) & (t < end))
             .group_by("day").agg(or_high=pl.col("high").max(), or_low=pl.col("low").min(),
                                  or_first=pl.col("ts").min(), or_bars=pl.len())
             .sort("day"))
    orb = orb.with_columns(
        or_mid=((pl.col("or_high") + pl.col("or_low")) / 2 / p.tick).floor() * p.tick,
        or_width_ticks=(pl.col("or_high") - pl.col("or_low")) / p.tick,
        or_ready=pl.col("day").dt.combine(end),
        or_complete=(pl.col("or_first").dt.time() <= time(9, 30, 5)) & (pl.col("or_bars") >= 100),
    )
    # percentile of today's width vs. the previous N days only (no look-ahead)
    w = orb["or_width_ticks"].to_numpy()
    pct = np.full(len(w), np.nan)
    for i in range(len(w)):
        prev = w[max(0, i - p.or_pctl_days):i]
        if len(prev) >= 20:
            pct[i] = (prev <= w[i]).mean() * 100
    return orb.with_columns(or_width_pctl=pl.Series(pct).fill_nan(None))


# --------------------------------------------------------------------------- #
# Entry fill helper
# --------------------------------------------------------------------------- #
def fill_at_next_second(events: pl.DataFrame, sec: pl.DataFrame, at: str) -> pl.DataFrame:
    """Entry = open of the first 1-second bar at/after `at` (same day, within 60 s)."""
    s = sec.select("day", pl.col("ts").alias("entry_ts"), pl.col("open").alias("entry_px"))
    return (events.sort(at)
                  .join_asof(s, left_on=at, right_on="entry_ts", by="day",
                             strategy="forward", tolerance="60s")
                  .drop_nulls("entry_px"))


# --------------------------------------------------------------------------- #
# Strategy A - VWAP mean reversion
# --------------------------------------------------------------------------- #
def events_a(sec: pl.DataFrame, b5: pl.DataFrame, b15: pl.DataFrame, p: Params) -> pl.DataFrame:
    ctx = b15.select("day", pl.col("bar_end").alias("b15_end"), pl.col("rsi").alias("rsi15"),
                     pl.col("close_dev").alias("close_dev15"))
    b = b5.sort("bar_end").join_asof(ctx.sort("b15_end"), left_on="bar_end", right_on="b15_end",
                                     by="day", strategy="backward")  # last CLOSED 15m bar
    et = pl.col("bar_end").dt.time()
    win = (et >= _t(p.a_start)) & (et <= _t(p.a_end))
    k, tick = p.a_k, p.tick

    long_ = b.filter(win & (pl.col("lo_dev_min") <= -k) & (pl.col("close_dev") > -k)
                     & (pl.col("rsi") < p.a_rsi_long) & (pl.col("rsi") > pl.col("rsi_prev")))
    long_ = long_.with_columns(
        side=pl.lit(1),
        stop=pl.col("low") - tick,                                        # below signal bar
        stop_band=((pl.col("vwap") - (k + 1) * pl.col("sd")) / tick).floor() * tick,
        tp_bar=pl.col("high"),                                            # signal bar high
        tp_vwap=(pl.col("vwap") / tick).floor() * tick)

    short = b.filter(win & (pl.col("hi_dev_max") >= k) & (pl.col("close_dev") < k)
                     & (pl.col("rsi") > p.a_rsi_short) & (pl.col("rsi") < pl.col("rsi_prev")))
    short = short.with_columns(
        side=pl.lit(-1),
        stop=pl.col("high") + tick,
        stop_band=((pl.col("vwap") + (k + 1) * pl.col("sd")) / tick).ceil() * tick,
        tp_bar=pl.col("low"),
        tp_vwap=(pl.col("vwap") / tick).ceil() * tick)

    ev = pl.concat([long_, short]).select(
        "day", pl.col("bar_end").alias("signal_ts"), "side", "stop", "stop_band", "tp_bar", "tp_vwap",
        "close", "vwap", "sd", "close_dev", "lo_dev_min", "hi_dev_max", "rsi", "rsi_prev",
        "rsi15", "close_dev15")
    ev = fill_at_next_second(ev, sec, "signal_ts")
    return ev.with_columns(
        risk_ticks=(pl.col("entry_px") - pl.col("stop")) * pl.col("side") / tick,
        tp_bar_ticks=(pl.col("tp_bar") - pl.col("entry_px")) * pl.col("side") / tick,
        tp_vwap_ticks=(pl.col("tp_vwap") - pl.col("entry_px")) * pl.col("side") / tick,
    ).filter(pl.col("risk_ticks") > 0).sort("entry_ts")


# --------------------------------------------------------------------------- #
# Strategy B - opening range break + retest
# --------------------------------------------------------------------------- #
def _events_b_side(sec, b5, orb, p: Params, side: int) -> pl.DataFrame:
    tick = p.tick
    edge = "or_high" if side == 1 else "or_low"
    b = b5.join(orb.filter("or_complete"), on="day", how="inner").filter(
        pl.col("bar_start") >= pl.col("or_ready"))
    outside = (pl.col("close") > pl.col(edge)) if side == 1 else (pl.col("close") < pl.col(edge))
    inside = (pl.col("close") < pl.col(edge)) if side == 1 else (pl.col("close") > pl.col(edge))

    brk = (b.filter(outside & (pl.col("bar_end").dt.time() <= _t(p.b_last_break)))
             .group_by("day").agg(pl.all().sort_by("bar_end").first())
             .select("day", pl.col("bar_end").alias("break_ts"),
                     pl.col("high" if side == 1 else "low").alias("break_ext"),
                     "or_high", "or_low", "or_mid", "or_width_ticks", "or_width_pctl"))
    # first 5m close back inside the range after the break -> setup invalid from that bar_end
    fail = (b.join(brk.select("day", "break_ts"), on="day")
              .filter((pl.col("bar_end") > pl.col("break_ts")) & inside)
              .group_by("day").agg(fail_ts=pl.col("bar_end").min()))
    brk = brk.join(fail, on="day", how="left").with_columns(
        win_end=pl.min_horizontal(pl.col("break_ts") + pl.duration(minutes=p.b_window_min),
                                  pl.col("fail_ts").fill_null(pl.col("break_ts") + pl.duration(days=1))))

    s = (sec.select("day", "ts", "open", "high", "low")
            .join(brk, on="day", how="inner")
            .filter((pl.col("ts") >= pl.col("break_ts")) & (pl.col("ts") < pl.col("win_end"))))
    ext_col = "high" if side == 1 else "low"
    run = (pl.col(ext_col).cum_max() if side == 1 else pl.col(ext_col).cum_min()).shift(1).over("day")
    s = s.sort("ts").with_columns(ext_so_far=run)
    ext_so_far = (pl.max_horizontal("ext_so_far", "break_ext") if side == 1
                  else pl.min_horizontal("ext_so_far", "break_ext"))
    zone = pl.col(edge) + side * p.b_retest_ticks * tick           # limit price
    extended = (ext_so_far - pl.col(edge)) * side >= p.b_min_ext_ticks * tick
    touched = (pl.col("low") <= zone) if side == 1 else (pl.col("high") >= zone)

    ev = (s.with_columns(zone=zone).filter(extended & touched)
            .group_by("day").agg(pl.all().sort_by("ts").first()))
    # limit order at `zone`, filled at the limit or better if the second opened through it
    fill = pl.min_horizontal("open", "zone") if side == 1 else pl.max_horizontal("open", "zone")
    stop_far = (pl.col("or_low") - tick) if side == 1 else (pl.col("or_high") + tick)
    return ev.select(
        "day", "break_ts", pl.col("ts").alias("entry_ts"), pl.lit(side).alias("side"),
        fill.alias("entry_px"), pl.col("zone").alias("limit_px"),
        pl.col("or_mid").alias("stop"), stop_far.alias("stop_far"),
        "or_high", "or_low", "or_width_ticks", "or_width_pctl",
        ((pl.col("ts") - pl.col("break_ts")).dt.total_seconds()).alias("retest_delay_sec"))


def events_b(sec, b5, orb, p: Params) -> pl.DataFrame:
    ev = pl.concat([_events_b_side(sec, b5, orb, p, 1), _events_b_side(sec, b5, orb, p, -1)])
    return ev.with_columns(
        risk_ticks=(pl.col("entry_px") - pl.col("stop")) * pl.col("side") / p.tick,
        risk_far_ticks=(pl.col("entry_px") - pl.col("stop_far")) * pl.col("side") / p.tick,
    ).filter(pl.col("risk_ticks") > 0).sort("entry_ts")


# --------------------------------------------------------------------------- #
def summarize(name: str, ev: pl.DataFrame, p: Params) -> None:
    if ev.is_empty():
        print(f"\n{name}: no events")
        return
    days = ev["day"].n_unique()
    r = ev["risk_ticks"]
    print(f"\n{name}: {ev.height} events on {days} days | long {int((ev['side'] == 1).sum())}"
          f" / short {int((ev['side'] == -1).sum())}")
    print(f"  risk ticks  median {r.median():.0f} | p75 {r.quantile(0.75):.0f} | p95 {r.quantile(0.95):.0f}")
    # $ per tick: NQ $5.00, MNQ $0.50 (ES $12.50, MES $1.25)
    print(f"  median risk $  NQ ${r.median() * 5:,.0f}  |  MNQ ${r.median() * 0.5:,.0f}")
    by_m = ev.group_by(pl.col("day").dt.strftime("%Y-%m").alias("month")).len().sort("month")
    print("  per month: " + ", ".join(f"{m}:{n}" for m, n in by_m.iter_rows()))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: features + entries for A (VWAP) and B (ORB)")
    ap.add_argument("--data", required=True, help="folder with Stage 1 parquet files")
    ap.add_argument("--out", required=True)
    for f in fields(Params):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = ap.parse_args()
    p = Params(**{f.name: getattr(a, f.name) for f in fields(Params)})

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sec = add_vwap(load(Path(a.data).expanduser()), p.tick)
    b5, b15 = resample(sec, 5, p), resample(sec, 15, p)
    orb = opening_range(sec, p)
    ev_a, ev_b = events_a(sec, b5, b15, p), events_b(sec, b5, orb, p)

    for name, frame in [("bars_5m", b5), ("bars_15m", b15), ("orb_daily", orb),
                        ("events_A", ev_a), ("events_B", ev_b)]:
        frame.write_parquet(out / f"{name}.parquet")
    print(f"{sec.height:,} seconds, {sec['day'].n_unique()} days | params: {p}")
    summarize("A - VWAP mean reversion", ev_a, p)
    summarize("B - ORB break + retest", ev_b, p)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
