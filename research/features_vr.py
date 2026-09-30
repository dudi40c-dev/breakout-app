#!/usr/bin/env python3
"""
Stage 2 - Strategy R: VWAP rejection (pullback to VWAP in the direction of the trend).

Short setup (long side mirrored, --r-sides both|short|long):
  1. Daily context (previous days only): continuous lower highs or close < falling SMA.
     Stored as columns; used as a filter only with --r-daily-ctx 1.
  2. Intraday trend: earlier today a 5m bar closed at least --r-min-dev sigma below VWAP.
  3. Pullback: the previous 5m bar closed below VWAP (price approaching from below).
  4. Signal bar (5m): touches VWAP (any second in the bar at/above VWAP), closes below it,
     and is a rejection candle - upper wick >= --r-wick of range (pin bar) or a doji
     (body <= --r-doji of range). Range >= --r-min-range ticks.
  5. Entry: sell-stop 1 tick below the signal bar low, valid for --r-trigger-bars 5m bars,
     cancelled if price trades through the stop level first. Fill = min(open, trigger).
  6. Stop: 1 tick above the signal bar high.
  Targets: R multiples / scale-out in evaluate_events.py, plus structural levels here:
     tp_pdl  = previous day low (high for longs), tp_round = next round number (--r-round-pts).

Input : Stage 1 parquet folder.  Output: events_R.parquet (+ daily_ctx.parquet) in --out.

Usage:
    python features_vr.py --data ~/nq_data/clean --out ~/nq_data/features_R
    python features_vr.py --data ... --out ... --r-daily-ctx 1 --r-sides short
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_ab import Params as ABParams  # noqa: E402
from features_ab import _t, add_vwap, load, resample, summarize  # noqa: E402


@dataclass
class Params:
    tick: float = 0.25
    r_sides: str = "both"          # both | short | long
    r_daily_ctx: int = 0           # 1 = require daily trend context in the trade direction
    r_sma_days: int = 10
    r_min_dev: float = 1.0         # earlier today: a 5m close at least N sigma beyond VWAP
    r_wick: float = 0.5            # pin bar: rejection wick >= this share of the bar range
    r_doji: float = 0.2            # doji: body <= this share of the bar range
    r_min_range: int = 8           # signal bar range, ticks
    r_trigger_bars: int = 2        # entry order valid for N 5m bars after the signal
    r_start: str = "09:50"
    r_end: str = "15:30"
    r_round_pts: float = 100.0     # round-number spacing for tp_round (NQ: 100 points)


def daily_context(sec: pl.DataFrame, p: Params) -> pl.DataFrame:
    """Per day, using only PREVIOUS days' RTH bars (no look-ahead)."""
    d = (sec.group_by("day").agg(d_high=pl.col("high").max(), d_low=pl.col("low").min(),
                                 d_close=pl.col("close").last(), d_open=pl.col("open").first())
            .sort("day"))
    rng = pl.col("d_high") - pl.col("d_low")
    d = d.with_columns(
        sma=pl.col("d_close").rolling_mean(p.r_sma_days),
        avg_rng=rng.rolling_mean(p.r_sma_days),
        rng=rng,
    ).with_columns(
        lower_highs=(pl.col("d_high") < pl.col("d_high").shift(1))
        & (pl.col("d_high").shift(1) < pl.col("d_high").shift(2)),
        higher_lows=(pl.col("d_low") > pl.col("d_low").shift(1))
        & (pl.col("d_low").shift(1) > pl.col("d_low").shift(2)),
        below_falling_sma=(pl.col("d_close") < pl.col("sma")) & (pl.col("sma") < pl.col("sma").shift(5)),
        above_rising_sma=(pl.col("d_close") > pl.col("sma")) & (pl.col("sma") > pl.col("sma").shift(5)),
        # expansion bar: range > 1.5x average and close in the outer 25% of the range
        exp_down=(pl.col("rng") > 1.5 * pl.col("avg_rng"))
        & (pl.col("d_close") <= pl.col("d_low") + 0.25 * pl.col("rng")),
        exp_up=(pl.col("rng") > 1.5 * pl.col("avg_rng"))
        & (pl.col("d_close") >= pl.col("d_high") - 0.25 * pl.col("rng")),
    )
    # shift everything by one day: the context for day D comes from D-1 and earlier
    ctx_cols = ["d_high", "d_low", "lower_highs", "higher_lows", "below_falling_sma",
                "above_rising_sma", "exp_down", "exp_up"]
    d = d.with_columns([pl.col(c).shift(1).alias(f"prev_{c}") for c in ctx_cols])
    return d.select(
        "day", pl.col("prev_d_high").alias("pdh"), pl.col("prev_d_low").alias("pdl"),
        ctx_down=(pl.col("prev_lower_highs") | pl.col("prev_below_falling_sma")).fill_null(False),
        ctx_up=(pl.col("prev_higher_lows") | pl.col("prev_above_rising_sma")).fill_null(False),
        prev_exp_down=pl.col("prev_exp_down").fill_null(False),
        prev_exp_up=pl.col("prev_exp_up").fill_null(False))


def signals(b5: pl.DataFrame, ctx: pl.DataFrame, p: Params) -> pl.DataFrame:
    tick = p.tick
    rng = pl.col("high") - pl.col("low")
    body = (pl.col("close") - pl.col("open")).abs()
    up_wick = pl.col("high") - pl.max_horizontal("open", "close")
    dn_wick = pl.min_horizontal("open", "close") - pl.col("low")
    doji = body <= p.r_doji * rng
    et = pl.col("bar_end").dt.time()

    b = (b5.sort("bar_start")
           .with_columns(prev_close_dev=pl.col("close_dev").shift(1).over("day"),
                         # extreme close_dev of EARLIER bars today (excludes the signal bar)
                         min_dev_before=pl.col("close_dev").cum_min().shift(1).over("day"),
                         max_dev_before=pl.col("close_dev").cum_max().shift(1).over("day"))
           .join(ctx, on="day", how="left")
           .filter((et >= _t(p.r_start)) & (et <= _t(p.r_end)) & (rng >= p.r_min_range * tick)))

    short = b.filter((pl.col("min_dev_before") <= -p.r_min_dev) & (pl.col("prev_close_dev") < 0)
                     & (pl.col("hi_dev_max") >= 0) & (pl.col("close_dev") < 0)
                     & ((up_wick >= p.r_wick * rng) | doji))
    long_ = b.filter((pl.col("max_dev_before") >= p.r_min_dev) & (pl.col("prev_close_dev") > 0)
                     & (pl.col("lo_dev_min") <= 0) & (pl.col("close_dev") > 0)
                     & ((dn_wick >= p.r_wick * rng) | doji))
    if p.r_daily_ctx:
        short, long_ = short.filter("ctx_down"), long_.filter("ctx_up")
    parts = []
    if p.r_sides in ("both", "short"):
        parts.append(short.with_columns(side=pl.lit(-1), trigger=pl.col("low") - tick,
                                        stop=pl.col("high") + tick, ctx_ok=pl.col("ctx_down"),
                                        prev_exp=pl.col("prev_exp_down"), tp_pdl=pl.col("pdl")))
    if p.r_sides in ("both", "long"):
        parts.append(long_.with_columns(side=pl.lit(1), trigger=pl.col("high") + tick,
                                        stop=pl.col("low") - tick, ctx_ok=pl.col("ctx_up"),
                                        prev_exp=pl.col("prev_exp_up"), tp_pdl=pl.col("pdh")))
    if not parts:
        raise SystemExit("--r-sides must be both, short or long")
    return pl.concat(parts).select(
        "day", pl.col("bar_end").alias("signal_ts"), "side", "trigger", "stop", "tp_pdl",
        pl.col("open").alias("sig_open"), pl.col("high").alias("sig_high"),
        pl.col("low").alias("sig_low"), pl.col("close").alias("sig_close"),
        "vwap", "sd", "close_dev", "ctx_ok", "prev_exp")


def fill_stop_entries(sig: pl.DataFrame, sec: pl.DataFrame, p: Params) -> pl.DataFrame:
    """Stop-entry order at `trigger`, live for r_trigger_bars*5 min, cancelled if `stop` trades first."""
    win = pl.duration(minutes=5 * p.r_trigger_bars)
    s = (sec.select("day", "ts", "open", "high", "low")
            .join(sig.with_row_index("sig_id"), on="day", how="inner")
            .filter((pl.col("ts") >= pl.col("signal_ts")) & (pl.col("ts") < pl.col("signal_ts") + win))
            .sort("sig_id", "ts"))
    side = pl.col("side")
    trig_hit = pl.when(side == 1).then(pl.col("high") >= pl.col("trigger")).otherwise(pl.col("low") <= pl.col("trigger"))
    stop_hit = pl.when(side == 1).then(pl.col("low") <= pl.col("stop")).otherwise(pl.col("high") >= pl.col("stop"))
    s = s.with_columns(trig_hit=trig_hit, stop_hit=stop_hit)
    first_trig = s.filter("trig_hit").group_by("sig_id").agg(entry_ts=pl.col("ts").min())
    first_stop = s.filter("stop_hit").group_by("sig_id").agg(inval_ts=pl.col("ts").min())
    ok = (first_trig.join(first_stop, on="sig_id", how="left")
                    .filter(pl.col("inval_ts").is_null() | (pl.col("entry_ts") < pl.col("inval_ts"))))
    ev = (s.join(ok, on="sig_id").filter(pl.col("ts") == pl.col("entry_ts"))
           .with_columns(entry_px=pl.when(side == 1).then(pl.max_horizontal("open", "trigger"))
                         .otherwise(pl.min_horizontal("open", "trigger"))))
    return ev.drop("ts", "open", "high", "low", "trig_hit", "stop_hit", "inval_ts")


def events_r(sec: pl.DataFrame, b5: pl.DataFrame, p: Params) -> tuple[pl.DataFrame, pl.DataFrame]:
    ctx = daily_context(sec, p)
    sig = signals(b5, ctx, p)
    ev = fill_stop_entries(sig, sec, p)
    step, tick = p.r_round_pts, p.tick
    rnd = pl.when(pl.col("side") == 1).then(
        ((pl.col("entry_px") + 2 * tick) / step).ceil() * step).otherwise(
        ((pl.col("entry_px") - 2 * tick) / step).floor() * step)
    ev = ev.with_columns(tp_round=rnd).with_columns(
        # structural targets must be on the profit side of the entry
        tp_pdl=pl.when((pl.col("tp_pdl") - pl.col("entry_px")) * pl.col("side") >= 4 * tick)
        .then(pl.col("tp_pdl")),
        risk_ticks=(pl.col("entry_px") - pl.col("stop")) * pl.col("side") / tick,
    )
    ev = ev.filter(pl.col("risk_ticks") > 0).drop("sig_id").sort("entry_ts")
    # one entry per 5m signal bar and side; the evaluator enforces one position at a time
    return ev.unique(["signal_ts", "side"], keep="first").sort("entry_ts"), ctx


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 2: VWAP rejection (strategy R) events")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    for f in fields(Params):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    a = ap.parse_args()
    p = Params(**{f.name: getattr(a, f.name) for f in fields(Params)})

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    sec = add_vwap(load(Path(a.data).expanduser()), p.tick)
    b5 = resample(sec, 5, ABParams(tick=p.tick))
    ev, ctx = events_r(sec, b5, p)
    ev.write_parquet(out / "events_R.parquet")
    ctx.write_parquet(out / "daily_ctx.parquet")
    print(f"{sec.height:,} seconds, {sec['day'].n_unique()} days | params: {p}")
    summarize("R - VWAP rejection", ev, ABParams(tick=p.tick))
    if not ev.is_empty():
        print(f"  with daily context in trade direction: {int(ev['ctx_ok'].sum())} / {ev.height}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
