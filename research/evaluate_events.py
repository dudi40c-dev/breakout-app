#!/usr/bin/env python3
"""
Stage 3 - Evaluate Stage 2 entry events on 1-second data (edge detection).

For every event and every combination of
    stop column  (any "stop*" column in the events file)
    target       (any "tp_*" column + fixed R multiples)
    max hold     (minutes, then exit at market)
the trade is walked second by second from the entry second:
  * stop hit when low <= stop (long) / high >= stop (short); gap-through fills at the open
  * target (limit) filled only when price trades THROUGH it by --limit-through ticks
  * stop and target in the same second -> counted as STOP (conservative)
  * slippage: --slip ticks on market entry, stop exit and time exit (limit entries: 0)
  * one position at a time, max trades per day, daily stop in R
Nothing is optimized here beyond reporting every configuration; pick robust ones, not the top one.

Usage:
    python evaluate_events.py --data ~/nq_data/clean --events ~/nq_data/features/events_A.parquet
    python evaluate_events.py --data ... --events .../events_B_cont.parquet --out ~/nq_data/eval_B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

TICK_VALUE = {"nq": (5.0, 0.5), "es": (12.5, 1.25)}   # $ per tick: (full, micro)


def load_prices(data: Path) -> dict[str, np.ndarray]:
    df = pl.read_parquet(data / "*.parquet", columns=["ts", "open", "high", "low", "close"]).sort("ts")
    sec = df["ts"].dt.epoch("s").to_numpy()
    return {"ts": sec, **{c: df[c].to_numpy() for c in ("open", "high", "low", "close")}}


def walk(px, i0: int, i1: int, side: int, stop: float, tgt: float, through: float):
    """First stop / target index in [i0, i1) (relative), -1 if none."""
    lo, hi = px["low"][i0:i1], px["high"][i0:i1]
    if side == 1:
        s_hit, t_hit = lo <= stop, hi >= tgt + through
    else:
        s_hit, t_hit = hi >= stop, lo <= tgt - through
    s = int(np.argmax(s_hit)) if s_hit.any() else -1
    t = int(np.argmax(t_hit)) if t_hit.any() else -1
    return s, t


def simulate(px, ev: pl.DataFrame, a) -> pl.DataFrame:
    tick, slip = a.tick, a.slip * a.tick
    entry_slip = 0.0 if "limit_px" in ev.columns else slip
    stop_cols = [c for c in ev.columns if c.startswith("stop")]
    tp_cols = [c for c in ev.columns if c.startswith("tp_") and not c.endswith("_ticks")]
    targets = tp_cols + [f"R{r:g}" for r in a.r_targets]
    holds = sorted(a.holds)
    ts = px["ts"]

    rows = []
    for e in ev.iter_rows(named=True):
        side = e["side"]
        t0 = int(np.datetime64(e["entry_ts"], "s").astype(np.int64))
        i0 = int(np.searchsorted(ts, t0))
        if i0 >= len(ts) or ts[i0] != t0:
            continue
        day_end = int(np.datetime64(e["day"], "s").astype(np.int64)) + 16 * 3600
        i_day = int(np.searchsorted(ts, day_end))
        entry = e["entry_px"] + side * entry_slip
        for sc in stop_cols:
            stop = e[sc]
            if stop is None:
                continue
            risk = (entry - stop) * side
            if risk < tick:
                continue
            for tg in targets:
                if tg.startswith("R"):
                    raw = entry + side * float(tg[1:]) * risk
                    tgt = (np.ceil(raw / tick) if side == 1 else np.floor(raw / tick)) * tick
                else:
                    tgt = e[tg]
                if tgt is None or (tgt - entry) * side < tick:
                    continue
                i_max = min(i_day, int(np.searchsorted(ts, t0 + holds[-1] * 60)))
                if i_max <= i0:
                    continue
                s, t = walk(px, i0, i_max, side, stop, tgt, a.limit_through * tick)
                for h in holds:
                    i_h = min(i_day, int(np.searchsorted(ts, t0 + h * 60))) - i0
                    s_h = s if 0 <= s < i_h else -1
                    t_h = t if 0 <= t < i_h else -1
                    if s_h >= 0 and (t_h < 0 or s_h <= t_h):          # stop first / same second
                        k = i0 + s_h
                        o = px["open"][k]
                        gap = min(stop, o) if side == 1 else max(stop, o)
                        exit_px, how = gap - side * slip, "stop"
                    elif t_h >= 0:
                        k, exit_px, how = i0 + t_h, tgt, "target"
                    else:
                        k = i0 + i_h - 1
                        exit_px, how = px["close"][k] - side * slip, "time"
                    pnl = (exit_px - entry) * side / tick
                    rows.append((sc, tg, h, e["day"], t0, int(ts[k]) + 1, side,
                                 pnl, pnl / (risk / tick), risk / tick, how))
    return pl.DataFrame(rows, orient="row", schema=[
        "stop_col", "target", "hold_min", "day", "entry_s", "exit_s", "side",
        "pnl_ticks", "r", "risk_ticks", "exit_how"])


def select_trades(tr: pl.DataFrame, a) -> pl.DataFrame:
    """One position at a time, max trades/day, daily stop in R (applied per configuration)."""
    keep = np.zeros(tr.height, dtype=bool)
    tr = tr.sort("stop_col", "target", "hold_min", "entry_s")
    cfg = tr.select("stop_col", "target", "hold_min").rows()
    entry, exit_, day, r = (tr[c].to_list() for c in ("entry_s", "exit_s", "day", "r"))
    prev_cfg, busy_until, cur_day, n_day, r_day = None, -1, None, 0, 0.0
    for i in range(tr.height):
        if cfg[i] != prev_cfg:
            prev_cfg, busy_until, cur_day = cfg[i], -1, None
        if day[i] != cur_day:
            cur_day, n_day, r_day = day[i], 0, 0.0
        if entry[i] < busy_until or n_day >= a.max_trades_day or r_day <= -a.max_daily_loss_r:
            continue
        keep[i] = True
        busy_until, n_day, r_day = exit_[i], n_day + 1, r_day + r[i]
    return tr.filter(pl.Series(keep))


def _max_dd(x: np.ndarray) -> float:
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(x) else 0.0


def _max_streak(losses: np.ndarray) -> int:
    best = cur = 0
    for v in losses:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def metrics(tr: pl.DataFrame, a) -> pl.DataFrame:
    full_tv, micro_tv = TICK_VALUE[a.instrument]
    out = []
    for key, g in tr.group_by("stop_col", "target", "hold_min"):
        g = g.sort("entry_s")
        r, pnl = g["r"].to_numpy(), g["pnl_ticks"].to_numpy()
        wins, losses = r[r > 0], r[r <= 0]
        half = len(r) // 2
        daily = g.group_by("day").agg(pl.col("pnl_ticks").sum(), pl.col("r").sum())
        months = g["day"].dt.strftime("%Y-%m").n_unique()
        full_usd = pnl * full_tv - a.comm_full
        micro_usd = pnl * micro_tv - a.comm_micro
        out.append({
            "stop_col": key[0], "target": key[1], "hold_min": key[2],
            "trades": len(r), "per_month": round(len(r) / max(months, 1), 1),
            "win_rate": round(float((r > 0).mean()) * 100, 1),
            "avg_win_R": round(float(wins.mean()), 2) if len(wins) else 0.0,
            "avg_loss_R": round(float(losses.mean()), 2) if len(losses) else 0.0,
            "exp_R": round(float(r.mean()), 3),
            # |t| < 2: expectancy is not distinguishable from zero (noise)
            "t_stat": round(float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))), 2) if len(r) > 2 else None,
            "exp_R_1st_half": round(float(r[:half].mean()), 3) if half else None,
            "exp_R_2nd_half": round(float(r[half:].mean()), 3) if half else None,
            "exp_ticks": round(float(pnl.mean()), 2),
            "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
            "max_dd_R": round(_max_dd(r), 1),
            "max_loss_streak": _max_streak(r <= 0),
            "worst_day_R": round(float(daily["r"].min()), 2),
            "time_exits_%": round(float((g["exit_how"] == "time").mean()) * 100, 1),
            "avg_hold_min": round(float((g["exit_s"] - g["entry_s"]).mean()) / 60, 1),
            "median_risk_ticks": float(np.median(g["risk_ticks"].to_numpy())),
            "net_per_trade_micro_$": round(float(micro_usd.mean()), 2),
            "net_per_trade_full_$": round(float(full_usd.mean()), 2),
            "worst_day_micro_$": round(float(daily["pnl_ticks"].min()) * micro_tv, 0),
            "max_dd_micro_$": round(_max_dd(micro_usd), 0),
        })
    return pl.DataFrame(out).sort("exp_R", descending=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 3: evaluate entry events on 1-second data")
    ap.add_argument("--data", required=True, help="Stage 1 parquet folder")
    ap.add_argument("--events", required=True, help="events_*.parquet from Stage 2")
    ap.add_argument("--out", default=None, help="output folder (default: next to events)")
    ap.add_argument("--instrument", choices=list(TICK_VALUE), default="nq")
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--slip", type=float, default=1.0, help="ticks per market fill")
    ap.add_argument("--limit-through", type=float, default=1.0, help="ticks a limit must trade through")
    ap.add_argument("--comm-full", type=float, default=4.0, help="$ round turn, NQ/ES (ASSUMPTION - verify)")
    ap.add_argument("--comm-micro", type=float, default=1.0, help="$ round turn, MNQ/MES (ASSUMPTION - verify)")
    ap.add_argument("--r-targets", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--holds", type=int, nargs="+", default=[10, 20, 30, 60], help="max hold, minutes")
    ap.add_argument("--max-trades-day", type=int, default=3)
    ap.add_argument("--max-daily-loss-r", type=float, default=2.0)
    ap.add_argument("--min-trades", type=int, default=50, help="hide configs with fewer trades")
    a = ap.parse_args()

    ev_path = Path(a.events).expanduser()
    out = Path(a.out).expanduser() if a.out else ev_path.parent / f"eval_{ev_path.stem}"
    out.mkdir(parents=True, exist_ok=True)

    px = load_prices(Path(a.data).expanduser())
    ev = pl.read_parquet(ev_path).sort("entry_ts")
    raw = simulate(px, ev, a)
    if raw.is_empty():
        print("no simulated trades")
        return 1
    trades = select_trades(raw, a)
    res = metrics(trades, a)
    res.write_csv(out / "results.csv")
    trades.write_parquet(out / "trades_all_configs.parquet")

    shown = res.filter(pl.col("trades") >= a.min_trades)
    print(f"{ev.height} events | {res.height} configurations | "
          f"{shown.height} with >= {a.min_trades} trades | slip {a.slip}t, "
          f"comm ${a.comm_micro}/{a.comm_full} RT (micro/full)\n")
    cols = ["stop_col", "target", "hold_min", "trades", "per_month", "win_rate", "exp_R", "t_stat",
            "exp_R_1st_half", "exp_R_2nd_half", "profit_factor", "max_dd_R", "max_loss_streak",
            "net_per_trade_micro_$", "worst_day_micro_$"]
    with pl.Config(tbl_rows=15, tbl_cols=len(cols), tbl_width_chars=250):
        print(shown.select(cols).head(15))
    positive = shown.filter((pl.col("exp_R") > 0) & (pl.col("exp_R_1st_half") > 0)
                            & (pl.col("exp_R_2nd_half") > 0))
    print(f"\nconfigs positive in BOTH halves: {positive.height} / {shown.height} | "
          f"with t_stat >= 2: {positive.filter(pl.col('t_stat') >= 2).height}")
    print("note: configs share the same trades - many 'positive' rows are NOT independent evidence")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
