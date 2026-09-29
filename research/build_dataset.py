#!/usr/bin/env python3
"""
Stage 1 - Build a clean 1-second dataset (Parquet, one file per month).

  * loads every file found by the Stage 0 loader (CSV / Parquet / NinjaTrader)
  * converts timestamps to naive US/Eastern (ET)
  * removes duplicate seconds, keeps the front month (highest daily volume)
    when a symbol column exists
  * keeps RTH only (09:30:00-15:59:59 ET, Mon-Fri) unless --keep-eth

Usage:
    python build_dataset.py --src ~/Downloads --out ~/nq_data/clean --tz-in America/New_York
    python build_dataset.py --src ~/Downloads --out ~/nq_data/clean --tz-in UTC
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_integrity_check import FILE_PATTERNS, load_normalized  # noqa: E402

ET = "America/New_York"


def to_et(ts: pl.Expr, tz_in: str) -> pl.Expr:
    if tz_in == ET:
        return ts
    return (ts.dt.replace_time_zone(tz_in, ambiguous="earliest", non_existent="null")
              .dt.convert_time_zone(ET).dt.replace_time_zone(None))


def build(src: Path, out: Path, tz_in: str, keep_eth: bool) -> pl.DataFrame:
    skip = {"integrity_report.csv", "daily_coverage.csv"}
    files = sorted({f for p in FILE_PATTERNS for f in src.rglob(p) if f.name not in skip})
    frames = []
    for f in files:
        try:
            df, _ = load_normalized(f)
            frames.append(df)
            print(f"loaded  {f.name:<40} {df.height:>12,} rows")
        except Exception as exc:
            print(f"skipped {f.name:<40} {exc}")
    if not frames:
        raise SystemExit("no usable files")

    df = pl.concat(frames, how="diagonal_relaxed").drop_nulls("ts")
    if "volume" not in df.columns:
        raise SystemExit("volume column is required (VWAP)")
    df = df.with_columns(ts=to_et(pl.col("ts"), tz_in)).drop_nulls("ts")

    if "symbol" in df.columns:
        # front month = symbol with the highest volume on each day
        df = df.with_columns(day=pl.col("ts").dt.date())
        front = (df.group_by("day", "symbol").agg(pl.col("volume").sum())
                   .sort("volume", descending=True).unique("day", keep="first")
                   .select("day", "symbol"))
        df = df.join(front, on=["day", "symbol"], how="inner").drop("day")

    df = df.unique("ts", keep="first").sort("ts")
    df = df.filter(pl.col("ts").dt.weekday() <= 5)
    if not keep_eth:
        t = pl.col("ts").dt.time()
        df = df.filter((t >= pl.time(9, 30)) & (t < pl.time(16, 0)))

    out.mkdir(parents=True, exist_ok=True)
    df = df.with_columns(ym=pl.col("ts").dt.strftime("%Y-%m"))
    for (ym,), part in df.group_by("ym", maintain_order=True):
        part.drop("ym").write_parquet(out / f"{ym}.parquet")
    print(f"\n{df.height:,} rows, {df['ts'].dt.date().n_unique()} days -> {out}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 1: clean 1-second dataset")
    ap.add_argument("--src", default=str(Path.home() / "Downloads"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tz-in", default=ET, help="timezone of the raw timestamps (e.g. UTC, Asia/Jerusalem)")
    ap.add_argument("--keep-eth", action="store_true", help="keep overnight session")
    a = ap.parse_args()
    build(Path(a.src).expanduser(), Path(a.out).expanduser(), a.tz_in, a.keep_eth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
