#!/usr/bin/env python3
"""
Stage 0 - Data integrity check for 1-second futures bars (NQ / MNQ / ES / MES).

Scans a folder (default: ~/Downloads) for CSV / CSV.GZ / TXT / Parquet files,
auto-detects the column layout, and reports per file:
  * schema, row count, date range, trading days
  * duplicate timestamps, unsorted rows, nulls
  * OHLC consistency (high >= max(open, close), low <= min(open, close))
  * tick-size alignment (0.25 for NQ/MNQ/ES/MES)
  * abnormal jumps between consecutive bars (possible roll / bad ticks)
  * large intraday gaps during RTH
  * timezone guess (minute of the day with the highest volume = cash open)

Usage:
    pip install polars pyarrow
    python data_integrity_check.py                       # scans ~/Downloads
    python data_integrity_check.py --path "D:/data/nq"   # custom folder
    python data_integrity_check.py --tick 0.25 --jump-ticks 40

Output: console summary + integrity_report.csv + daily_coverage.csv
        (written into the scanned folder, or --out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

FILE_PATTERNS = ("*.csv", "*.csv.gz", "*.txt", "*.parquet")

# Candidate column names (lower-case) for auto-detection
COLUMN_ALIASES = {
    "ts": ["timestamp", "datetime", "date_time", "time", "ts", "ts_event", "date"],
    "open": ["open", "o", "open_price"],
    "high": ["high", "h", "high_price"],
    "low": ["low", "l", "low_price"],
    "close": ["close", "c", "last", "close_price"],
    "volume": ["volume", "vol", "v", "total_volume"],
    "bid_vol": ["bid_volume", "bidvolume", "sell_volume", "bid_vol"],
    "ask_vol": ["ask_volume", "askvolume", "buy_volume", "ask_vol"],
    "symbol": ["symbol", "ticker", "instrument", "contract"],
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_raw(path: Path) -> pl.DataFrame:
    """Read a file without assumptions. Handles NinjaTrader ';' exports without header."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)

    import gzip
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:  # peek at first line
        head = fh.read(4096)
    first_line = head.decode("utf-8", errors="ignore").splitlines()[0]
    sep = ";" if first_line.count(";") >= 4 else ("\t" if first_line.count("\t") >= 4 else ",")
    has_header = any(ch.isalpha() for ch in first_line.split(sep)[0].replace("T", ""))

    df = pl.read_csv(path, separator=sep, has_header=has_header, infer_schema_length=10_000,
                     try_parse_dates=False)
    if not has_header:
        # NinjaTrader 8 bar export: "yyyyMMdd HHmmss;O;H;L;C;V"
        names = ["timestamp", "open", "high", "low", "close", "volume"][: df.width]
        df.columns = names + [f"extra_{i}" for i in range(df.width - len(names))]
    return df


def _detect_columns(df: pl.DataFrame) -> dict[str, str | None]:
    lower = {c.lower().strip(): c for c in df.columns}
    mapping: dict[str, str | None] = {}
    for key, aliases in COLUMN_ALIASES.items():
        mapping[key] = next((lower[a] for a in aliases if a in lower), None)
    # Separate date + time columns (e.g. "Date","Time")
    if "date" in lower and "time" in lower:
        mapping["ts"] = None
        mapping["_date"], mapping["_time"] = lower["date"], lower["time"]
    return mapping


def _parse_timestamp(df: pl.DataFrame, cmap: dict) -> pl.Series:
    if cmap.get("_date"):
        raw = (df[cmap["_date"]].cast(pl.Utf8) + " " + df[cmap["_time"]].cast(pl.Utf8))
    else:
        raw = df[cmap["ts"]]

    if raw.dtype in (pl.Datetime, pl.Datetime("ns"), pl.Datetime("us"), pl.Datetime("ms")):
        return raw.dt.replace_time_zone(None) if raw.dtype.time_zone else raw  # type: ignore[attr-defined]

    if raw.dtype.is_numeric():  # epoch - detect unit by magnitude
        mx = raw.max()
        unit = "ns" if mx > 1e17 else "us" if mx > 1e14 else "ms" if mx > 1e11 else "s"
        mult = {"s": 1_000_000, "ms": 1_000, "us": 1, "ns": None}[unit]
        if unit == "ns":
            return pl.from_epoch(raw, time_unit="ns")
        return pl.from_epoch(raw * mult, time_unit="us")

    s = raw.cast(pl.Utf8).str.strip_chars()
    formats = ["%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S",
               "%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"]
    for fmt in formats:
        parsed = s.str.strptime(pl.Datetime("us"), fmt, strict=False)
        if parsed.null_count() < 0.01 * len(s):
            return parsed
    # Last resort: ISO with timezone
    parsed = s.str.to_datetime(strict=False, time_zone="UTC").dt.replace_time_zone(None)
    return parsed


def load_normalized(path: Path) -> tuple[pl.DataFrame, dict]:
    raw = _read_raw(path)
    cmap = _detect_columns(raw)
    missing = [k for k in ("open", "high", "low", "close") if cmap.get(k) is None]
    if (cmap.get("ts") is None and not cmap.get("_date")) or missing:
        raise ValueError(f"cannot map columns {raw.columns} (missing: ts/{missing})")

    cols = {"ts": _parse_timestamp(raw, cmap)}
    for k in ("open", "high", "low", "close", "volume", "bid_vol", "ask_vol"):
        if cmap.get(k):
            cols[k] = raw[cmap[k]].cast(pl.Float64, strict=False)
    if cmap.get("symbol"):
        cols["symbol"] = raw[cmap["symbol"]].cast(pl.Utf8)
    return pl.DataFrame(cols), {"raw_columns": raw.columns, "mapping": cmap}


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
def guess_timezone(df: pl.DataFrame) -> str:
    """Cash open (09:30 ET) = the largest minute-over-minute volume increase of the day."""
    if "volume" not in df.columns:
        return "unknown (no volume)"
    vol = (df.with_columns(mod=(pl.col("ts").dt.hour().cast(pl.Int32) * 60
                                + pl.col("ts").dt.minute().cast(pl.Int32)))
             .group_by("mod").agg(pl.col("volume").sum()))
    full = pl.DataFrame({"mod": pl.int_range(0, 1440, dtype=pl.Int32, eager=True)}).join(
        vol, on="mod", how="left").fill_null(0).sort("mod")
    jump = full["volume"] - full["volume"].shift(1, fill_value=full["volume"][-1])
    peak = int(full["mod"][int(jump.arg_max())])
    hhmm = f"{peak // 60:02d}:{peak % 60:02d}"
    guesses = {9 * 60 + 30: "US/Eastern (ET)", 8 * 60 + 30: "US/Central (CT - exchange)",
               13 * 60 + 30: "UTC (summer)", 14 * 60 + 30: "UTC (winter) / ET+5",
               16 * 60 + 30: "Asia/Jerusalem"}
    near = [v for k, v in guesses.items() if abs(k - peak) <= 1]
    return f"open-jump minute {hhmm} -> {near[0] if near else 'check manually'}"


def check_file(path: Path, tick: float, jump_ticks: int, gap_sec: int) -> tuple[dict, pl.DataFrame]:
    df, meta = load_normalized(path)
    n = df.height
    rep: dict = {"file": path.name, "rows": n, "columns": ",".join(meta["raw_columns"])}

    rep["ts_parse_fail"] = df["ts"].null_count()
    df = df.drop_nulls("ts")
    rep["unsorted"] = bool((df["ts"].diff().dt.total_microseconds() < 0).any())
    df = df.sort("ts")
    rep["start"], rep["end"] = str(df["ts"].min()), str(df["ts"].max())
    rep["duplicate_ts"] = n - rep["ts_parse_fail"] - df["ts"].n_unique()
    rep["sub_second_ts"] = int((df["ts"].dt.microsecond() != 0).sum())
    rep["null_prices"] = int(df.select(pl.sum_horizontal(
        pl.col(c).is_null() for c in ("open", "high", "low", "close"))).to_series().gt(0).sum())

    ohlc_bad = df.filter(
        (pl.col("high") < pl.max_horizontal("open", "close"))
        | (pl.col("low") > pl.min_horizontal("open", "close"))
        | (pl.col("low") > pl.col("high"))
        | (pl.col("low") <= 0))
    rep["ohlc_violations"] = ohlc_bad.height

    off_tick = df.filter(((pl.col("close") / tick).round(0) * tick - pl.col("close")).abs() > 1e-6)
    rep["off_tick_prices"] = off_tick.height

    if "volume" in df.columns:
        rep["zero_volume_bars"] = int((df["volume"] <= 0).sum())
        rep["median_volume"] = float(df["volume"].median() or 0)
    rep["has_bid_ask_volume"] = "bid_vol" in df.columns and "ask_vol" in df.columns
    rep["has_symbol_col"] = "symbol" in df.columns

    # Jumps between consecutive bars (bad ticks or contract roll without back-adjust)
    # (session-open gaps between different days are excluded)
    jumps = df.with_columns(jump=(pl.col("open") - pl.col("close").shift(1)).abs() / tick,
                            same_day=pl.col("ts").dt.date() == pl.col("ts").dt.date().shift(1)) \
              .filter(pl.col("same_day") & (pl.col("jump") > jump_ticks))
    rep["big_jumps"] = jumps.height
    rep["big_jumps_sample"] = "; ".join(
        f"{r['ts']} ({r['jump']:.0f}t)" for r in jumps.head(5).iter_rows(named=True))

    # Daily coverage + intraday gaps (only gaps inside the same day)
    df = df.with_columns(day=pl.col("ts").dt.date(),
                         gap=pl.col("ts").diff().dt.total_seconds())
    daily = (df.group_by("day").agg(
        bars=pl.len(), first=pl.col("ts").min(), last=pl.col("ts").max(),
        max_gap_sec=pl.col("gap").filter(pl.col("day") == pl.col("day").shift(1)).max(),
        volume=pl.col("volume").sum() if "volume" in df.columns else pl.lit(None))
        .sort("day").with_columns(file=pl.lit(path.name)))
    rep["trading_days"] = daily.height
    rep["median_bars_per_day"] = int(daily["bars"].median() or 0)
    rep["thin_days(<20% median)"] = int((daily["bars"] < 0.2 * rep["median_bars_per_day"]).sum())

    weekdays = pl.date_range(daily["day"].min(), daily["day"].max(), "1d", eager=True)
    weekdays = weekdays.filter(weekdays.dt.weekday() <= 5)
    rep["missing_weekdays"] = len(set(weekdays.to_list()) - set(daily["day"].to_list()))
    rep["days_with_gap>{}s".format(gap_sec)] = int((daily["max_gap_sec"].fill_null(0) > gap_sec).sum())
    rep["timezone_guess"] = guess_timezone(df)
    return rep, daily


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="1-second futures data integrity check")
    ap.add_argument("--path", default=str(Path.home() / "Downloads"))
    ap.add_argument("--out", default=None, help="output folder (default: --path)")
    ap.add_argument("--tick", type=float, default=0.25, help="tick size (NQ/ES = 0.25)")
    ap.add_argument("--jump-ticks", type=int, default=40, help="flag bar-to-bar jumps above N ticks")
    ap.add_argument("--gap-sec", type=int, default=300, help="flag intraday gaps above N seconds")
    args = ap.parse_args()

    folder = Path(args.path).expanduser()
    own_outputs = {"integrity_report.csv", "daily_coverage.csv"}
    files = sorted({f for p in FILE_PATTERNS for f in folder.rglob(p) if f.name not in own_outputs})
    if not files:
        print(f"No data files found in {folder}")
        return 1
    print(f"Found {len(files)} file(s) in {folder}\n")

    reports, dailies = [], []
    for f in files:
        size_mb = f.stat().st_size / 1e6
        print(f"-> {f.name} ({size_mb:,.1f} MB)")
        try:
            rep, daily = check_file(f, args.tick, args.jump_ticks, args.gap_sec)
            rep["size_mb"] = round(size_mb, 1)
            reports.append(rep)
            dailies.append(daily)
            for k, v in rep.items():
                if k not in ("file", "columns"):
                    print(f"     {k:<26} {v}")
            print(f"     {'columns':<26} {rep['columns']}")
        except Exception as exc:  # keep scanning other files
            print(f"     SKIPPED: {exc}")
            reports.append({"file": f.name, "error": str(exc)})
        print()

    out = Path(args.out).expanduser() if args.out else folder
    pl.DataFrame(reports, infer_schema_length=None).write_csv(out / "integrity_report.csv")
    if dailies:
        all_days = pl.concat(dailies, how="diagonal_relaxed")
        all_days.write_csv(out / "daily_coverage.csv")
        overlap = all_days.group_by("day").agg(pl.len().alias("n_files")).filter(pl.col("n_files") > 1)
        print(f"Total unique trading days: {all_days['day'].n_unique()}  "
              f"| days present in >1 file (possible overlap/roll): {overlap.height}")
    print(f"Reports written to {out / 'integrity_report.csv'} and {out / 'daily_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
