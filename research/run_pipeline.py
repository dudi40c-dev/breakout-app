#!/usr/bin/env python3
"""
One command: Stage 0 (integrity) -> 1 (clean dataset) -> 2 (features/events) -> 3 (evaluation)
for strategy A (VWAP mean reversion) and optionally B.

Writes <out>/summary_for_claude.txt - a short text file to paste back into the chat.

Usage (Windows / macOS / Linux):
    pip install polars pyarrow numpy
    python run_pipeline.py                                  # ~/Downloads, timezone auto-detected
    python run_pipeline.py --src "D:/data/nq" --tz-in UTC   # explicit folder / timezone
    python run_pipeline.py --with-b                         # also evaluate ORB (B) variants
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_integrity_check import FILE_PATTERNS, check_file  # noqa: E402

TZ_MAP = {"US/Eastern": "America/New_York", "US/Central": "America/Chicago",
          "UTC": "UTC", "Asia/Jerusalem": "Asia/Jerusalem"}

# Strategy A variants from the source rules (see features_ab.py for parameters)
A_VARIANTS = {
    "A_k2_rsi5m": [],
    "A_k2_rsi15m": ["--a-rsi-tf", "15m"],
    "A_k3_rsi15m": ["--a-k", "3", "--a-rsi-tf", "15m"],
}


def run(script: str, *args: str) -> str:
    cmd = [sys.executable, str(HERE / script), *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"\n{script} failed:\n{res.stdout}\n{res.stderr}")
    return res.stdout


def stage0(src: Path, out: Path) -> tuple[str, str | None]:
    skip = {"integrity_report.csv", "daily_coverage.csv"}
    files = sorted({f for p in FILE_PATTERNS for f in src.rglob(p) if f.name not in skip})
    lines, tz_votes = [], Counter()
    for f in files:
        try:
            with redirect_stdout(io.StringIO()):
                rep, _ = check_file(f, 0.25, 40, 300)
        except Exception as exc:
            lines.append(f"  {f.name}: SKIPPED ({str(exc)[:120]})")
            continue
        tz = next((v for k, v in TZ_MAP.items() if k in rep["timezone_guess"]), None)
        if tz:
            tz_votes[tz] += rep["rows"]
        lines.append(
            f"  {f.name}: rows {rep['rows']:,} | {rep['start'][:10]}..{rep['end'][:10]} | days {rep['trading_days']}"
            f" | dup {rep['duplicate_ts']} | ohlc_bad {rep['ohlc_violations']} | jumps {rep['big_jumps']}"
            f" | bid/ask vol {rep['has_bid_ask_volume']} | symbol col {rep['has_symbol_col']}"
            f" | {rep['timezone_guess']} | columns: {rep['columns']}")
    if not files:
        raise SystemExit(f"no data files in {src}")
    tz = tz_votes.most_common(1)[0][0] if tz_votes else None
    return "\n".join(lines), tz


def top_rows(results_csv: Path, min_trades: int, n: int = 8) -> str:
    d = pl.read_csv(results_csv).filter(pl.col("trades") >= min_trades)
    if d.is_empty():
        return f"    (no configuration with >= {min_trades} trades)"
    both = d.filter((pl.col("exp_R_1st_half") > 0) & (pl.col("exp_R_2nd_half") > 0))
    cols = ["stop_col", "target", "hold_min", "trades", "win_rate", "exp_R", "t_stat",
            "exp_R_1st_half", "exp_R_2nd_half", "profit_factor", "max_dd_R", "max_loss_streak",
            "net_per_trade_micro_$", "worst_day_micro_$"]
    rows = [" | ".join(cols)]
    rows += [" | ".join(str(v) for v in r) for r in d.sort("exp_R", descending=True).select(cols).head(n).iter_rows()]
    rows.append(f"configs: {d.height} | positive in both halves: {both.height} | "
                f"of those t>=2: {both.filter(pl.col('t_stat') >= 2).height} | t>=3: "
                f"{both.filter(pl.col('t_stat') >= 3).height} | mean exp_R all configs: {d['exp_R'].mean():.3f}")
    return "\n".join("    " + r for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Stages 0-3 and write summary_for_claude.txt")
    ap.add_argument("--src", default=str(Path.home() / "Downloads"))
    ap.add_argument("--out", default=str(Path.home() / "nq_research"))
    ap.add_argument("--tz-in", default="auto", help="auto | America/New_York | UTC | Asia/Jerusalem ...")
    ap.add_argument("--with-b", action="store_true", help="also evaluate ORB variants")
    ap.add_argument("--min-trades", type=int, default=50)
    ap.add_argument("--comm-micro", default="1.0", help="$ round turn MNQ (verify with TPT)")
    ap.add_argument("--comm-full", default="4.0", help="$ round turn NQ (verify with TPT)")
    a = ap.parse_args()

    src, out = Path(a.src).expanduser(), Path(a.out).expanduser()
    clean = out / "clean"
    out.mkdir(parents=True, exist_ok=True)
    summary = [f"SOURCE: {src}"]

    print("Stage 0: integrity check ...")
    report, tz_guess = stage0(src, out)
    tz = tz_guess if a.tz_in == "auto" else a.tz_in
    if tz is None:
        raise SystemExit("timezone could not be detected - rerun with --tz-in (e.g. UTC or America/New_York)")
    summary += ["\n[STAGE 0] files", report, f"timezone used: {tz} ({'auto' if a.tz_in == 'auto' else 'manual'})"]

    print(f"Stage 1: clean dataset (tz {tz}) ...")
    s1 = run("build_dataset.py", "--src", str(src), "--out", str(clean), "--tz-in", tz)
    summary += ["\n[STAGE 1]", s1.strip().splitlines()[-1]]

    variants = dict(A_VARIANTS)
    if a.with_b:
        variants["B"] = []
    for name, extra in variants.items():
        print(f"Stage 2+3: {name} ...")
        feat = out / f"features_{name}"
        s2 = run("features_ab.py", "--data", str(clean), "--out", str(feat), *extra)
        prefix = "A -" if name.startswith("A") else "B -"
        lines = s2.splitlines()
        ev_lines = [ln for i, ln in enumerate(lines)
                    if ln.startswith(prefix) or (i and lines[i - 1].startswith(prefix) and "risk ticks" in ln)]
        summary += [f"\n[{name}] stage 2 args: {' '.join(extra) or '(defaults)'}", *("  " + ln for ln in ev_lines)]
        files = ["events_A"] if name.startswith("A") else ["events_B", "events_B_cont"]
        for ev in files:
            ev_path = feat / f"{ev}.parquet"
            if pl.read_parquet(ev_path).is_empty():
                summary.append(f"  {ev}: no events")
                continue
            ev_out = out / f"eval_{name}_{ev}"
            run("evaluate_events.py", "--data", str(clean), "--events", str(ev_path), "--out", str(ev_out),
                "--min-trades", str(a.min_trades), "--comm-micro", a.comm_micro, "--comm-full", a.comm_full)
            summary += [f"  [STAGE 3] {ev} (slip 1t, comm ${a.comm_micro}/{a.comm_full})",
                        top_rows(ev_out / "results.csv", a.min_trades)]

    txt = out / "summary_for_claude.txt"
    txt.write_text("\n".join(summary), encoding="utf-8")
    print(f"\nDone. Paste the contents of this file into the chat:\n  {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
