"""Aggregate samples.csv files into Markdown pivot tables. Pure pandas.

Examples (see README for more):
  python src/eval/aggregate.py --task sr --metric lpips --pivot run_id --columns split,zl_bucket
  python src/eval/aggregate.py --task op --metric psnr  --pivot exp    --columns split,regime
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import List

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def discover_runs(results_root: str) -> List[str]:
    """Return paths to every run directory containing samples.csv."""
    pattern = os.path.join(results_root, "**", "samples.csv")
    return sorted(glob.glob(pattern, recursive=True))


def load_run(samples_csv: str) -> pd.DataFrame:
    df = pd.read_csv(samples_csv)
    run_dir = os.path.dirname(samples_csv)
    run_id  = os.path.basename(run_dir)

    run_meta_path = os.path.join(run_dir, "run.json")
    if os.path.isfile(run_meta_path):
        with open(run_meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}

    df["run_id"]  = run_id
    df["task"]    = meta.get("task", "")
    df["exp"]     = meta.get("exp")  or ""
    df["stage"]   = meta.get("stage") or ""
    df["model"]   = (meta.get("extra") or {}).get("model", "")
    df["scale"]   = (meta.get("extra") or {}).get("scale", "")
    df["regime"]  = (meta.get("extra") or {}).get("regime", "")
    df["cfg"]     = (meta.get("config") or {}).get("cfg", "")

    if df.get("split") is None or df["split"].fillna("").eq("").all():
        df["split"] = meta.get("split", "")
    return df


def add_zl_bucket(df: pd.DataFrame, threshold: int = 15) -> pd.DataFrame:
    """SR convention: ↓zl = parent_zl ≤ 15, ↑zl = > 15."""
    if "parent_zl" in df.columns:
        df = df.copy()
        df["zl_bucket"] = df["parent_zl"].apply(
            lambda z: "↓zl" if pd.notna(z) and int(z) <= threshold else
                      ("↑zl" if pd.notna(z) else "")
        )
    return df


def add_zoom_bucket(df: pd.DataFrame, threshold: int = 15) -> pd.DataFrame:
    """OP convention: bucket on tile zoom column (`zoom`)."""
    if "zoom" in df.columns:
        df = df.copy()
        df["zoom_bucket"] = df["zoom"].apply(
            lambda z: "↓zoom" if pd.notna(z) and z != "" and int(z) <= threshold else
                      ("↑zoom" if pd.notna(z) and z != "" else "")
        )
    return df


def aggregate(
    df: pd.DataFrame,
    *,
    metric: str,
    pivot: str,
    columns: List[str],
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate `metric` over (pivot, *columns) groups."""
    df = df.copy()
    df = df[df[metric].astype(str).ne("")]                 # drop blank
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])

    keys = [pivot] + list(columns)
    grouped = df.groupby(keys, dropna=False)[metric]
    if agg == "mean":
        out = grouped.mean()
    elif agg == "median":
        out = grouped.median()
    elif agg == "std":
        out = grouped.std()
    elif agg == "count":
        out = grouped.count()
    elif agg == "mean±std":
        m = grouped.mean()
        s = grouped.std()
        out = m.combine(s, lambda mv, sv: f"{mv:.4f} ± {sv:.4f}")
    else:
        raise ValueError(f"--agg {agg!r} not in {{mean, median, std, count, mean±std}}")

    # Build wide table.
    if not columns:
        wide = out.to_frame(name=metric)
    else:
        wide = out.unstack(columns)
    return wide


def to_markdown(wide: pd.DataFrame, *, decimals: int = 4) -> str:
    """Pretty Markdown table with rounded floats."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v) if v is not None else ""
    # pandas ≥ 2.1 exposes DataFrame.map; older versions only have applymap.
    try:
        df = wide.map(fmt)
    except AttributeError:
        df = wide.applymap(fmt)
    return df.to_markdown()


def get_args():
    p = argparse.ArgumentParser("Aggregate per-sample eval CSVs into Markdown tables.")
    p.add_argument("--results_root", default=os.path.join(_THIS_DIR, "results"),
                   help="Root dir to scan for */samples.csv. Default: src/eval/results")
    p.add_argument("--task", default=None, choices=(None, "sr", "op"),
                   help="Optional filter by task.")
    p.add_argument("--metric", default="psnr",
                   help="Metric column to aggregate. Use any column from samples.csv.")
    p.add_argument("--pivot", default="run_id",
                   help='Row dimension. Common choices: "run_id", "exp", "model", "zl_bucket".')
    p.add_argument("--columns", default="split",
                   help='Comma-separated column dimensions. e.g. "split,zl_bucket" or "regime".')
    p.add_argument("--agg", default="mean",
                   choices=("mean", "median", "std", "count", "mean±std"))
    p.add_argument("--decimals", default=4, type=int)
    p.add_argument("--filter", default=None,
                   help="Optional pandas query string applied before aggregation.")
    p.add_argument("--zl_threshold", default=15, type=int,
                   help="Cutoff for ↓zl/↑zl bucket (SR) or ↓zoom/↑zoom (OP).")
    return p.parse_args()


def main():
    args = get_args()
    runs = discover_runs(args.results_root)
    if not runs:
        print(f"[aggregate] No samples.csv found under {args.results_root}", file=sys.stderr)
        sys.exit(1)

    frames = []
    for r in runs:
        try:
            df = load_run(r)
            frames.append(df)
        except Exception as e:
            print(f"[aggregate] WARN: could not load {r}: {e}", file=sys.stderr)
    df = pd.concat(frames, ignore_index=True)
    if args.task:
        df = df[df["task"] == args.task]
    df = add_zl_bucket(df, threshold=args.zl_threshold)
    df = add_zoom_bucket(df, threshold=args.zl_threshold)
    if args.filter:
        df = df.query(args.filter)

    cols = [c.strip() for c in args.columns.split(",") if c.strip()]
    if args.metric not in df.columns:
        print(f"[aggregate] Metric column {args.metric!r} not in CSV. Available: "
              f"{[c for c in df.columns if c not in ('run_id','task','exp','stage','model','scale','regime','cfg')]}",
              file=sys.stderr)
        sys.exit(2)

    table = aggregate(df, metric=args.metric, pivot=args.pivot, columns=cols, agg=args.agg)
    print(f"# {args.metric}  (pivot={args.pivot}, columns={cols}, agg={args.agg}, n_runs={df['run_id'].nunique()})\n")
    print(to_markdown(table, decimals=args.decimals))


if __name__ == "__main__":
    main()
