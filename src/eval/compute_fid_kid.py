"""Compute whole-image FID per `inference_result_*` directory.

For each OP/SR result dir, pools all shard-level `predictions/*.png` and
`targets/*.png`, runs them through Inception-v3 (via torchmetrics), and
writes `fid_kid.json` in the dir's root containing:

  {
    "n_pred": ..., "n_target": ...,
    "fid":    ...,
    "feature_dim": 2048, "model": "inception_v3 (torchmetrics default)"
  }

Notes
-----
* "Whole-image" FID: predictions are the model's full composited output
  (model fill + paste-back of known region for OP), matching how the rest of
  the inpainting/OP literature reports FID.
* Per-region (hole-only) FID is NOT computed here — torchmetrics' FID expects
  Inception-natural-sized inputs, and arbitrary hole-cropped regions would
  introduce shape and statistic mismatches that make the number harder to
  interpret. Hole-aware quality is already covered by LPIPS/DISTS in the
  per-sample CSVs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

GENESIS_ROOT = os.environ.get(
    "GENESIS_RESULTS_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)


def list_image_paths(run_root: str, kind: str) -> List[str]:
    pat = os.path.join(run_root, "**", kind, "*.png")
    return sorted(glob.glob(pat, recursive=True))


def load_batch(paths: List[str], device, target_size: int = 299) -> torch.Tensor:
    """Load N PNGs as a uint8 N×3×H×W tensor on `device`. Resizes if needed
    (Inception expects 299×299; torchmetrics auto-resizes but doing it here
    keeps batching uniform)."""
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.size != (target_size, target_size):
            im = im.resize((target_size, target_size), Image.BICUBIC)
        imgs.append(np.asarray(im))
    arr = np.stack(imgs, axis=0)                       # N,H,W,3 uint8
    t   = torch.from_numpy(arr).permute(0, 3, 1, 2)    # N,3,H,W
    return t.to(device)


def _fid_from_paths(pred_paths, target_paths, device, batch_size):
    """Core compute: returns dict (fid, n_pred, n_target, elapsed_seconds) for
    the given image-path pair."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def feed(paths, real: bool):
        for i in range(0, len(paths), batch_size):
            batch = load_batch(paths[i:i + batch_size], device)
            fid.update(batch, real=real)

    t0 = time.time()
    feed(target_paths, real=True)
    feed(pred_paths,   real=False)
    fid_val = float(fid.compute().item())
    return {
        "n_pred":   len(pred_paths),
        "n_target": len(target_paths),
        "fid":      fid_val,
        "elapsed_seconds": time.time() - t0,
    }


def compute_partitioned(run_root: str, device, batch_size: int = 32,
                        force: bool = False,
                        partition_col: str = "parent_zl") -> dict:
    """Compute per-`partition_col` FID by walking shard CSVs and grouping
    rows by their `partition_col` value. Required columns: `partition_col`,
    `prediction_path`, `target_path`. Writes `fid_kid_by_<col>.json` in
    `run_root` with shape `{"<col_value>": {fid_dict}, ...}`.
    """
    out_path = os.path.join(run_root, f"fid_kid_by_{partition_col}.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            return {"cached": True, "path": out_path, "by": json.load(f)}

    csvs = sorted(c for c in glob.glob(os.path.join(run_root, "**", "samples.csv"),
                                       recursive=True)
                  if "/merged/" not in c)
    if not csvs:
        return {"skipped": True, "reason": "no shard CSVs", "path": out_path}

    import pandas as pd
    dfs = []
    for c in csvs:
        try:
            dfs.append(pd.read_csv(c))
        except Exception as e:
            print(f"  [warn] could not read {c}: {e}", file=sys.stderr)
    if not dfs:
        return {"skipped": True, "reason": "all CSVs empty", "path": out_path}
    df = pd.concat(dfs, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")]
    needed = {partition_col, "prediction_path", "target_path"}
    if not needed.issubset(df.columns):
        return {"skipped": True,
                "reason": f"missing columns: {needed - set(df.columns)}",
                "path": out_path}

    out_by = {}
    for val, sub in df.groupby(partition_col, dropna=True):
        try:
            key = int(val)
        except (TypeError, ValueError):
            continue
        preds   = [p for p in sub["prediction_path"].tolist() if isinstance(p, str) and os.path.isfile(p)]
        targets = [p for p in sub["target_path"].tolist()    if isinstance(p, str) and os.path.isfile(p)]
        if len(preds) < 2 or len(targets) < 2:
            print(f"  [skip] {partition_col}={key}: too few images "
                  f"(pred={len(preds)}, target={len(targets)})", flush=True)
            continue
        print(f"  computing {partition_col}={key}: pred={len(preds)} target={len(targets)}...",
              flush=True)
        result = _fid_from_paths(preds, targets, device, batch_size)
        out_by[str(key)] = result
    if not out_by:
        return {"skipped": True, "reason": "no partitions had data", "path": out_path}
    with open(out_path, "w") as f:
        json.dump(out_by, f, indent=2)
    return {"path": out_path, "by": out_by}


def compute_one(run_root: str, device, batch_size: int = 32,
                force: bool = False) -> dict:
    out_path = os.path.join(run_root, "fid_kid.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            cached = json.load(f)
        return {"cached": True, "path": out_path, **cached}

    pred_paths   = list_image_paths(run_root, "predictions")
    target_paths = list_image_paths(run_root, "targets")
    if not pred_paths or not target_paths:
        return {"skipped": True, "reason": "no images", "n_pred": len(pred_paths),
                "n_target": len(target_paths), "path": out_path}

    core = _fid_from_paths(pred_paths, target_paths, device, batch_size)
    out = {
        **core,
        "feature_dim": 2048,
        "image_size_inception": 299,
        "model": "inception_v3 (torchmetrics default)",
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return {"path": out_path, **out}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=GENESIS_ROOT)
    p.add_argument("--pattern", default="inference_result_*",
                   help="Glob under --root for run dirs.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", default=32, type=int)
    p.add_argument("--force", action="store_true", help="Overwrite existing fid_kid.json.")
    p.add_argument("--only", default=None,
                   help="Optional substring filter — only compute dirs whose basename matches.")
    p.add_argument("--partition", default=None,
                   help="If set (e.g. 'parent_zl'), compute per-partition FID by "
                        "grouping shard-CSV rows on this column (uses prediction_path/"
                        "target_path) instead of the dir-level aggregate.")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[fid] device={device}", flush=True)

    dirs = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    dirs = [d for d in dirs if os.path.isdir(d)
            and "smoke" not in os.path.basename(d).lower()]
    if args.only:
        dirs = [d for d in dirs if args.only in os.path.basename(d)]

    print(f"[fid] found {len(dirs)} dirs (partition={args.partition or '<dir-level>'})", flush=True)
    for d in dirs:
        name = os.path.basename(d)
        print(f"\n[fid] === {name} ===", flush=True)
        try:
            if args.partition:
                r = compute_partitioned(d, device, batch_size=args.batch_size,
                                        force=args.force, partition_col=args.partition)
                if r.get("cached"):
                    print(f"  cached partitions: {sorted(r['by'].keys())}", flush=True)
                elif r.get("skipped"):
                    print(f"  SKIPPED ({r.get('reason')})", flush=True)
                else:
                    for k in sorted(r["by"].keys(), key=lambda x: int(x)):
                        v = r["by"][k]
                        print(f"  {args.partition}={k}: n={v['n_pred']}  "
                              f"FID={v['fid']:.3f}  ({v['elapsed_seconds']:.1f}s)", flush=True)
            else:
                r = compute_one(d, device, batch_size=args.batch_size, force=args.force)
                if r.get("cached"):
                    print(f"  cached  FID={r.get('fid', '?'):.3f}", flush=True)
                elif r.get("skipped"):
                    print(f"  SKIPPED ({r.get('reason')})", flush=True)
                else:
                    print(f"  n_pred={r['n_pred']} n_target={r['n_target']}  "
                          f"FID={r['fid']:.3f}  ({r['elapsed_seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
