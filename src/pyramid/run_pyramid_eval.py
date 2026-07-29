"""Batch driver for the Dense500 pyramid evaluation (see demos/evaluate_pyramid.md).

Generates a full depth-4 pyramid for each (protocol, window, site) with the
``unified`` strategy and saves the **complete pyramid** (all 85 tiles, every
level) in a smart, self-describing, re-loadable format:

    <out_root>/
        config.json               # run-level metadata (config + schema)
        manifest.tsv              # the 1500 work units (worker mode)
        proto<P>/W<w>_l<l>/
            tiles/<site_id>.parquet   # 85 rows: gen image bytes + coords + provenance
            mosaic/<site_id>.jpg      # stitched leaf-level mosaic (quick visual)
        .worker_logs/worker_<id>.jsonl   # per-site timing / model-call counts / seeds
        .claims/                  # atomic work-claim markers (worker mode)

The per-tile parquet mirrors the dense500 schema (+ ``protocol`` / ``is_seed`` /
``source``) so evaluation re-loads generated tiles columnar and joins to GT by
``(zoom, tile_x, tile_y)``. Per-site files make the run trivially resumable
(an existing parquet is skipped).

Per-subtree RNG matches the spec: ``Random(global_seed*1_000_003 + idx)`` where
``idx = (zoom_window-1)*100 + site_index_in_window`` (the benchmark's own stable,
shard-independent site index).

Two run modes:

  • single cell (used for smoke tests):
      python -m pyramid.run_pyramid_eval --protocol A --window 4 --sites 0:2 ...

  • worker (the campaign): N workers drain a shared manifest, each loading the
    model once and claiming units atomically (auto-balances fast/slow GPUs):
      python -m pyramid.run_pyramid_eval --worker --manifest <root>/manifest.tsv \
          --out_root <root> --worker_id h100_0 ...
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from typing import Dict, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .dataset import GlobalDense, Subtree, get_protocol
from .geometry import stitch_level
from .models import ModelBundle
from .strategies import GenContext, get_strategy
from .tiles import TileStore, subtree_tiles

WINDOW_TO_L = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}

_PARQUET_COLS = ("site_id", "zoom_window", "site_index_in_window", "protocol",
                 "zoom", "tile_x", "tile_y", "is_seed", "source", "image")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset indexing
# ─────────────────────────────────────────────────────────────────────────────

def build_window_index(gd: GlobalDense) -> Dict[str, Tuple[int, int]]:
    """site_id -> (zoom_window, site_index_in_window) from the parquet metadata."""
    out: Dict[str, Tuple[int, int]] = {}
    for f in gd.files:
        t = pq.read_table(f, columns=["site_id", "zoom_window",
                                      "site_index_in_window"]).to_pydict()
        for sid, w, idx in zip(t["site_id"], t["zoom_window"],
                               t["site_index_in_window"]):
            out[sid] = (int(w), int(idx))
    return out


def build_subtree_index(gd: GlobalDense, widx: Dict[str, Tuple[int, int]],
                        depth: int) -> Dict[Tuple[int, int], Subtree]:
    """(zoom_window, site_index_in_window) -> Subtree (carved once)."""
    out: Dict[Tuple[int, int], Subtree] = {}
    for s in gd.iter_subtrees(depth=depth):
        if s.site_id in widx:
            out[widx[s.site_id]] = s
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def _encode(img: Image.Image, fmt: str, q: int) -> bytes:
    buf = io.BytesIO()
    if fmt == "jpeg":
        img.convert("RGB").save(buf, "JPEG", quality=q)
    else:
        img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def save_site_parquet(path: str, sub: Subtree, store: TileStore,
                      meta: Dict, fmt: str, q: int) -> Dict[str, int]:
    """Write all resolved subtree tiles + provenance; return per-source counts."""
    rows: Dict[str, list] = {k: [] for k in _PARQUET_COLS}
    src_counts: Dict[str, int] = {}
    for t in subtree_tiles(sub.root, sub.depth):
        rec = store.get(t)
        if rec is None or rec.image is None:
            continue
        src = rec.state.value if hasattr(rec.state, "value") else str(rec.state)
        src_counts[src] = src_counts.get(src, 0) + 1
        rows["site_id"].append(sub.site_id)
        rows["zoom_window"].append(meta["zoom_window"])
        rows["site_index_in_window"].append(meta["site_index_in_window"])
        rows["protocol"].append(meta["protocol"])
        rows["zoom"].append(t[0])
        rows["tile_x"].append(t[1])
        rows["tile_y"].append(t[2])
        rows["is_seed"].append(bool(store.is_seed(t)))
        rows["source"].append(src)
        rows["image"].append(_encode(rec.image, fmt, q))
    tmp = path + ".tmp"
    pq.write_table(pa.table(rows), tmp)
    os.replace(tmp, path)            # atomic: a half-written file never looks done
    return src_counts


def cell_dir(out_root: str, protocol: str, window: int) -> str:
    return os.path.join(out_root, f"proto{protocol}", f"W{window}_l{WINDOW_TO_L[window]}")


# ─────────────────────────────────────────────────────────────────────────────
# One generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_unit(gd: GlobalDense, sub: Subtree, protocol: str, window: int,
                  site_index: int, bundle: ModelBundle, params: Dict,
                  out_root: str, fmt: str, q: int, save_cmp: bool,
                  global_seed: int) -> Optional[Dict]:
    """Generate one site's full pyramid and persist tiles + mosaic. None if already done."""
    site = sub.site_id
    cdir = cell_dir(out_root, protocol, window)
    for sd in ("tiles", "mosaic", "cmp"):
        os.makedirs(os.path.join(cdir, sd), exist_ok=True)
    pq_path = os.path.join(cdir, "tiles", f"{site}.parquet")
    mo_path = os.path.join(cdir, "mosaic", f"{site}.jpg")
    if os.path.isfile(pq_path) and os.path.isfile(mo_path):
        return None

    gidx = (window - 1) * 100 + site_index
    rng = random.Random(global_seed * 1_000_003 + gidx)
    seeds = get_protocol(protocol)(sub, rng)

    store = TileStore()
    for s in seeds:
        store.put_seed(s, gd.image_by_row(sub.row_index[s]).convert("RGB"))

    bundle.sr_calls = bundle.op_calls = 0
    ctx = GenContext(bundle=bundle, params=params, _log=None)
    t0 = time.time()
    get_strategy("unified").generate(store, sub, seeds, ctx)
    elapsed = time.time() - t0

    meta = {"protocol": protocol, "zoom_window": window, "site_index_in_window": site_index}
    src_counts = save_site_parquet(pq_path, sub, store, meta, fmt, q)

    leaf = sub.leaf_zoom
    gen = stitch_level({t: store.image(t) for t in sub.at_zoom(leaf)
                        if store.resolved(t)}, sub, leaf)
    gen.save(mo_path, "JPEG", quality=92)
    if save_cmp:
        os.makedirs(os.path.join(cdir, "cmp"), exist_ok=True)
        gt = stitch_level({t: gd.image_by_row(sub.row_index[t])
                           for t in sub.at_zoom(leaf)}, sub, leaf)
        canvas = Image.new("RGB", (gen.width * 2 + 8, gen.height), (0, 0, 0))
        canvas.paste(gen, (0, 0))
        canvas.paste(gt, (gen.width + 8, 0))
        canvas.save(os.path.join(cdir, "cmp", f"{site}.jpg"), "JPEG", quality=88)

    return {"site_id": site, "protocol": protocol, "global_idx": gidx,
            "zoom_window": window, "site_index_in_window": site_index,
            "seeds": [list(s) for s in seeds], "n_tiles": sum(src_counts.values()),
            "src_counts": src_counts, "elapsed_s": round(elapsed, 2),
            **bundle.call_counts()}


# ─────────────────────────────────────────────────────────────────────────────
# Run-level metadata
# ─────────────────────────────────────────────────────────────────────────────

def write_config(out_root: str, args, params: Dict) -> None:
    cfg = {
        "dataset": "dcher95/dense500",
        "strategy": "unified",
        "algo_source": "github.com/mvrl/genesis@unified",
        "algo_commit": args.algo_commit,
        "config_provenance": "demo defaults (demos/app_pyramid_unified.py): "
                             "JiT-H/16, steps=50, cfg=2.5, overlap=32, uni_t=0.40, "
                             "reconcile_skip_unblended=True; op_batch/guidance/sync = strategy defaults",
        "arch": args.arch, "steps": args.steps, "cfg": args.cfg,
        "uni_overlap": args.uni_overlap, "uni_t": args.uni_t,
        "reconcile_skip_unblended": True,
        "depth": args.depth, "global_seed": args.global_seed,
        "img_format": args.img_format, "img_quality": args.img_quality,
        "protocols": {"A": "single seed, level ratio 1:4:8",
                      "B": "3 seeds, one per non-root level, independent",
                      "C": "4 seeds: 2 leaf + 1 D-1 + 1 D-2, independent"},
        "window_to_l": WINDOW_TO_L,
        "parquet_schema": {
            "row": "one tile of a site's depth-4 quadtree (85 rows/site)",
            "columns": list(_PARQUET_COLS),
            "image": f"{args.img_format} bytes, 256x256 RGB",
            "source": "provenance: seed | sr | outpaint | downsample | refined",
            "join_to_GT": "by (zoom, tile_x, tile_y) against dcher95/dense500",
        },
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # mode selection
    ap.add_argument("--worker", action="store_true", help="worker mode: drain --manifest")
    ap.add_argument("--manifest", default=None, help="manifest.tsv (worker mode)")
    ap.add_argument("--worker_id", default="0", help="label for this worker's log")
    # single-cell mode
    ap.add_argument("--protocol", help="seed protocol (single-cell mode)")
    ap.add_argument("--window", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--sites", default="0:100", help="site_index range lo:hi (single-cell)")
    # shared config
    ap.add_argument("--global_seed", type=int, default=0)
    ap.add_argument("--arch", default="JiT-H/16")
    ap.add_argument("--steps", type=int, default=50)   # demo default: app_pyramid_unified.py _DEF_STEPS
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--uni_overlap", type=int, default=32)
    ap.add_argument("--uni_t", type=float, default=0.40)
    # Optional seam-fix knobs (ablation only). Unset => strategy defaults
    # (seam_thresh=14, band=64, max=64), i.e. canonical demo behavior unchanged.
    ap.add_argument("--seam_thresh", type=float, default=None)
    ap.add_argument("--seam_band", type=int, default=None)
    ap.add_argument("--seam_max", type=int, default=None)
    ap.add_argument("--uni_guidance", type=float, default=None,
                    help="EXPLORATORY: late-step HF guidance scale (needs the strategy patch)")
    ap.add_argument("--uni_sync", type=float, default=None,
                    help="EXPLORATORY: MultiDiffusion overlap-band velocity-sync scale (needs the strategy patch)")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out_root", default="outputs/dense500_eval")
    ap.add_argument("--img_format", choices=["jpeg", "png"], default="jpeg")
    ap.add_argument("--img_quality", type=int, default=95)
    ap.add_argument("--save_comparison", action="store_true")
    ap.add_argument("--data_dir", default=None, help="override dense500 data dir")
    ap.add_argument("--algo_commit", default="",
                    help="unified commit hash of the algorithm code (recorded in config.json for provenance)")
    args = ap.parse_args()

    gd = GlobalDense(data_dir=args.data_dir) if args.data_dir else GlobalDense()
    widx = build_window_index(gd)
    sub_index = build_subtree_index(gd, widx, args.depth)
    params = {"uni_overlap": args.uni_overlap, "uni_t": args.uni_t,
              "reconcile_skip_unblended": True}
    if args.seam_thresh is not None:
        params["seam_thresh"] = args.seam_thresh
    if args.seam_band is not None:
        params["seam_band"] = args.seam_band
    if args.seam_max is not None:
        params["seam_max"] = args.seam_max
    if args.uni_guidance is not None:
        params["uni_guidance"] = args.uni_guidance
    if args.uni_sync is not None:
        params["uni_sync"] = args.uni_sync
    write_config(args.out_root, args, params)
    bundle = ModelBundle(arch=args.arch, steps=args.steps, cfg=args.cfg)

    def _gen(protocol, window, site_index):
        sub = sub_index.get((window, site_index))
        if sub is None:
            return {"skip": f"no site (W{window}, idx{site_index})"}
        return generate_unit(gd, sub, protocol, window, site_index, bundle, params,
                             args.out_root, args.img_format, args.img_quality,
                             args.save_comparison, args.global_seed)

    if args.worker:
        _run_worker(args, sub_index, _gen)
    else:
        _run_single_cell(args, _gen)


def _run_single_cell(args, _gen) -> None:
    assert args.protocol and args.window, "single-cell mode needs --protocol and --window"
    lo, hi = (int(x) for x in args.sites.split(":"))
    cdir = cell_dir(args.out_root, args.protocol, args.window)
    print(f"=== proto{args.protocol} W{args.window} sites[{lo}:{hi}] -> {cdir} ===", flush=True)
    log = os.path.join(cdir, "runs.jsonl")
    os.makedirs(cdir, exist_ok=True)
    done = 0
    with open(log, "a") as rf:
        for s in range(lo, hi):
            rec = _gen(args.protocol, args.window, s)
            if rec is None:
                print(f"  skip idx{s} (done)", flush=True); continue
            if "skip" in rec:
                print(f"  {rec['skip']}", flush=True); continue
            rf.write(json.dumps(rec) + "\n"); rf.flush(); done += 1
            print(f"  [{done}] {rec['site_id']} {rec['elapsed_s']}s "
                  f"sr={rec['sr_calls']} op={rec['op_calls']} {rec['src_counts']}", flush=True)
    print(f"=== CELL DONE ({done} new) ===", flush=True)


def _run_worker(args, sub_index, _gen) -> None:
    assert args.manifest, "worker mode needs --manifest"
    claim_dir = os.path.join(args.out_root, ".claims")
    log_dir = os.path.join(args.out_root, ".worker_logs")
    os.makedirs(claim_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"worker_{args.worker_id}.jsonl")

    def claim(unit: str) -> bool:
        try:
            os.mkdir(os.path.join(claim_dir, unit)); return True
        except FileExistsError:
            return False

    with open(args.manifest) as mf:
        units = [ln.split() for ln in mf if ln.strip() and not ln.startswith("idx")]

    print(f"=== worker {args.worker_id}: {len(units)} units in manifest ===", flush=True)
    done = 0
    with open(log_path, "a") as rf:
        for row in units:
            _, P, W, S = row[0], row[1], int(row[2]), int(row[3])
            sub = sub_index.get((W, S))
            if sub is None:
                continue
            pqp = os.path.join(cell_dir(args.out_root, P, W), "tiles", f"{sub.site_id}.parquet")
            if os.path.isfile(pqp):
                continue                              # already generated (resume)
            if not claim(f"{P}_W{W}_s{S}"):
                continue                              # another worker owns it
            rec = _gen(P, W, S)
            if rec is None or "skip" in rec:
                continue
            rec["worker"] = args.worker_id
            rf.write(json.dumps(rec) + "\n"); rf.flush(); done += 1
            print(f"  [{args.worker_id} #{done}] proto{P} W{W} s{S} {rec['site_id']} "
                  f"{rec['elapsed_s']}s sr={rec['sr_calls']} op={rec['op_calls']}", flush=True)
    print(f"=== worker {args.worker_id} DONE ({done} generated) ===", flush=True)


if __name__ == "__main__":
    main()
