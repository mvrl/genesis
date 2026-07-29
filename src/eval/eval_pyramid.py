#!/usr/bin/env python
"""Evaluate Dense500 full-pyramid generations against ground truth.

Consumes the generation output written by ``pyramid.run_pyramid_eval``
(``<gen_root>/proto<P>/W<w>_l<l>/tiles/<site>.parquet`` — 85 tiles/site, each
row = image bytes + ``(zoom,tile_x,tile_y)`` + ``is_seed`` + ``source``) and the
GT benchmark (``dcher95/dense500`` via ``pyramid.dataset.GlobalDense``).

Two metric families, both backed by **standard library implementations** (no
re-implementation):

  • Reference (per generated tile, vs the GT tile at the same coord) — via
    ``src/eval/common.compute_metrics`` (torchmetrics): ``psnr`` (data_range=1),
    ``ssim`` (11×11 gaussian), ``lpips`` (squeeze, inputs in [-1,1]).
  • Distribution (set of generated tiles vs set of GT tiles) — torchmetrics
    ``FrechetInceptionDistance`` (feature=2048).

Dependencies: ``torchmetrics`` + ``torch-fidelity`` (the FID backend).

Seed tiles are excluded from scoring (they are real GT pasted into the pyramid).
Results are stratified overall / by protocol / by window / by pyramid level
(``level = zoom − root_zoom``: 0=root … 3=leaf) / by protocol×level.

Adding a metric:
  • reference metric → add it to ``compute_metrics`` in ``common.py`` (one place),
    then pass its name in ``--ref_metrics``.
  • distribution metric → add an entry to ``DISTRIBUTION_METRICS`` below
    (a callable returning a {name: value} dict from two uint8 NCHW batches).

Run (full, on a GPU):
  PYTHONPATH=src python src/eval/eval_pyramid.py --gen_root outputs/dense500_eval \
      --out_dir outputs/dense500_eval/_metrics --device cuda
Smoke (small subset, CPU):
  PYTHONPATH=src python src/eval/eval_pyramid.py --max_sites_per_cell 3 \
      --fid_strata overall --device cpu
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `common`
import common  # noqa: E402

from pyramid.dataset import GlobalDense  # noqa: E402  (PYTHONPATH=src)

WINDOW_TO_L = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
REF_METRICS_DEFAULT = ["psnr", "ssim", "lpips"]
_INCEPTION_SIZE = 299


# ─────────────────────────────────────────────────────────────────────────────
# Distribution metrics (registry) — standard torchmetrics implementations.
# ─────────────────────────────────────────────────────────────────────────────

class FidAccumulator:
    """One stratum's FID, fed uint8 NCHW batches incrementally.

    Standard impl: torchmetrics FrechetInceptionDistance (Inception-v3, 2048-d).
    ``normalize=False`` → inputs are uint8 in [0,255].
    """

    def __init__(self, device: torch.device):
        from torchmetrics.image.fid import FrechetInceptionDistance
        self.fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        self.n_real = self.n_fake = 0

    def update(self, imgs_u8: torch.Tensor, real: bool) -> None:
        self.fid.update(imgs_u8, real=real)
        if real:
            self.n_real += imgs_u8.shape[0]
        else:
            self.n_fake += imgs_u8.shape[0]

    def compute(self) -> Dict:
        out: Dict = {"n_real": self.n_real, "n_fake": self.n_fake}
        try:
            out["fid"] = float(self.fid.compute().item())
        except Exception as e:               # too few samples, etc.
            out["fid"] = None
            out["fid_error"] = str(e)[:120]
        return out


# To add another distribution metric (e.g. precision/recall), register a builder
# here that exposes .update(imgs_u8, real) / .compute() -> dict, and wire it in
# build_accumulators().
DISTRIBUTION_METRICS = {"fid": FidAccumulator}


def _pil_to_u8_nchw(pils: List[Image.Image], device: torch.device) -> torch.Tensor:
    """List of PIL RGB → uint8 N×3×299×299 tensor (Inception input size)."""
    arrs = []
    for im in pils:
        if im.size != (_INCEPTION_SIZE, _INCEPTION_SIZE):
            im = im.resize((_INCEPTION_SIZE, _INCEPTION_SIZE), Image.BICUBIC)
        arrs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()
    return t.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def load_gen_parquet(path: str) -> Tuple[Dict, Dict[Tuple[int, int, int], Tuple[Image.Image, bool, str]]]:
    t = pq.read_table(path).to_pydict()
    meta = {"site_id": t["site_id"][0], "protocol": t["protocol"][0],
            "zoom_window": int(t["zoom_window"][0])}
    tiles: Dict[Tuple[int, int, int], Tuple[Image.Image, bool, str]] = {}
    for i in range(len(t["zoom"])):
        key = (int(t["zoom"][i]), int(t["tile_x"][i]), int(t["tile_y"][i]))
        tiles[key] = (Image.open(io.BytesIO(t["image"][i])).convert("RGB"),
                      bool(t["is_seed"][i]), str(t["source"][i]))
    return meta, tiles


def discover_parquets(gen_root: str, protocols: List[str], windows: List[int],
                      max_per_cell: int) -> List[Tuple[str, int, str]]:
    out = []
    for P in protocols:
        for W in windows:
            cell = os.path.join(gen_root, f"proto{P}", f"W{W}_l{WINDOW_TO_L[W]}", "tiles")
            ps = sorted(glob.glob(os.path.join(cell, "*.parquet")))
            if max_per_cell:
                ps = ps[:max_per_cell]
            out += [(P, W, p) for p in ps]
    return out


def _strata_for(P: str, W: int, level: int, leaf: bool) -> List[Tuple]:
    """Distribution-metric strata a tile belongs to. Leaf-only for overall/protocol/window."""
    s: List[Tuple] = [("level", level)]
    if leaf:
        s += [("overall",), ("protocol", P), ("window", W)]
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_root", default="outputs/dense500_eval")
    ap.add_argument("--out_dir", default=None, help="default: <gen_root>/_metrics")
    ap.add_argument("--ref_metrics", default=",".join(REF_METRICS_DEFAULT))
    ap.add_argument("--fid_strata", choices=["none", "overall", "full"], default="full")
    ap.add_argument("--max_sites_per_cell", type=int, default=0, help="0 = all (smoke knob)")
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--windows", default="1,2,3,4,5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lpips_max_size", type=int, default=0, help="0 = no downsample (256² tiles)")
    args = ap.parse_args()

    device = torch.device(args.device)
    ref_metrics = [m.strip() for m in args.ref_metrics.split(",") if m.strip()]
    out_dir = args.out_dir or os.path.join(args.gen_root, "_metrics")
    os.makedirs(out_dir, exist_ok=True)
    protos = [p for p in args.protocols.split(",") if p]
    windows = [int(w) for w in args.windows.split(",") if w]

    gd = GlobalDense()
    sid2sub = {s.site_id: s for s in gd.iter_subtrees(depth=4)}
    parquets = discover_parquets(args.gen_root, protos, windows, args.max_sites_per_cell)
    print(f"[eval-pyramid] {len(parquets)} site-parquets | device={device} | "
          f"ref={ref_metrics} | fid_strata={args.fid_strata}", flush=True)
    if not parquets:
        print("[eval-pyramid] nothing to evaluate."); return

    t0 = time.time()

    # ── pass 1: count non-seed tiles per distribution stratum (metadata only) ──
    counts: Dict[Tuple, int] = defaultdict(int)
    if args.fid_strata != "none":
        for (P, W, p) in parquets:
            d = pq.read_table(p, columns=["zoom", "is_seed"]).to_pydict()
            rz = min(d["zoom"])
            for z, sd in zip(d["zoom"], d["is_seed"]):
                if sd:
                    continue
                lvl = z - rz
                for st in _strata_for(P, W, lvl, leaf=(lvl == 3)):
                    counts[st] += 1

    def stratum_enabled(st: Tuple) -> bool:
        if args.fid_strata == "none":
            return False
        if args.fid_strata == "overall":
            return st == ("overall",)
        return True

    accs: Dict[Tuple, FidAccumulator] = {}
    for st, n in counts.items():
        if stratum_enabled(st) and n >= 2:
            accs[st] = FidAccumulator(device)
    print(f"[eval-pyramid] distribution strata: {len(accs)} "
          f"({sorted(str(s) for s in accs)[:6]}{' ...' if len(accs) > 6 else ''})", flush=True)

    # ── pass 2: per-tile reference metrics + feed distribution accumulators ──
    sw = common.SampleWriter(os.path.join(out_dir, "samples.csv"))
    cols = ["site_id", "protocol", "window", "zoom", "level", "tile_x", "tile_y",
            "source"] + ref_metrics
    sw.open(cols)
    # stratum -> metric -> [sum, count]
    agg: Dict[Tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    n_tiles = n_err = 0

    for si, (P, W, p) in enumerate(parquets):
        meta, tiles = load_gen_parquet(p)
        sub = sid2sub.get(meta["site_id"])
        if sub is None:
            n_err += 1
            continue
        gt = gd.load_gt(sub)                       # {(z,x,y): PIL} decoded once
        rz = sub.root[0]
        # collect distribution batches for this site, per stratum / real|fake
        dist_real: Dict[Tuple, list] = defaultdict(list)
        dist_fake: Dict[Tuple, list] = defaultdict(list)

        for key, (img, is_seed, src) in tiles.items():
            if is_seed:
                continue
            gtim = gt.get(key)
            if gtim is None:
                n_err += 1
                continue
            z = key[0]
            lvl = z - rz
            m = common.compute_metrics(img, gtim, metrics=ref_metrics, device=device,
                                       lpips_max_size=args.lpips_max_size)
            sw.append({"site_id": meta["site_id"], "protocol": P, "window": W,
                       "zoom": z, "level": lvl, "tile_x": key[1], "tile_y": key[2],
                       "source": src, **{k: round(v, 6) for k, v in m.items()}})
            n_tiles += 1
            for st in [("overall",), ("protocol", P), ("window", W),
                       ("level", lvl), ("proto_level", P, lvl)]:
                for k, v in m.items():
                    if v == v:                     # skip NaN
                        agg[st][k][0] += v
                        agg[st][k][1] += 1
            for st in _strata_for(P, W, lvl, leaf=(lvl == 3)):
                if st in accs:
                    dist_fake[st].append(img)
                    dist_real[st].append(gtim)

        for st, pics in dist_real.items():
            accs[st].update(_pil_to_u8_nchw(pics, device), real=True)
        for st, pics in dist_fake.items():
            accs[st].update(_pil_to_u8_nchw(pics, device), real=False)

        if (si + 1) % 25 == 0 or si + 1 == len(parquets):
            print(f"  [{si+1}/{len(parquets)}] tiles_scored={n_tiles}", flush=True)

    sw.close()

    # ── finalize ──
    def fmt_agg(d):
        return {st_key(st): {m: round(s / c, 6) for m, (s, c) in mm.items() if c}
                | {"n": max((c for _, c in mm.values()), default=0)}
                for st, mm in d.items()}

    def st_key(st):
        return st[0] if len(st) == 1 else ":".join(str(x) for x in st)

    reference = fmt_agg(agg)
    distribution = {st_key(st): acc.compute() for st, acc in accs.items()}

    aggregate = {"reference": reference, "distribution": distribution}
    with open(os.path.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    # flat CSV of aggregates for quick reading
    import csv
    with open(os.path.join(out_dir, "aggregate.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["family", "stratum", "metric", "value", "n"])
        for st, mm in reference.items():
            n = mm.get("n", "")
            for k, v in mm.items():
                if k != "n":
                    wr.writerow(["reference", st, k, v, n])
        for st, dd in distribution.items():
            if dd.get("fid") is not None:
                wr.writerow(["distribution", st, "fid", dd["fid"], dd.get("n_fake", "")])

    common.dump_run_meta(
        out_dir, task="pyramid", exp=None, stage=None, ckpt="", split="dense500",
        extra={"gen_root": args.gen_root, "protocols": protos, "windows": windows,
               "max_sites_per_cell": args.max_sites_per_cell},
        config={"ref_metrics": ref_metrics, "fid_strata": args.fid_strata,
                "lpips_max_size": args.lpips_max_size},
        n_samples=n_tiles, n_errors=n_err, elapsed_seconds=round(time.time() - t0, 1),
        device=str(device), metrics=ref_metrics + (["fid"] if accs else []))

    print(f"\n[eval-pyramid] DONE: {n_tiles} tiles scored, {n_err} errors, "
          f"{round(time.time()-t0,1)}s -> {out_dir}", flush=True)
    print("=== reference (overall) ===", flush=True)
    print(json.dumps(reference.get("overall", {}), indent=2), flush=True)
    if "overall" in distribution:
        print("=== distribution (overall, leaf) ===", flush=True)
        print(json.dumps(distribution["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
