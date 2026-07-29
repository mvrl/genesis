#!/usr/bin/env python
"""Driver for the extended Dense500 metrics (see metrics_ext.py).

Reuses the generation parquets + GlobalDense GT exactly like eval_pyramid.py.
Families are selected with --families and written to separate files so each can
run as an independent GPU job:

  semantic    -> samples_sem.csv  (per non-seed tile: CLIP-I + DINOv3-sat cosines)
  consistency -> consistency_samples.csv (per internal tile: LR-PSNR box,
                 xLR-PSNR, RAPSD)

Aggregates (mean, stratified overall/protocol/window/level) -> aggregate_<fam>.json.

  PYTHONPATH=src python src/eval/eval_ext.py --gen_root outputs/dense500_eval_new \
      --families consistency,semantic --device cuda
"""
from __future__ import annotations

import argparse, csv, glob, io, json, os, sys, time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_ext as MX                                   # noqa: E402
from pyramid.dataset import GlobalDense                    # noqa: E402

WINDOW_TO_L = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}


def load_gen(path):
    t = pq.read_table(path).to_pydict()
    meta = {"site_id": t["site_id"][0], "protocol": t["protocol"][0],
            "window": int(t["zoom_window"][0])}
    tiles = {}
    for i in range(len(t["zoom"])):
        k = (int(t["zoom"][i]), int(t["tile_x"][i]), int(t["tile_y"][i]))
        tiles[k] = (Image.open(io.BytesIO(t["image"][i])).convert("RGB"),
                    bool(t["is_seed"][i]), str(t["source"][i]))
    return meta, tiles


def discover(gen_root, protos, windows, max_per_cell):
    out = []
    for P in protos:
        for W in windows:
            cell = os.path.join(gen_root, f"proto{P}", f"W{W}_l{WINDOW_TO_L[W]}", "tiles")
            ps = sorted(glob.glob(os.path.join(cell, "*.parquet")))
            if max_per_cell:
                ps = ps[:max_per_cell]
            out += [(P, W, p) for p in ps]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_root", default="outputs/dense500_eval_new")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--families", default="consistency,semantic")
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--windows", default="1,2,3,4,5")
    ap.add_argument("--max_sites_per_cell", type=int, default=0)
    ap.add_argument("--encoders", default="clip,dinov3sat")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    fams = set(f.strip() for f in args.families.split(",") if f.strip())
    out_dir = args.out_dir or os.path.join(args.gen_root, "_metrics_ext")
    os.makedirs(out_dir, exist_ok=True)
    protos = [p for p in args.protocols.split(",") if p]
    windows = [int(w) for w in args.windows.split(",") if w]

    gd = GlobalDense()
    sid2sub = {s.site_id: s for s in gd.iter_subtrees(depth=4)}
    parquets = discover(args.gen_root, protos, windows, args.max_sites_per_cell)
    print(f"[eval-ext] {len(parquets)} sites | families={sorted(fams)} | device={device}", flush=True)

    # lazy scorers
    emb = (MX.EmbedScorer(device, [e for e in args.encoders.split(",") if e])
           if "semantic" in fams else None)

    writers: Dict[str, csv.DictWriter] = {}
    files: Dict[str, object] = {}
    def W(name, fields):
        f = open(os.path.join(out_dir, name), "w", newline="")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        files[name] = f; writers[name] = w; return w

    base_cols = ["site_id", "protocol", "window", "zoom", "level", "tile_x", "tile_y", "source"]
    if "semantic" in fams:
        sem_cols = base_cols + [f"{e}_cos" for e in (emb.enc if emb else [])]
        W("samples_sem.csv", sem_cols)
    if "consistency" in fams:
        W("consistency_samples.csv", ["site_id", "protocol", "window", "level", "zoom",
                                      "lrpsnr_self_box", "lrpsnr_cross_box", "rapsd_sd"])

    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # (fam,stratum)->metric->[sum,n]
    def acc(fam, P, W_, lvl, d):
        for st in [("overall",), ("protocol", P), ("window", W_), ("level", lvl)]:
            key = (fam, st)
            for k, v in d.items():
                if isinstance(v, float) and v == v:
                    agg[key][k][0] += v; agg[key][k][1] += 1

    t0 = time.time()
    for si, (P, Wn, p) in enumerate(parquets):
        meta, tiles = load_gen(p)
        sub = sid2sub.get(meta["site_id"])
        if sub is None:
            continue
        gt = gd.load_gt(sub)
        rz = sub.root[0]

        # ---- consistency (per internal tile; needs the full gen pyramid) ----
        if "consistency" in fams:
            geni = {k: im for k, (im, _s, _src) in tiles.items()}
            for row in MX.consistency_for_site(geni, gt, rz, sub.depth):
                row.update({"site_id": meta["site_id"], "protocol": P, "window": Wn})
                writers["consistency_samples.csv"].writerow(row)
                acc("consistency", P, Wn, row["level"],
                    {k: v for k, v in row.items() if k.startswith(("lrpsnr", "rapsd"))})

        # ---- per-tile semantic (batched embeddings per site) ----
        if "semantic" in fams:
            sem_preds, sem_gts, sem_meta = [], [], []
            for k, (img, is_seed, src) in tiles.items():
                if is_seed:
                    continue
                z = k[0]; lvl = z - rz; gtim = gt.get(k)
                if gtim is None:
                    continue
                rowbase = {"site_id": meta["site_id"], "protocol": P, "window": Wn,
                           "zoom": z, "level": lvl, "tile_x": k[1], "tile_y": k[2], "source": src}
                sem_preds.append(img); sem_gts.append(gtim); sem_meta.append((rowbase, lvl))
            if sem_preds and emb and emb.enc:
                cos = emb.cosine_batch(sem_preds, sem_gts)
                for j, (rowbase, lvl) in enumerate(sem_meta):
                    cj = {kk: vv[j] for kk, vv in cos.items()}
                    writers["samples_sem.csv"].writerow({**rowbase, **cj})
                    acc("semantic", rowbase["protocol"], rowbase["window"], lvl, cj)

        if (si + 1) % 25 == 0 or si + 1 == len(parquets):
            print(f"  [{si+1}/{len(parquets)}] {round(time.time()-t0)}s", flush=True)

    for f in files.values():
        f.close()

    # aggregate
    def stkey(st): return st[0] if len(st) == 1 else ":".join(map(str, st))
    out = defaultdict(dict)
    for (fam, st), mm in agg.items():
        out[fam][stkey(st)] = {k: round(s / n, 5) for k, (s, n) in mm.items() if n} | \
                              {"n": max((n for _, n in mm.values()), default=0)}
    agg_name = "aggregate_ext_" + "-".join(sorted(fams)) + ".json"   # family-specific => parallel-safe
    with open(os.path.join(out_dir, agg_name), "w") as f:
        json.dump(dict(out), f, indent=2)
    print(f"[eval-ext] DONE {round(time.time()-t0)}s -> {out_dir}", flush=True)
    for fam in out:
        print(f"=== {fam} (overall) ===", json.dumps(out[fam].get("overall", {})), flush=True)


if __name__ == "__main__":
    main()
