"""Verify the global-dense dataset: quadtree completeness + image decode sanity.

    PYTHONPATH=src python -m pyramid.verify_data
"""
from __future__ import annotations

import argparse
import json
import random

from .dataset import GlobalDense, get_protocol, PROTOCOLS
from .tiles import lca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--depth", type=int, default=4, help="subtree depth to demo-carve")
    args = ap.parse_args()

    gd = GlobalDense(args.data_dir) if args.data_dir else GlobalDense()
    print(f"shards: {len(gd.files)}")

    report = gd.verify()
    print("VERIFY:", json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=2))
    if not report["ok"]:
        print("FIRST PROBLEMS:")
        for p in report["problems"]:
            print("  ", p)

    # Subtree carving at the requested depth.
    subs = gd.iter_subtrees(depth=args.depth)
    print(f"\ndepth-{args.depth} subtrees carved: {len(subs)}")
    by_tag: dict = {}
    for s in subs:
        by_tag[s.tag] = by_tag.get(s.tag, 0) + 1
    print("  by tag:", by_tag)
    if subs:
        s0 = subs[0]
        print(f"  example: site={s0.site_id} tag={s0.tag} root={s0.root} "
              f"depth={s0.depth} n_tiles={len(s0.tiles)} leaves={len(s0.leaves())}")

    # Seed protocols + LCA sanity on the example subtree.
    if subs:
        rng = random.Random(0)
        s0 = subs[0]
        print("\nseed protocols on example subtree:")
        for name in PROTOCOLS:
            seeds = get_protocol(name)(s0, random.Random(0))
            l = lca(seeds) if seeds else None
            print(f"  {name:20s} -> {len(seeds)} seed(s); lca={l}")

    # Image decode sanity: pull a few GT images for the example subtree.
    if subs:
        s0 = subs[0]
        some = s0.tiles[:3] + s0.leaves()[:2]
        for t in some:
            img = gd.image_by_row(s0.row_index[t])
            print(f"  decoded {t}: size={img.size} mode={img.mode}")
    print("VERIFY_DATA_DONE")


if __name__ == "__main__":
    main()
