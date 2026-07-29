#!/usr/bin/env python
"""Caption-agreement metric (scene level) for Dense500 generations.

The literature-correct "caption agreement" (vs CLIPScore, which is image<->text):
caption the generated scene AND the GT scene with a VLM, embed both captions with
a sentence encoder, and report their cosine. We caption the full LEAF MOSAIC per
site (the whole synthesized scene) — captioning all 123k tiles is infeasible, and
the scene is the natural semantic unit.

VLM: Qwen2.5-VL-3B-Instruct (cached). Text encoder: sentence-transformers
all-MiniLM-L6-v2. We also report a SHUFFLED baseline (gen-caption_i vs gt-caption_j)
so the agreement number is calibrated against chance.

  PYTHONPATH=src python src/eval/caption_agree.py --gen_root outputs/dense500_eval_new \
      --sites_per_cell 10 --device cuda
"""
from __future__ import annotations
import argparse, csv, glob, io, json, os, sys
from collections import defaultdict
import numpy as np
import torch
from PIL import Image
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyramid.dataset import GlobalDense
from pyramid.geometry import stitch_level

WIN_L = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
PROMPT = ("Describe this overhead satellite/aerial image in one sentence, focusing on "
          "land-cover types, built structures, roads, vegetation and overall spatial layout.")
VLM = "Qwen/Qwen2.5-VL-3B-Instruct"


class Captioner:
    def __init__(self, device, max_new_tokens=70):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        self.dev = device; self.mnt = max_new_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VLM, torch_dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(VLM)

    @torch.no_grad()
    def caption(self, pil: Image.Image) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[pil.convert("RGB")], return_tensors="pt").to(self.dev)
        out = self.model.generate(**inp, max_new_tokens=self.mnt, do_sample=False)
        trimmed = out[:, inp.input_ids.shape[1]:]
        return self.proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip().replace("\n", " ")


def leaf_mosaic_gen(path, sub):
    t = pq.read_table(path).to_pydict()
    leaf_z = sub.leaf_zoom
    imgs = {}
    for i in range(len(t["zoom"])):
        if int(t["zoom"][i]) == leaf_z:
            imgs[(leaf_z, int(t["tile_x"][i]), int(t["tile_y"][i]))] = \
                Image.open(io.BytesIO(t["image"][i])).convert("RGB")
    return stitch_level(imgs, sub, leaf_z)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen_root", default="outputs/dense500_eval_new")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--protocols", default="A,B,C")
    ap.add_argument("--windows", default="1,2,3,4,5")
    ap.add_argument("--sites_per_cell", type=int, default=10)
    ap.add_argument("--resize", type=int, default=768)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = torch.device(args.device)
    out_dir = args.out_dir or os.path.join(args.gen_root, "_metrics_ext")
    os.makedirs(out_dir, exist_ok=True)
    gd = GlobalDense(); sid2sub = {s.site_id: s for s in gd.iter_subtrees(depth=4)}

    cap = Captioner(dev)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2", device=str(dev))

    rows = []
    protos = [p for p in args.protocols.split(",") if p]
    windows = [int(w) for w in args.windows.split(",") if w]
    for P in protos:
        for Wn in windows:
            cell = os.path.join(args.gen_root, f"proto{P}", f"W{Wn}_l{WIN_L[Wn]}", "tiles")
            ps = sorted(glob.glob(os.path.join(cell, "*.parquet")))[: args.sites_per_cell]
            for p in ps:
                sid = os.path.basename(p)[:-8]
                sub = sid2sub.get(sid)
                if sub is None:
                    continue
                try:
                    gen_m = leaf_mosaic_gen(p, sub).resize((args.resize, args.resize))
                    gt = gd.load_gt(sub)
                    gt_m = stitch_level({t: gt[t] for t in sub.at_zoom(sub.leaf_zoom) if t in gt},
                                        sub, sub.leaf_zoom).resize((args.resize, args.resize))
                    cg, ct = cap.caption(gen_m), cap.caption(gt_m)
                    e = enc.encode([cg, ct], normalize_embeddings=True)
                    cos = float(e[0] @ e[1])
                    rows.append({"site_id": sid, "protocol": P, "window": Wn,
                                 "caption_cos": round(cos, 5), "gen_caption": cg, "gt_caption": ct})
                    print(f"  {P}/W{Wn}/{sid} cos={cos:.3f}", flush=True)
                except Exception as ex:                    # noqa: BLE001
                    print(f"  SKIP {P}/W{Wn}/{sid}: {ex}", flush=True)

    with open(os.path.join(out_dir, "caption_samples.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "protocol", "window", "caption_cos",
                                          "gen_caption", "gt_caption"])
        w.writeheader(); w.writerows(rows)

    # shuffled baseline: gen-caption_i vs gt-caption_j (i!=j) — chance-level agreement
    gen_caps = [r["gen_caption"] for r in rows]; gt_caps = [r["gt_caption"] for r in rows]
    base = float("nan")
    if len(rows) > 2:
        eg = enc.encode(gen_caps, normalize_embeddings=True)
        et = enc.encode(gt_caps, normalize_embeddings=True)
        perm = np.roll(np.arange(len(rows)), 1)
        base = float(np.mean([eg[i] @ et[perm[i]] for i in range(len(rows))]))

    def by(keyf):
        b = defaultdict(list)
        for r in rows:
            b[keyf(r)].append(r["caption_cos"])
        return {k: round(float(np.mean(v)), 5) for k, v in sorted(b.items())}

    agg = {"overall": round(float(np.mean([r["caption_cos"] for r in rows])), 5) if rows else None,
           "shuffled_baseline": round(base, 5), "n": len(rows),
           "by_protocol": by(lambda r: r["protocol"]),
           "by_window": by(lambda r: r["window"])}
    with open(os.path.join(out_dir, "aggregate_caption.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== caption agreement ===", json.dumps(agg, indent=2), flush=True)


if __name__ == "__main__":
    main()
