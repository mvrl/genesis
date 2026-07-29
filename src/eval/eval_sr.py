"""SR evaluation: per-sample metrics → <out_dir>/samples.csv + run.json.
Single-step ×2 by default; `--scale {2,3,4}` for iterative ×4/×8/×16 (uses
SubtreeDataset, which only emits parents whose full 4ⁿ-leaf subtree is in
the requested split)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_THIS, "..", "superresolution"))
sys.path.insert(0, os.path.join(_THIS, ".."))

from common import (   # noqa: E402
    DEFAULT_HINDEX_PKL,
    DEFAULT_META_FILE,
    QUAD_ORDER,
    SampleWriter,
    auto_tag_dir,
    compute_metrics,
    default_results_root,
    dump_run_meta,
    load_sr_denoiser,
    meta_columns,
    resolve_ckpt,
    resolve_dino_weights,
    resolve_meta_path,
    tensor_to_pil,
)


_TILE_SIZE = 256


def save_sr_comparison(path: str, parent_pil: Image.Image,
                       pred_pil: Image.Image, gt_pil: Image.Image) -> None:
    """3-panel: parent context | prediction | GT."""
    w, h = gt_pil.size
    parent_vis = parent_pil.convert("RGB").resize((w, h), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(parent_vis, (0, 0))
    canvas.paste(pred_pil.convert("RGB"), (w, 0))
    canvas.paste(gt_pil.convert("RGB"), (w * 2, 0))
    canvas.save(path)


# ── Datasets ────────────────────────────────────────────────────────────────

def _make_scale1_dataset(args, splits: Dict[str, List[int]], index_to_meta: Dict[str, Dict]):
    """Wrap HFParentChildDataset for ×2 (single-step) evaluation."""
    from dataloader import HFParentChildDataset
    return HFParentChildDataset(
        hindex_pkl   = args.hindex_pkl,
        indices      = splits[args.split],
        index_to_meta= index_to_meta,
        cache_dir    = args.hf_data_cache,
        ensure_size  = _TILE_SIZE,
        require_complete_quad = True,
    )


def child_zl_group_to_parent_zls(group: str) -> Optional[set[int]]:
    group = (group or "").strip().lower()
    if not group:
        return None
    mapping = {
        "z12-15": {11, 12, 13, 14},
        "childz12-15": {11, 12, 13, 14},
        "child-z12-15": {11, 12, 13, 14},
        "z16-17": {15, 16},
        "childz16-17": {15, 16},
        "child-z16-17": {15, 16},
        "z18": {17},
        "childz18": {17},
        "child-z18": {17},
    }
    if group not in mapping:
        raise ValueError(
            f"Unsupported --child_zl_group={group!r}. Use z12-15, z16-17, or z18."
        )
    return mapping[group]


class SubtreeDataset(Dataset):
    """For ×4 / ×8 / ×16: yields a parent + the full 4^scale-leaf descendant subtree.

    Reuses the same `hierarchy_index_quad.pkl` as `HFParentChildDataset` and
    only emits parents whose entire subtree (down to zoom = parent_zl+scale)
    lives in the requested split.

    __getitem__ returns:
      parent_rgb  : (3, S, S) in [-1, 1]
      gt_leaves   : (4**scale, 3, S, S) in [-1, 1] in row-major QUAD_ORDER (DFS over children)
      parent_loc  : str
      parent_zl   : int
      parent_idx  : int  (HF dataset index)
    """

    def __init__(self, hindex_pkl: str, indices: Sequence[int],
                 index_to_meta: Dict[str, Dict], scale: int, cache_dir: str,
                 hf_name: str = "lcybuaa/Git-10M"):
        import pickle
        from datasets import load_dataset

        assert scale >= 1
        self.scale = int(scale)
        self.tile  = _TILE_SIZE
        print(f"[eval-sr] Loading HF dataset (cache={cache_dir})...", flush=True)
        self.ds = load_dataset(hf_name, split="train", cache_dir=cache_dir)

        print(f"[eval-sr] Loading hierarchy from {hindex_pkl}...", flush=True)
        with open(hindex_pkl, "rb") as f:
            self.hier = pickle.load(f)

        # Build location → HF index over all tiles, then filter to current split.
        all_loc_to_idx = {m["location"]: int(k) for k, m in index_to_meta.items() if m.get("location")}
        split_set      = set(indices)
        loc_to_idx     = {loc: i for loc, i in all_loc_to_idx.items() if i in split_set}

        # A parent is eligible if every leaf in its 4^scale subtree at zoom = zp+scale
        # lives in this split. We expand DFS in QUAD_ORDER.
        eligible = []
        for parent_loc, parent_idx in loc_to_idx.items():
            zp, _, _ = _parse_loc(parent_loc)
            leaves = self._enumerate_leaves(parent_loc, self.scale)
            if leaves is None:
                continue   # subtree partial
            leaf_idxs = []
            ok = True
            for leaf_loc in leaves:
                li = all_loc_to_idx.get(leaf_loc)
                if li is None or li not in split_set:
                    ok = False
                    break
                leaf_idxs.append(li)
            if ok:
                eligible.append({
                    "parent_loc": parent_loc,
                    "parent_idx": parent_idx,
                    "parent_zl":  zp,
                    "leaf_idxs":  leaf_idxs,
                })

        self.records = eligible
        print(f"[eval-sr] Subtree-complete parents at scale={scale}: "
              f"{len(self.records):,} / {len(loc_to_idx):,} candidates", flush=True)

    def _enumerate_leaves(self, parent_loc: str, scale: int) -> Optional[List[str]]:
        """Return all 4**scale leaf locations under `parent_loc`, in DFS QUAD_ORDER.
        Returns None if the parent_loc isn't in the hierarchy."""
        out: List[str] = []

        def recurse(loc: str, depth: int):
            if depth == 0:
                out.append(loc)
                return
            zc, xc, yc = _parse_loc(loc)
            for qx, qy in QUAD_ORDER:
                child = f"{zc + 1}_{2 * xc + qx}_{2 * yc + qy}"
                recurse(child, depth - 1)

        recurse(parent_loc, scale)
        return out

    def _process_img(self, pil_img: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil_img.convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0
        if t.shape[1] != self.tile or t.shape[2] != self.tile:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.tile, self.tile),
                mode="bicubic", align_corners=False,
            ).squeeze(0)
        return t

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        parent_rgb = self._process_img(self.ds[r["parent_idx"]]["image"])
        leaves     = torch.stack([self._process_img(self.ds[i]["image"]) for i in r["leaf_idxs"]], dim=0)
        return {
            "parent_rgb": parent_rgb,
            "gt_leaves":  leaves,
            "parent_loc": r["parent_loc"],
            "parent_zl":  int(r["parent_zl"]),
            "parent_idx": int(r["parent_idx"]),
        }


def _parse_loc(loc: str):
    z, x, y = loc.split("_")
    return int(z), int(x), int(y)


def _scale1_collate(batch):
    """Collate the HFParentChildDataset records DenoiserSR.generate expects."""
    return {
        "parent_rgb":     torch.stack([b["parent_rgb"]     for b in batch]),
        "children_rgb":   torch.stack([b["children_rgb"]   for b in batch]),
        "children_valid": torch.stack([b["children_valid"] for b in batch]),
        "location":       [b["location"] for b in batch],
        "zoom_level":     torch.stack([b["zoom_level"] for b in batch]),
    }


def _subtree_collate(batch):
    return {
        "parent_rgb": torch.stack([b["parent_rgb"] for b in batch]),
        "gt_leaves":  torch.stack([b["gt_leaves"]  for b in batch]),
        "parent_loc": [b["parent_loc"] for b in batch],
        "parent_zl":  torch.tensor([b["parent_zl"]  for b in batch], dtype=torch.long),
        "parent_idx": torch.tensor([b["parent_idx"] for b in batch], dtype=torch.long),
    }


# ── Iterative SR ─────────────────────────────────────────────────────────────

@torch.no_grad()
def run_sr_iter(denoiser, parent_rgb_b: torch.Tensor, parent_zl_b: torch.Tensor,
                scale: int, device: torch.device) -> torch.Tensor:
    """
    Apply DenoiserSR `scale` times. Returns a (B, 3, T, T) tensor with
    T = 2 * (2 ** (scale - 1)) * (tile/2)... essentially T = tile * 2 ** scale
    for tile=_TILE_SIZE/2 inputs after each iteration. Concretely:

        scale=1: T = 512   (one ×2 step)
        scale=2: T = 1024
        scale=3: T = 2048
        scale=4: T = 4096
    """
    cur = parent_rgb_b.to(device)               # (B, 3, 256, 256) at zp
    target_zoom = parent_zl_b.to(device).float() + 1.0
    for _ in range(scale):
        # Run the SR step on each item individually to keep memory bounded
        # (children are concatenated along H/W; we re-feed them per-iter).
        out = []
        for b in range(cur.shape[0]):
            single = cur[b:b + 1]               # (1, 3, S, S) where S is the current parent size
            # If S != 256, downsample so the model sees its native input size.
            if single.shape[-1] != _TILE_SIZE:
                # Split into 256-sized parents — each block becomes a separate parent.
                blocks = _split_into_blocks(single, _TILE_SIZE)   # (n*n, 3, 256, 256)
                # Run blocks in mini-batches; reuse `denoiser.generate` directly.
                outs = []
                for k in range(blocks.shape[0]):
                    bb = blocks[k:k + 1]
                    z = float(target_zoom[b].item())
                    res = denoiser.generate({
                        "parent_rgb": bb,
                        "zoom_level": torch.tensor([z], device=device),
                    })   # (1, 3, 512, 512)
                    outs.append(res)
                merged = torch.cat(outs, dim=0)                       # (n*n, 3, 512, 512)
                out.append(_merge_blocks(merged, blocks_per_side=_blocks_per_side(single, _TILE_SIZE)))
            else:
                z = float(target_zoom[b].item())
                res = denoiser.generate({
                    "parent_rgb": single,
                    "zoom_level": torch.tensor([z], device=device),
                })
                out.append(res[0])
        cur = torch.stack(out, dim=0)
        target_zoom = target_zoom + 1.0
    return cur


def _blocks_per_side(t: torch.Tensor, block: int) -> int:
    return t.shape[-1] // block


def _split_into_blocks(t: torch.Tensor, block: int) -> torch.Tensor:
    """(1, 3, H, W) → (n*n, 3, block, block) where n = H/block."""
    _, _, H, W = t.shape
    assert H == W and H % block == 0, f"can't split {H}x{W} into {block} blocks"
    n = H // block
    out = []
    for j in range(n):
        for i in range(n):
            out.append(t[0, :, j * block:(j + 1) * block, i * block:(i + 1) * block])
    return torch.stack(out, dim=0)


def _merge_blocks(blocks: torch.Tensor, blocks_per_side: int) -> torch.Tensor:
    """(n*n, 3, B, B) → (3, n*B, n*B), in row-major order matching _split_into_blocks."""
    n = blocks_per_side
    _, C, B, _ = blocks.shape
    out = torch.zeros(C, n * B, n * B, dtype=blocks.dtype, device=blocks.device)
    k = 0
    for j in range(n):
        for i in range(n):
            out[:, j * B:(j + 1) * B, i * B:(i + 1) * B] = blocks[k]
            k += 1
    return out


# ── GT mosaic stitching ──────────────────────────────────────────────────────

def gt_mosaic_from_children(children_rgb: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """children_rgb (4,3,S,S) + valid (4,) → (3, 2S, 2S) — same logic the engine uses for x1."""
    s = children_rgb.shape[-1]
    out = torch.zeros(3, 2 * s, 2 * s, dtype=children_rgb.dtype, device=children_rgb.device)
    placements = [(0, 0), (s, 0), (0, s), (s, s)]
    for c, v, (px, py) in zip(children_rgb, valid, placements):
        if bool(v):
            out[:, py:py + s, px:px + s] = c
    return out


def gt_mosaic_from_leaves(gt_leaves: torch.Tensor, scale: int) -> torch.Tensor:
    """(4**scale, 3, S, S) DFS-QUAD_ORDER → stitched (3, S * 2**scale, S * 2**scale)."""
    n_leaves = 4 ** scale
    assert gt_leaves.shape[0] == n_leaves
    s = gt_leaves.shape[-1]

    def recurse(start: int, depth: int) -> torch.Tensor:
        if depth == 0:
            return gt_leaves[start]
        step = 4 ** (depth - 1)
        ul = recurse(start + 0 * step, depth - 1)
        ur = recurse(start + 1 * step, depth - 1)
        ll = recurse(start + 2 * step, depth - 1)
        lr = recurse(start + 3 * step, depth - 1)
        h = ul.shape[-1]
        out = torch.zeros(3, 2 * h, 2 * h, dtype=ul.dtype, device=ul.device)
        out[:, :h, :h]   = ul
        out[:, :h, h:]   = ur
        out[:, h:, :h]   = ll
        out[:, h:, h:]   = lr
        return out

    return recurse(0, scale)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_metric_list(s: str) -> List[str]:
    s = (s or "").strip().lower()
    if s in ("", "all"):
        return ["psnr", "ssim", "lpips", "dists"]
    return [m.strip() for m in s.split(",") if m.strip()]


def get_args():
    p = argparse.ArgumentParser("Genesis SR evaluation")

    p.add_argument("--ckpt", default=None, help="Explicit ckpt path. If unset, --exp + --stage are used.")
    p.add_argument("--exp",  default=None, help="Experiment name: abl_dino_lpips, main_b16, main_h16, ...")
    p.add_argument("--stage", default=3, type=int, choices=(2, 3),
                   help="Curriculum stage of the ckpt (3 for SR ablations, 2 or 3 for main).")
    p.add_argument("--model", default="JiT-B/16", help="Architecture for the engine (B16/H16).")

    p.add_argument("--split", default="val_random",
                   choices=("val_random", "val_spatial", "random_test", "spatial_test"))
    p.add_argument("--scale", default=1, type=int, choices=(1, 2, 3, 4),
                   help="1 = single ×2 (ZL→ZL+1); 2/3/4 = iterative ×4/×8/×16")

    p.add_argument("--cfg", default=1.0, type=float)
    p.add_argument("--num_sampling_steps", default=50, type=int)
    p.add_argument("--bicubic_init", action="store_true")
    p.add_argument("--bicubic_init_t_start", default=0.5, type=float)

    p.add_argument("--max_samples", default=0, type=int, help="0 = full split")
    p.add_argument("--start_sample", default=0, type=int)
    p.add_argument("--sample_count", default=0, type=int)
    p.add_argument("--parent_zl", default=0, type=int,
                   help="Optional parent zoom-level filter for scale=1. 0 = no filter.")
    p.add_argument("--child_zl_group", default="",
                   help="Optional target/child zoom bucket for scale=1: z12-15, z16-17, or z18.")
    p.add_argument("--batch_size", default=4, type=int)
    p.add_argument("--num_workers", default=4, type=int)

    p.add_argument("--metrics", default="psnr,ssim,lpips,dists",
                   help='Comma-separated, or "all".')
    p.add_argument("--lpips_max_size", default=1024, type=int,
                   help="Cap LPIPS/DISTS input side length (0 = no cap). "
                        "Useful at large iterative-SR scales.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int)

    p.add_argument("--meta_file_path", default=DEFAULT_META_FILE,
                   help="Git-10M quad meta; auto-pulls from the Google Drive release if missing.")
    p.add_argument("--hindex_pkl",    default=DEFAULT_HINDEX_PKL,
                   help="Hierarchy index; auto-pulls from the Google Drive release if missing.")
    p.add_argument("--hf_data_cache", default=os.environ.get("HF_DATASETS_CACHE") or None,
                   help="HF datasets cache dir for lcybuaa/Git-10M (default: HF's own cache).")
    p.add_argument("--dino_weights",  default="", help="Empty = auto-resolve / auto-pull.")
    p.add_argument("--no_dino", action="store_true",
                   help="Disable DINO conditioning. Required for no-DINO ablation ckpts "
                        "(abl_nodino_nolpips, abl_nodino_lpips); the engine will be built "
                        "without parent_proj so the ckpt's state_dict matches.")

    p.add_argument("--out_dir", default=None,
                   help="Output dir. Default: <repo>/logs/eval_results/<auto-tag>/")
    p.add_argument("--save_images", default=-1, type=int,
                   help="Save first N pred/target/comparison PNGs per shard. "
                        "-1 = save all, 0 = no images, N>0 = first N.")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metrics = parse_metric_list(args.metrics)
    if args.start_sample < 0 or args.sample_count < 0:
        raise ValueError("--start_sample and --sample_count must be non-negative.")
    parent_zl_values = child_zl_group_to_parent_zls(args.child_zl_group)
    if args.parent_zl > 0 and parent_zl_values is not None:
        raise ValueError("Use either --parent_zl or --child_zl_group, not both.")
    if args.scale != 1 and (args.parent_zl > 0 or parent_zl_values is not None):
        raise ValueError("--parent_zl/--child_zl_group are only supported for --scale 1.")

    # ── Resolve ckpt + dino, build model ────────────────────────────────────
    ckpt_path = resolve_ckpt("sr", args.ckpt, args.exp, args.stage)
    dino      = "" if args.no_dino else resolve_dino_weights(args.dino_weights)

    denoiser, eng_args = load_sr_denoiser(
        ckpt_path, args.model, dino, device,
        cfg=args.cfg, num_sampling_steps=args.num_sampling_steps,
        bicubic_init=args.bicubic_init, bicubic_init_t_start=args.bicubic_init_t_start,
    )

    # ── Build dataset ───────────────────────────────────────────────────────
    # Git-10M meta + hierarchy index auto-pull from the Google Drive release if absent.
    meta_path = resolve_meta_path(args.meta_file_path, "shared/git10m_quad_meta.json", "Git-10M meta")
    args.hindex_pkl = resolve_meta_path(args.hindex_pkl, "shared/hierarchy_index_quad.pkl", "hierarchy index")
    print(f"[eval-sr] Loading meta from {meta_path}...", flush=True)
    with open(meta_path) as f:
        meta = json.load(f)
    splits        = meta["splits"]
    index_to_meta = meta["index_to_meta"]

    if args.scale == 1:
        ds = _make_scale1_dataset(args, splits, index_to_meta)
        if args.parent_zl > 0:
            before = len(ds.pairs)
            ds.pairs = [
                p for p in ds.pairs
                if int((index_to_meta.get(str(p["parent_idx"])) or {}).get("zoom", -1)) == args.parent_zl
            ]
            print(
                f"[eval-sr] parent_zl={args.parent_zl} filter: {before} -> {len(ds.pairs)} pairs",
                flush=True,
            )
        if parent_zl_values is not None:
            before = len(ds.pairs)
            ds.pairs = [p for p in ds.pairs if int(p["parent_zl"]) in parent_zl_values]
            print(
                f"[eval-sr] child_zl_group={args.child_zl_group} "
                f"(parent_zl={sorted(parent_zl_values)}) filter: {before} -> {len(ds.pairs)} pairs",
                flush=True,
            )
        if args.start_sample or args.sample_count:
            shard_end = args.start_sample + args.sample_count if args.sample_count > 0 else len(ds.pairs)
            original_n = len(ds.pairs)
            ds.pairs = ds.pairs[args.start_sample:min(shard_end, original_n)]
            print(
                f"[eval-sr] shard slice start={args.start_sample} "
                f"end={min(shard_end, original_n)} count={len(ds.pairs)} from built_pairs={original_n}",
                flush=True,
            )
        loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            shuffle=False, collate_fn=_scale1_collate)
    else:
        ds = SubtreeDataset(
            hindex_pkl    = args.hindex_pkl,
            indices       = splits[args.split],
            index_to_meta = index_to_meta,
            scale         = args.scale,
            cache_dir     = args.hf_data_cache,
        )
        if args.start_sample or args.sample_count:
            shard_end = args.start_sample + args.sample_count if args.sample_count > 0 else len(ds.records)
            original_n = len(ds.records)
            ds.records = ds.records[args.start_sample:min(shard_end, original_n)]
            print(
                f"[eval-sr] shard slice start={args.start_sample} "
                f"end={min(shard_end, original_n)} count={len(ds.records)} from built_records={original_n}",
                flush=True,
            )
        bs = 1 if args.scale >= 3 else args.batch_size  # ×8/×16 force batch_size=1
        loader = DataLoader(ds, batch_size=bs, num_workers=args.num_workers,
                            shuffle=False, collate_fn=_subtree_collate)

    # ── Output dir ──────────────────────────────────────────────────────────
    out_root = args.out_dir or default_results_root()
    extra = f"scale{args.scale}"
    if args.bicubic_init:
        extra += f"_binit{args.bicubic_init_t_start}"
    out_dir = auto_tag_dir(out_root, [
        "sr", args.exp or "ckpt", f"s{args.stage}", args.split,
        extra, f"cfg{args.cfg}",
    ])
    print(f"[eval-sr] Writing results under {out_dir}", flush=True)
    pred_dir = os.path.join(out_dir, "predictions")
    target_dir = os.path.join(out_dir, "targets")
    comp_dir = os.path.join(out_dir, "comparisons")
    if args.save_images != 0:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)

    # ── CSV columns: identifiers + per-metric ──────────────────────────────
    cols = ["sample_idx",
            "parent_loc", "parent_zl", "child_zl", "scale",
            "n_gt_leaves",
            "hf_idx", "location", "satellite", "split",
            "lat", "lon", "gsd", "zoom"] + metrics + [
                "prediction_path", "target_path", "comparison_path", "error"
            ]
    writer = SampleWriter(os.path.join(out_dir, "samples.csv"))
    writer.open(cols)

    # ── Eval loop ───────────────────────────────────────────────────────────
    n_done = n_err = 0
    t0 = time.time()
    # Clamp so progress prints don't claim more samples than the split contains.
    target_total = min(args.max_samples, len(ds)) if args.max_samples > 0 else len(ds)
    if target_total < 1:
        if args.start_sample > 0 or args.sample_count > 0 or args.parent_zl > 0 or parent_zl_values is not None:
            print(
                f"[eval-sr] no samples in this shard (start={args.start_sample}, "
                f"parent_zl={args.parent_zl}, child_zl_group={args.child_zl_group}); exiting cleanly.",
                flush=True,
            )
            return
        raise ValueError("No SR samples found for this split/filter.")

    if args.scale == 1:
        # Iterate using HFParentChildDataset's batch contract (children_rgb + valid).
        # `pairs` carries parent_loc / parent_zl / parent_idx — pull them in order.
        pairs = ds.pairs

        for batch_idx, batch in enumerate(loader):
            if n_done >= target_total:
                break
            B = batch["parent_rgb"].shape[0]
            with torch.no_grad():
                # SR engine signature only needs parent_rgb + optional zoom_level.
                pred = denoiser.generate({
                    "parent_rgb": batch["parent_rgb"].to(device),
                    "zoom_level": batch["zoom_level"].to(device),
                })   # (B, 3, 512, 512)

            for k in range(B):
                if n_done >= target_total:
                    break
                pair_idx = batch_idx * args.batch_size + k
                global_sample_idx = args.start_sample + pair_idx
                rec = pairs[pair_idx]
                parent_zl = int(rec["parent_zl"])
                pred_pil = tensor_to_pil(pred[k].cpu())
                gt_pil   = tensor_to_pil(
                    gt_mosaic_from_children(batch["children_rgb"][k], batch["children_valid"][k])
                )

                row: Dict[str, Any] = {
                    "sample_idx": global_sample_idx,
                    "parent_loc": rec["parent_loc"],
                    "parent_zl":  parent_zl,
                    "child_zl":   parent_zl + 1,
                    "scale":      1,
                    "n_gt_leaves": int(batch["children_valid"][k].sum().item()),
                }
                row.update(meta_columns(index_to_meta, rec["parent_idx"]))
                if args.save_images != 0 and (args.save_images < 0 or n_done < args.save_images):
                    stem = (
                        f"sample{global_sample_idx:06d}_parent{int(rec['parent_idx']):08d}"
                        f"_loc{rec['parent_loc']}"
                    )
                    pred_path = os.path.join(pred_dir, f"{stem}_pred.png")
                    target_path = os.path.join(target_dir, f"{stem}_target.png")
                    comp_path = os.path.join(comp_dir, f"{stem}_parent_pred_target.png")
                    pred_pil.save(pred_path)
                    gt_pil.save(target_path)
                    parent_pil = tensor_to_pil(batch["parent_rgb"][k])
                    save_sr_comparison(comp_path, parent_pil, pred_pil, gt_pil)
                    row.update({
                        "prediction_path": pred_path,
                        "target_path": target_path,
                        "comparison_path": comp_path,
                    })
                try:
                    m = compute_metrics(
                        pred_pil, gt_pil, mask=None, base=None,
                        metrics=metrics, lpips_max_size=args.lpips_max_size, device=device,
                    )
                    row.update(m)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    n_err += 1
                writer.append(row)
                n_done += 1

            elapsed = time.time() - t0
            print(f"[eval-sr] {n_done}/{target_total}  ({elapsed/max(n_done,1):.2f}s/sample)", flush=True)
    else:
        for batch in loader:
            if n_done >= target_total:
                break
            with torch.no_grad():
                pred = run_sr_iter(denoiser, batch["parent_rgb"], batch["parent_zl"],
                                   args.scale, device)   # (B, 3, T, T)

            B = pred.shape[0]
            for k in range(B):
                if n_done >= target_total:
                    break
                parent_zl = int(batch["parent_zl"][k].item())
                parent_idx = int(batch["parent_idx"][k].item())
                pred_pil = tensor_to_pil(pred[k].cpu())
                gt_pil   = tensor_to_pil(gt_mosaic_from_leaves(batch["gt_leaves"][k], args.scale))

                row = {
                    "sample_idx":   args.start_sample + n_done,
                    "parent_loc":   batch["parent_loc"][k],
                    "parent_zl":    parent_zl,
                    "child_zl":     parent_zl + args.scale,
                    "scale":        args.scale,
                    "n_gt_leaves":  4 ** args.scale,
                }
                row.update(meta_columns(index_to_meta, parent_idx))
                if args.save_images != 0 and (args.save_images < 0 or n_done < args.save_images):
                    stem = (
                        f"sample{args.start_sample + n_done:06d}_parent{parent_idx:08d}"
                        f"_loc{batch['parent_loc'][k]}_scale{args.scale}"
                    )
                    pred_path = os.path.join(pred_dir, f"{stem}_pred.png")
                    target_path = os.path.join(target_dir, f"{stem}_target.png")
                    comp_path = os.path.join(comp_dir, f"{stem}_parent_pred_target.png")
                    pred_pil.save(pred_path)
                    gt_pil.save(target_path)
                    parent_pil = tensor_to_pil(batch["parent_rgb"][k])
                    save_sr_comparison(comp_path, parent_pil, pred_pil, gt_pil)
                    row.update({
                        "prediction_path": pred_path,
                        "target_path": target_path,
                        "comparison_path": comp_path,
                    })
                try:
                    m = compute_metrics(
                        pred_pil, gt_pil, mask=None, base=None,
                        metrics=metrics, lpips_max_size=args.lpips_max_size, device=device,
                    )
                    row.update(m)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    n_err += 1
                writer.append(row)
                n_done += 1

            elapsed = time.time() - t0
            print(f"[eval-sr] {n_done}/{target_total}  ({elapsed/max(n_done,1):.2f}s/sample)", flush=True)

    n_written = writer.close()
    elapsed = time.time() - t0

    dump_run_meta(
        out_dir,
        task="sr",
        exp=args.exp,
        stage=args.stage,
        ckpt=ckpt_path,
        split=args.split,
        extra={"scale": args.scale,
               "bicubic_init": args.bicubic_init,
               "bicubic_init_t_start": args.bicubic_init_t_start,
               "lpips_max_size": args.lpips_max_size,
               "model": args.model,
               "no_dino": args.no_dino,
               "max_samples": args.max_samples,
               "start_sample": args.start_sample,
               "sample_count": args.sample_count,
               "parent_zl": args.parent_zl,
               "child_zl_group": args.child_zl_group,
               "save_images": args.save_images,
               "parent_zl_values": sorted(parent_zl_values) if parent_zl_values is not None else None},
        config={"cfg": args.cfg, "num_sampling_steps": args.num_sampling_steps,
                "batch_size": args.batch_size, "seed": args.seed},
        n_samples=n_written,
        n_errors=n_err,
        elapsed_seconds=elapsed,
        device=str(device),
        metrics=metrics,
    )
    print(f"[eval-sr] DONE — {n_written} samples ({n_err} errors) in {elapsed:.1f}s.", flush=True)
    print(f"[eval-sr] CSV : {os.path.join(out_dir, 'samples.csv')}", flush=True)
    print(f"[eval-sr] meta: {os.path.join(out_dir, 'run.json')}", flush=True)


if __name__ == "__main__":
    main()
