"""
probe_dataset.py — visualize parent-child quadtree samples from HFParentChildDataset.

Usage:
    python probe_dataset.py \
        --meta_file  data/git10m_quad_meta.json \
        --hindex_pkl data/hierarchy_index_quad.pkl \
        --cache_dir  data/hf_cache \
        --n_samples  4 \
        --split      train \
        --out        probe_output.png

For train: shows partial quads (1–4 children) by default so the per-tile
loss setup is visible.  Pass --complete_only to restrict to full 2×2 quads.

For eval splits (val_random, val_spatial, random_test, spatial_test):
require_complete is always forced True — the build script guarantees every
eval quad has all 4 children, so partial results would only be artefacts
from cross-split parent lookups (ZL=N eval tile whose ZL=N-1 parent is in
train), and build_pairs now rejects those by requiring the parent itself to
be in the queried split.

For each sample the figure shows:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Parent (256×256, ZL) │ TL │ TR │ BL │ BR │ Bicubic ↑512 │ Stitched GT │
    │  [N/4 children]       │    │    │    │    │  (no detail) │  (ZL+1)     │
    │  solid border=present │    ABSENT placeholder if missing                │
    │  dashed=absent quad   │    black in stitched = absent quadrant          │
    └─────────────────────────────────────────────────────────────────────────┘
Bicubic ↑512 and Stitched GT are placed side by side for direct comparison.

Note: absent quadrants appear black in the Stitched GT panel for display only.
During training the engine crops each valid quadrant from the 512×512 predicted
image and computes velocity loss + LPIPS directly against the individual child
tiles — no masked composites are ever passed to the perceptual network.
"""

import argparse
import json
import pickle
import sys
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))


# ── helpers ─────────────────────────────────────────────────────────────────

def _parse_loc(loc):
    if isinstance(loc, str):
        z, x, y = loc.split('_')
    else:
        z, x, y = loc
    return int(z), int(x), int(y)


def tensor_to_numpy(t):
    """[-1,1] CHW tensor → HWC uint8."""
    img = (t * 0.5 + 0.5).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def process_img(pil_img, size=256):
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = t * 2.0 - 1.0
    if t.shape[1] != size or t.shape[2] != size:
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(size, size), mode="bicubic", align_corners=False
        ).squeeze(0)
    return t


# ── build pairs (mirrors HFParentChildDataset logic) ────────────────────────

def build_pairs(indices, index_to_meta, hierarchy, max_pairs=None,
                require_complete=False):
    """
    Build parent→children pairs from a split's index list.

    Only pairs where the PARENT tile is also in the split are included.
    This prevents spurious "partial quads" caused by cross-split parent
    lookups: e.g. a ZL=13 tile in spatial_test has a ZL=12 parent in
    train; without this guard the probe would show it as a 1-child quad.

    require_complete=False (default): include partial quads (≥1 child).
        Missing child positions are left absent in `rec['children']`.
        For train this reflects genuinely absent tiles in Git-10M.
    require_complete=True: complete 2×2 quads only (all 4 children present).
        Always used automatically for eval splits in main().
    """
    split_idx_set = set(indices)
    # location → dataset index, restricted to this split
    loc_to_idx = {}
    for idx_str, meta in index_to_meta.items():
        idx = int(idx_str)
        if idx not in split_idx_set:
            continue
        loc = meta.get('location')
        if loc:
            loc_to_idx[loc] = idx

    parent_map = {}
    for child_loc, child_idx in loc_to_idx.items():
        parent_loc = hierarchy.get_parent(child_loc)
        if parent_loc is None:
            continue
        # Parent tile must itself be in the split to form a valid pair.
        # Skipping cross-split parents eliminates false partial-quad
        # artefacts that appear when multi-level quad chains cross split
        # boundaries (e.g. ZL=12 train-parent of a ZL=13 eval tile).
        if parent_loc not in loc_to_idx:
            continue
        zp, xp, yp = _parse_loc(parent_loc)
        zc, xc, yc = _parse_loc(child_loc)
        qx = xc - 2 * xp
        qy = yc - 2 * yp
        if not (0 <= qx <= 1 and 0 <= qy <= 1):
            continue
        if parent_loc not in parent_map:
            parent_map[parent_loc] = {'parent_idx': loc_to_idx[parent_loc], 'children': {}}
        parent_map[parent_loc]['children'][(qx, qy)] = child_idx

    pairs = []
    for parent_loc, data in parent_map.items():
        is_complete = all((qx, qy) in data['children'] for qx in range(2) for qy in range(2))
        if require_complete and not is_complete:
            continue
        pairs.append({
            'parent_idx':  data['parent_idx'],
            'parent_loc':  parent_loc,
            'children':    data['children'],
            'is_complete': is_complete,
        })
        if max_pairs and len(pairs) >= max_pairs:
            break
    return pairs


# ── plotting ─────────────────────────────────────────────────────────────────

QUAD_LABELS = {
    (0, 0): "TL (qx=0,qy=0)",
    (1, 0): "TR (qx=1,qy=0)",
    (0, 1): "BL (qx=0,qy=1)",
    (1, 1): "BR (qx=1,qy=1)",
}
QUAD_COLORS = {
    (0, 0): "#e63946",
    (1, 0): "#457b9d",
    (0, 1): "#2a9d8f",
    (1, 1): "#e9c46a",
}
_ABSENT_BG = 35   # dark-gray fill for absent-child panels


def plot_sample(ds, rec, ax_row, sample_idx):
    """Fill one row of axes: [parent | TL | TR | BL | BR | bicubic↑512 | stitched]."""
    ax_parent, ax_tl, ax_tr, ax_bl, ax_br, ax_bicubic, ax_stitch = ax_row

    parent_loc = rec['parent_loc']
    zp, xp, yp = _parse_loc(parent_loc)
    n_children  = len(rec['children'])

    # ── parent ──────────────────────────────────────────────────────────────
    parent_t  = process_img(ds[rec['parent_idx']]['image'])
    parent_np = tensor_to_numpy(parent_t)

    ax_parent.imshow(parent_np)
    ax_parent.set_title(
        f"Sample {sample_idx + 1}\nParent ZL={zp}  ({xp},{yp})\n[{n_children}/4 children]",
        fontsize=7, pad=3,
    )
    ax_parent.axis("off")

    # Colored quadrant outlines: solid for present, dashed+faded for absent
    h, w = parent_np.shape[:2]
    for (qx, qy), color in QUAD_COLORS.items():
        present = (qx, qy) in rec['children']
        rect = mpatches.Rectangle(
            (qx * w // 2, qy * h // 2), w // 2, h // 2,
            linewidth=2, edgecolor=color, facecolor="none",
            linestyle="solid" if present else "dashed",
            alpha=1.0 if present else 0.35,
        )
        ax_parent.add_patch(rect)

    # ── individual children ──────────────────────────────────────────────────
    child_axes  = {(0, 0): ax_tl, (1, 0): ax_tr, (0, 1): ax_bl, (1, 1): ax_br}
    child_tiles = {}

    for (qx, qy), ax in child_axes.items():
        zc, xc, yc = zp + 1, 2 * xp + qx, 2 * yp + qy

        if (qx, qy) in rec['children']:
            # Present child — show real tile
            tile_t  = process_img(ds[rec['children'][(qx, qy)]]['image'])
            child_tiles[(qx, qy)] = tile_t
            tile_np = tensor_to_numpy(tile_t)
            ax.imshow(tile_np)
            ax.set_title(
                f"ZL={zc} ({xc},{yc})\n{QUAD_LABELS[(qx, qy)]}",
                fontsize=6, pad=3,
                color=QUAD_COLORS[(qx, qy)],
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(QUAD_COLORS[(qx, qy)])
                spine.set_linewidth(3)
                spine.set_linestyle("solid")
        else:
            # Absent child — gray placeholder with hatching
            placeholder = np.full((256, 256, 3), _ABSENT_BG, dtype=np.uint8)
            ax.imshow(placeholder)
            ax.text(
                0.5, 0.5,
                f"ABSENT\n({xc},{yc})\nnot in Git-10M",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=7, color="#cc2222", fontweight="bold",
            )
            # Diagonal-hatch overlay (uses Axes coordinates so it fills the panel)
            hatch_rect = mpatches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                linewidth=0, facecolor="none",
                hatch="///", edgecolor=QUAD_COLORS[(qx, qy)], alpha=0.25,
            )
            ax.add_patch(hatch_rect)
            ax.set_title(
                f"ZL={zc} ({xc},{yc})\n{QUAD_LABELS[(qx, qy)]}  ✗",
                fontsize=6, pad=3,
                color="#777777",
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(QUAD_COLORS[(qx, qy)])
                spine.set_linewidth(2)
                spine.set_linestyle("dashed")

        ax.set_xticks([])
        ax.set_yticks([])

    # ── bicubic upsample of parent: 256×256 → 512×512 ───────────────────────
    bicubic_t  = torch.nn.functional.interpolate(
        parent_t.unsqueeze(0), size=(512, 512), mode="bicubic", align_corners=False
    ).squeeze(0).clamp(-1, 1)
    bicubic_np = tensor_to_numpy(bicubic_t)

    ax_bicubic.imshow(bicubic_np)
    ax_bicubic.set_title(
        f"Bicubic ↑512×512\n(parent ZL={zp} → ZL={zp+1})\nno learned detail",
        fontsize=7, pad=3,
    )
    for spine in ax_bicubic.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(1)
        spine.set_linestyle("dashed")
    ax_bicubic.set_xticks([])
    ax_bicubic.set_yticks([])

    # ── stitched 512×512 ─────────────────────────────────────────────────────
    tile_size = 256
    stitched  = torch.zeros(3, tile_size * 2, tile_size * 2)
    for (qx, qy), tile_t in child_tiles.items():
        stitched[:, qy * tile_size:(qy + 1) * tile_size,
                    qx * tile_size:(qx + 1) * tile_size] = tile_t
    stitch_np = tensor_to_numpy(stitched)

    n_present = len(child_tiles)
    missing_note = "\nblack = absent (no loss there)" if n_present < 4 else ""
    ax_stitch.imshow(stitch_np)
    ax_stitch.set_title(
        f"GT Stitched 512×512  ({n_present}/4 children)\nSR target ZL={zp+1}{missing_note}",
        fontsize=7, pad=3,
    )
    for spine in ax_stitch.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(1)
    ax_stitch.set_xticks([])
    ax_stitch.set_yticks([])


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    print("Loading metadata …")
    with open(args.meta_file, 'r') as f:
        meta = json.load(f)
    index_to_meta = meta['index_to_meta']

    split_indices = sorted(
        int(k) for k, v in index_to_meta.items() if v.get('split') == args.split
    )
    print(f"  {len(split_indices):,} indices in split='{args.split}'")

    print(f"Loading hierarchy from {args.hindex_pkl} …")
    with open(args.hindex_pkl, 'rb') as f:
        hierarchy = pickle.load(f)

    # Eval splits are guaranteed complete by the build script — enforce it here
    # too so cross-split parent lookups can never produce spurious partial quads.
    _eval_splits = {"val_random", "val_spatial", "random_test", "spatial_test"}
    require_complete = args.complete_only or (args.split in _eval_splits)
    if args.split in _eval_splits and not args.complete_only:
        print(f"  [auto] require_complete=True for eval split '{args.split}'")

    quad_mode = "complete quads only" if require_complete else "partial quads (≥1 child)"
    print(f"Building pairs ({quad_mode}, scanning up to {args.n_samples * 20} pairs) …")
    pairs = build_pairs(
        split_indices, index_to_meta, hierarchy,
        max_pairs=args.n_samples * 20,
        require_complete=require_complete,
    )
    if len(pairs) < args.n_samples:
        print(f"Warning: only {len(pairs)} pairs found; plotting all of them.")

    if args.seed is not None:
        import random
        random.seed(args.seed)
        random.shuffle(pairs)
    pairs = pairs[:args.n_samples]

    # Sort so partial quads (interesting for debugging) appear first when not shuffled
    if args.seed is None:
        pairs.sort(key=lambda r: len(r['children']), reverse=False)

    n_complete = sum(1 for r in pairs if r['is_complete'])
    n_partial  = len(pairs) - n_complete
    print(f"  Using {len(pairs)} pairs — {n_complete} complete, {n_partial} partial")

    print(f"Loading HuggingFace dataset from cache={args.cache_dir} …")
    ds = load_dataset("lcybuaa/Git-10M", split="train", cache_dir=args.cache_dir)

    n = len(pairs)
    col_labels = [
        f"Parent (ZL)  split={args.split}",
        "TL child", "TR child", "BL child", "BR child",
        "Bicubic ↑512 (baseline)",
        "GT Stitched (ZL+1)",
    ]
    fig, axes = plt.subplots(
        n, 7,
        figsize=(2.4 + 5 * 1.8 + 2 * 3.6, n * 3.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.45, "width_ratios": [1, 1, 1, 1, 1, 2, 2]},
    )
    if n == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Column headers on first row
    for ax, label in zip(axes[0], col_labels):
        ax.set_title(label, fontsize=8, color="#222222", pad=10, fontweight="bold")

    for i, rec in enumerate(pairs):
        plot_sample(ds, rec, axes[i], i)

    out_path = args.out
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Probe HFParentChildDataset — quadtree visualizer")
    parser.add_argument("--meta_file",    default=os.environ.get("GENESIS_GIT10M_META", "data/git10m_quad_meta.json"))
    parser.add_argument("--hindex_pkl",   default=os.environ.get("GENESIS_HINDEX", "data/hierarchy_index_quad.pkl"))
    parser.add_argument("--cache_dir",    default=os.environ.get("HF_DATASETS_CACHE", "data/hf_cache"))
    parser.add_argument("--split",        default="train",
                        choices=["train", "val_random", "val_spatial", "random_test", "spatial_test"],
                        help="Which metadata split to sample from (default: train)")
    parser.add_argument("--n_samples",    default=3, type=int,
                        help="Number of quad samples to visualize")
    parser.add_argument("--complete_only", action="store_true",
                        help="Restrict to complete 2×2 quads (all 4 children present). "
                             "Default: show partial quads too, with absent children as gray placeholders.")
    parser.add_argument("--seed",         default=99, type=int,
                        help="Random seed for sample selection (None = first N pairs found)")
    parser.add_argument("--out",          default="sample_example.png",
                        help="Output PNG path")
    main(parser.parse_args())
