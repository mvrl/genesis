"""Outpainting evaluation: per-sample metrics → <out_dir>/samples.csv + run.json.
Quadrant-only regimes: ``quad1 / quad2_adj / quad2_diag / quad3``
(see ``make_mask_for_regime`` and ``docs/METRICS.md``). Masks are sampled
deterministically per ``(seed, idx)``. Reference-based metrics
(psnr/ssim/lpips/dists/boundary_diff) are restricted to the
hole region — the model overwrites the known region with GT, so
full-image scores would be trivially inflated."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_THIS, "..", "outpainting"))
sys.path.insert(0, os.path.join(_THIS, "..", "superresolution"))   # for model_sr.get_dino_embeddings
sys.path.insert(0, os.path.join(_THIS, ".."))

from common import (   # noqa: E402
    DEFAULT_META_FILE,
    SampleWriter,
    auto_tag_dir,
    compute_metrics,
    default_results_root,
    dump_run_meta,
    load_op_denoiser,
    meta_columns,
    resolve_ckpt,
    resolve_dino_weights,
    resolve_meta_path,
    tensor_to_pil,
)
# `tensor_to_pil` from common handles both (3,H,W) and (1,3,H,W); aliased
# so the eval-loop reads naturally.
tensor_to_image_pil = tensor_to_pil


_TILE = 256


# ── Mask generation (deterministic per (seed, idx)) ──────────────────────────

QUAD_BOXES = {
    "UL": (0, 0, _TILE // 2, _TILE // 2),
    "UR": (_TILE // 2, 0, _TILE, _TILE // 2),
    "LL": (0, _TILE // 2, _TILE // 2, _TILE),
    "LR": (_TILE // 2, _TILE // 2, _TILE, _TILE),
}


_QUAD_KEYS = ("UL", "UR", "LL", "LR")
_ADJ_PAIRS = (("UL", "UR"), ("LL", "LR"), ("UL", "LL"), ("UR", "LR"))  # top/bottom/left/right pairs
_DIAG_PAIRS = (("UL", "LR"), ("UR", "LL"))


def make_mask_for_regime(regime: str, idx: int, seed: int) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return (mask (1,H,W) float, mask_params dict).
    mask: 1 = hole, 0 = known. Reproducible per (seed, idx).

    Genesis OP eval is quadrant-only. Regimes parametrize the number of
    *known* quadrants — at least one quadrant must be known (the model
    needs spatial context to generate from; the 0-known case degenerates
    to unconditional generation and is not supported):

      quad1       : 1 known, 3 hole       (75% hole — Genesis pyramid setting)
      quad2_adj   : 2 known adjacent      (50% hole — connected rectangular hole)
      quad2_diag  : 2 known diagonal      (50% hole — two disconnected quadrants)
      quad3       : 3 known, 1 hole       (25% hole — single-quadrant fill)
      quad        : alias for quad1 (legacy)

    Which specific quadrants are picked is uniformly sampled per
    ``(seed, idx)`` so the same regime+seed combination reproduces the same
    masks across runs.
    """
    rng = np.random.default_rng(seed * 1_000_003 + int(idx))
    mask = np.zeros((1, _TILE, _TILE), dtype=np.float32)

    def _hole_quads(known_set):
        """Fill `mask` with 1s, then carve out the known quadrants."""
        mask[:] = 1.0
        for q in known_set:
            x1, y1, x2, y2 = QUAD_BOXES[q]
            mask[:, y1:y2, x1:x2] = 0.0
        return torch.from_numpy(mask), {
            "kind": "quad",
            "known_count": len(known_set),
            "known": sorted(known_set),
        }

    if regime in ("quad1", "quad"):
        kept = _QUAD_KEYS[int(rng.integers(0, 4))]
        return _hole_quads({kept})

    if regime == "quad2_adj":
        pair = _ADJ_PAIRS[int(rng.integers(0, 4))]
        return _hole_quads(set(pair))

    if regime == "quad2_diag":
        pair = _DIAG_PAIRS[int(rng.integers(0, 2))]
        return _hole_quads(set(pair))

    if regime == "quad3":
        hole_quad = _QUAD_KEYS[int(rng.integers(0, 4))]
        known = set(_QUAD_KEYS) - {hole_quad}
        return _hole_quads(known)

    raise ValueError(
        f"Unknown --regime {regime!r}; expected one of "
        "quad1 / quad2_adj / quad2_diag / quad3 / quad (alias of quad1)."
    )


# ── Eval dataset ─────────────────────────────────────────────────────────────

class EvalOutpaintingDataset(Dataset):
    """Reads tiles from HF dataset by `indices`; stamps deterministic regime-specific masks.

    Yields:
      image       (3, 256, 256) in [-1, 1]
      roi_mask    (1, 256, 256) — 1=hole, 0=known
      zoom_level  scalar float — tile zoom from index_to_meta
      hf_idx      int
      mask_params dict (mask spec for CSV)
    """

    def __init__(self, hf_dataset, indices: List[int], index_to_meta: Dict[str, Dict],
                 regime: str, seed: int):
        self.hf  = hf_dataset
        self.idx = list(indices)
        self.meta = index_to_meta
        self.regime = regime
        self.seed = int(seed)
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((_TILE, _TILE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        hf_i = self.idx[i]
        item = self.hf[hf_i]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_t = self.transform(img)
        if not torch.isfinite(image_t).all():
            image_t = torch.zeros_like(image_t)

        mask, params = make_mask_for_regime(self.regime, i, self.seed)
        m = self.meta.get(str(hf_i)) or self.meta.get(hf_i) or {}
        zoom = float(m.get("zoom") or 0.0)

        return {
            "image":       image_t,
            "roi_mask":    mask,
            "zoom_level":  torch.tensor(zoom),
            "hf_idx":      hf_i,
            "mask_params": json.dumps(params, sort_keys=True),
            # Git-10M text / quality / resolution metadata. Legacy evals that
            # don't reference these keys via the collate function will silently
            # drop them, so adding them here is backwards-compatible.
            "caption":     item.get("caption") or "",
            "text":        item.get("text") or "",
            "img_quality": float(item.get("img_quality_score") or 0.0),
            "resolution":  item.get("resolution") or "",
        }


def _collate(batch):
    return {
        "image":       torch.stack([b["image"]      for b in batch]),
        "roi_mask":    torch.stack([b["roi_mask"]   for b in batch]),
        "zoom_level":  torch.stack([b["zoom_level"] for b in batch]),
        "hf_idx":      torch.tensor([b["hf_idx"]    for b in batch], dtype=torch.long),
        "mask_params": [b["mask_params"] for b in batch],
        "caption":     [b["caption"]     for b in batch],
        "text":        [b["text"]        for b in batch],
        "img_quality": [b["img_quality"] for b in batch],
        "resolution":  [b["resolution"]  for b in batch],
    }


# ── Image helpers ────────────────────────────────────────────────────────────

def tensor_to_mask_pil(mask_t: torch.Tensor) -> Image.Image:
    """(1, H, W) float in {0, 1} → "L" PIL where 255=hole, 0=known."""
    arr = (mask_t.detach().cpu().numpy()[0] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def make_masked_input(gt_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """GT with the hole zeroed, matching the model's masked image input."""
    gt_arr = np.asarray(gt_pil.convert("RGB")).copy()
    hole = np.asarray(mask_pil) > 127
    gt_arr[hole] = 0
    return Image.fromarray(gt_arr, "RGB")


def save_op_comparison(path: str, masked_in_pil: Image.Image,
                       pred_pil: Image.Image, gt_pil: Image.Image) -> None:
    """3-panel: masked input | prediction | GT."""
    w, h = gt_pil.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(masked_in_pil.convert("RGB"), (0, 0))
    canvas.paste(pred_pil.convert("RGB"), (w, 0))
    canvas.paste(gt_pil.convert("RGB"), (w * 2, 0))
    canvas.save(path)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_metric_list(s: str) -> List[str]:
    s = (s or "").strip().lower()
    if s in ("", "all"):
        return ["psnr", "ssim", "lpips", "dists", "boundary_diff"]
    return [m.strip() for m in s.split(",") if m.strip()]


def get_args():
    p = argparse.ArgumentParser("Genesis outpainting evaluation")

    p.add_argument("--ckpt", default=None)
    p.add_argument("--exp",  default=None,
                   help="Experiment name: main_b16 or main_h16. OP is ablation-free.")
    p.add_argument("--stage", default=3, type=int, choices=(3,),
                   help="Stage 3 end-of-curriculum ckpt (the only supported OP eval stage).")
    p.add_argument("--model", default="JiT-B/16")

    p.add_argument("--split", default="val_random",
                   choices=("val_random", "val_spatial", "random_test", "spatial_test"))
    p.add_argument(
        "--regime",
        default="quad1",
        choices=("quad1", "quad2_adj", "quad2_diag", "quad3", "quad"),
        help="Quadrant-known regime: 1/2-adj/2-diag/3 known quadrants. "
             "`quad` is a legacy alias for `quad1`. (0-known is not supported "
             "— with no spatial context the model has nothing to generate from.)",
    )

    p.add_argument("--cfg", default=1.0, type=float)
    p.add_argument("--num_sampling_steps", default=50, type=int)

    p.add_argument("--max_samples", default=0, type=int, help="0 = full split")
    p.add_argument("--start_sample", default=0, type=int)
    p.add_argument("--sample_count", default=0, type=int,
                   help="Number of samples in this shard; 0 = use the remaining split.")
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--num_workers", default=4, type=int)

    p.add_argument("--metrics", default="psnr,ssim,lpips,dists,boundary_diff",
                   help='Comma-separated, or "all".')
    p.add_argument("--lpips_max_size", default=0, type=int,
                   help="Cap LPIPS/DISTS input side length (0 = no cap; OP outputs are 256).")

    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int,
                   help="RNG seed for mask sampling. Same (seed, idx) → same mask across runs.")

    p.add_argument("--meta_file_path", default=DEFAULT_META_FILE,
                   help="Git-10M quad meta; auto-pulls from the Google Drive release if missing.")
    p.add_argument("--hf_data_cache", default=os.environ.get("HF_DATASETS_CACHE") or None,
                   help="HF datasets cache dir for lcybuaa/Git-10M (default: HF's own cache).")
    p.add_argument("--dino_weights", default="", help="Empty = auto-resolve / auto-pull from Google Drive.")
    p.add_argument("--no_dino", action="store_true",
                   help="Disable DINO conditioning at inference. Required for the main OP "
                        "ckpts (which are no-DINO): the architecture is identical, the "
                        "generate() path just skips the DINO embedding pass.")

    p.add_argument("--out_dir", default=None)
    p.add_argument("--save_images", default=-1, type=int,
                   help="Save first N pred/target/masked_input/comparison PNGs per shard. "
                        "-1 = save all, 0 = no images, N>0 = first N.")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    metrics = parse_metric_list(args.metrics)
    if args.start_sample < 0 or args.sample_count < 0:
        raise ValueError("--start_sample and --sample_count must be non-negative.")

    # The main OP checkpoints are trained no-DINO; attaching DINO would route
    # features through an untrained projection. Force the paper configuration.
    if (args.exp or "").startswith("main_") and not args.no_dino:
        print("[eval-op] main OP checkpoints are no-DINO — forcing --no_dino.", flush=True)
        args.no_dino = True

    ckpt_path = resolve_ckpt("op", args.ckpt, args.exp, args.stage)
    dino      = "" if args.no_dino else resolve_dino_weights(args.dino_weights)

    denoiser, eng_args = load_op_denoiser(
        ckpt_path, args.model, dino, device,
        cfg=args.cfg, num_sampling_steps=args.num_sampling_steps,
    )

    # ── Build dataset ───────────────────────────────────────────────────────
    meta_path = resolve_meta_path(args.meta_file_path, "shared/git10m_quad_meta.json", "Git-10M meta")
    print(f"[eval-op] Loading meta from {meta_path}...", flush=True)
    with open(meta_path) as f:
        meta = json.load(f)
    splits        = meta["splits"]
    index_to_meta = meta["index_to_meta"]

    from datasets import load_dataset
    print(f"[eval-op] Loading HF dataset (cache={args.hf_data_cache})...", flush=True)
    hf_ds = load_dataset("lcybuaa/Git-10M", split="train", cache_dir=args.hf_data_cache)

    ds = EvalOutpaintingDataset(
        hf_dataset    = hf_ds,
        indices       = splits[args.split],
        index_to_meta = index_to_meta,
        regime        = args.regime,
        seed          = args.seed,
    )

    original_n = len(ds)
    if args.start_sample > 0 or args.sample_count > 0:
        end = (args.start_sample + args.sample_count) if args.sample_count > 0 else original_n
        end = min(end, original_n)
        ds.idx = ds.idx[args.start_sample:end]
        print(
            f"[eval-op] shard slice start={args.start_sample} "
            f"end={end} count={len(ds)} from full_split={original_n}",
            flush=True,
        )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, collate_fn=_collate)

    # ── Output dir ──────────────────────────────────────────────────────────
    out_root = args.out_dir or default_results_root()
    out_dir  = auto_tag_dir(out_root, [
        "op", args.exp or "ckpt", f"s{args.stage}", args.split,
        f"r{args.regime}", f"cfg{args.cfg}", f"seed{args.seed}",
    ])
    print(f"[eval-op] Writing results under {out_dir}", flush=True)
    pred_dir = os.path.join(out_dir, "predictions")
    target_dir = os.path.join(out_dir, "targets")
    masked_dir = os.path.join(out_dir, "masked_inputs")
    comp_dir = os.path.join(out_dir, "comparisons")
    if args.save_images != 0:
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(masked_dir, exist_ok=True)
        os.makedirs(comp_dir, exist_ok=True)

    # ── CSV columns ─────────────────────────────────────────────────────────
    cols = ["sample_idx", "regime", "mask_params", "hole_fraction",
            "hf_idx", "location", "satellite", "split",
            "lat", "lon", "gsd", "zoom"] + metrics + [
                "prediction_path", "target_path", "masked_input_path",
                "comparison_path", "error"
            ]
    writer = SampleWriter(os.path.join(out_dir, "samples.csv"))
    writer.open(cols)

    # ── Eval loop ───────────────────────────────────────────────────────────
    n_done = n_err = 0
    t0 = time.time()
    # Clamp so progress prints don't claim more samples than the split contains.
    target_total = min(args.max_samples, len(ds)) if args.max_samples > 0 else len(ds)
    if target_total < 1:
        if args.start_sample > 0 or args.sample_count > 0:
            print(
                f"[eval-op] no samples in this shard (start={args.start_sample}, "
                f"regime={args.regime}); exiting cleanly.",
                flush=True,
            )
            return
        raise ValueError("No OP samples found for this split.")

    for batch in loader:
        if n_done >= target_total:
            break
        with torch.no_grad():
            pred = denoiser.generate({
                "image":      batch["image"].to(device),
                "roi_mask":   batch["roi_mask"].to(device),
                "zoom_level": batch["zoom_level"].to(device),
            })   # (B, 3, 256, 256), known region overwritten with GT internally

        B = pred.shape[0]
        for k in range(B):
            if n_done >= target_total:
                break
            hf_idx   = int(batch["hf_idx"][k].item())
            mask_pil = tensor_to_mask_pil(batch["roi_mask"][k])
            base_pil = tensor_to_image_pil(batch["image"][k])    # GT tile
            pred_pil = tensor_to_image_pil(pred[k].cpu())

            hole_arr = np.asarray(mask_pil) > 127
            hole_frac = float(hole_arr.mean())

            row: Dict[str, Any] = {
                "sample_idx":     args.start_sample + n_done,
                "regime":         args.regime,
                "mask_params":    batch["mask_params"][k],
                "hole_fraction":  hole_frac,
            }
            row.update(meta_columns(index_to_meta, hf_idx))
            if args.save_images != 0 and (args.save_images < 0 or n_done < args.save_images):
                global_sample_idx = args.start_sample + n_done
                stem = f"sample{global_sample_idx:06d}_hf{hf_idx:08d}_{args.regime}"
                pred_path = os.path.join(pred_dir, f"{stem}_pred.png")
                target_path = os.path.join(target_dir, f"{stem}_target.png")
                masked_path = os.path.join(masked_dir, f"{stem}_masked_input.png")
                comp_path = os.path.join(comp_dir, f"{stem}_masked_pred_target.png")
                masked_in_pil = make_masked_input(base_pil, mask_pil)
                pred_pil.save(pred_path)
                base_pil.save(target_path)
                masked_in_pil.save(masked_path)
                save_op_comparison(comp_path, masked_in_pil, pred_pil, base_pil)
                row.update({
                    "prediction_path": pred_path,
                    "target_path": target_path,
                    "masked_input_path": masked_path,
                    "comparison_path": comp_path,
                })
            try:
                m = compute_metrics(
                    pred_pil, base_pil,
                    mask=mask_pil,            # restricts PSNR/SSIM to hole pixels
                    base=base_pil,            # for boundary_diff
                    metrics=metrics,
                    lpips_max_size=args.lpips_max_size,
                    device=device,
                )
                row.update(m)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                n_err += 1
            writer.append(row)
            n_done += 1

        elapsed = time.time() - t0
        print(f"[eval-op] {n_done}/{target_total}  ({elapsed/max(n_done,1):.2f}s/sample)", flush=True)

    n_written = writer.close()
    elapsed = time.time() - t0

    dump_run_meta(
        out_dir,
        task="op",
        exp=args.exp,
        stage=args.stage,
        ckpt=ckpt_path,
        split=args.split,
        extra={"regime": args.regime,
               "model": args.model,
               "no_dino": args.no_dino,
               "max_samples": args.max_samples,
               "start_sample": args.start_sample,
               "sample_count": args.sample_count,
               "lpips_max_size": args.lpips_max_size,
               "save_images": args.save_images},
        config={"cfg": args.cfg, "num_sampling_steps": args.num_sampling_steps,
                "batch_size": args.batch_size, "seed": args.seed},
        n_samples=n_written,
        n_errors=n_err,
        elapsed_seconds=elapsed,
        device=str(device),
        metrics=metrics,
    )
    print(f"[eval-op] DONE — {n_written} samples ({n_err} errors) in {elapsed:.1f}s.", flush=True)
    print(f"[eval-op] CSV : {os.path.join(out_dir, 'samples.csv')}", flush=True)
    print(f"[eval-op] meta: {os.path.join(out_dir, 'run.json')}", flush=True)


if __name__ == "__main__":
    main()
