#!/usr/bin/env python
"""Extended Dense500 metrics: semantic alignment and pyramid consistency.

Only the metrics reported in the paper are computed here. All scorers lazy-load
only the models their family needs, so families can run as independent jobs.

Families
--------
semantic     full-reference per tile : CLIP-I, DINOv3-sat cosine            (vs GT)
consistency  per internal tile       : quadtree LR-PSNR box (self) + xLR-PSNR
                                        (cross-scale) + RAPSD spectral distance
caption      per site (scene)        : VLM caption(gen|gt) -> sentence cosine
                                        (see caption_agree.py)

Conventions: tiles are 256x256 RGB PIL, on [0,1] tensors. Higher better:
CLIP-I / DINOv3-sat cosines / LR-PSNR / xLR-PSNR. Lower better: RAPSD-distance.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

TILE_PX = 256


# ─────────────────────────────────────────────────────────────────────────────
# tensor helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_t01(img: Image.Image, device) -> torch.Tensor:
    """PIL RGB -> (1,3,H,W) float in [0,1]."""
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(device)


def box_down2(img: Image.Image) -> Image.Image:
    """Physically-correct quadtree downsample: 2x2 box average (area)."""
    return img.resize((img.width // 2, img.height // 2), Image.Resampling.BOX)


def stitch_quad(children: Dict[Tuple[int, int], Image.Image], px: int = TILE_PX) -> Image.Image:
    """Stitch 4 children keyed by (dx,dy) in {0,1}^2 into a 2px x 2px mosaic."""
    m = Image.new("RGB", (2 * px, 2 * px))
    for (dx, dy), im in children.items():
        m.paste(im.convert("RGB").resize((px, px)), (dx * px, dy * px))
    return m


def rapsd(img: Image.Image) -> np.ndarray:
    """Radially-averaged power spectral density of the grayscale tile.

    FFT2 -> fftshift -> |.|^2 -> azimuthal average over integer radii. Returns the
    1-D profile (length ~ H/2). Standard GAN/diffusion spectral-artifact probe
    (Durall 2020; Schwarz 2021). Compare gen-vs-GT profiles per zoom level.
    """
    a = np.asarray(img.convert("L"), np.float64)
    a = a - a.mean()
    f = np.fft.fftshift(np.fft.fft2(a))
    p = np.abs(f) ** 2
    h, w = p.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.hypot(x - cx, y - cy).astype(int)
    nbin = min(cy, cx)
    tbin = np.bincount(r.ravel(), p.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin[:nbin] / np.maximum(nr[:nbin], 1)
    return prof


# ─────────────────────────────────────────────────────────────────────────────
# Semantic embedding cosine: CLIP-I (DreamBooth-standard) + DINOv3-sat
# ─────────────────────────────────────────────────────────────────────────────

class EmbedScorer:
    """Image-image cosine in several encoders. Higher = more semantically aligned.

    encoders: 'clip' (open_clip ViT-B-32 openai, DreamBooth-standard CLIP-I),
              'dinov3sat' (in-repo satellite DINOv3 ViT-L/16; domain instance-level).
    """

    def __init__(self, device, encoders=("clip", "dinov3sat")):
        self.device = device
        self.enc: Dict[str, tuple] = {}
        for e in encoders:
            try:
                self.enc[e] = self._build(e)
                print(f"[embed] loaded {e}", flush=True)
            except Exception as ex:                       # noqa: BLE001
                print(f"[embed] SKIP {e}: {ex}", flush=True)

    def _build(self, e):
        import open_clip
        if e == "clip":
            # quickgelu variant matches the openai CLIP weights exactly (DreamBooth CLIP-I).
            m, _, pp = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
            return ("clip", m.visual.to(self.device).eval(), pp, m.to(self.device).eval())
        if e == "dinov3sat":
            import torchvision.transforms as T
            w = (os.environ.get("GENESIS_DINO_WEIGHTS") or "").strip()
            if not w:
                # Auto-resolve (and download from the HF release if missing) — same
                # path the SR eval uses, so a bare `run_all_dense500.sh` works.
                from common import resolve_dino_weights
                w = resolve_dino_weights(None)
            model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16", weights=w)
            model = model.to(self.device).eval()
            pp = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            return ("dino", model, pp, None)
        raise ValueError(e)

    @torch.no_grad()
    def _embed(self, kind, visual, pp, full, imgs: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([pp(im.convert("RGB")) for im in imgs]).to(self.device)
        if kind == "clip":
            f = full.encode_image(batch)
        else:                                              # dino: CLS token
            f = visual(batch)
            if isinstance(f, dict):
                f = f.get("x_norm_clstoken", f.get("cls_token"))
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f

    @torch.no_grad()
    def cosine_batch(self, preds: List[Image.Image], gts: List[Image.Image]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for e, (kind, visual, pp, full) in self.enc.items():
            fp = self._embed(kind, visual, pp, full, preds)
            fg = self._embed(kind, visual, pp, full, gts)
            cos = (fp * fg).sum(-1).clamp(-1, 1).cpu().tolist()
            out[f"{e}_cos"] = cos
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quadtree pyramid consistency  (LR-PSNR box, xLR-PSNR, RAPSD)
# ─────────────────────────────────────────────────────────────────────────────

def _psnr01(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), np.float64) / 255.0
    y = np.asarray(b.convert("RGB"), np.float64) / 255.0
    mse = float(((x - y) ** 2).mean())
    return 99.0 if mse <= 1e-12 else float(-10.0 * np.log10(mse))


def consistency_for_site(gen: Dict[Tuple[int, int, int], Image.Image],
                         gt: Dict[Tuple[int, int, int], Image.Image],
                         root_z: int, depth: int) -> List[Dict]:
    """Per internal tile (zoom < leaf): quadtree LR-consistency.

    LR-PSNR box (self):  parent_gen  vs  box_down2(stitch(4 gen children))
    xLR-PSNR (cross):    box_down2(stitch gen children) vs box_down2(stitch gt children)
    RAPSD:               spectral distance between the gen and gt child mosaics.
    Returns one row per internal tile with its level (=z-root_z) and zoom.
    """
    leaf_z = root_z + depth - 1
    rows: List[Dict] = []
    # group by parent: a parent at (z,x,y) has children (z+1, 2x+dx, 2y+dy)
    coords = sorted(gen.keys())
    for (z, x, y) in coords:
        if z >= leaf_z:
            continue
        kids = {(dx, dy): (z + 1, 2 * x + dx, 2 * y + dy) for dx in (0, 1) for dy in (0, 1)}
        if not all(c in gen for c in kids.values()):
            continue
        gen_kids = {dxy: gen[c] for dxy, c in kids.items()}
        mos_gen = stitch_quad(gen_kids)
        dg_box = box_down2(mos_gen)
        parent_gen = gen.get((z, x, y))
        row = {"level": z - root_z, "zoom": z,
               "lrpsnr_self_box": _psnr01(dg_box, parent_gen)}
        if all(c in gt for c in kids.values()):
            mos_gt = stitch_quad({dxy: gt[c] for dxy, c in kids.items()})
            dgt_box = box_down2(mos_gt)
            row["lrpsnr_cross_box"] = _psnr01(dg_box, dgt_box)          # xLR-PSNR
            # spectral consistency: |log RAPSD_gen - log RAPSD_gt| over the child mosaic
            pg, pt = rapsd(mos_gen), rapsd(mos_gt)
            n = min(len(pg), len(pt))
            row["rapsd_sd"] = float(np.abs(np.log(pg[1:n] + 1e-12) - np.log(pt[1:n] + 1e-12)).mean())
        rows.append(row)
    return rows
