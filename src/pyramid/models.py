"""Thin model wrapper over the demo inference code (`genesis_common`).

Reuses `demos/utils/genesis_common.py` for checkpoint loading (EMA) and the SR/OP
`generate` calls, with true-zoom conditioning (the actual tile zoom feeds the
model's zoom embedder). Checkpoints resolve from local disk first (set
GENESIS_CKPT_BASE / GENESIS_DINO_WEIGHTS *before* importing genesis_common, whose
path constants are computed at import), else auto-pull from the Google Drive
release folder via rclone.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

from .tiles import Tile, QUAD_ORDER, children_of

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TILE_PX = 256

# Per-arch relative checkpoint paths under a checkpoint root (gdrive layout).
_SR_REL = {
    "JiT-B/16": ("main", "superresolution", "main_B16", "sr-full-tile-stage3-step0800000.ckpt"),
    "JiT-H/16": ("main", "superresolution", "main_H16", "sr-full-tile-stage3-step0800000.ckpt"),
}
_OP_REL = {
    "JiT-B/16": ("main", "outpainting", "main_B16", "op-new-stage3-step0800000.ckpt"),
    "JiT-H/16": ("main", "outpainting", "main_H16", "op-new-stage3-step0800000.ckpt"),
}

# Candidate checkpoint roots, in priority order. The user's shell may set
# GENESIS_CKPT_BASE one level too high (…/gdrive_ckpts instead of
# …/gdrive_ckpts/genesis), so we also probe a "genesis" subdir and the repo
# mirror. The first root that actually contains the file wins.
def _candidate_roots() -> list:
    roots = []
    env = (os.environ.get("GENESIS_CKPT_BASE") or "").strip()
    if env:
        roots += [env, os.path.join(env, "genesis")]
    roots += [
        os.path.join(_REPO, "gdrive_ckpts", "genesis"),
        os.path.join(_REPO, "gdrive_ckpts"),
        os.path.join(_REPO, "logs", "ckpts"),
    ]
    # de-dup, preserve order
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r); out.append(r)
    return out


def _resolve_ckpt(rel: tuple, arch: str, task: str) -> str:
    # 1) Local probe across all candidate roots (fast path; no network).
    for root in _candidate_roots():
        p = os.path.join(root, *rel)
        if os.path.isfile(p):
            return p
    # 2) Not on disk anywhere → auto-pull from the Google Drive release folder.
    #    `rel` already mirrors the gdrive layout (e.g. main/superresolution/...),
    #    so the rclone source is just gdrive:<rel> (anchored at the release folder
    #    id by genesis_common). Download into genesis_common's _CKPT_BASE so the
    #    local mirror matches the gdrive tree and later runs hit the fast path.
    target = os.path.join(gc._CKPT_BASE, *rel)
    gdrive_src = f"{gc.GDRIVE_REMOTE}:" + "/".join(rel)
    try:
        return gc._gdrive_pull_if_missing(target, gdrive_src,
                                          label=f"{task} checkpoint ({arch})")
    except FileNotFoundError as e:
        tried = "\n  ".join(os.path.join(r, *rel) for r in _candidate_roots())
        raise FileNotFoundError(
            f"{task} checkpoint for {arch} not found locally and the Google Drive "
            f"auto-pull failed.\nProbed local roots:\n  {tried}\n\n{e}") from e


# Pin GENESIS_CKPT_BASE to the resolved local mirror so genesis_common's own
# path defaults (and any internal DINO resolution) line up — set BEFORE import.
for _r in _candidate_roots():
    if os.path.isdir(os.path.join(_r, "main")):
        os.environ["GENESIS_CKPT_BASE"] = _r
        break

# DINOv3 satellite weights used by SR (avoid the rclone auto-pull).
_DINO_DEFAULT = os.environ.get("GENESIS_DINO_WEIGHTS", "")
if not (os.environ.get("GENESIS_DINO_WEIGHTS") or "").strip() and os.path.isfile(_DINO_DEFAULT):
    os.environ["GENESIS_DINO_WEIGHTS"] = _DINO_DEFAULT

# Import the demo inference helpers (standalone module, no relative imports).
sys.path.insert(0, os.path.join(_REPO, "demos", "utils"))
import genesis_common as gc  # noqa: E402


def pick_devices() -> Tuple[torch.device, torch.device]:
    """(sr_device, op_device). Both on cuda:0 when only one GPU is allocated."""
    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cpu")
    dev = torch.device("cuda:0")
    return dev, dev


class ModelBundle:
    """Lazily-loaded SR + OP models with the quad-SR / quadrant-OP primitives.

    A strategy receives one ``ModelBundle`` and calls ``sr_children`` /
    ``op_fill``. Loading is deferred so non-model baselines (bicubic) cost nothing.
    """

    def __init__(self, arch: str = "JiT-B/16", steps: int = 50, cfg: float = 1.0,
                 use_dino_sr: bool = True,
                 bicubic_init: bool = False,
                 bicubic_init_t_start: float = 0.5,
                 sr_device: Optional[torch.device] = None,
                 op_device: Optional[torch.device] = None):
        self.arch = arch
        self.steps = steps
        self.cfg = cfg
        self.use_dino_sr = use_dino_sr
        self.bicubic_init = bicubic_init
        self.bicubic_init_t_start = bicubic_init_t_start
        d0, d1 = pick_devices()
        self.sr_device = sr_device or d0
        self.op_device = op_device or d1
        self._sr = None
        self._op = None
        self.sr_calls = 0
        self.op_calls = 0

    # ── loaders ───────────────────────────────────────────────────────────────
    def _ensure_sr(self):
        if self._sr is None:
            ckpt = _resolve_ckpt(_SR_REL[self.arch], self.arch, "SR")
            dino = gc.resolved_dino_weights_path() if self.use_dino_sr else ""
            gc.build_sr_args(self.arch, dino_weights=dino, steps=self.steps, cfg=self.cfg,
                             bicubic_init=self.bicubic_init,
                             bicubic_init_t_start=self.bicubic_init_t_start)
            self._sr, _ = gc.load_sr_denoiser(ckpt, self.arch, dino_weights=dino, device=self.sr_device)
            self._sr.args.num_sampling_steps = self.steps
            self._sr.args.cfg = self.cfg
            self._sr.args.bicubic_init = self.bicubic_init
            self._sr.args.bicubic_init_t_start = self.bicubic_init_t_start
        return self._sr

    def _ensure_op(self):
        if self._op is None:
            ckpt = _resolve_ckpt(_OP_REL[self.arch], self.arch, "OP")
            # OP main config is locked no-DINO.
            self._op, _ = gc.load_op_denoiser(ckpt, self.arch, dino_weights="", device=self.op_device)
            self._op.args.num_sampling_steps = self.steps
            self._op.args.cfg = self.cfg
        return self._op

    # ── primitives ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sr_children(self, parent_img: Image.Image, parent: Tile) -> Dict[Tile, Image.Image]:
        """One SR call on ``parent`` → dict of all four child tiles → 256² images.

        The model returns a 512² mosaic (UL, UR, LL, LR). We split it into the
        four children at ``parent_z + 1`` so siblings share one stochastic draw.
        """
        sr = self._ensure_sr()
        child_z = parent[0] + 1
        mosaic = gc.run_superresolution(sr, parent_img, self.sr_device, target_zoom=child_z)
        self.sr_calls += 1
        if mosaic.size != (2 * TILE_PX, 2 * TILE_PX):
            mosaic = mosaic.resize((2 * TILE_PX, 2 * TILE_PX), Image.Resampling.BILINEAR)
        out: Dict[Tile, Image.Image] = {}
        for child, (rx, ry) in zip(children_of(parent), QUAD_ORDER):
            box = (rx * TILE_PX, ry * TILE_PX, (rx + 1) * TILE_PX, (ry + 1) * TILE_PX)
            out[child] = mosaic.crop(box)
        return out

    @torch.no_grad()
    def sr_level_md(self, coarse_pil: Image.Image, target_zoom: Optional[int],
                    t_start: float,
                    seeds: Optional[Dict] = None,
                    stride_frac: float = 0.5) -> Image.Image:
        """MultiDiffusion joint 2x SR of a whole level mosaic (seam-free).

        ``coarse_pil`` is the G*256 mosaic at the parent level; returns the
        2G*256 mosaic at ``target_zoom``. ``seeds`` maps child-grid (col,row) →
        256² PIL of known leaf tiles to anchor. Counts one SR call per window
        (compute-fair accounting).
        """
        from .multidiffusion import multidiffusion_sr_level, _window_origins, WIN
        sr = self._ensure_sr()
        dino = gc.resolved_dino_weights_path() if self.use_dino_sr else ""
        out = multidiffusion_sr_level(
            sr, coarse_pil, self.sr_device,
            target_zoom=target_zoom, steps=self.steps, cfg=self.cfg,
            t_start=t_start, noise_scale=getattr(sr, "noise_scale", 1.0),
            dino_weights=dino, seeds=seeds, stride_frac=stride_frac,
        )
        stride = max(1, int(round(WIN * stride_frac)))
        self.sr_calls += len(_window_origins(coarse_pil.size[0], stride)) ** 2
        return out

    @torch.no_grad()
    def sr_level_unified(self, coarse_pil: Image.Image, target_zoom: Optional[int],
                         t_start: float, seeds: Optional[Dict] = None,
                         overlap_px: int = 32,
                         init_pil: Optional[Image.Image] = None,
                         guidance_scale: float = 0.0,
                         guidance_t_min: float = 0.75,
                         guidance_mode: str = "hf",
                         sync_scale: float = 0.0) -> Image.Image:
        """Unified single-canvas MultiDiffusion 2x SR of a whole level mosaic.

        Same mechanism at every level; ``overlap_px`` is the sharpness knob.
        Counts one SR call per window (compute-fair accounting).
        """
        from .unified_sr import unified_sr_level, count_windows
        sr = self._ensure_sr()
        dino = gc.resolved_dino_weights_path() if self.use_dino_sr else ""
        out = unified_sr_level(
            sr, coarse_pil, self.sr_device,
            target_zoom=target_zoom, steps=self.steps, cfg=self.cfg,
            t_start=t_start, noise_scale=getattr(sr, "noise_scale", 1.0),
            dino_weights=dino, seeds=seeds, overlap_px=overlap_px,
            init_pil=init_pil, guidance_scale=guidance_scale,
            guidance_t_min=guidance_t_min, guidance_mode=guidance_mode,
            sync_scale=sync_scale,
        )
        self.sr_calls += count_windows(coarse_pil.size[0], overlap_px)
        return out

    @torch.no_grad()
    def op_fill(self, canvas: Image.Image, mask: Image.Image, tile_zoom: Optional[int]) -> Image.Image:
        """Outpaint: ``canvas`` known pixels + ``mask`` (white=hole) → filled tile.

        Known pixels are frozen by the model; only the hole is generated.
        Masks should be quadrant-aligned to stay in-distribution (IDEAS.md §0.2).
        """
        op = self._ensure_op()
        out = gc.run_outpainting(op, canvas, mask, self.op_device, tile_zoom=tile_zoom)
        self.op_calls += 1
        return out

    @torch.no_grad()
    def op_fill_batch(self, items, tile_zoom: Optional[int], max_batch: int = 16):
        """Batched ``op_fill``: fill many (canvas, mask) tiles per sampler pass.

        Splits ``items`` into ``max_batch`` chunks (bounds GPU memory); returns the
        filled 256² PILs in order. Caller must only batch independent fills (no
        shared output pixels) so the result equals filling sequentially.
        """
        if not items:
            return []
        op = self._ensure_op()
        outs = []
        for i in range(0, len(items), max(1, max_batch)):
            chunk = items[i:i + max(1, max_batch)]
            outs.extend(gc.run_outpainting_batch(
                op, [c for c, _ in chunk], [m for _, m in chunk],
                self.op_device, tile_zoom=tile_zoom))
        self.op_calls += len(items)
        return outs

    def call_counts(self) -> Dict[str, int]:
        return {"sr_calls": self.sr_calls, "op_calls": self.op_calls}
