"""Shared eval helpers: ckpt resolve + Google Drive auto-pull, model loaders,
metrics, results I/O.

Metrics are library-backed wherever a clean library implementation exists:

- ``psnr``           ``torchmetrics.functional.peak_signal_noise_ratio``
- ``ssim``           ``torchmetrics.image.StructuralSimilarityIndexMeasure`` (11×11 window)
- ``lpips``          ``torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity`` (SqueezeNet)
- ``dists``          ``DISTS_pytorch.DISTS``
- ``boundary_diff``  ``scipy.ndimage.binary_erosion`` + ``torch L1`` (task-specific)

OP hole-awareness: pair-comparison metrics (SSIM / LPIPS / DISTS) score per
region returned by ``_hole_scoring_regions`` — quadrant decomposition if
the mask aligns to the 4 quadrants, else a single hole bounding box —
and average across regions. PSNR uses per-pixel mask gather.
``boundary_diff`` uses a 4-pixel ring inside the hole edge.

See ``docs/METRICS.md`` for the full per-metric breakdown.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ── Paths / Google Drive auto-pull ───────────────────────────────────────────

_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SRC       = os.path.abspath(os.path.join(_REPO_ROOT, "src"))

# Same convention as demos: under <repo>/logs/ckpts/ unless overridden.
_DEFAULT_CKPT_BASE = os.path.join(_REPO_ROOT, "logs", "ckpts")
_CKPT_BASE = (os.environ.get("GENESIS_CKPT_BASE") or "").strip() or _DEFAULT_CKPT_BASE

# Public release folder on Google Drive — main ckpts + DINOv3 + shared/.
# "Anyone with the link can view", so any authenticated Google account
# with a configured rclone `gdrive:` remote can pull from it.
GDRIVE_REMOTE = (os.environ.get("GENESIS_GDRIVE_REMOTE") or "gdrive").strip() or "gdrive"
GDRIVE_FOLDER_ID = (
    (os.environ.get("GENESIS_GDRIVE_FOLDER_ID") or "").strip()
    or "1OKe2YRJFkwwFD3GWXVJMEr6qKGJz0qyA"
)
GDRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"


def _gdrive_pull_if_missing(local_path: str, gdrive_path: str, label: str = "checkpoint") -> str:
    """rclone copy `gdrive_path` → `local_path` if the local file is missing.

    `gdrive_path` is of the form ``gdrive:<rel>``; the remote is anchored at
    the release folder via ``--drive-root-folder-id``.
    """
    if os.path.isfile(local_path):
        return local_path
    if not shutil.which("rclone"):
        raise FileNotFoundError(
            f"{label} missing locally and `rclone` not on PATH:\n"
            f"  expected: {local_path}\n"
            f"  source:   {gdrive_path}\n"
            f"  release:  {GDRIVE_FOLDER_URL}\n"
            f"Install rclone (https://rclone.org/) and configure a Google Drive remote named `{GDRIVE_REMOTE}:`."
        )
    dst_dir = os.path.dirname(local_path) or "."
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[eval] {label} not found locally — pulling from {gdrive_path}", flush=True)
    rc = subprocess.run(
        [
            "rclone", "copy", gdrive_path, dst_dir,
            "--drive-root-folder-id", GDRIVE_FOLDER_ID,
            "--progress",
        ],
        check=False,
    )
    if rc.returncode != 0 or not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"{label} download failed (rclone exit {rc.returncode}).\n"
            f"  expected: {local_path}\n"
            f"  source:   {gdrive_path}\n"
            f"  release:  {GDRIVE_FOLDER_URL}\n"
            f"Verify the `{GDRIVE_REMOTE}:` rclone remote and that the source exists."
        )
    return local_path


def _gdrive_path_for(local_path: str) -> str:
    """Map a local path under `_CKPT_BASE` → its Google Drive source.
    Falls back to basename when the path isn't under `_CKPT_BASE`."""
    try:
        rel = os.path.relpath(local_path, _CKPT_BASE)
        if rel.startswith(".."):
            raise ValueError
        return f"{GDRIVE_REMOTE}:{rel.replace(os.sep, '/')}"
    except ValueError:
        return f"{GDRIVE_REMOTE}:{os.path.basename(local_path)}"


# ── Default meta-file paths + auto-pull helper ───────────────────────────────
# The Google Drive release folder publishes the eval metadata under shared/.
# `resolve_meta_path` lets a CLI argument default to the gdrive layout while
# still respecting an explicit override (e.g. a lab-local path).

DEFAULT_META_FILE  = os.path.join(_CKPT_BASE, "shared", "git10m_quad_meta.json")
DEFAULT_HINDEX_PKL = os.path.join(_CKPT_BASE, "shared", "hierarchy_index_quad.pkl")


def resolve_meta_path(local_path: str, gdrive_rel: str, label: str) -> str:
    """Return `local_path`, auto-pulling from `gdrive:<gdrive_rel>` if missing.

    Unlike `_gdrive_path_for`, the gdrive-side relative path is supplied
    explicitly because meta files may live outside `_CKPT_BASE` (e.g. a
    lab-shared directory). Existing local files are returned as-is —
    download only happens when the file isn't already cached.
    """
    if os.path.isfile(local_path):
        return local_path
    return _gdrive_pull_if_missing(
        local_path,
        f"{GDRIVE_REMOTE}:{gdrive_rel}",
        label=label,
    )


# ── Experiment registry / ckpt resolution ────────────────────────────────────

_SR_ABL_FNAME = {
    3: "sr-full-tile-stage3-step0500000.ckpt",
}
_SR_MAIN_FNAME = {
    3: "sr-full-tile-stage3-step0800000.ckpt",
}
_OP_MAIN_FNAME = {
    3: "op-new-stage3-step0800000.ckpt",
}

_MAIN_EXP_TO_ARCH_DIR = {
    "main_b16": "main_B16",
    "main_h16": "main_H16",
}


def _ckpt_path_for_exp(task: str, exp: str, stage: int) -> str:
    """Build the local ckpt path for `(task, exp, stage)`.
    `exp` is `abl_<cell>` (SR only) or `main_b16` / `main_h16`."""
    task_dir = "superresolution" if task == "sr" else "outpainting"

    if exp.startswith("abl_"):
        if task != "sr":
            raise ValueError(
                f"OP is ablation-free (no-DINO + no-time-weight is the locked "
                f"main config); --exp {exp!r} is only valid for task=sr."
            )
        if stage not in _SR_ABL_FNAME:
            raise ValueError(f"SR ablations only ship at stage 3 (got --stage {stage}).")
        fname = _SR_ABL_FNAME[stage]
        # Ablation ckpts are published on Google Drive under `ablations/<exp>/`
        # (SR-only, so no per-task subfolder) and auto-pull like the main ckpts.
        return os.path.join(_CKPT_BASE, "ablations", exp, fname)

    if exp in _MAIN_EXP_TO_ARCH_DIR:
        arch_dir = _MAIN_EXP_TO_ARCH_DIR[exp]
        if task == "sr":
            if stage not in _SR_MAIN_FNAME:
                raise ValueError(f"SR main: only stage 3 (800k) is published (got --stage {stage}).")
            fname = _SR_MAIN_FNAME[stage]
        else:
            if stage not in _OP_MAIN_FNAME:
                raise ValueError(f"OP main: only stage 3 (800k) is published (got --stage {stage}).")
            fname = _OP_MAIN_FNAME[stage]
        return os.path.join(_CKPT_BASE, "main", task_dir, arch_dir, fname)

    raise ValueError(
        f"Unrecognised --exp {exp!r}. Expected: "
        f"abl_{{nodino_nolpips,dino_nolpips,nodino_lpips,dino_lpips}} (SR only) / "
        f"main_b16 / main_h16."
    )


def resolve_ckpt(task: str, ckpt: Optional[str], exp: Optional[str], stage: Optional[int]) -> str:
    """Return a local ckpt path; auto-pulls from Google Drive if missing.

    Both main (`--exp main_b16 / main_h16`) and SR ablation (`--exp abl_*`)
    checkpoints are published in the Google Drive release and auto-pull the
    same way.
    """
    if ckpt:
        if os.path.isfile(ckpt):
            return os.path.abspath(ckpt)
        if ckpt.startswith(_CKPT_BASE):
            return _gdrive_pull_if_missing(ckpt, _gdrive_path_for(ckpt), label=f"{task} checkpoint")
        raise FileNotFoundError(f"--ckpt does not exist: {ckpt}")
    if exp is None or stage is None:
        raise ValueError("Must pass either --ckpt or both --exp and --stage.")
    local = _ckpt_path_for_exp(task, exp, stage)
    return _gdrive_pull_if_missing(local, _gdrive_path_for(local), label=f"{task} checkpoint")


# DINOv3 weights — same auto-pull as demos. Google Drive source is
# `gdrive:<fname>` (anchored at the release folder id).
_DINO_FNAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DEFAULT_DINO_WEIGHTS = os.path.join(_CKPT_BASE, _DINO_FNAME)


def resolve_dino_weights(arg: Optional[str]) -> str:
    """Resolve DINOv3 weights path; auto-pulls from Google Drive if not local."""
    p = (arg or "").strip()
    if not p:
        p = (os.environ.get("GENESIS_DINO_WEIGHTS") or "").strip()
    if p:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"DINO weights path not found: {p}")
        return p
    return _gdrive_pull_if_missing(
        DEFAULT_DINO_WEIGHTS,
        f"{GDRIVE_REMOTE}:{_DINO_FNAME}",
        label="DINOv3 weights",
    )


# ── Model loaders ────────────────────────────────────────────────────────────

def _build_sr_args(model_name: str, dino_weights: str, *, cfg: float, num_sampling_steps: int,
                   bicubic_init: bool, bicubic_init_t_start: float, output_size: int = 512) -> Namespace:
    return Namespace(
        model=model_name,
        output_size=output_size,
        use_dino=bool(dino_weights),
        dino_weights=dino_weights or "",
        P_mean=-0.8, P_std=0.8, noise_scale=1.0, t_eps=1e-5,
        num_sampling_steps=int(num_sampling_steps),
        cfg=float(cfg),
        bicubic_init=bool(bicubic_init),
        bicubic_init_t_start=float(bicubic_init_t_start),
        lpips_weight=0.0,
        ema_decay1=0.9999, ema_decay2=0.9996,
    )


def _build_op_args(model_name: str, dino_weights: str, *, cfg: float, num_sampling_steps: int,
                   img_size: int = 256) -> Namespace:
    return Namespace(
        model=model_name,
        img_size=img_size,
        dino_weights=dino_weights or "",
        P_mean=-0.8, P_std=0.8, noise_scale=1.0, t_eps=1e-5,
        num_sampling_steps=int(num_sampling_steps),
        cfg=float(cfg),
        enable_time_weight=True,
        ema_decay1=0.9999, ema_decay2=0.9996,
    )


def _strip_lightning_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            k = k[len("model._orig_mod."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def _load_with_ema(module: torch.nn.Module, ck: dict, label: str) -> str:
    """Load state_dict + copy EMA params (ema_params1) into live params if present."""
    online_only = (os.environ.get("GENESIS_CKPT_WEIGHTS") or "").strip().lower() == "online"
    sd = _strip_lightning_prefix(ck["state_dict"])
    module.load_state_dict(sd, strict=False)
    if online_only:
        return "online"
    ema = ck.get("ema_params1")
    if ema is None:
        return "online"
    trainable = [p for p in module.parameters() if p.requires_grad]
    if len(ema) != len(trainable):
        print(f"[eval] {label}: EMA length mismatch ({len(ema)} ema vs {len(trainable)} "
              f"trainable params) — using online weights.", flush=True)
        return "online"
    with torch.no_grad():
        for p, t in zip(trainable, ema):
            te = t.detach().to(dtype=p.dtype, device=p.device)
            if te.shape != p.shape:
                print(f"[eval] {label}: EMA shape mismatch — using online weights.", flush=True)
                return "online"
            p.data.copy_(te)
    return "ema"


def load_sr_denoiser(ckpt_path: str, model_name: str, dino_weights: str, device: torch.device,
                     *, cfg: float, num_sampling_steps: int,
                     bicubic_init: bool = False, bicubic_init_t_start: float = 0.5):
    """Build DenoiserSR from src/superresolution/ and load EMA weights from `ckpt_path`."""
    sys.path.insert(0, os.path.join(_SRC, "superresolution"))
    sys.modules.pop("engine", None)
    from engine import DenoiserSR  # noqa: E402

    args = _build_sr_args(model_name, dino_weights, cfg=cfg, num_sampling_steps=num_sampling_steps,
                          bicubic_init=bicubic_init, bicubic_init_t_start=bicubic_init_t_start)
    m = DenoiserSR(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = _load_with_ema(m, ck, label="SR")
    print(f"[eval] SR: loaded {src} weights from {ckpt_path}", flush=True)
    m.eval()
    m.to(device)
    return m, args


def load_op_denoiser(ckpt_path: str, model_name: str, dino_weights: str, device: torch.device,
                     *, cfg: float, num_sampling_steps: int):
    """Build DenoiserOutpainting from src/outpainting/ and load EMA weights from `ckpt_path`."""
    sys.path.insert(0, os.path.join(_SRC, "outpainting"))
    sys.modules.pop("engine", None)
    from engine import DenoiserOutpainting  # noqa: E402

    args = _build_op_args(model_name, dino_weights, cfg=cfg, num_sampling_steps=num_sampling_steps)
    m = DenoiserOutpainting(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = _load_with_ema(m, ck, label="OP")
    print(f"[eval] OP: loaded {src} weights from {ckpt_path}", flush=True)
    m.eval()
    m.to(device)
    return m, args


# ── Metrics ──────────────────────────────────────────────────────────────────
#
# Inputs at the call site are PIL.Image (uint8 RGB). `mask` is PIL "L" with
# >127 = hole. The mask drives hole-aware scoring (per-region averaging for
# windowed pair-comparison metrics; pixel-gather for PSNR; ring
# extraction for boundary_diff).
#
# See `docs/METRICS.md` for the full per-metric breakdown.

_DISTS_FN: Optional[torch.nn.Module] = None
_SSIM_FN:  Optional[torch.nn.Module] = None
_LPIPS_FN: Optional[torch.nn.Module] = None


def _get_ssim(device: torch.device) -> torch.nn.Module:
    """Lazy-init torchmetrics SSIM, data_range=1.0, default 11×11 window."""
    global _SSIM_FN
    if _SSIM_FN is None:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        _SSIM_FN = StructuralSimilarityIndexMeasure(data_range=1.0)
    return _SSIM_FN.to(device)


def _get_lpips(device: torch.device) -> torch.nn.Module:
    """Lazy-init torchmetrics LPIPS with the SqueezeNet backbone. Inputs in [-1, 1]."""
    global _LPIPS_FN
    if _LPIPS_FN is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        _LPIPS_FN = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")
    return _LPIPS_FN.to(device)


def _hole_scoring_regions(
    mb: Optional[np.ndarray], h: int, w: int
) -> List[Tuple[int, int, int, int]]:
    """Decompose `mb` into hole-only rectangular scoring regions.

    Returns a list of ``(y0, y1, x0, x1)`` tuples.

    - ``mb`` is ``None`` or all-zero → full image (no mask).
    - ``mb`` aligns to the 4 quadrant cells (each quadrant is uniformly hole
      or uniformly known) → one rectangle per *hole* quadrant. Handles every
      quadrant-known configuration (0/1/2/3 known).
    - Otherwise (rectangular non-quadrant masks) → single bbox of the hole.
    """
    if mb is None or not mb.any():
        return [(0, h, 0, w)]

    half_h, half_w = h // 2, w // 2
    quad_boxes = [
        (0, half_h, 0, half_w),         # UL
        (0, half_h, half_w, w),         # UR
        (half_h, h, 0, half_w),         # LL
        (half_h, h, half_w, w),         # LR
    ]

    def _quad_status(box):
        y0, y1, x0, x1 = box
        sub = mb[y0:y1, x0:x1]
        if sub.all():
            return "hole"
        if not sub.any():
            return "known"
        return "mixed"

    statuses = [_quad_status(b) for b in quad_boxes]
    if all(s != "mixed" for s in statuses):
        hole_quads = [b for b, s in zip(quad_boxes, statuses) if s == "hole"]
        return hole_quads if hole_quads else [(0, h, 0, w)]

    # Defensive fallback: single hole bbox.
    ys, xs = np.where(mb)
    return [(int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)]


def _ring_inside_hole(mb: np.ndarray, iterations: int = 4) -> np.ndarray:
    """4-pixel ring just inside the hole edge (returns boolean mask)."""
    from scipy.ndimage import binary_erosion
    return mb & ~binary_erosion(mb, iterations=iterations)


def compute_metrics(
    pred: Image.Image,
    gt: Image.Image,
    *,
    mask: Optional[Image.Image] = None,
    base: Optional[Image.Image] = None,
    metrics: Sequence[str],
    lpips_max_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Per-sample metrics.

    Recognised metric names: ``psnr``, ``ssim``, ``lpips``, ``dists``,
    ``boundary_diff``.

    ``mask`` (PIL "L", >127 = hole) drives mask-aware computation:
      - PSNR uses per-pixel mask gathering.
      - SSIM / LPIPS / DISTS use per-region averaging via
        ``_hole_scoring_regions`` (quadrant decomposition or single bbox).
      - boundary_diff: 4-px ring inside the hole edge, abs-diff vs ``base``.

    ``lpips_max_size`` caps the longest side of the LPIPS / DISTS input
    crop (bilinear resize). Set ``0`` to disable.
    """
    global _DISTS_FN
    out: Dict[str, float] = {}
    metrics = [m.lower() for m in metrics]

    # Make sizes match.
    if gt.size != pred.size:
        gt = gt.resize(pred.size, Image.Resampling.BILINEAR)
    if mask is not None and mask.size != pred.size:
        mask = mask.resize(pred.size, Image.Resampling.NEAREST)

    w, h = pred.size  # PIL is (W, H); numpy/torch are (H, W)

    # Float arrays in [0, 1] for use across metrics.
    pa = np.asarray(pred.convert("RGB"), dtype=np.float32) / 255.0  # (H, W, 3)
    ga = np.asarray(gt.convert("RGB"),   dtype=np.float32) / 255.0  # (H, W, 3)
    mb: Optional[np.ndarray] = None
    if mask is not None:
        mb = np.asarray(mask.convert("L")) > 127  # (H, W) bool

    pa_t_full = torch.from_numpy(pa).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    ga_t_full = torch.from_numpy(ga).permute(2, 0, 1).unsqueeze(0).to(device)

    regions = _hole_scoring_regions(mb, h, w)

    def _crop(t: torch.Tensor, y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
        return t[..., y0:y1, x0:x1]

    def _maybe_downsample(t: torch.Tensor) -> torch.Tensor:
        """Cap longest side to `lpips_max_size`. No-op if size already <= cap."""
        if not lpips_max_size:
            return t
        H, W = t.shape[-2], t.shape[-1]
        long_side = max(H, W)
        if long_side <= lpips_max_size:
            return t
        scale = lpips_max_size / long_side
        return F.interpolate(t, size=(int(H * scale), int(W * scale)),
                             mode="bilinear", align_corners=False)

    # ── PSNR ────────────────────────────────────────────────────────────────
    if "psnr" in metrics:
        from torchmetrics.functional import peak_signal_noise_ratio
        if mb is not None:
            sel = mb[..., None].repeat(3, axis=2)
            if sel.any():
                pa_hole = torch.from_numpy(pa[sel]).to(device).reshape(-1)
                ga_hole = torch.from_numpy(ga[sel]).to(device).reshape(-1)
                out["psnr"] = float(peak_signal_noise_ratio(
                    pa_hole, ga_hole, data_range=1.0,
                ).item())
            else:
                out["psnr"] = float("nan")
        else:
            out["psnr"] = float(peak_signal_noise_ratio(
                pa_t_full, ga_t_full, data_range=1.0,
            ).item())

    # ── SSIM (windowed, per-region) ─────────────────────────────────────────
    if "ssim" in metrics:
        ssim_m = _get_ssim(device)
        vals = []
        for (y0, y1, x0, x1) in regions:
            pa_c = _crop(pa_t_full, y0, y1, x0, x1)
            ga_c = _crop(ga_t_full, y0, y1, x0, x1)
            # torchmetrics SSIM uses an 11×11 window — skip tiny regions.
            if pa_c.shape[-1] < 11 or pa_c.shape[-2] < 11:
                continue
            with torch.no_grad():
                vals.append(float(ssim_m(pa_c, ga_c).item()))
            ssim_m.reset()
        out["ssim"] = float(np.mean(vals)) if vals else float("nan")

    # ── LPIPS (per-region, inputs in [-1, 1]) ───────────────────────────────
    if "lpips" in metrics:
        lpips_m = _get_lpips(device)
        vals = []
        for (y0, y1, x0, x1) in regions:
            pa_c = _maybe_downsample(_crop(pa_t_full, y0, y1, x0, x1))
            ga_c = _maybe_downsample(_crop(ga_t_full, y0, y1, x0, x1))
            with torch.no_grad():
                vals.append(float(lpips_m(pa_c * 2 - 1, ga_c * 2 - 1).item()))
            lpips_m.reset()
        out["lpips"] = float(np.mean(vals)) if vals else float("nan")

    # ── DISTS (per-region, inputs in [0, 1]) ────────────────────────────────
    if "dists" in metrics:
        try:
            if _DISTS_FN is None:
                from DISTS_pytorch import DISTS
                _DISTS_FN = DISTS().to(device)
            vals = []
            for (y0, y1, x0, x1) in regions:
                pa_c = _maybe_downsample(_crop(pa_t_full, y0, y1, x0, x1))
                ga_c = _maybe_downsample(_crop(ga_t_full, y0, y1, x0, x1))
                with torch.no_grad():
                    vals.append(float(_DISTS_FN(pa_c, ga_c).item()))
            out["dists"] = float(np.mean(vals)) if vals else float("nan")
        except Exception as e:
            print(f"[eval] DISTS unavailable: {e}", flush=True)
            out["dists"] = float("nan")

    # ── Boundary diff (4-px ring just inside the hole, abs-diff vs base) ───
    if "boundary_diff" in metrics:
        if mb is None or base is None:
            out["boundary_diff"] = float("nan")
        else:
            ring = _ring_inside_hole(mb, iterations=4)
            if ring.any():
                ba = np.asarray(
                    base.convert("RGB").resize(pred.size, Image.Resampling.BILINEAR),
                    dtype=np.float32,
                ) / 255.0
                sel = ring[..., None].repeat(3, axis=2)
                pa_ring = torch.from_numpy(pa[sel]).to(device)
                ba_ring = torch.from_numpy(ba[sel]).to(device)
                # L1 in [0,1] × 255 to keep the value range of the old metric.
                out["boundary_diff"] = float(
                    F.l1_loss(pa_ring, ba_ring).item() * 255.0
                )
            else:
                out["boundary_diff"] = float("nan")

    return out


# ── Image helpers ────────────────────────────────────────────────────────────

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """t: (1, 3, H, W) or (3, H, W) in [-1, 1] → PIL RGB."""
    if t.dim() == 4:
        t = t[0]
    x = t.detach().float().cpu().permute(1, 2, 0).numpy()
    x = (x * 0.5 + 0.5).clip(0, 1) * 255.0
    return Image.fromarray(x.astype(np.uint8), "RGB")


def pil_to_model_tensor(pil: Image.Image, size: int = 256) -> torch.Tensor:
    """PIL → (1, 3, size, size) float in [-1, 1]."""
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return (t * 2.0 - 1.0).unsqueeze(0)


# QUAD_ORDER mirrors src/superresolution/dataloader.py: UL, UR, LL, LR.
QUAD_ORDER: Tuple[Tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1), (1, 1))


def stitch_quad_mosaic(children: Sequence[torch.Tensor], tile_size: int) -> torch.Tensor:
    """[UL, UR, LL, LR] tensors (3, S, S) → (3, 2S, 2S) stitched mosaic."""
    s = tile_size
    out = torch.zeros(3, 2 * s, 2 * s, dtype=children[0].dtype, device=children[0].device)
    placements = [(0, 0), (s, 0), (0, s), (s, s)]   # matches QUAD_ORDER
    for child, (px, py) in zip(children, placements):
        out[:, py:py + s, px:px + s] = child
    return out


# ── Streaming results writer ─────────────────────────────────────────────────

@dataclass
class SampleWriter:
    """One CSV row per sample, written incrementally so partial runs survive crashes."""
    path: str
    columns: List[str] = field(default_factory=list)
    _f: Any = None
    _writer: Any = None
    _rows_written: int = 0

    def open(self, columns: Sequence[str]) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.columns = list(columns)
        self._f = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=self.columns, extrasaction="ignore")
        self._writer.writeheader()
        self._f.flush()

    def append(self, row: Dict[str, Any]) -> None:
        for col in self.columns:
            row.setdefault(col, "")
        self._writer.writerow(row)
        self._rows_written += 1
        if self._rows_written % 32 == 0:
            self._f.flush()

    def close(self) -> int:
        if self._f is not None:
            self._f.flush()
            self._f.close()
        return self._rows_written


def dump_run_meta(
    out_dir: str,
    *,
    task: str,
    exp: Optional[str],
    stage: Optional[int],
    ckpt: str,
    split: str,
    extra: Dict[str, Any],
    config: Dict[str, Any],
    n_samples: int,
    n_errors: int,
    elapsed_seconds: float,
    device: str,
    metrics: Sequence[str],
) -> None:
    """Write run.json next to samples.csv with the run-level metadata."""
    payload = {
        "task": task,
        "exp": exp,
        "stage": stage,
        "ckpt": ckpt,
        "split": split,
        "extra": extra,
        "config": config,
        "metrics": list(metrics),
        "n_samples": n_samples,
        "n_errors": n_errors,
        "elapsed_seconds": elapsed_seconds,
        "device": device,
        "git_hash": _git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(payload, f, indent=2)


def _git_hash() -> Optional[str]:
    try:
        rc = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=_REPO_ROOT, capture_output=True, text=True, check=False)
        h = rc.stdout.strip()
        return h or None
    except Exception:
        return None


# ── Identifier extraction ────────────────────────────────────────────────────

def meta_columns(index_to_meta: Dict[str, Dict[str, Any]], idx: int) -> Dict[str, Any]:
    m = index_to_meta.get(str(idx)) or index_to_meta.get(idx) or {}
    return {
        "hf_idx":    int(idx),
        "location":  m.get("location") or "",
        "satellite": m.get("satellite") or "",
        "split":     m.get("split") or "",
        "lat":       float(m["lat"])  if "lat"  in m else "",
        "lon":       float(m["lon"])  if "lon"  in m else "",
        "gsd":       float(m["gsd"])  if "gsd"  in m else "",
        "zoom":      int(m["zoom"])   if "zoom" in m else "",
    }


# ── Output-dir auto-tagging ──────────────────────────────────────────────────

def auto_tag_dir(base: str, parts: Sequence[Optional[str]]) -> str:
    tag = "_".join(str(p) for p in parts if p not in (None, "", False))
    out_dir = os.path.join(base, tag) if tag else base
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def default_results_root() -> str:
    """Default eval/inference output root: ``$GENESIS_RESULTS_ROOT`` when set,
    else the gitignored ``src/eval/results/`` (portable for a public checkout)."""
    return (os.environ.get("GENESIS_RESULTS_ROOT") or "").strip() or os.path.join(_THIS_DIR, "results")
