"""Unified single-canvas MultiDiffusion super-resolution.

One global noisy canvas and one denoising trajectory per level: overlapping
windows read crops, the model predicts a per-window velocity, and predictions are
fused on the shared canvas → seam-free by construction. To stay sharp, fusion
uses a *partition of unity* that is flat (=1) over each tile interior and ramps
only within a narrow overlap band, so interiors get a single prediction and only
seams are blended. Fidelity knobs: ``overlap_px`` (band width) and ``t_start``
(SDEdit warm-start).

Engine convention (pixel-space flow matching, no VAE):
    zt = t * x1 + (1 - t) * x0,   t: t_start -> 1,  x1 = clean, x0 = noise
    v  = (net(z, t) - z) / (1 - t);   z <- z + dt * v
Seeds are re-anchored on the canvas every step at the correct noise level, so any
seed layout is honored exactly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

TILE_PX = 256
WIN = TILE_PX           # parent-window size on the coarse mosaic (model input)
OUT = 2 * TILE_PX       # SR output per window (model upscales 2x)


def _pil_to_tensor(pil: Image.Image, device) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (t * 2.0 - 1.0).to(device)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    x = ((x * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(x)


def window_origins(coarse_side: int, stride: int) -> List[int]:
    """Top-left coords for WIN-sized windows tiling [0, coarse_side] at ``stride``;
    always includes the flush-right window so the far edge is covered."""
    if coarse_side <= WIN:
        return [0]
    xs = list(range(0, coarse_side - WIN + 1, stride))
    if xs[-1] != coarse_side - WIN:
        xs.append(coarse_side - WIN)
    return xs


def _plateau_taper(size: int, ramp: int, device, floor: float = 1e-3) -> torch.Tensor:
    """1D->2D partition weight: ==1 in the interior, linear ramp to ~0 over the
    outer ``ramp`` px on each side. Flat interiors keep single-window ownership
    (sharp); only the ramp band participates in cross-window blending (seam-free).
    """
    if ramp <= 0:
        w1 = torch.ones(size, device=device)
    else:
        idx = torch.arange(size, device=device, dtype=torch.float32)
        left = torch.clamp((idx + 0.5) / ramp, 0.0, 1.0)
        right = torch.clamp((size - 0.5 - idx) / ramp, 0.0, 1.0)
        w1 = torch.minimum(left, right).clamp_min(floor)
    w2 = torch.outer(w1, w1)
    return w2.view(1, 1, size, size)


def count_windows(coarse_side: int, overlap_px: int) -> int:
    stride = max(1, WIN - int(overlap_px))
    return len(window_origins(coarse_side, stride)) ** 2


def _gaussian_blur(x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    ax = torch.arange(k, device=x.device, dtype=torch.float32) - (k - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    g2 = (g[:, None] * g[None, :]).view(1, 1, k, k)
    g2 = g2.expand(x.shape[1], 1, k, k)
    pad = k // 2
    return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), g2, groups=x.shape[1])


def _hf_guidance_nudge(
    x: torch.Tensor,
    bic: torch.Tensor,
    scale: float,
    t: float,
    t_min: float,
) -> torch.Tensor:
    """Late-step nudge: add high-frequency component relative to bicubic init (detail guidance)."""
    if scale <= 0 or t < t_min:
        return x
    with torch.enable_grad():
        xg = x.detach().float().requires_grad_(True)
        hp = xg - _gaussian_blur(xg)
        hp_b = bic - _gaussian_blur(bic)
        # Encourage ||highpass(x)|| >= ||highpass(bic)|| on average (sharper than warm-start).
        loss = -(hp.pow(2).mean() - hp_b.pow(2).mean()).clamp(max=0)
        if loss.abs() < 1e-8:
            return x
        grad, = torch.autograd.grad(loss, xg, retain_graph=False)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        gnorm = grad.norm().clamp_min(1e-8)
        grad = grad * (0.05 / gnorm).clamp(max=1.0)  # cap step magnitude
        w = scale * (t - t_min) / max(1e-5, 1.0 - t_min)
        return (x - w * grad).detach()


_LAPLACIAN_K: torch.Tensor | None = None


def _laplacian_k(device) -> torch.Tensor:
    global _LAPLACIAN_K
    if _LAPLACIAN_K is None or _LAPLACIAN_K.device != device:
        k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device)
        _LAPLACIAN_K = k.view(1, 1, 3, 3).expand(3, 1, 3, 3)
    return _LAPLACIAN_K


def _apply_guidance_nudge(
    x: torch.Tensor,
    bic: torch.Tensor,
    scale: float,
    t: float,
    t_min: float,
    mode: str,
) -> torch.Tensor:
    """Late-step loss-guided nudges. mode: off | hf | lap | highpass."""
    mode = (mode or "off").lower()
    if scale <= 0 or t < t_min or mode in ("off", "none", "0"):
        return x
    if mode == "hf":
        return _hf_guidance_nudge(x, bic, scale, t, t_min)
    with torch.enable_grad():
        xg = x.detach().float().requires_grad_(True)
        if mode == "lap":
            lap = F.conv2d(xg, _laplacian_k(x.device), padding=1, groups=3)
            loss = -lap.pow(2).mean()
        elif mode == "highpass":
            hp = xg - _gaussian_blur(xg)
            loss = -hp.abs().mean()
        else:
            return x
        if loss.abs() < 1e-8:
            return x
        grad, = torch.autograd.grad(loss, xg, retain_graph=False)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        gnorm = grad.norm().clamp_min(1e-8)
        grad = grad * (0.05 / gnorm).clamp(max=1.0)
        w = scale * (t - t_min) / max(1e-5, 1.0 - t_min)
        return (x - w * grad).detach()


def _sync_overlap_velocity(
    v_acc: torch.Tensor,
    w_acc: torch.Tensor,
    win_pos: List[Tuple[int, int]],
    taper: torch.Tensor,
    v_windows: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """SyncDiffusion-lite: in overlap bands (w_acc>1), pull fused v toward unweighted mean of window v."""
    if scale <= 0:
        return v_acc / w_acc.clamp_min(1e-6)
    fused = v_acc / w_acc.clamp_min(1e-6)
    overlap = (w_acc > 1.02).expand_as(v_acc)
    if not overlap.any():
        return fused
    v_mean = torch.zeros_like(v_acc)
    n_mean = torch.zeros_like(w_acc)
    b = v_windows.shape[0]
    for j in range(b):
        ox, oy = win_pos[j]
        cy, cx = 2 * oy, 2 * ox
        patch = v_windows[j : j + 1]
        v_mean[:, :, cy : cy + OUT, cx : cx + OUT] += patch
        n_mean[:, :, cy : cy + OUT, cx : cx + OUT] += taper
    v_avg = v_mean / n_mean.clamp_min(1e-6)
    return torch.where(overlap, (1.0 - scale) * fused + scale * v_avg, fused)


@torch.no_grad()
def unified_sr_level(
    denoiser,
    coarse_pil: Image.Image,
    device: torch.device,
    *,
    target_zoom: Optional[int],
    steps: int,
    cfg: float,
    t_start: float,
    noise_scale: float,
    dino_weights: str,
    seeds: Optional[Dict[Tuple[int, int], Image.Image]] = None,
    overlap_px: int = 32,
    batch: int = 16,   # SR windows per forward; half the SR training batch (32),
                       # sized to co-exist with the OP H/16 model in memory
    init_pil: Optional[Image.Image] = None,
    guidance_scale: float = 0.0,
    guidance_t_min: float = 0.75,
    guidance_mode: str = "hf",
    sync_scale: float = 0.0,
) -> Image.Image:
    """Joint 2× SR of a G*256 level mosaic via single-canvas MultiDiffusion → 2G*256.

    target_zoom: child zoom for GSD conditioning (None = unknown). seeds: {(col,row)
    on the 2G child grid: 256² PIL} to anchor. overlap_px: sharpness knob (smaller =
    sharper, narrower seam; stride = 256-overlap). init_pil: warm-start for refinement.
    guidance_scale/_t_min/_mode: optional late-step detail guidance (hf|lap|highpass).
    sync_scale: overlap-band velocity averaging (0=off).
    """
    net = denoiser.net
    use_dino = bool(getattr(net, "use_dino", False))
    coarse = _pil_to_tensor(coarse_pil, device)
    c = coarse.shape[-1]
    out_side = 2 * c
    stride = max(1, WIN - int(overlap_px))

    origins = window_origins(c, stride)
    win_pos: List[Tuple[int, int]] = [(ox, oy) for oy in origins for ox in origins]
    n = len(win_pos)

    parent_crops = torch.cat(
        [coarse[:, :, oy:oy + WIN, ox:ox + WIN] for (ox, oy) in win_pos], dim=0
    )

    dino_feats_all = None
    if use_dino:
        from model_sr import get_dino_embeddings
        feats = [get_dino_embeddings(parent_crops[i:i + batch], weights_path=dino_weights)
                 for i in range(0, n, batch)]
        dino_feats_all = torch.cat(feats, dim=0)

    zoom_vec = torch.full((n,), float(target_zoom), device=device) if target_zoom is not None else None

    # Partition of unity: ramp band = overlap on the child canvas (2 * overlap_px).
    taper = _plateau_taper(OUT, 2 * int(overlap_px), device)

    # Single global canvas, SDEdit warm-start from bicubic (or refinement init).
    bic = F.interpolate(coarse, size=(out_side, out_side), mode="bicubic", align_corners=False)
    if init_pil is not None:
        init_t = _pil_to_tensor(init_pil, device)
        if init_t.shape[-1] != out_side:
            init_t = F.interpolate(init_t, size=(out_side, out_side), mode="bicubic", align_corners=False)
        x0 = init_t
    else:
        x0 = bic
    x = t_start * x0 + (1.0 - t_start) * noise_scale * torch.randn(
        1, 3, out_side, out_side, device=device
    )

    # Seed anchoring on the shared canvas (correct noise level every step).
    seed_specs: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
    if seeds:
        for (col, row), spil in seeds.items():
            y0, x0 = row * TILE_PX, col * TILE_PX
            x1c = _pil_to_tensor(spil, device)
            x0n = noise_scale * torch.randn_like(x1c)
            seed_specs.append((y0, x0, x1c, x0n))
            x[:, :, y0:y0 + TILE_PX, x0:x0 + TILE_PX] = t_start * x1c + (1.0 - t_start) * x0n

    timesteps = torch.linspace(t_start, 1.0, steps + 1, device=device)
    dt = (1.0 - t_start) / steps

    for si in range(steps):
        t = timesteps[si]
        inv = 1.0 / (1.0 - t + 1e-5)
        v_acc = torch.zeros_like(x)
        w_acc = torch.zeros(1, 1, out_side, out_side, device=device)
        v_windows: List[torch.Tensor] = []

        # Accumulate every window's velocity onto the shared canvas BEFORE stepping,
        # so the global Euler update is applied exactly once per timestep regardless
        # of how many batches the windows are split across.
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            zb = torch.cat([x[:, :, 2 * oy:2 * oy + OUT, 2 * ox:2 * ox + OUT]
                            for (ox, oy) in win_pos[sl]], dim=0)
            b = zb.shape[0]
            tb = t.expand(b)
            cond = {"parent_img": parent_crops[sl]}
            if use_dino:
                cond["dino_feats"] = dino_feats_all[sl]
            if zoom_vec is not None:
                cond["zoom_level"] = zoom_vec[sl]
            x_cond = net(zb, tb, conditions=cond)
            v = (x_cond - zb) * inv
            if cfg > 1.0:
                uncond = {"dino_feats": None, "parent_img": None, "zoom_level": None}
                x_un = net(zb, tb, conditions=uncond)
                v = (x_un - zb) * inv + cfg * (v - (x_un - zb) * inv)
            for j, (ox, oy) in enumerate(win_pos[sl]):
                cy, cx = 2 * oy, 2 * ox
                v_acc[:, :, cy:cy + OUT, cx:cx + OUT] += v[j:j + 1] * taper
                w_acc[:, :, cy:cy + OUT, cx:cx + OUT] += taper
            if sync_scale > 0:
                v_windows.append(v)

        if sync_scale > 0:
            v_step = _sync_overlap_velocity(
                v_acc, w_acc, win_pos, taper, torch.cat(v_windows, dim=0), sync_scale,
            )
        else:
            v_step = v_acc / w_acc.clamp_min(1e-6)
        x = x + dt * v_step
        x = _apply_guidance_nudge(
            x, bic, guidance_scale, float(t), guidance_t_min, guidance_mode,
        )

        if seed_specs:
            tn = timesteps[si + 1]
            for (y0, x0, x1c, x0n) in seed_specs:
                x[:, :, y0:y0 + TILE_PX, x0:x0 + TILE_PX] = tn * x1c + (1.0 - tn) * x0n

    return _tensor_to_pil(x)
