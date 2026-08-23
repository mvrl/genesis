"""
Shared utilities for Genesis Gradio demos: Esri tile fetch, SR / outpainting inference.

Wires the demos to the SR / outpainting code under `src/superresolution/`
and `src/outpainting/` (full-tile 256→512 SR, mask-based outpainting).
"""
from __future__ import annotations

import io
import math
import os
import shutil
import sys
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

# Repo roots  (this file lives in demos/utils/ → demos/ is one level up)
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEMOS_DIR = os.path.dirname(_UTILS_DIR)
_REPO_ROOT = os.path.abspath(os.path.join(_DEMOS_DIR, ".."))
_SRC = os.path.abspath(os.path.join(_REPO_ROOT, "src"))

# Default ckpt root: $GENESIS_CKPT_BASE if set, otherwise <repo>/logs/ckpts —
# `logs/` is git-ignored, so anything pulled here stays out of the repo.
_DEFAULT_CKPT_BASE = os.path.join(_REPO_ROOT, "logs", "ckpts")
_CKPT_BASE = (os.environ.get("GENESIS_CKPT_BASE") or "").strip() or _DEFAULT_CKPT_BASE

# Hugging Face release repo for auto-downloaded checkpoints. The local layout
# under `_CKPT_BASE` mirrors the HF repo: a missing local file at
# `<_CKPT_BASE>/X` is fetched from `https://huggingface.co/<repo>/blob/main/X`.
# Public repo, so no authentication or extra tooling is needed.
HF_REPO_ID = (os.environ.get("GENESIS_HF_REPO") or "").strip() or "MVRL/genesis"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_ID}"


def _resolve_path(*candidates: str) -> str:
    """Return the first candidate that exists as a file; fall back to the first candidate."""
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[0] if candidates else ""


def _hf_pull_if_missing(local_path: str, hf_rel: str, label: str = "checkpoint") -> str:
    """Ensure `local_path` exists, downloading `hf_rel` from the HF repo if not.

    `hf_rel` is the repo-relative path, e.g. "main/.../foo.ckpt". Files land
    directly under `_CKPT_BASE` (which mirrors the repo layout), so no
    separate cache copy is kept. Returns local_path.
    """
    if os.path.isfile(local_path):
        return local_path
    from huggingface_hub import hf_hub_download

    print(f"[Genesis demo] {label} not found locally — downloading {hf_rel} from {HF_REPO_URL}", flush=True)
    try:
        got = hf_hub_download(repo_id=HF_REPO_ID, filename=hf_rel, local_dir=_CKPT_BASE)
    except Exception as e:
        raise FileNotFoundError(
            f"{label} download failed ({type(e).__name__}).\n"
            f"  expected: {local_path}\n"
            f"  source:   {HF_REPO_URL}/blob/main/{hf_rel}\n"
            f"Check network access and that the file exists in the release repo."
        ) from e
    if os.path.abspath(got) != os.path.abspath(local_path):
        # Rare: caller expects the file outside `_CKPT_BASE` (custom override).
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        shutil.move(got, local_path)
    return local_path


def _hf_rel_for(local_path: str) -> str:
    """Map a local path under `_CKPT_BASE` back onto its repo-relative Hub path.
    Falls back to the file's basename when the path isn't under `_CKPT_BASE`."""
    try:
        rel = os.path.relpath(local_path, _CKPT_BASE)
        if rel.startswith(".."):
            raise ValueError
        return rel.replace(os.sep, "/")
    except ValueError:
        return os.path.basename(local_path)


# Training-style tile zoom levels (Git-10M / slippy map pyramid)
TRAINING_ZOOM_LEVELS = tuple(range(10, 20))  # pick closest in UI

# Default checkpoint paths mirror the HF release-repo layout:
#   $GENESIS_CKPT_BASE/main/{superresolution,outpainting}/main_{B16,H16}/*.ckpt
# Per-checkpoint env overrides take precedence over the default layout:
#   GENESIS_SR_B16_CKPT, GENESIS_SR_H16_CKPT,
#   GENESIS_OP_B16_CKPT, GENESIS_OP_H16_CKPT
def _ckpt_default(env_var: str, *fallbacks: str) -> str:
    p = (os.environ.get(env_var) or "").strip()
    return p if p else _resolve_path(*fallbacks)


_SR_MAIN_FNAME = "sr-full-tile-stage3-step0800000.ckpt"   # DINO + LPIPS
_OP_MAIN_FNAME = "op-new-stage3-step0800000.ckpt"         # quadrant masks, no DINO

SR_KEY_B16_MAIN = "JiT-B/16 · main (DINO+LPIPS)"
SR_KEY_H16_MAIN = "JiT-H/16 · main (DINO+LPIPS)"

DEFAULT_SR_CKPT = {
    SR_KEY_B16_MAIN: _ckpt_default(
        "GENESIS_SR_B16_CKPT",
        os.path.join(_CKPT_BASE, "main", "superresolution", "main_B16", _SR_MAIN_FNAME),
    ),
    SR_KEY_H16_MAIN: _ckpt_default(
        "GENESIS_SR_H16_CKPT",
        os.path.join(_CKPT_BASE, "main", "superresolution", "main_H16", _SR_MAIN_FNAME),
    ),
}

# OP model is trained on quadrant masks only; the demos serve quadrant masks.
OP_KEY_B16_MAIN = "JiT-B/16 · main (quadrant)"
OP_KEY_H16_MAIN = "JiT-H/16 · main (quadrant)"

DEFAULT_OP_CKPT = {
    OP_KEY_B16_MAIN: _ckpt_default(
        "GENESIS_OP_B16_CKPT",
        os.path.join(_CKPT_BASE, "main", "outpainting", "main_B16", _OP_MAIN_FNAME),
    ),
    OP_KEY_H16_MAIN: _ckpt_default(
        "GENESIS_OP_H16_CKPT",
        os.path.join(_CKPT_BASE, "main", "outpainting", "main_H16", _OP_MAIN_FNAME),
    ),
}

SR_MODEL_KEYS = list(DEFAULT_SR_CKPT.keys())
OP_MODEL_KEYS = list(DEFAULT_OP_CKPT.keys())


def model_arch(key: str) -> str:
    """Strip the ' · <variant>' suffix from a dropdown key, leaving the engine arch.

    e.g. "JiT-B/16 · ablation D (DINO+LPIPS)" → "JiT-B/16".
    """
    return key.split(" · ")[0].strip()

# DINOv3 weights — satellite-finetuned variant used by training. Stored under
# _CKPT_BASE so the auto-download mirrors the HF repo layout.
_DINO_FNAME = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
DEFAULT_DINO_WEIGHTS = os.path.join(_CKPT_BASE, _DINO_FNAME)


def resolved_dino_weights_path() -> str:
    """Resolve the DINOv3 weights path; auto-download from the HF Hub if missing.

    Resolution order:
      1. $GENESIS_DINO_WEIGHTS → returned as-is.
      2. $GENESIS_CKPT_BASE/<_DINO_FNAME> → auto-fetched from the HF repo if missing.
    """
    env = (os.environ.get("GENESIS_DINO_WEIGHTS") or "").strip()
    if env:
        return env
    return _hf_pull_if_missing(DEFAULT_DINO_WEIGHTS, _DINO_FNAME, label="DINOv3 weights")


def inference_device() -> torch.device:
    """CUDA if available, else CPU. Override: GENESIS_DEVICE=cuda|cpu"""
    o = (os.environ.get("GENESIS_DEVICE") or "").strip().lower()
    if o == "cpu":
        return torch.device("cpu")
    if o == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def inference_device_label(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "CUDA"
    return "CPU"


ESRI_WORLD_IMAGERY = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


def lonlat_to_tile_xy(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    """Web Mercator tile (x,y) for zoom level z; y origin north (OSM/Google style)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    # Wrap longitude into [-180, 180) so scrolling past the antimeridian still works
    lon = (lon + 180.0) % 360.0 - 180.0
    n = 2.0**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_xy_to_lonlat_center(x: int, y: int, zoom: int) -> Tuple[float, float]:
    n = 2.0**zoom
    lon = (x + 0.5) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def fetch_tile_pil(z: int, x: int, y: int, timeout: float = 30.0) -> Image.Image:
    url = ESRI_WORLD_IMAGERY.format(z=z, y=y, x=x)
    r = requests.get(
        url,
        headers={"User-Agent": "Genesis-Demo/1.0 (research)"},
        timeout=timeout,
    )
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def fetch_tile_mosaic_center(
    lon: float, lat: float, zoom: int, grid: int = 3, out_size: int = 256
) -> Tuple[Image.Image, int, int, int]:
    """
    Return a (grid*256) mosaic centered on (lon,lat), and the child tile (zc,xc,yc)
    containing that point at zoom `zoom`.
    """
    xc, yc = lonlat_to_tile_xy(lon, lat, zoom)
    half = grid // 2
    tiles = []
    for dy in range(-half, half + 1):
        row = []
        for dx in range(-half, half + 1):
            tx, ty = xc + dx, yc + dy
            # wrap x
            n = 2**zoom
            tx = tx % n
            ty = max(0, min(n - 1, ty))
            row.append(fetch_tile_pil(zoom, tx, ty))
        tiles.append(row)
    w, h = tiles[0][0].size
    mosaic = Image.new("RGB", (w * grid, h * grid))
    for i, row in enumerate(tiles):
        for j, im in enumerate(row):
            mosaic.paste(im, (j * w, i * h))
    mosaic = mosaic.resize((out_size, out_size), Image.Resampling.BILINEAR)
    return mosaic, zoom, xc, yc


def fetch_tile_at_zoom(lon: float, lat: float, zoom: int, size: int = 256) -> Image.Image:
    """Single Esri tile covering (lon,lat), resized to size x size."""
    xc, yc = lonlat_to_tile_xy(lon, lat, zoom)
    im = fetch_tile_pil(zoom, xc, yc)
    return im.resize((size, size), Image.Resampling.BILINEAR)


def nearest_training_zoom(requested_z: int) -> int:
    return min(TRAINING_ZOOM_LEVELS, key=lambda z: abs(z - requested_z))


def parent_zoom_for_sr(map_zoom: float, zoom_diff: int) -> tuple[int, int]:
    """
    From map zoom level (any float) and upsampling steps d, pick parent zp in 10–19
    closest to map_zoom, then clamp so child zc = zp + d <= 19. Returns (zp, zc).
    """
    d = int(zoom_diff)
    d = max(1, min(4, d))
    snap = min(TRAINING_ZOOM_LEVELS, key=lambda z: abs(float(z) - float(map_zoom)))
    zp = min(snap, 19 - d)
    zp = max(zp, 10)
    zc = zp + d
    return zp, zc


def child_bbox_in_parent(
    parent_z: int, parent_x: int, parent_y: int,
    child_z: int, child_x: int, child_y: int,
    zoom_diff: int,
) -> torch.Tensor:
    """Normalized [x0,y0,x1,y1] of child in parent tile (same as dataloader)."""
    zp, xp, yp = parent_z, parent_x, parent_y
    zc, xc, yc = child_z, child_x, child_y
    s = 1 << zoom_diff
    rx = xc - (xp << zoom_diff)
    ry = yc - (yp << zoom_diff)
    sz = 1.0 / float(s)
    x0, y0 = rx * sz, ry * sz
    return torch.tensor([x0, y0, x0 + sz, y0 + sz], dtype=torch.float32)


def pil_to_model_tensor(pil: Image.Image, size: int = 256) -> torch.Tensor:
    """[1,3,H,W] in [-1, 1]."""
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t * 2.0 - 1.0
    return t.unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """t: [1,3,H,W] roughly [-1,1]."""
    x = t.detach().float().cpu()[0].permute(1, 2, 0).numpy()
    x = (x * 0.5 + 0.5).clip(0, 1)
    x = (x * 255).astype(np.uint8)
    return Image.fromarray(x)


def strip_lightning_compile_prefix(state_dict: dict) -> dict:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            k = k[len("model._orig_mod."):]
        elif k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


def _demo_ckpt_use_online_only() -> bool:
    """Set GENESIS_CKPT_WEIGHTS=online to load raw state_dict instead of EMA."""
    return (os.environ.get("GENESIS_CKPT_WEIGHTS") or "").strip().lower() == "online"


def load_denoiser_weights_from_ckpt(module: torch.nn.Module, ck: dict, label: str = "model") -> str:
    """Load weights into `module`, preferring EMA params. Returns 'ema' or 'online'.

    Lit checkpoints store ema_params1 in the order of the *trainable* params
    (`[p for p in model.parameters() if p.requires_grad]`), excluding pos_embed
    and frozen LPIPS weights. The live module must be filtered the same way
    before zipping or the EMA copy is silently skipped.
    """
    if _demo_ckpt_use_online_only():
        sd = strip_lightning_compile_prefix(ck["state_dict"])
        module.load_state_dict(sd, strict=False)
        return "online"

    # Always load state_dict first so buffers (e.g. BatchNorm) match the checkpoint.
    sd = strip_lightning_compile_prefix(ck["state_dict"])
    module.load_state_dict(sd, strict=False)

    ema_list = ck.get("ema_params1")
    trainable = [p for p in module.parameters() if p.requires_grad]
    if ema_list is not None and len(ema_list) == len(trainable):
        # Two-pass: validate/convert everything first, then copy — so a mid-list
        # mismatch can never leave the module half-EMA/half-online.
        converted, ok = [], True
        for p, t in zip(trainable, ema_list):
            try:
                te = t.detach().to(dtype=p.dtype, device=p.device)
            except Exception:
                ok = False
                break
            if te.shape != p.shape:
                ok = False
                break
            converted.append(te)
        if ok:
            with torch.no_grad():
                for p, te in zip(trainable, converted):
                    p.data.copy_(te)
            return "ema"

    if ema_list is not None:
        print(
            f"[Genesis demo] {label}: EMA length/shape mismatch "
            f"({len(ema_list)} ema vs {len(trainable)} trainable) — using online weights.",
            flush=True,
        )
    return "online"


def build_sr_args(
    model_name: str,
    dino_weights: str = "",
    output_size: int = 512,
    steps: int = 50,
    cfg: float = 1.0,
    bicubic_init: bool = False,
    bicubic_init_t_start: float = 0.5,
) -> Namespace:
    """Args namespace for `src/superresolution/engine.DenoiserSR`.

    `use_dino` is auto-set from whether `dino_weights` is non-empty, so
    passing `dino_weights=""` disables DINO conditioning.
    """
    return Namespace(
        model=model_name,
        output_size=output_size,
        use_dino=bool(dino_weights),
        dino_weights=dino_weights or "",
        # Diffusion noise schedule (matches main.py defaults)
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=1e-5,
        # Sampling
        num_sampling_steps=steps,
        cfg=cfg,
        bicubic_init=bicubic_init,
        bicubic_init_t_start=bicubic_init_t_start,
        # Loss-only fields (init reads them; not used at inference)
        lpips_weight=0.0,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
    )


def build_op_args(
    model_name: str,
    img_size: int = 256,
    dino_weights: str = "",
    steps: int = 50,
    cfg: float = 1.0,
    enable_time_weight: bool = True,
) -> Namespace:
    """Args namespace for `src/outpainting/engine.DenoiserOutpainting`.

    `dino_weights=""` disables DINO conditioning at inference.
    """
    return Namespace(
        model=model_name,
        img_size=img_size,
        dino_weights=dino_weights or "",
        # Diffusion noise schedule
        P_mean=-0.8,
        P_std=0.8,
        noise_scale=1.0,
        t_eps=1e-5,
        # Sampling
        num_sampling_steps=steps,
        cfg=cfg,
        # Loss-only fields (init reads them; not used at inference)
        enable_time_weight=enable_time_weight,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
    )


def load_sr_denoiser(ckpt_path: str, model_name: str, dino_weights: str, device: torch.device):
    """Build `DenoiserSR` from src/superresolution/ and load EMA (or online) weights.

    Auto-fetches the ckpt from Google Drive if `ckpt_path` is missing locally
    and lives under `$GENESIS_CKPT_BASE`.
    """
    ckpt_path = _hf_pull_if_missing(ckpt_path, _hf_rel_for(ckpt_path), label="SR checkpoint")

    sys.path.insert(0, os.path.join(_SRC, "superresolution"))
    sys.modules.pop("engine", None)   # evict any previously-cached engine (SR or OP share the name)
    from engine import DenoiserSR  # noqa: E402

    args = build_sr_args(model_name, dino_weights=dino_weights)
    m = DenoiserSR(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = load_denoiser_weights_from_ckpt(m, ck, label="SR")
    if src == "ema":
        print(f"[Genesis demo] SR: loaded EMA weights (ema_params1) from {ckpt_path}", flush=True)
    m.eval()
    m.to(device)
    return m, args


def load_op_denoiser(ckpt_path: str, model_name: str, dino_weights: str, device: torch.device):
    """Build `DenoiserOutpainting` from src/outpainting/ and load EMA (or online) weights.

    Auto-fetches the ckpt from Google Drive if `ckpt_path` is missing locally.
    """
    ckpt_path = _hf_pull_if_missing(ckpt_path, _hf_rel_for(ckpt_path), label="OP checkpoint")

    sys.path.insert(0, os.path.join(_SRC, "outpainting"))
    sys.modules.pop("engine", None)   # evict any previously-cached engine (SR or OP share the name)
    from engine import DenoiserOutpainting  # noqa: E402

    args = build_op_args(model_name, dino_weights=dino_weights)
    m = DenoiserOutpainting(args)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = load_denoiser_weights_from_ckpt(m, ck, label="outpainting")
    if src == "ema":
        print(f"[Genesis demo] Outpainting: loaded EMA weights (ema_params1) from {ckpt_path}", flush=True)
    m.eval()
    m.to(device)
    return m, args


@torch.no_grad()
def run_superresolution(
    denoiser,
    parent_pil: Image.Image,
    device: torch.device,
    target_zoom: int | None = None,
) -> Image.Image:
    """Generate the full 512×512 4-tile mosaic at ZL+1 from a 256×256 parent at ZL.

    `target_zoom` is the child tile zoom level (= parent ZL + 1), fed through
    the model's zoom embedder; pass `None` (or 0) for the "unknown GSD" sentinel.
    """
    parent_rgb = pil_to_model_tensor(parent_pil).to(device)
    batch = {"parent_rgb": parent_rgb}
    if target_zoom is not None:
        batch["zoom_level"] = torch.tensor([float(target_zoom)], device=device)
    out = denoiser.generate(batch)   # (1, 3, 512, 512)
    return tensor_to_pil(out)


def outpaint_context_white_holes_preview(base: Image.Image, mask_pil: Image.Image, size: int = 256) -> Image.Image:
    """RGB tile: known pixels from base; hole (mask>0) filled white — shows region the model must fill."""
    base = base.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    m = np.array(mask_pil.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 128
    arr = np.array(base, dtype=np.uint8).copy()
    arr[m] = 255
    return Image.fromarray(arr, mode="RGB")


@torch.no_grad()
def run_outpainting(
    denoiser,
    image_pil: Image.Image,
    mask_pil: Image.Image,
    device: torch.device,
    tile_zoom: int | None = None,
) -> Image.Image:
    """
    mask_pil: white / non-black = inpaint (hole). Same size as image (256 recommended).
    tile_zoom: tile zoom level used as the GSD-conditioning scalar; pass `None`
               (or 0) to fall back to the model's "unknown GSD" sentinel.
    """
    image_pil = image_pil.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    mask_pil = mask_pil.convert("L").resize((256, 256), Image.Resampling.NEAREST)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
    roi_mask = torch.from_numpy(mask_arr).view(1, 1, 256, 256).to(device)
    roi_mask = (roi_mask > 0.25).float()

    img_t = pil_to_model_tensor(image_pil).to(device)
    batch = {"image": img_t, "roi_mask": roi_mask}
    if tile_zoom is not None:
        batch["zoom_level"] = torch.tensor([float(tile_zoom)], device=device)
    out = denoiser.generate(batch)
    return tensor_to_pil(out)


@torch.no_grad()
def run_outpainting_batch(
    denoiser,
    image_pils: list,
    mask_pils: list,
    device: torch.device,
    tile_zoom: int | None = None,
) -> list:
    """Batched outpainting: fill N (image, mask) pairs in one sampler trajectory.

    Same result as N ``run_outpainting`` calls (the sampler broadcasts over dim 0)
    but in one batched pass. Caller must only batch independent generations.
    Returns N filled 256² PIL tiles in input order.
    """
    assert len(image_pils) == len(mask_pils)
    if not image_pils:
        return []
    imgs, masks = [], []
    for image_pil, mask_pil in zip(image_pils, mask_pils):
        image_pil = image_pil.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        mask_pil = mask_pil.convert("L").resize((256, 256), Image.Resampling.NEAREST)
        mask_arr = np.array(mask_pil).astype(np.float32) / 255.0
        rm = torch.from_numpy(mask_arr).view(1, 1, 256, 256)
        masks.append((rm > 0.25).float())
        imgs.append(pil_to_model_tensor(image_pil))
    img_t = torch.cat(imgs, dim=0).to(device)
    roi_mask = torch.cat(masks, dim=0).to(device)
    batch = {"image": img_t, "roi_mask": roi_mask}
    if tile_zoom is not None:
        batch["zoom_level"] = torch.full((img_t.shape[0],), float(tile_zoom), device=device)
    out = denoiser.generate(batch)              # (N, 3, 256, 256)
    return [tensor_to_pil(out[i:i + 1]) for i in range(out.shape[0])]


def esri_viewer_url(lon: float, lat: float, zoom: int) -> str:
    return (
        f"https://livingatlas.arcgis.com/wayback/#mapCenter={lon:.5f},{lat:.5f},{zoom}"
    )
