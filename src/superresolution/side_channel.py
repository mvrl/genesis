"""Side-information-conditioned super-resolution for the Genesis SR operator.

Genesis's ``DenoiserSR`` super-resolves an RGB quad-tile from its RGB parent.  This
module adds the machinery to super-resolve *any* single- or multi-channel raster while
conditioning on a **co-registered auxiliary raster** ("side channel") that may live at a
different resolution than the parent — the motivating case being

    high-resolution optical  +  low-resolution DSM   ->   high-resolution DSM

which is the mechanism that neither Genesis nor TerraDiT currently implements.  The
architectural support lives in :class:`model_sr.SR_JiT` (``side_channels`` /
``side_fusion`` / ``parent_size`` / ``parent_channels``) and
:class:`engine.DenoiserSR` (the generic ``target``/``parent``/``side`` batch protocol).
What lives here is the glue a caller needs:

* :class:`SideSRConfig`  — a plain dataclass that is duck-type compatible with the
  ``args`` namespace ``DenoiserSR`` expects, so no argparse plumbing is required.
* :func:`build_denoiser` — config -> ``DenoiserSR``.
* :func:`train_side_sr` — a compact, **wall-clock-budgeted** training loop (AdamW +
  bf16 autocast + grad clip + Genesis's own dual EMA).  Budgeting on wall clock rather
  than a step count is what lets several architecture arms be compared under an
  *equal compute* budget, which is the only fair protocol when the arms have different
  per-step costs.
* :func:`evaluate_side_sr` — held-out evaluation producing both readouts described in
  :meth:`engine.DenoiserSR.conditional_mean` and
  :meth:`engine.DenoiserSR.generate_generic`.
* :func:`raster_metrics` — RMSE/MAE plus the terrain-aware slope/aspect errors and an
  edge-restricted RMSE, all in physical units.

Residual parameterisation
-------------------------
When ``residual=True`` (the default) the diffusion target is

    x1 = (hr_norm - bicubic_up(lr_norm)) / residual_scale

i.e. the model predicts the *detail bicubic misses*, and the trivial all-zero solution
reproduces bicubic exactly.  This removes the "an under-trained generative model loses
to a closed-form interpolator for reasons unrelated to the hypothesis" failure mode and
puts the scientific weight where it belongs: on the trained-vs-trained comparison
(side-conditioned vs. no side channel).  Set ``residual=False`` for arms that have no
low-resolution parent to build a base from.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F

from engine import DenoiserSR

__all__ = [
    "SideSRConfig",
    "build_denoiser",
    "train_side_sr",
    "evaluate_side_sr",
    "raster_metrics",
    "bicubic_upsample",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SideSRConfig:
    """Everything :class:`engine.DenoiserSR` reads off ``args``, plus training knobs.

    Attribute names intentionally mirror ``main.py``'s argparse flags so that a config
    built here and a namespace built there are interchangeable.
    """

    # ---- backbone geometry ----
    model: str = "JiT-B/16"
    output_size: int = 512          # target raster size
    in_channels: int = 1            # target channels (1 = DSM)
    parent_channels: int = 1        # low-res conditioning raster channels
    parent_size: int = 128          # low-res conditioning raster size
    side_channels: int = 0          # 0 disables the auxiliary path
    side_size: int = 512            # auxiliary raster size (may equal output_size)
    side_fusion: str = "none"       # none | early | crossattn | late
    side_late_frac: float = 0.25
    cond_drop_prob: float = 0.1
    use_dino: bool = False          # DINOv3 is an RGB-parent feature; off for raster SR
    dino_weights: str = ""
    # free-form size overrides (only honoured when model == "JiT-custom")
    depth: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    bottleneck_dim: Optional[int] = None

    # ---- flow-matching / diffusion ----
    P_mean: float = -0.8
    P_std: float = 0.8
    noise_scale: float = 1.0
    t_eps: float = 1e-5
    num_sampling_steps: int = 16
    cfg: float = 1.0
    lpips_weight: float = 0.0        # kept for API parity; unused (see grad_loss_weight)
    grad_loss_weight: float = 0.0    # Sobel-gradient edge loss weight (dependency-free
                                      # substitute for LPIPS on physical rasters, see
                                      # engine.DenoiserSR._grad_loss_term). This is the
                                      # loss term the `no_lpips_loss` ablation drops.
    ema_decay1: float = 0.999
    ema_decay2: float = 0.9996

    # ---- optimisation ----
    lr: float = 2e-4
    weight_decay: float = 0.0
    batch_size: int = 8
    grad_clip: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10_000_000     # effectively unbounded; the budget is wall clock
    train_seconds: float = 1080.0   # <- the real stopping criterion
    compile: bool = False
    seed: int = 0

    # ---- task parameterisation ----
    residual: bool = True
    residual_scale: float = 1.0

    def model_summary(self) -> Dict[str, Any]:
        return asdict(self)


def build_denoiser(cfg: SideSRConfig) -> DenoiserSR:
    """Instantiate a :class:`DenoiserSR` from a :class:`SideSRConfig`."""
    return DenoiserSR(cfg)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------


def bicubic_upsample(lr: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(lr, size=(size, size), mode="bicubic", align_corners=False)


def area_downsample(hr: torch.Tensor, factor: int) -> torch.Tensor:
    """Box-average downsample — the standard, non-cheating LR simulation."""
    return F.avg_pool2d(hr, kernel_size=factor, stride=factor)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _lr_at(step: int, cfg: SideSRConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    return cfg.lr


def train_side_sr(
    model: DenoiserSR,
    loader: Iterable[Dict[str, torch.Tensor]],
    cfg: SideSRConfig,
    device: torch.device | str = "cuda",
    log_every: int = 50,
    on_log: Optional[Callable[[Dict[str, float]], None]] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Train under a **wall-clock budget**.

    Stops at the first of: ``cfg.train_seconds`` elapsed, or ``cfg.max_steps`` optimiser
    steps.  Returns a history dict including ``steps`` and ``train_seconds_used`` so a
    downstream analysis can verify that every arm actually consumed its budget (rather
    than exiting after a token number of steps, which silently turns a method
    comparison into a "who was allowed to train" comparison).

    ``loader`` must be *restartable* (a DataLoader is); it is re-iterated until the
    budget is exhausted.
    """
    device = torch.device(device)
    model.to(device).train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    model.ema_params1 = [p.detach().clone() for p in trainable]
    model.ema_params2 = [p.detach().clone() for p in trainable]

    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = device.type == "cuda"

    t0 = time.time()
    step, epoch = 0, 0
    loss_hist: list[Tuple[int, float]] = []
    running, running_n = 0.0, 0

    while step < cfg.max_steps and (time.time() - t0) < cfg.train_seconds:
        epoch += 1
        for batch in loader:
            if step >= cfg.max_steps or (time.time() - t0) >= cfg.train_seconds:
                break
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, cfg)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _aux = model(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            opt.step()
            model.update_ema()
            step += 1
            running += float(loss.detach()); running_n += 1
            if step % log_every == 0:
                rec = {"step": step, "loss": running / max(running_n, 1),
                       "elapsed_s": time.time() - t0}
                loss_hist.append((step, rec["loss"]))
                running, running_n = 0.0, 0
                if on_log is not None:
                    on_log(rec)
        else:
            continue
        break

    elapsed = time.time() - t0
    # Derive the binding constraint from the final state rather than from wherever the
    # loop happened to break: an epoch can finish exactly as the clock runs out, in
    # which case the `while` guard (not the inner `break`) is what stops training.
    stopped_on = "max_steps" if step >= cfg.max_steps else "train_seconds"
    return {
        "steps": step,
        "epochs": epoch,
        "train_seconds_used": elapsed,
        "train_seconds_budget": cfg.train_seconds,
        "budget_fraction_used": elapsed / max(cfg.train_seconds, 1e-9),
        "stopped_on": stopped_on,
        "samples_seen": step * cfg.batch_size,
        "loss_history": loss_hist,
        "trainable_params": count_params(model),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _slope_aspect(a: torch.Tensor, gsd: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central-difference slope (degrees) and aspect (degrees, 0-360) of a height map."""
    sx = (a[..., :, 2:] - a[..., :, :-2]) / (2.0 * gsd)
    sy = (a[..., 2:, :] - a[..., :-2, :]) / (2.0 * gsd)
    sx = sx[..., 1:-1, :]
    sy = sy[..., :, 1:-1]
    slope = torch.rad2deg(torch.atan(torch.sqrt(sx ** 2 + sy ** 2)))
    aspect = torch.rad2deg(torch.atan2(sy, sx)) % 360.0
    return slope, aspect


def raster_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gsd: float = 0.35,
    edge_quantile: float = 0.95,
    aspect_slope_min_deg: float = 10.0,
) -> Dict[str, float]:
    """Physical-unit accuracy metrics for a predicted height raster.

    ``pred`` / ``gt``: (B, 1, H, W) in metres.

    Returns
    -------
    rmse_m, mae_m
        Whole-tile errors.
    edge_rmse_m
        RMSE restricted to the top ``1 - edge_quantile`` of GT gradient magnitude —
        i.e. building/vegetation boundaries, where an optical side channel should help
        and where bicubic is worst.
    flat_rmse_m
        RMSE on the complement (should be ~unchanged by side information).
    slope_rmse_deg, slope_mae_deg
        Terrain-aware surface-orientation error.
    aspect_mae_deg
        Circular mean absolute aspect error, restricted to GT pixels whose slope
        exceeds ``aspect_slope_min_deg`` (aspect is undefined on flat ground).
    """
    pred = pred.float()
    gt = gt.float()
    err = pred - gt

    gm = torch.zeros_like(gt)
    gm[..., :, :-1] += (gt[..., :, 1:] - gt[..., :, :-1]).abs()
    gm[..., :-1, :] += (gt[..., 1:, :] - gt[..., :-1, :]).abs()
    flat_gm = gm.flatten()
    # torch.quantile has a 2**24-element input limit; subsample deterministically.
    if flat_gm.numel() > 1_000_000:
        flat_gm = flat_gm[:: max(1, flat_gm.numel() // 1_000_000)]
    thr = torch.quantile(flat_gm, edge_quantile)
    edge = gm > thr

    s_p, a_p = _slope_aspect(pred, gsd)
    s_g, a_g = _slope_aspect(gt, gsd)
    d_asp = (a_p - a_g).abs()
    d_asp = torch.minimum(d_asp, 360.0 - d_asp)
    asp_mask = s_g > aspect_slope_min_deg

    out = {
        "rmse_m": float(err.pow(2).mean().sqrt()),
        "mae_m": float(err.abs().mean()),
        "edge_rmse_m": float(err[edge].pow(2).mean().sqrt()) if edge.any() else float("nan"),
        "flat_rmse_m": float(err[~edge].pow(2).mean().sqrt()) if (~edge).any() else float("nan"),
        "slope_rmse_deg": float((s_p - s_g).pow(2).mean().sqrt()),
        "slope_mae_deg": float((s_p - s_g).abs().mean()),
        "aspect_mae_deg": float(d_asp[asp_mask].mean()) if asp_mask.any() else float("nan"),
    }
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def decode_prediction(
    x1_pred: torch.Tensor,
    base_norm: Optional[torch.Tensor],
    cfg: SideSRConfig,
    denorm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Map a network x1-prediction back to physical units.

    ``base_norm`` is the bicubic base in *normalised* units (required when
    ``cfg.residual``); ``denorm`` converts normalised heights to metres.
    """
    if cfg.residual:
        if base_norm is None:
            raise ValueError("residual=True requires base_norm")
        return denorm(base_norm + x1_pred * cfg.residual_scale)
    return denorm(x1_pred)


@torch.no_grad()
def evaluate_side_sr(
    model: DenoiserSR,
    loader: Iterable[Dict[str, torch.Tensor]],
    cfg: SideSRConfig,
    denorm: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device | str = "cuda",
    gsd: float = 0.35,
    cond_mean_draws: int = 2,
    run_flow_sampling: bool = True,
    max_batches: Optional[int] = None,
    flow_max_batches: Optional[int] = None,
    seed: int = 12345,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Evaluate both readouts on a held-out loader.

    Produces ``cond_mean`` metrics (deterministic conditional-mean readout; the primary
    accuracy number) and, when ``run_flow_sampling``, ``flow`` metrics (Genesis-faithful
    Euler sampling).  Per-tile predictions are accumulated on CPU so the metrics are
    computed once over the whole split rather than averaged over batches.

    ``max_batches`` caps the cheap ``cond_mean`` readout; ``flow_max_batches``
    *separately* caps the ~``num_sampling_steps``x more expensive ``flow`` readout.
    Keeping these independent matters: tying them together would silently shrink the
    primary metric's evaluation set to whatever the sampling budget allowed, making it
    incomparable with a baseline measured over the full split.  The returned
    ``n_tiles`` / ``n_tiles_flow`` record what each number was actually computed on.
    """
    device = torch.device(device)
    model.to(device).eval()
    use_amp = device.type == "cuda"

    preds_cm, preds_fl, gts, gts_fl = [], [], [], []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        gen = torch.Generator(device=device).manual_seed(seed + i)
        base = batch.get("base")
        want_flow = run_flow_sampling and (flow_max_batches is None or i < flow_max_batches)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            cm = model.conditional_mean(batch, generator=gen, n_draws=cond_mean_draws)
            fl = model.generate_generic(batch, generator=gen) if want_flow else None
        gt_batch = denorm(batch["target_hr_norm"]).cpu()
        preds_cm.append(decode_prediction(cm.float(), base, cfg, denorm).cpu())
        gts.append(gt_batch)
        if fl is not None:
            preds_fl.append(decode_prediction(fl.float(), base, cfg, denorm).cpu())
            gts_fl.append(gt_batch)

    gt = torch.cat(gts)
    out: Dict[str, Any] = {
        "n_tiles": int(gt.shape[0]),
        "cond_mean": raster_metrics(torch.cat(preds_cm), gt, gsd=gsd),
    }
    if preds_fl:
        gt_fl = torch.cat(gts_fl)
        out["n_tiles_flow"] = int(gt_fl.shape[0])
        out["flow"] = raster_metrics(torch.cat(preds_fl), gt_fl, gsd=gsd)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Self-test (no dataset required)
# ---------------------------------------------------------------------------


def _synthetic_batch(cfg: SideSRConfig, b: int = 2, device="cpu"):
    hr = torch.randn(b, cfg.in_channels, cfg.output_size, cfg.output_size, device=device)
    lr = F.avg_pool2d(hr, cfg.output_size // cfg.parent_size)
    base = bicubic_upsample(lr, cfg.output_size)
    batch = {
        "target": ((hr - base) / cfg.residual_scale) if cfg.residual else hr,
        "parent": lr,
        "base": base,
        "target_hr_norm": hr,
    }
    if cfg.side_channels > 0:
        batch["side"] = torch.randn(b, cfg.side_channels, cfg.side_size,
                                    cfg.side_size, device=device)
    return batch


def smoke_test(device: str | None = None) -> None:
    """End-to-end shape/optimisation check for every fusion mechanism (tiny + fast)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    denorm = lambda x: x * 30.0  # noqa: E731 — arbitrary scale for the self-test

    specs = [
        ("dsm_only", 0, "none"),
        ("early", 3, "early"),
        ("crossattn", 3, "crossattn"),
        ("late", 3, "late"),
    ]
    for label, sc, fusion in specs:
        cfg = SideSRConfig(
            model="JiT-custom", depth=2, hidden_size=128, num_heads=4,
            output_size=64, parent_size=16, in_channels=1, parent_channels=1,
            side_channels=sc, side_size=64, side_fusion=fusion,
            num_sampling_steps=2, train_seconds=2.0, batch_size=2,
            warmup_steps=1, residual_scale=0.8,
        )
        model = build_denoiser(cfg)
        batch = _synthetic_batch(cfg, device=device)
        loader = [batch] * 4
        hist = train_side_sr(model, loader, cfg, device=device, log_every=1)
        res = evaluate_side_sr(model, [batch], cfg, denorm, device=device,
                               cond_mean_draws=1, run_flow_sampling=True)
        assert hist["steps"] > 0, label
        assert math.isfinite(res["cond_mean"]["rmse_m"]), (label, res)
        assert math.isfinite(res["flow"]["rmse_m"]), (label, res)
        print(f"[{label:10s}] params {hist['trainable_params']/1e6:6.2f}M  "
              f"steps {hist['steps']:3d}  "
              f"rmse(cond_mean) {res['cond_mean']['rmse_m']:.3f}  "
              f"rmse(flow) {res['flow']['rmse_m']:.3f}  "
              f"slope {res['cond_mean']['slope_rmse_deg']:.2f} deg")

    # Every fusion mechanism must start as an exact no-op relative to side_fusion="none"
    # (see SR_JiT.initialize_weights).  Verify for the two modes that promise bit-parity.
    for fusion in ("early", "late"):
        kw = dict(model="JiT-custom", depth=2, hidden_size=128, num_heads=4,
                  output_size=64, parent_size=16, in_channels=1, parent_channels=1)
        torch.manual_seed(0); m_none = build_denoiser(SideSRConfig(**kw)).eval()
        torch.manual_seed(0); m_side = build_denoiser(
            SideSRConfig(side_channels=3, side_size=64, side_fusion=fusion, **kw)).eval()
        x = torch.randn(2, 1, 64, 64); t = torch.rand(2)
        p = torch.randn(2, 1, 16, 16); s = torch.randn(2, 3, 64, 64)
        a = m_none.net(x, t, conditions={"parent_img": p})
        b = m_side.net(x, t, conditions={"parent_img": p, "side": s})
        assert torch.allclose(a, b, atol=1e-5), (fusion, (a - b).abs().max())
        print(f"[init-parity] side_fusion={fusion:5s} matches 'none' at init "
              f"(max|diff| = {(a - b).abs().max():.2e})")

    print("✓ side_channel smoke test passed")


if __name__ == "__main__":
    smoke_test()
