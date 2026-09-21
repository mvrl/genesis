"""Tests for the auxiliary side-channel SR path (``model_sr`` + ``engine`` + ``side_channel``).

The repo has no pytest dependency, so both entry points work:

    cd src/superresolution && python test_side_channel.py
    cd src/superresolution && pytest -q test_side_channel.py

All tests are CPU-only and finish in a few seconds.

Coverage
--------
Backbone geometry
  * the RGB-only default path is unchanged from pre-existing Genesis behaviour;
  * parent channels/size are decoupled, enabling N× (not just 2×) SR;
  * all three fusion mechanisms run, are distinct modules, and `early`/`late` are
    *initialised as exact no-ops* so the mechanism comparison is not confounded by
    a different random init;
  * misconfigurations raise rather than silently disabling the side channel.

DenoiserSR generic protocol
  * ``target``/``parent``/``side`` batches produce finite losses and both readouts;
  * the original quad (RGB pyramid) protocol still works — regression guard;
  * ``grad_loss_weight`` (the term the `no_lpips_loss` ablation drops) actually
    changes the loss and only then appears in the aux dict;
  * the ``optical_only_no_dsm`` scenario (no parent at all, ``residual=False``);
  * a data-level shuffled-side-channel control composes with the model — the
    precondition for KC2 being a meaningful test downstream in cards/573.

Budgeted training + metrics
  * the wall-clock budget binds and is reported honestly (this is what lets a
    reviewer verify every arm consumed a comparable, non-trivial fraction of its
    compute allowance instead of exiting after a token number of steps);
  * ``max_steps`` can bind instead;
  * ``raster_metrics`` is exact on analytically-known signals.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from model_sr import SR_JiT, JiT_models
from side_channel import (
    SideSRConfig,
    build_denoiser,
    evaluate_side_sr,
    raster_metrics,
    train_side_sr,
    _synthetic_batch,
)

# Tiny backbone shared by most tests: 64×64 target, 16×16 parent (4× SR), 1-ch DSM.
TINY_MODEL = dict(output_size=64, parent_size=16, in_channels=1, parent_channels=1,
                  depth=2, hidden_size=128, num_heads=4)
TINY_CFG = dict(model="JiT-custom", **TINY_MODEL)
DENORM = lambda x: x * 30.0  # noqa: E731 — arbitrary metres-per-unit for the tests


# ---------------------------------------------------------------------------
# Backbone geometry
# ---------------------------------------------------------------------------

def test_rgb_default_is_unchanged():
    """A default-constructed SR_JiT must behave exactly as before this feature."""
    m = SR_JiT(output_size=512, use_dino=False, depth=2, hidden_size=128, num_heads=4)
    assert m.side_channels == 0 and m.side_fusion == "none"
    assert m.parent_size == 256 and m.parent_channels == 3
    assert m.parent_patch_size == 16
    out = m(torch.randn(1, 3, 512, 512), torch.rand(1),
            conditions={"parent_img": torch.randn(1, 3, 256, 256)})
    assert out.shape == (1, 3, 512, 512)


def test_parent_geometry_is_decoupled():
    """Any SR factor / parent channel count still yields a 256-token parent context."""
    for parent_size in (128, 64, 32):
        m = SR_JiT(output_size=512, use_dino=False, depth=2, hidden_size=128, num_heads=4,
                   in_channels=1, parent_channels=1, parent_size=parent_size)
        assert m.parent_patch_size * 16 == parent_size, parent_size
        out = m(torch.randn(1, 1, 512, 512), torch.rand(1),
                conditions={"parent_img": torch.randn(1, 1, parent_size, parent_size)})
        assert out.shape == (1, 1, 512, 512)


def test_all_fusions_run_and_are_distinct_modules():
    param_counts = {}
    for fusion in ("early", "crossattn", "late"):
        m = SR_JiT(use_dino=False, side_channels=3, side_size=64,
                   side_fusion=fusion, **TINY_MODEL)
        out = m(torch.randn(2, 1, 64, 64), torch.rand(2),
                conditions={"parent_img": torch.randn(2, 1, 16, 16),
                            "side": torch.randn(2, 3, 64, 64)})
        assert out.shape == (2, 1, 64, 64)
        param_counts[fusion] = sum(p.numel() for p in m.parameters())
    # each mechanism adds its own parameters; none is accidentally a no-op alias
    assert len(set(param_counts.values())) == 3, param_counts


def test_side_raster_at_other_resolution_is_resized():
    m = SR_JiT(use_dino=False, side_channels=3, side_size=64,
               side_fusion="crossattn", **TINY_MODEL)
    out = m(torch.randn(1, 1, 64, 64), torch.rand(1),
            conditions={"parent_img": torch.randn(1, 1, 16, 16),
                        "side": torch.randn(1, 3, 32, 32)})  # half resolution
    assert out.shape == (1, 1, 64, 64)


def test_bad_fusion_config_raises():
    for kwargs in ({"side_channels": 3, "side_fusion": "none"},
                   {"side_channels": 0, "side_fusion": "early"},
                   {"side_channels": 3, "side_fusion": "bogus"}):
        try:
            SR_JiT(use_dino=False, **{**TINY_MODEL, **kwargs})
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_init_parity_with_no_side_channel():
    """`early` and `late` must be bit-identical to `none` at initialisation.

    This is what makes the fusion-mechanism comparison a test of *what the model
    learns from the side raster* rather than of a different random initialisation.
    """
    x, t = torch.randn(2, 1, 64, 64), torch.rand(2)
    p, s = torch.randn(2, 1, 16, 16), torch.randn(2, 3, 64, 64)
    for fusion in ("early", "late"):
        torch.manual_seed(0)
        base = SR_JiT(use_dino=False, **TINY_MODEL).eval()
        torch.manual_seed(0)
        side = SR_JiT(use_dino=False, side_channels=3, side_size=64,
                      side_fusion=fusion, **TINY_MODEL).eval()
        a = base(x, t, conditions={"parent_img": p})
        b = side(x, t, conditions={"parent_img": p, "side": s})
        assert torch.allclose(a, b, atol=1e-5), (fusion, (a - b).abs().max())


def test_jit_custom_is_registered():
    assert "JiT-custom" in JiT_models
    m = JiT_models["JiT-custom"](depth=3, hidden_size=64, num_heads=4,
                                 output_size=64, use_dino=False)
    assert len(m.blocks) == 3 and m.hidden_size == 64


# ---------------------------------------------------------------------------
# DenoiserSR generic protocol
# ---------------------------------------------------------------------------

def test_generic_protocol_loss_and_readouts():
    cfg = SideSRConfig(side_channels=3, side_size=64, side_fusion="early",
                       num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    loss, aux = model(batch)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert aux == {}  # no auxiliary loss terms enabled by default
    cm = model.conditional_mean(batch, n_draws=2)
    fl = model.generate_generic(batch)
    assert cm.shape == fl.shape == (2, 1, 64, 64)
    assert torch.isfinite(cm).all() and torch.isfinite(fl).all()


def test_target_valid_mask_is_honoured():
    cfg = SideSRConfig(num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    batch["target_valid"] = torch.zeros(2, 1, 64, 64)
    batch["target_valid"][:, :, :32] = 1.0
    loss, _ = model(batch)
    assert torch.isfinite(loss)


def test_quad_protocol_still_works():
    """The original Genesis RGB quad path must be untouched by this change."""
    cfg = SideSRConfig(model="JiT-custom", depth=2, hidden_size=128, num_heads=4,
                       output_size=64, parent_size=32, in_channels=3,
                       parent_channels=3, num_sampling_steps=2)
    model = build_denoiser(cfg)
    batch = {
        "children_rgb": torch.randn(2, 4, 3, 32, 32),
        "children_valid": torch.ones(2, 4, dtype=torch.bool),
        "parent_rgb": torch.randn(2, 3, 32, 32),
    }
    loss, _aux = model(batch)
    assert torch.isfinite(loss)
    gen = model.generate(batch)
    assert gen.shape == (2, 3, 64, 64)


def test_grad_loss_ablation():
    """`grad_loss_weight` is the loss term the `no_lpips_loss` ablation drops.

    Same weights, same batch: enabling it must change the loss (so the ablation is
    not a silent no-op) and expose `grad_loss` in the aux dict.
    """
    base_kw = dict(side_channels=3, side_size=64, side_fusion="early",
                   num_sampling_steps=2, batch_size=2, residual_scale=0.8, **TINY_CFG)
    torch.manual_seed(0)
    cfg_off = SideSRConfig(grad_loss_weight=0.0, **base_kw)
    model_off = build_denoiser(cfg_off)
    batch = _synthetic_batch(cfg_off, b=2)
    torch.manual_seed(2)
    loss_off, aux_off = model_off(batch)
    assert "grad_loss" not in aux_off

    torch.manual_seed(0)
    cfg_on = SideSRConfig(grad_loss_weight=0.5, **base_kw)
    model_on = build_denoiser(cfg_on)
    model_on.load_state_dict(model_off.state_dict(), strict=False)
    torch.manual_seed(2)
    loss_on, aux_on = model_on(batch)
    assert "grad_loss" in aux_on and torch.isfinite(loss_on)
    assert not torch.allclose(loss_on, loss_off), (loss_on, loss_off)


def test_lpips_path_accepts_single_channel():
    try:
        import lpips  # noqa: F401
    except ImportError:
        return  # optional dependency; the DSM task uses grad_loss_weight instead
    cfg = SideSRConfig(lpips_weight=0.5, num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    loss, aux = model(_synthetic_batch(cfg, b=2))
    assert torch.isfinite(loss) and "lpips_loss" in aux


def test_optical_only_no_dsm():
    """No low-resolution parent at all — can the side channel alone drive the model?

    (Sanity ablation, not expected to win; `residual=False` because there is no
    low-res raster to build a bicubic base from.)
    """
    cfg = SideSRConfig(side_channels=3, side_size=64, side_fusion="early",
                       num_sampling_steps=2, batch_size=2, residual=False, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    del batch["parent"], batch["base"]
    loss, _ = model(batch)
    assert torch.isfinite(loss)
    res = evaluate_side_sr(model, [batch], cfg, DENORM, device="cpu",
                           cond_mean_draws=1, run_flow_sampling=True)
    assert math.isfinite(res["cond_mean"]["rmse_m"])
    assert math.isfinite(res["flow"]["rmse_m"])


def test_shuffled_side_channel_control():
    """KC2 plumbing: permuting the side raster across the batch breaks co-registration.

    The model must not care *how* ``batch["side"]`` was built, only its shape — that
    is what makes the shuffled control a pure data-level manipulation with an
    architecture identical to the true-conditioning arm.
    """
    cfg = SideSRConfig(side_channels=3, side_size=64, side_fusion="early",
                       num_sampling_steps=2, batch_size=4, residual_scale=0.8, **TINY_CFG)
    torch.manual_seed(1)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=4)
    shuffled = dict(batch)
    shuffled["side"] = batch["side"][torch.tensor([2, 3, 0, 1])]  # no fixed point
    torch.manual_seed(2)
    loss_paired, _ = model(batch)
    torch.manual_seed(2)
    loss_shuffled, _ = model(shuffled)
    assert torch.isfinite(loss_paired) and torch.isfinite(loss_shuffled)


# ---------------------------------------------------------------------------
# Budgeted training loop + metrics
# ---------------------------------------------------------------------------

def test_wall_clock_budget_is_respected_and_reported():
    cfg = SideSRConfig(train_seconds=1.5, warmup_steps=1, batch_size=2,
                       num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    hist = train_side_sr(model, [batch] * 4, cfg, device="cpu", log_every=1)
    assert hist["steps"] > 0
    assert hist["stopped_on"] == "train_seconds", hist
    assert hist["budget_fraction_used"] >= 0.5, hist
    assert hist["samples_seen"] == hist["steps"] * cfg.batch_size


def test_max_steps_can_bind_before_wall_clock():
    cfg = SideSRConfig(train_seconds=600.0, max_steps=3, warmup_steps=1,
                       num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    hist = train_side_sr(model, [batch] * 8, cfg, device="cpu")
    assert hist["steps"] == 3 and hist["stopped_on"] == "max_steps", hist


def test_raster_metrics_are_exact_on_a_known_signal():
    gt = torch.zeros(1, 1, 16, 16)
    gt[..., 8:, :] = 4.0                 # a 4 m step edge across the tile
    pred = gt + 0.5                      # uniform +0.5 m bias
    m = raster_metrics(pred, gt, gsd=1.0, edge_quantile=0.90)
    assert math.isclose(m["rmse_m"], 0.5, abs_tol=1e-5)
    assert math.isclose(m["mae_m"], 0.5, abs_tol=1e-5)
    # a constant offset leaves gradients (hence slope and aspect) untouched
    assert math.isclose(m["slope_rmse_deg"], 0.0, abs_tol=1e-4)
    # the edge/flat split partitions the tile and both sides see the same bias
    assert math.isclose(m["edge_rmse_m"], 0.5, abs_tol=1e-5)
    assert math.isclose(m["flat_rmse_m"], 0.5, abs_tol=1e-5)


def test_raster_metrics_detect_a_smoothed_edge():
    """Sanity: blurring the step edge must inflate edge RMSE and slope error."""
    gt = torch.zeros(1, 1, 32, 32)
    gt[..., 16:, :] = 4.0
    blurred = torch.nn.functional.avg_pool2d(gt, 4)
    blurred = torch.nn.functional.interpolate(blurred, size=(32, 32), mode="bicubic",
                                              align_corners=False)
    sharp = raster_metrics(gt, gt, gsd=1.0, edge_quantile=0.90)
    soft = raster_metrics(blurred, gt, gsd=1.0, edge_quantile=0.90)
    assert sharp["rmse_m"] < 1e-5
    assert soft["edge_rmse_m"] > soft["flat_rmse_m"] > 0
    assert soft["slope_rmse_deg"] > 1.0


def test_evaluate_produces_both_readouts():
    cfg = SideSRConfig(side_channels=3, side_size=64, side_fusion="crossattn",
                       num_sampling_steps=2, residual_scale=0.8, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    res = evaluate_side_sr(model, [batch], cfg, DENORM, device="cpu", cond_mean_draws=1)
    assert res["n_tiles"] == 2
    for key in ("cond_mean", "flow"):
        assert math.isfinite(res[key]["rmse_m"]), (key, res)
        assert math.isfinite(res[key]["slope_rmse_deg"]), (key, res)


def test_flow_budget_does_not_shrink_the_cond_mean_split():
    """Regression: capping the expensive flow readout must NOT cap ``cond_mean``.

    Tying the two together silently evaluates the *primary* metric on whatever subset
    the sampling budget allowed, which makes it incomparable with a baseline measured
    over the full split -- a comparison bug, not just a precision loss.
    """
    cfg = SideSRConfig(side_channels=3, side_size=64, side_fusion="early",
                       num_sampling_steps=2, residual_scale=0.8, **TINY_CFG)
    model = build_denoiser(cfg)
    batches = [_synthetic_batch(cfg, b=2) for _ in range(4)]   # 8 tiles over 4 batches
    res = evaluate_side_sr(model, batches, cfg, DENORM, device="cpu",
                           cond_mean_draws=1, run_flow_sampling=True,
                           flow_max_batches=1)
    assert res["n_tiles"] == 8, res            # cond_mean saw every tile
    assert res["n_tiles_flow"] == 2, res       # flow saw only the first batch
    assert math.isfinite(res["cond_mean"]["rmse_m"])
    assert math.isfinite(res["flow"]["rmse_m"])


def test_max_batches_still_caps_both():
    cfg = SideSRConfig(num_sampling_steps=2, residual_scale=0.8, **TINY_CFG)
    model = build_denoiser(cfg)
    batches = [_synthetic_batch(cfg, b=2) for _ in range(4)]
    res = evaluate_side_sr(model, batches, cfg, DENORM, device="cpu",
                           cond_mean_draws=1, run_flow_sampling=True, max_batches=2)
    assert res["n_tiles"] == 4 and res["n_tiles_flow"] == 4, res


def test_residual_false_needs_no_base():
    cfg = SideSRConfig(residual=False, num_sampling_steps=2, **TINY_CFG)
    model = build_denoiser(cfg)
    batch = _synthetic_batch(cfg, b=2)
    batch.pop("base")
    res = evaluate_side_sr(model, [batch], cfg, DENORM, device="cpu",
                           cond_mean_draws=1, run_flow_sampling=False)
    assert math.isfinite(res["cond_mean"]["rmse_m"])


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
