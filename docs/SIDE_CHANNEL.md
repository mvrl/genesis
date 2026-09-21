# Auxiliary side-channel super-resolution

The stock Genesis SR operator (`src/superresolution/`) super-resolves an **RGB** quad-tile
conditioned on its **RGB** parent. This page documents the optional extension that lets the
same JiT operator super-resolve an arbitrary raster while conditioning on a **co-registered
auxiliary raster** ("side channel") that may live at a *different* resolution than the parent.

The motivating regime, and the one the implementation is tuned for, is

```
high-resolution optical  +  low-resolution DSM   -->   high-resolution DSM
```

i.e. the side channel is at the *target* resolution and carries the high-frequency structure
that the low-resolution parent has lost. Nothing in the code is optical/DSM-specific — any
co-registered raster pair works.

Everything here is **off by default**: with `side_channels=0` and the default
`parent_channels`/`parent_size`, `SR_JiT` and `DenoiserSR` are byte-for-byte the previous
behaviour (regression-guarded by `test_rgb_default_is_unchanged` and
`test_quad_protocol_still_works`).

---

## 1. Backbone: `model_sr.SR_JiT`

New constructor arguments:

| Argument | Default | Meaning |
|---|---|---|
| `in_channels` | `3` | channels of the SR **target** (1 for a DSM) |
| `parent_channels` | `= in_channels` | channels of the low-res conditioning raster |
| `parent_size` | `output_size // 2` | spatial size of the low-res conditioning raster |
| `side_channels` | `0` | channels of the auxiliary raster; `0` disables the path |
| `side_size` | `output_size` | spatial size of the auxiliary raster |
| `side_fusion` | `"none"` | `none` \| `early` \| `crossattn` \| `late` |
| `side_late_frac` | `0.25` | fraction of *trailing* blocks that get the `late` injection |

**Parent geometry is now decoupled.** The parent patch size is derived as
`parent_size // 16`, so the parent context is always a 16×16 = 256-token grid whatever the
SR factor. `parent_size = output_size // N` therefore gives an N× operator:

```python
SR_JiT(output_size=512, parent_size=128, ...)   # 4x
SR_JiT(output_size=512, parent_size=64,  ...)   # 8x
```

### The three fusion mechanisms

```
                       ┌──────────────── side raster (B, Cs, Ss, Ss) ────────────────┐
                       │                        │                        │           │
                  [early]                 [crossattn]                 [late]         │
                       │                        │                        │           │
  noisy target ──concat┴──> x_embedder ──> block 0 ... block k ──> block k+1 ... ──> decoder
                                              ▲  ▲                     ▲
                       parent tokens ─────────┘  │                     │
                       side tokens ──────────────┘ (appended to the    │
                                                    cross-attn context)│
                       side tokens ───────────────────────────────────┘ (added to the
                                                                         token stream)
```

* **`early`** — the side raster is resized to `output_size` and channel-concatenated with the
  noisy target before `x_embedder`. Cheapest and highest-bandwidth: every patch token sees its
  own spatially-aligned side-channel content. The side slice of `x_embedder.proj1` is
  zero-initialised, so the model *starts* bit-identical to `side_fusion="none"`.
* **`crossattn`** — the side raster is patch-embedded into 256 tokens which are appended to the
  `sr_cross_attn` context alongside the parent tokens. This mirrors TerraDiT's conditioning
  pattern (a zero-initialised cross-attention residual branch, see
  `terradit/models/omega.py:GeometryAwareLocalAttention`) and is the most flexible mechanism —
  attention chooses where to look rather than relying on positional alignment — but it is also
  the lowest-bandwidth, since 256 tokens must summarise a full-resolution raster.
* **`late`** — the side raster is patch-embedded into 256 spatially-aligned tokens and added
  (through a zero-initialised linear) to the token stream at the start of the trailing
  `side_late_frac` of blocks. Tests the hypothesis that the side signal is only needed for
  high-frequency refinement, not for global layout.

`early` and `late` are *exactly* no-ops at initialisation (`test_init_parity_with_no_side_channel`),
which keeps a fusion-mechanism comparison free of an initialisation confound. `crossattn` starts
with zero-valued side tokens but those tokens still participate in the attention softmax, so it
is near-parity rather than exact parity.

---

## 2. Engine: `engine.DenoiserSR` generic batch protocol

`DenoiserSR` now accepts two batch protocols and picks automatically:

| Protocol | Keys | Selected when |
|---|---|---|
| **Quad** (original) | `children_rgb`, `children_valid`, `parent_rgb` | `"target"` not in batch |
| **Generic** (new) | `target`, `parent`, optional `side`, optional `target_valid` | `"target"` in batch |

Generic-protocol methods:

* `forward_generic(batch)` → `(loss, aux)` — flow-matching v-loss, optional validity mask.
* `conditional_mean(batch, generator=None, n_draws=1)` — **deterministic** readout. Evaluates
  the net at `t = 0`, where `z_t` is pure noise and therefore carries no information about the
  target, so the network's x1-prediction *is* its estimate of `E[x1 | conditioning]`. This is
  the RMSE-optimal point estimate; it costs one forward pass instead of `num_sampling_steps`
  and has far lower variance, which makes it the right primary readout when the question is
  "does the side channel carry exploitable signal?" rather than "how good are the samples?".
* `generate_generic(batch, generator=None)` — Genesis-faithful Euler flow sampling (with CFG
  when `cfg > 1`), for the generative-quality readout.

### Auxiliary loss terms

* `lpips_weight` — the paper's t-scaled LPIPS term. Works for 1-channel targets (replicated to
  3 channels), but requires the `lpips` package **and a network fetch of its AlexNet weights**.
* `grad_loss_weight` — a dependency-free Sobel gradient-magnitude L1 term that plays the same
  "don't blur the edges" role. Preferred for single-channel physical rasters (LPIPS is
  calibrated for 8-bit natural RGB) and for environments where job code may not download
  weights.

---

## 3. Glue: `src/superresolution/side_channel.py`

```python
from side_channel import SideSRConfig, build_denoiser, train_side_sr, evaluate_side_sr

cfg = SideSRConfig(
    model="JiT-B/16", output_size=512,
    in_channels=1, parent_channels=1, parent_size=128,     # 4x DSM SR
    side_channels=3, side_size=512, side_fusion="early",   # high-res optical side channel
    residual=True, residual_scale=0.81,                    # predict the bicubic residual
    train_seconds=1020, batch_size=8, lr=2e-4,
)
model = build_denoiser(cfg)
hist  = train_side_sr(model, train_loader, cfg, device="cuda")
res   = evaluate_side_sr(model, val_loader, cfg, denorm=lambda x: (x + 1) * 30.0)
```

* **`SideSRConfig`** is a dataclass that is duck-type compatible with the `args` namespace
  `DenoiserSR` reads, so no argparse plumbing is needed.
* **`train_side_sr`** budgets on **wall clock** (`train_seconds`), not on a step count, and
  returns `steps`, `train_seconds_used`, `budget_fraction_used` and `stopped_on`. Budgeting on
  wall clock is the only fair protocol when comparing architectures with different per-step
  costs, and reporting the realised budget lets a reviewer verify that each arm actually
  trained rather than exiting after a token number of steps.
* **`evaluate_side_sr`** returns both readouts (`cond_mean`, `flow`) with physical-unit metrics
  from `raster_metrics`: `rmse_m`, `mae_m`, `edge_rmse_m` (top 5% of GT gradient magnitude —
  building/vegetation boundaries, where interpolation is worst), `flat_rmse_m`,
  `slope_rmse_deg`, `slope_mae_deg`, `aspect_mae_deg`.

### Residual parameterisation

With `residual=True` (default) the diffusion target is

```
x1 = (hr_norm - bicubic_up(lr_norm)) / residual_scale
```

so the trivial all-zero solution reproduces bicubic interpolation exactly and any learned
signal is strictly an improvement on it. This removes the "an under-trained generative model
loses to a closed-form interpolator for reasons unrelated to the question being asked" failure
mode and moves the scientific weight onto the trained-vs-trained comparison. Set
`residual=False` for arms that have no low-resolution parent to build a base from.

`residual_scale` should be the dataset standard deviation of that residual, so the diffusion
target is roughly unit variance and `noise_scale=1.0` remains sensible.

### Required batch keys (generic protocol)

| Key | Shape | Needed for |
|---|---|---|
| `target` | (B, C, S, S) | training loss (the *residual* when `residual=True`) |
| `parent` | (B, Cp, Sp, Sp) | conditioning (omit for a no-parent ablation) |
| `side` | (B, Cs, Ss, Ss) | conditioning (omit to ablate the side channel) |
| `base` | (B, C, S, S) | evaluation only — the bicubic base, in normalised units |
| `target_hr_norm` | (B, C, S, S) | evaluation only — the un-residualised GT, normalised |
| `target_valid` | (B, 1, S, S) | optional loss mask |

---

## 4. Tests

```bash
cd src/superresolution
PYTHONPATH=..:. python test_side_channel.py     # 20 CPU tests, a few seconds
PYTHONPATH=..:. python side_channel.py          # end-to-end smoke test of all fusions
PYTHONPATH=..:. python model_sr.py              # backbone shape check, RGB + side variants
```
