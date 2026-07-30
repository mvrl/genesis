# Genesis Eval Metrics

The per-sample metrics computed by [`common.py:compute_metrics`](../src/eval/common.py)
and written to `samples.csv`. Every metric is backed by a library
implementation; the math, code, and OP hole-awareness behaviour are
documented per-metric below.

## At a glance

| Column          | Library            | Direction | Range          | SR | OP | OP hole-aware |
|-----------------|--------------------|-----------|----------------|:--:|:--:|:--:|
| `psnr`          | `torchmetrics`     | ↑ better  | dB, typ. 10–35 | ✓  | ✓  | hole pixels only |
| `ssim`          | `torchmetrics`     | ↑ better  | 0..1           | ✓  | ✓  | per-region mean |
| `lpips`         | `torchmetrics`     | ↓ better  | ~0..1          | ✓  | ✓  | per-region mean |
| `dists`         | `DISTS_pytorch`    | ↓ better  | ~0..1          | ✓  | ✓  | per-region mean |
| `boundary_diff` | `scipy` + `torch`  | ↓ better  | 0..255         | —  | ✓  | 4-px ring inside hole edge |

## Hole-region decomposition (shared infrastructure)

OP eval must score only pixels the model actually generated — not the GT
quadrants the model was given (the OP `generate` call overwrites known
pixels with GT inside `denoiser.generate()` before metrics see the
output, so a naive full-image SSIM would be dominated by those copied
pixels).

`_hole_scoring_regions(mb, h, w)` ([`common.py:333`](../src/eval/common.py#L333))
takes the boolean hole mask and returns a list of rectangular regions
to score independently:

- mask is `None` or all-zero → `[(0, h, 0, w)]` (full image, the SR
  path).
- mask aligns to the 4 quadrant cells (each quadrant is uniformly hole
  or uniformly known) → one rectangle per **hole** quadrant. Handles
  every supported quadrant config: 1, 2-adjacent, 2-diagonal, 3 known.
- otherwise (defensive fallback) → single bbox of the hole.

For `quad1` (1 known, 3 hole) this returns three 128×128 rectangles:

```
mask for quad1, UL known:               _hole_scoring_regions:
+--------+--------+                     [(0, 128, 128, 256),     # UR
|   UL   |   UR   |                      (128, 256,  0, 128),    # LL
|  known |  hole  |                      (128, 256, 128, 256)]   # LR
+--------+--------+
|   LL   |   LR   |
|  hole  |  hole  |
+--------+--------+
```

The pair-comparison metrics (SSIM / LPIPS / DISTS) are computed
independently on each region and averaged. PSNR uses per-pixel mask
gather instead (no need for regions because it doesn't use spatial
windows / feature maps). `boundary_diff` constructs its own ring
directly from the mask.

See [`README.md`](../src/eval/README.md) for the per-regime mask layouts the eval
script samples.

---

## 1. `psnr` — Peak Signal-to-Noise Ratio

### Math

$$
\text{PSNR} = 10 \cdot \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right),
\qquad
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N} \left(\hat{x}_i - x_i\right)^2
$$

with $\text{MAX} = 1$ for inputs in $[0, 1]$. Direction: **↑ better**,
unbounded above, typically 10–35 dB on natural images.

### Code ([`common.py:446`](../src/eval/common.py#L446))

```python
from torchmetrics.functional import peak_signal_noise_ratio

if mb is not None:
    sel = mb[..., None].repeat(3, axis=2)              # (H, W, 3) boolean
    pa_hole = torch.from_numpy(pa[sel]).reshape(-1)
    ga_hole = torch.from_numpy(ga[sel]).reshape(-1)
    out["psnr"] = float(peak_signal_noise_ratio(
        pa_hole, ga_hole, data_range=1.0,
    ).item())
else:
    out["psnr"] = float(peak_signal_noise_ratio(
        pa_t_full, ga_t_full, data_range=1.0,
    ).item())
```

### OP hole-awareness

**Per-pixel mask gather.** We flatten the masked (H, W, 3) view into a
1-D tensor of only the hole pixels and pass that to
`peak_signal_noise_ratio`. The MSE is itself a mean, so per-pixel
gather gives the correct hole-only PSNR without needing per-region
averaging.

---

## 2. `ssim` — Structural Similarity Index (windowed)

### Math

For an 11×11 sliding Gaussian window over the image, with local means
$\mu_x, \mu_y$, variances $\sigma_x^2, \sigma_y^2$, and covariance
$\sigma_{xy}$:

$$
\text{SSIM}(x, y) =
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
     {(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

with $C_1 = (0.01 \cdot L)^2$, $C_2 = (0.03 \cdot L)^2$ and
$L = \text{data\_range} = 1.0$. Per-window SSIM is averaged across the
image.

Direction: **↑ better**, range roughly $[0, 1]$.

### Code ([`common.py:464`](../src/eval/common.py#L464))

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure
# _SSIM_FN = StructuralSimilarityIndexMeasure(data_range=1.0)  — cached in _get_ssim

if "ssim" in metrics:
    ssim_m = _get_ssim(device)
    vals = []
    for (y0, y1, x0, x1) in regions:
        pa_c = pa_t_full[..., y0:y1, x0:x1]
        ga_c = ga_t_full[..., y0:y1, x0:x1]
        if pa_c.shape[-1] < 11 or pa_c.shape[-2] < 11:   # need ≥ 11×11
            continue
        with torch.no_grad():
            vals.append(float(ssim_m(pa_c, ga_c).item()))
        ssim_m.reset()
    out["ssim"] = float(np.mean(vals)) if vals else float("nan")
```

### OP hole-awareness

**Per-region windowed SSIM, averaged across hole regions.** SSIM's
sliding window doesn't natively support a mask — sliding across an
L-shape hole would let the known quadrant dominate the score. Instead,
score each 128×128 hole quadrant independently and average. Tiny
regions (< 11 px on a side) are skipped because the window doesn't
fit.

---

## 3. `lpips` — Learned Perceptual Image Patch Similarity

### Math

LPIPS doesn't have a closed-form formula. Both images are pushed through
a frozen CNN (we use SqueezeNet), the per-layer feature maps are
unit-normalized along the channel dimension, and the squared difference
is summed across layers and spatial positions with per-channel learned
weights $w_l$:

$$
\text{LPIPS}(x, y) =
\sum_{l} \frac{1}{H_l W_l} \sum_{h, w}
\bigl\lVert w_l \odot
\bigl(\hat\phi_l(x)_{h, w} - \hat\phi_l(y)_{h, w}\bigr)\bigr\rVert_2^2
$$

where $\hat\phi_l$ is the unit-normalized feature map at layer $l$.
Direction: **↓ better**, range ~0.05–0.5 on natural images.

### Code ([`common.py:479`](../src/eval/common.py#L479))

```python
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# _LPIPS_FN = LearnedPerceptualImagePatchSimilarity(net_type="squeeze")  — cached

if "lpips" in metrics:
    lpips_m = _get_lpips(device)
    vals = []
    for (y0, y1, x0, x1) in regions:
        pa_c = _maybe_downsample(pa_t_full[..., y0:y1, x0:x1])
        ga_c = _maybe_downsample(ga_t_full[..., y0:y1, x0:x1])
        with torch.no_grad():
            vals.append(float(lpips_m(pa_c * 2 - 1, ga_c * 2 - 1).item()))  # [-1, 1]
        lpips_m.reset()
    out["lpips"] = float(np.mean(vals)) if vals else float("nan")
```

### OP hole-awareness

**Per-region perceptual distance, averaged.** Same idea as SSIM. The
`* 2 - 1` shifts the `[0, 1]` inputs to `[-1, 1]` as LPIPS expects.
`_maybe_downsample` caps the longest side at `lpips_max_size` for very
large crops (a no-op at our 128×128 quadrants).

---

## 4. `dists` — Deep Image Structure and Texture Similarity

### Math

DISTS pushes both images through VGG-16. For each layer $l$ it computes
two per-channel similarities — one over feature-map first moments
(**structure**) and one over feature-map covariances (**texture**) —
then combines them with learned weights $\alpha_l, \beta_l$:

$$
S_l^{\text{struct}} =
\frac{2\mu_{\phi_l(x)} \mu_{\phi_l(y)} + C_1}
     {\mu_{\phi_l(x)}^2 + \mu_{\phi_l(y)}^2 + C_1},
\quad
S_l^{\text{texture}} =
\frac{2\sigma_{\phi_l(x)\phi_l(y)} + C_2}
     {\sigma_{\phi_l(x)}^2 + \sigma_{\phi_l(y)}^2 + C_2}
$$

$$
\text{DISTS}(x, y) =
1 - \sum_{l} \bigl(\alpha_l \cdot S_l^{\text{struct}} +
                   \beta_l  \cdot S_l^{\text{texture}}\bigr)
$$

Direction: **↓ better**, range ~0..1.

### Code ([`common.py:491`](../src/eval/common.py#L491))

```python
from DISTS_pytorch import DISTS
# _DISTS_FN = DISTS()  — cached lazily inside the block

if "dists" in metrics:
    if _DISTS_FN is None:
        _DISTS_FN = DISTS().to(device)
    vals = []
    for (y0, y1, x0, x1) in regions:
        pa_c = _maybe_downsample(pa_t_full[..., y0:y1, x0:x1])
        ga_c = _maybe_downsample(ga_t_full[..., y0:y1, x0:x1])
        with torch.no_grad():
            vals.append(float(_DISTS_FN(pa_c, ga_c).item()))
    out["dists"] = float(np.mean(vals)) if vals else float("nan")
```

### OP hole-awareness

**Per-region perceptual distance, averaged.** Same pattern as LPIPS.
Inputs stay in `[0, 1]` (DISTS convention).

---

## 5. `boundary_diff` — Hole-edge continuity (OP only)

### Math

Let `mb` be the boolean hole mask and `base` be the model's input
(known region + zeros in the hole). Erode the hole inward by 4 pixels
and keep the **ring** that the erosion just removed — a 4-pixel-wide
annulus *inside* the hole, hugging the known/hole boundary:

$$
\text{ring} = mb \;\wedge\; \neg\, \text{erode}_{4}(mb)
$$

The metric is the mean absolute difference between the model's output
and `base` on those ring pixels, scaled to the 8-bit pixel range:

$$
\text{boundary\_diff} =
255 \cdot \frac{1}{\lvert\text{ring}\rvert}
\sum_{i \in \text{ring}} \lvert\hat{x}_i - b_i\rvert
$$

Lower = the model's fill stitches smoothly onto the known region.
Direction: **↓ better**, range 0–255.

### Code ([`common.py:529`](../src/eval/common.py#L529))

```python
from scipy.ndimage import binary_erosion

def _ring_inside_hole(mb: np.ndarray, iterations: int = 4) -> np.ndarray:
    return mb & ~binary_erosion(mb, iterations=iterations)

# inside compute_metrics ...
ring   = _ring_inside_hole(mb, iterations=4)
ba     = np.asarray(base.convert("RGB").resize(pred.size, ...),
                    dtype=np.float32) / 255.0
sel    = ring[..., None].repeat(3, axis=2)
pa_ring = torch.from_numpy(pa[sel]).to(device)
ba_ring = torch.from_numpy(ba[sel]).to(device)
out["boundary_diff"] = float(F.l1_loss(pa_ring, ba_ring).item() * 255.0)
```

### OP hole-awareness

**Mask-defined by construction.** The metric requires both `mask` and
`base`; SR passes neither and gets NaN. The ring construction is
topology-agnostic — works for any hole shape (L, disconnected, single
quadrant).

---

## Module-level caches

The three library networks live as module-level globals in `common.py`
so they're loaded once per process and reused across thousands of
samples. Initialized lazily on first use
([`common.py:284-320`](../src/eval/common.py#L284)):

```python
_DISTS_FN = None    # DISTS_pytorch.DISTS()
_SSIM_FN  = None    # torchmetrics SSIM
_LPIPS_FN = None    # torchmetrics LPIPS (SqueezeNet)
```

`_DISTS_FN` is initialized inline inside the `dists` branch; the other
two have small `_get_*` lazy initializers that also place the module on
the requested device.

---

## Metric family split

| Family | Examples | Hole-aware mechanism |
|---|---|---|
| Per-pixel | `psnr` | boolean mask gather |
| Per-region (windowed / feature-map) | `ssim`, `lpips`, `dists` | `_hole_scoring_regions` + average |
| Task-specific | `boundary_diff` | 4-px ring inside hole edge |

## Implementation notes

- The eval pipeline assumes the OP model output has the known region
  overwritten with GT before metric computation (this happens inside
  `denoiser.generate()`). If a downstream model doesn't do this,
  `boundary_diff` and the LPIPS / DISTS hole-region scoring would need
  to be revisited.
