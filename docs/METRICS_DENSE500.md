# Genesis dense500 Metrics

Metrics for the **multi-scale pyramid-completion** benchmark (the paper's dense500 results).
dense500 provides ground truth for every tile at every level, so all tiles are scored against
GT in three families: **image quality**, **semantic alignment**, and **pyramid consistency**.

> Tile-level SR/OP metrics (PSNR/SSIM/LPIPS/DISTS/B-diff) are in [`METRICS.md`](METRICS.md).

Implemented in [`eval_pyramid.py`](../src/eval/eval_pyramid.py) (quality),
[`metrics_ext.py`](../src/eval/metrics_ext.py) + [`eval_ext.py`](../src/eval/eval_ext.py)
(semantic + consistency), and [`caption_agree.py`](../src/eval/caption_agree.py) (caption).

## At a glance

| Metric | Family | Dir | Backend |
|---|---|---|---|
| PSNR | image quality | ↑ | `torchmetrics` |
| SSIM | image quality | ↑ | `torchmetrics` (11×11 gaussian) |
| LPIPS | image quality | ↓ | `torchmetrics` (SqueezeNet) |
| FID | image quality | ↓ | `torchmetrics` + `torch-fidelity` (Inception-2048) |
| CLIP-I | semantic (category) | ↑ | `open_clip` ViT-B/32 (OpenAI) |
| DINOv3-sat | semantic (instance) | ↑ | `torch.hub` DINOv3 ViT-L/16 (sat493m) |
| Caption | semantic (scene) | ↑ | Qwen2.5-VL-3B + Sentence-BERT (all-MiniLM-L6-v2) |
| LR-PSNRbox | consistency (self) | ↑ | numpy (box ↓2) |
| xLR-PSNR | consistency (cross) | ↑ | numpy |
| RAPSD | consistency (spectral) | ↓ | numpy FFT |

---

## 1. Image quality — PSNR, SSIM, LPIPS, FID

The same implementations as the tile metrics ([`METRICS.md`](METRICS.md)), applied to every
generated dense500 tile against its ground truth. PSNR / SSIM / LPIPS are full-reference,
per-tile; **FID** is distributional over Inception-2048 features of the generated-vs-real
tile sets. Higher PSNR / SSIM, lower LPIPS / FID.

---

## 2. Semantic alignment with the target

Do generated tiles depict the *same content* as the target (rather than matching
pixel-for-pixel)? Three measures at increasing strictness, all against held-out GT.

### CLIP-I — category level ↑
Cosine similarity of CLIP image embeddings (ViT-B/32, OpenAI weights), following the
image–image protocol of DreamBooth:

$$
\cos\big(\phi(\hat{x}),\, \phi(x)\big) =
\frac{\langle \phi(\hat{x}),\, \phi(x)\rangle}{\lVert\phi(\hat{x})\rVert\,\lVert\phi(x)\rVert}
$$

with $\phi$ the CLIP visual embedding. Trained on natural-image/text supervision, so it is
forgiving: it rewards depicting the same broad scene category (e.g. "farmland with roads")
and is largely invariant to the exact layout of fields, buildings, and parcels.

### DINOv3-sat — instance level ↑
CLIP saturates once the category is right, so we add a stricter, domain-specific embedding:
DINOv3 ViT-L/16 self-supervised on 493M satellite images. Its features are
instance-discriminative (trained to separate individual images, not text-aligned classes),
so the cosine between generated and target CLS tokens penalizes getting the *specific*
arrangement of ground structures wrong, not just the category. CLIP-I alongside DINOv3-sat
brackets "same kind of place" against "same place".

### Caption agreement — scene level ↑
Tests whether a human-readable description of the scene survives generation, in the
literature-correct **image-pair** form (comparing two captions, not an image to a text
prompt as in CLIPScore). Stitch the site's generated leaf tiles into one scene, do the same
for the GT leaves, caption both with a vision–language model (Qwen2.5-VL-3B-Instruct), and
embed the captions with a sentence encoder (Sentence-BERT, all-MiniLM-L6-v2):

$$
\cos\big(\psi(c(\hat{x})),\, \psi(c(x))\big)
$$

with $c(\cdot)$ the VLM caption and $\psi(\cdot)$ the sentence embedding. Because two
unrelated satellite scenes still share vocabulary ("fields", "roads"), the raw cosine has a
high floor; a **shuffled-pair** baseline (mean cosine over mismatched generated/target
captions) is reported as the chance level, and a score is meaningful only above that floor.

---

## 3. Pyramid / super-resolution consistency

A correct pyramid is *internally consistent*: on the Web-Mercator quadtree, every parent
tile should equal the 2×-downsampled mosaic of its four children (a parent pixel is the
average of the four child pixels it subtends). The quality metrics score each tile in
isolation and never test this cross-scale relationship — these do. For an internal tile $p$
with children $q$, let $M(\cdot)$ stitch the four children into a 512×512 mosaic and
$\downarrow_2$ denote 2× downsampling. Reported per quadtree level (distance from the root),
since consistency error compounds toward the leaves.

### LR-PSNRbox — intra-scale self-consistency ↑
How well a generated parent agrees with its *own* generated children (needs no GT):

$$
\mathrm{PSNR}\big(\hat{p},\; \downarrow_2 M(\hat{q})\big)
$$

Reported with **box** downsampling (area-average, the physically-correct quadtree
operator). ≳45 dB indicates a faithful
low-resolution match; low LR-PSNR means a parent and its children disagree — a seam or
content drift between zoom levels.

### xLR-PSNR — cross-scale fidelity ↑
Self-consistency can be high while parent and children are *jointly* wrong. Comparing the
generated child mosaic to the **ground-truth** child mosaic at the parent's resolution
separates consistency from correctness:

$$
\mathrm{PSNR}\big(\downarrow_2 M(\hat{q}),\; \downarrow_2 M(q)\big)
$$

Downsampling both sides removes the high-frequency detail the model must invent and isolates
whether the coarse structure is placed correctly.

### RAPSD — spectral distance ↓
Diffusion/GAN generators leave spectral signatures (a high-frequency deficit or excess vs
real imagery). RAPSD is the radially-averaged power spectral density; we report the mean
absolute log-power difference between the generated and GT child mosaics:

$$
\frac{1}{|F|}\sum_{f\in F}\big|\log S_{\hat{q}}(f) - \log S_{q}(f)\big|
$$

where $S(f)$ is the azimuthally-averaged power at radial frequency $f$. Catches
over-smoothing or texture hallucination that pixel metrics miss.

---

Seed protocols (A/B/C) and the dense500 generation config are documented in the
dense500 section of [`HYPERPARAMETERS.md`](HYPERPARAMETERS.md).
