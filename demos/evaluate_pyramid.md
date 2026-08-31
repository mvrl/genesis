# Genesis Pyramid — Evaluation Protocol

End-to-end evaluation of the `unified` pyramid-generation strategy across zoom windows and seed configurations.

---

## 1. Benchmark Dataset

Each benchmark **sample** is a complete depth-4 quadtree (85 tiles) with all tiles ground-truthed from Mapbox satellite imagery. Tile size is 256×256 RGB.

### Source dataset

A purpose-built benchmark dataset with **100 sites per zoom window × 5 windows = 500 sites globally**. Each site is a complete depth-4 quadtree (85 tiles) with all tiles sourced from ESRI/Mapbox satellite imagery. Sites are geographically distributed across continents to avoid regional bias. Each location is identified by its root tile coordinate `(l, x, y)` in the Web-Mercator slippy-map scheme.

Sites are loaded via the `GlobalDense` class in `src/pyramid/dataset.py`; subtrees are carved with `iter_subtrees(depth=4)`.

### Zoom windows

**100 sites per window**, five windows total:

| Window | l  | h = l+3 | Root tile covers | Sites |
|--------|----|---------|-----------------|-------|
| W1     | 11 | 14      | ~18 km across   | 100    |
| W2     | 12 | 15      | ~9 km across   | 100    |
| W3     | 13 | 16      | ~4 km across   | 100    |
| W4     | 14 | 17      | ~2 km across   | 100    |
| W5     | 15 | 18      | ~1 km across    | 100    |
|        |    |         | **Total**       | **500** |

### Pyramid structure (per sample)

```
Depth  Zoom   Grid     Tile count
  0     l      1×1         1       ← root
  1    l+1     2×2         4
  2    l+2     4×4        16       ← D-1 (penultimate)
  3     h      8×8        64       ← D   (leaf)
                        ────
                          85  tiles total
```

---

## 2. Seed Protocols

Three protocols are evaluated on every subtree. Seeds are always drawn from the ground-truth imagery of that subtree (i.e., the model never sees the GT tiles it must generate).

### Protocol A — Single seed, level sampled by ratio

One seed tile is placed per subtree. The **zoom level** of the seed is sampled stochastically according to the ratio **1 : 4 : 8**:

| Level | Zoom  | Grid            | Weight | P(chosen)     | Interpretation |
|-------|-------|-----------------|--------|---------------|----------------|
| D-2   | z=l+1 | 2×2 = 4 tiles   | 1      | 1/13 ≈ 7.7 %  | Coarse anchor; most generation done by OP |
| D-1   | z=l+2 | 4×4 = 16 tiles  | 4      | 4/13 ≈ 30.8 % | Penultimate anchor; seed lands directly on the quadrant grid |
| D     | z=h   | 8×8 = 64 tiles  | 8      | 8/13 ≈ 61.5 % | Leaf anchor; most realistic single-seed scenario |

The level weights are fixed at 1:4:8, favoring finer levels (roughly tracking tile count), so the evaluation distribution reflects how users are likely to place seeds in practice.

Once a level is chosen, one tile is selected **uniformly at random** from the `n×n` grid at that level.

**Per-run breakdown:** Because the level is stochastic, results are reported both as a pooled mean across all 100 subtrees and stratified by which level was actually drawn (D-2 / D-1 / D).

RNG is seeded per subtree as `rng = Random(global_seed × 1_000_003 + subtree_idx)` for full reproducibility.

---

### Protocol B — Three seeds, one per non-root level, no ancestor–descendant relationship

Three seed tiles are placed per subtree — exactly one at each non-root zoom level. The seeds are drawn so that **no tile is an ancestor or descendant of another**, guaranteeing they carry spatially independent information.

| Seed | Zoom  | Valid pool (after exclusions) |
|------|-------|-------------------------------|
| s₁   | z=l+1 | All 4 tiles in the 2×2 grid |
| s₂   | z=l+2 | 16 − 4 = **12 tiles**: exclude the 4 children of s₁ |
| s₃   | z=h   | 64 − 16 − 4 = **44 tiles**: exclude the 16 grandchildren of s₁ and the 4 children of s₂ (these two exclusion sets are disjoint because s₂ is not under s₁) |

**Sampling procedure:**
1. Draw s₁ uniformly from all 4 tiles at z=l+1.
2. Draw s₂ uniformly from the 12 tiles at z=l+2 that are **not** children of s₁.
3. Draw s₃ uniformly from the 44 tiles at z=h that are **not** descendants of s₁ or s₂.

This ensures no parent–child or grandparent–grandchild relationship exists among the three seeds. Each seed is an independent hint at a different resolution, testing whether the strategy can reconcile multi-scale anchors that point to unrelated spatial regions.

---

### Protocol C — Four seeds: two at D, one at D-1, one at D-2, all spatially independent

Four seed tiles are placed per subtree — **two at the leaf level (D)**, one at D-1, and one at D-2. No root tile is ever used as a seed. **No pair shares an ancestor–descendant relationship.**

Sampling proceeds bottom-up (leaf → D-1 → D-2) so the leaf-level anchors are placed first and coarser seeds are constrained to lie outside their ancestor chains.

| Seed | Level | Zoom  | Valid pool (after exclusions) |
|------|-------|-------|-------------------------------|
| s₁   | D     | z=h   | All **64** leaf tiles |
| s₂   | D     | z=h   | **63** tiles: exclude s₁ |
| s₃   | D-1   | z=l+2 | **≥ 14** tiles: exclude the D-1 parents of s₁ and s₂ (1 parent each; pool = 14 if parents differ, 15 if s₁ and s₂ share a parent — probability ≈ 5%) |
| s₄   | D-2   | z=l+1 | **1–3** tiles: exclude the D-2 ancestors of s₁, s₂, and s₃ (at most 3 distinct tiles out of 4; pool ≥ 1, always feasible) |

**Sampling procedure:**
1. Draw s₁ uniformly from all 64 leaf tiles at z=h.
2. Draw s₂ uniformly from the remaining 63 leaf tiles (exclude s₁).
3. Draw s₃ uniformly from the D-1 tiles at z=l+2 that are **not** the parent of s₁ or s₂.
4. Draw s₄ uniformly from the D-2 tiles at z=l+1 that are **not** a D-2 ancestor of s₁, s₂, or s₃.

s₁ and s₂ are at the same zoom level so they cannot be ancestors or descendants of each other — they only need to be distinct. The pool for s₄ is always non-empty because there are 4 D-2 tiles and at most 3 can be excluded.

---

## 3. Evaluation Matrix

```
             W1      W2      W3      W4      W5
           l=11    l=12    l=13    l=14    l=15
         ────────────────────────────────────────
Proto A  │  100    │  100    │  100    │  100    │  100    │  →   500 runs
Proto B  │  100    │  100    │  100    │  100    │  100    │  →   500 runs
Proto C  │  100    │  100    │  100    │  100    │  100    │  →   500 runs
         ────────────────────────────────────────
Total:    1 500 pyramid generations  (85 tiles each = 127 500 tiles)
```

Each cell (one protocol × one window) is run as a separate process and writes its own results directory.

---

## 4. Generation Settings

| Parameter | Value |
|-----------|-------|
| Strategy | `unified` |
| Architecture | JiT-H/16 |
| CFG scale | 2.5 |
| Sampling steps | 100 |
| SR window overlap | 32 px |
| SR warm-start t | 0.40 |
| Reconcile: keep unblended D-1 tiles | **on** (default) |
