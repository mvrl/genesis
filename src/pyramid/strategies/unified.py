"""unified: D-1 greedy outpainting → SR → free-floating seam-fix → reconcile.

Outpainting-driven pyramid filler: concentrate work at the penultimate (D-1)
level, super-resolve once to the leaf, then repair seams.

Pipeline (ctx.params: uni_overlap, uni_t, op_batch; seam knobs seam_thresh,
seam_band, seam_max):
  1. Seeds → D-1: bring each seed to the penultimate level (SR up if coarser,
     as-is at D-1, bicubic ÷2 if a leaf seed) as known 128² quadrants.
  2. Greedy outpaint the rest of D-1, most-constrained-first.
  3. One MultiDiffusion SR pass D-1 → leaf (leaf seeds anchored).
  4. Seam-fix: outpaint a band over high-discontinuity leaf boundaries, feather back.
  5. Reconcile: bicubic ÷2 so coarser levels are pixel-consistent with the leaf.
     By default (reconcile_skip_unblended=True) D-1 tiles whose leaf children were
     not seam-fixed are kept as-is rather than overwritten with ÷2(leaf).

Outpaint fills are batched (op_batch tiles/pass); only cell-disjoint windows are
ever batched, so the output is identical to filling them one-by-one.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
from PIL import Image, ImageFilter

from ..dataset import Subtree
from ..geometry import level_grid_n, stitch_level, tile_pos
from ..models import TILE_PX
from ..tiles import (
    QUAD_ORDER, Tile, TileState, TileStore, children_of, parent_of, quadrant_in_parent,
)
from .base import HALF, NEUTRAL, GenContext, Strategy, reconcile_up, register


def _greedy_fill_quads(store: TileStore, sub: Subtree, ctx: GenContext, z: int,
                       quads: Dict[Tuple[int, int], Image.Image], qn: int) -> None:
    """Greedy lateral outpainting over the ``qn × qn`` grid of 128² quads at level z.

    Sweeps overlapping 2×2 quadrant windows (each a 256² tile) most-constrained-
    first until no fillable hole remains, storing each tile once its 4 quadrants
    exist (streamed tile-by-tile to the UI).
    """
    tiles_at: Dict[Tuple[int, int], Tile] = {tile_pos(sub, t): t for t in sub.at_zoom(z)}
    done: set = set()

    def qok(qx: int, qy: int) -> bool:
        return (qx, qy) in quads

    def try_store(col: int, row: int) -> None:
        if (col, row) in done:
            return
        t = tiles_at.get((col, row))
        if t is None or store.is_seed(t):
            done.add((col, row))
            return
        if not all((2 * col + rx, 2 * row + ry) in quads for (rx, ry) in QUAD_ORDER):
            return
        canvas = Image.new("RGB", (TILE_PX, TILE_PX), NEUTRAL)
        for (rx, ry) in QUAD_ORDER:
            canvas.paste(quads[(2 * col + rx, 2 * row + ry)], (rx * HALF, ry * HALF))
        store.put(t, canvas, TileState.OUTPAINT, source="dm1")
        done.add((col, row))
        ctx.log(f"   [unified] D-1 tile {t[1]},{t[2]}")   # space-prefixed → live snapshot

    for (col, row) in list(tiles_at):       # tiles already complete from seeds
        try_store(col, row)

    op_batch = int(ctx.params.get("op_batch", 16))

    while True:
        candidates: List[Tuple[int, int, int]] = []
        for wy in range(qn - 1):
            for wx in range(qn - 1):
                cells = [(wx + dx, wy + dy) for dy in (0, 1) for dx in (0, 1)]
                k = sum(qok(*c) for c in cells)
                if 1 <= k <= 3:
                    candidates.append((k, wx, wy))
        if not candidates:
            break
        candidates.sort(reverse=True)   # most-constrained first

        # Pick a maximal set of windows that share NO quad cell: their fills are
        # mutually independent (disjoint inputs *and* outputs), so batching them in
        # one sampler pass is identical to filling them sequentially. Highest-k
        # first preserves the most-constrained-first priority. Remaining/contending
        # windows are revisited next round, when these fills have become context.
        used: set = set()
        specs: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        items: List[Tuple[Image.Image, Image.Image]] = []
        for k, wx, wy in candidates:
            cells = [(wx + dx, wy + dy) for dy in (0, 1) for dx in (0, 1)]
            if any(c in used for c in cells):
                continue
            holes = [c for c in cells if not qok(*c)]
            if not holes or len(holes) == 4:
                continue
            canvas = Image.new("RGB", (TILE_PX, TILE_PX), NEUTRAL)
            mask = np.full((TILE_PX, TILE_PX), 255, np.uint8)
            for (cqx, cqy) in cells:
                lx, ly = cqx - wx, cqy - wy
                if qok(cqx, cqy):
                    canvas.paste(quads[(cqx, cqy)], (lx * HALF, ly * HALF))
                    mask[ly * HALF:(ly + 1) * HALF, lx * HALF:(lx + 1) * HALF] = 0
            used.update(cells)
            specs.append((wx, wy, holes))
            items.append((canvas, Image.fromarray(mask, "L")))

        if not items:
            break

        # Independence guarantee: only batch windows that are mutually cell-disjoint
        # (no shared input context, no shared output quad). If this ever fails, a
        # dependent fill would have been batched — refuse rather than corrupt.
        all_cells = [(wx + dx, wy + dy)
                     for (wx, wy, _h) in specs for dy in (0, 1) for dx in (0, 1)]
        assert len(set(all_cells)) == len(all_cells) == 4 * len(specs), \
            "greedy OP batch is not cell-disjoint (dependent windows would be batched)"

        outs = ctx.bundle.op_fill_batch(items, tile_zoom=z, max_batch=op_batch)
        # Commit every batched quad first, THEN surface completed tiles, so a tile
        # whose 4 quads span two windows of this (independent) batch is only stored
        # once all of them exist.
        for (wx, wy, holes), out in zip(specs, outs):
            for (cqx, cqy) in holes:
                lx, ly = cqx - wx, cqy - wy
                quads[(cqx, cqy)] = out.crop((lx * HALF, ly * HALF, (lx + 1) * HALF, (ly + 1) * HALF))
        for (wx, wy, holes) in specs:
            for (cqx, cqy) in holes:
                try_store(cqx // 2, cqy // 2)

    for (col, row) in tiles_at:             # any stragglers
        try_store(col, row)


@register("unified")
class UnifiedStrategy(Strategy):
    """D-1 greedy outpainting → SR to leaf → free-floating seam-fix → reconcile."""

    def generate(self, store: TileStore, sub: Subtree, seeds: List[Tile], ctx: GenContext) -> None:
        overlap = int(ctx.params.get("uni_overlap", 32))
        t_start = float(ctx.params.get("uni_t", 0.5))
        skip_unblended = bool(ctx.params.get("reconcile_skip_unblended", True))

        leaf_z = sub.leaf_zoom
        dm1_z = leaf_z - 1                       # the penultimate (D-1) level
        if dm1_z < sub.root[0]:                  # depth-1 subtree: nothing to do
            return
        N = level_grid_n(sub, dm1_z)
        qn = 2 * N                               # D-1 as a quadrant grid
        ctx.log(f"[unified] {len(seeds)} seed(s); D-1=z{dm1_z} ({N}×{N}), "
                f"leaf=z{leaf_z}; overlap={overlap} t={t_start}")

        # 1. Bring every seed to the D-1 quadrant grid (SR up / as-is / downsample).
        quads: Dict[Tuple[int, int], Image.Image] = {}
        for s in seeds:
            self._seed_to_dm1_quads(store, sub, ctx, s, dm1_z, qn, quads, t_start, overlap)
        ctx.log(f"[unified] seeds → D-1: {len(quads)} known quadrant(s)")

        # 2. Greedy outpaint the rest of D-1 (streams each tile to the store live).
        _greedy_fill_quads(store, sub, ctx, dm1_z, quads, qn)
        ctx.log(f"[unified] D-1 greedy fill complete (z{dm1_z})")

        # 3. Super-resolve D-1 → leaf (leaf seeds anchored & preserved).
        self._sr_level(store, sub, ctx, dm1_z, t_start, overlap)
        ctx.log(f"[unified] SR z{dm1_z}→{leaf_z}")

        # 4. Seam-fix on the leaf mosaic (free-floating outpaint windows).
        fixed = self._seam_fix_leaf(store, sub, ctx)
        ctx.log(f"[unified] seam-fix: {len(fixed)} leaf tile(s) touched")

        # 5. Reconcile: coarser levels = bicubic ÷2 of the leaf. Default
        #    (reconcile_skip_unblended=True): keep the original D-1 (penultimate)
        #    tiles whose leaf children were NOT seam-fixed — preserves their
        #    pre-SR content instead of overwriting with ÷2(leaf).
        keep = None
        if skip_unblended:
            def keep(t: Tile) -> bool:
                return (t[0] == leaf_z - 1 and store.resolved(t)
                        and not any(c in fixed for c in children_of(t)))
        r = reconcile_up(store, sub, keep=keep)
        ctx.log(f"[unified] reconcile: {r} tile(s)" + (" (kept unblended D-1)" if skip_unblended else ""))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _seed_to_dm1_quads(self, store: TileStore, sub: Subtree, ctx: GenContext,
                           seed: Tile, dm1_z: int, qn: int,
                           quads: Dict[Tuple[int, int], Image.Image],
                           t_start: float, overlap: int) -> None:
        """Place a seed's content onto the D-1 quadrant grid."""
        z_s = seed[0]
        img = store.image(seed)
        if img is None:
            return
        img = img.convert("RGB")

        def place_tile(tile: Tile, im: Image.Image) -> None:
            col, row = tile_pos(sub, tile)
            im = im.resize((TILE_PX, TILE_PX))
            for (rx, ry) in QUAD_ORDER:
                qx, qy = 2 * col + rx, 2 * row + ry
                if 0 <= qx < qn and 0 <= qy < qn:
                    quads[(qx, qy)] = im.crop((rx * HALF, ry * HALF, (rx + 1) * HALF, (ry + 1) * HALF))

        if z_s == dm1_z:                          # already a D-1 tile
            place_tile(seed, img)
        elif z_s > dm1_z:                         # leaf seed → ÷2 downsample → one quadrant
            parent = parent_of(seed)
            col, row = tile_pos(sub, parent)
            qrx, qry = quadrant_in_parent(seed)
            qx, qy = 2 * col + qrx, 2 * row + qry
            if 0 <= qx < qn and 0 <= qy < qn:
                quads[(qx, qy)] = img.resize((HALF, HALF), Image.Resampling.BICUBIC)
        else:                                     # coarser seed → SR up to D-1
            for tile, im in self._sr_up(ctx, seed, img, dm1_z, t_start, overlap).items():
                place_tile(tile, im)

    def _sr_up(self, ctx: GenContext, tile: Tile, img: Image.Image, target_z: int,
               t_start: float, overlap: int) -> Dict[Tile, Image.Image]:
        """SR-upsample a seed up to ``target_z``; return its (contiguous) tiles.

        Each level is stitched into ONE coarse mosaic and super-resolved in a
        single MultiDiffusion ``sr_level_unified`` call, so the seed's footprint
        is upscaled jointly (cross-window blend) rather than tile-by-tile.
        """
        cur: Dict[Tile, Image.Image] = {tile: img}
        z = tile[0]
        while z < target_z:
            # The footprint is a contiguous G×G block; stitch it into one mosaic.
            xs = [t[1] for t in cur]
            ys = [t[2] for t in cur]
            x0, y0, G = min(xs), min(ys), max(xs) - min(xs) + 1
            coarse = Image.new("RGB", (G * TILE_PX, G * TILE_PX))
            for (tz, tx, ty), im in cur.items():
                coarse.paste(im.convert("RGB").resize((TILE_PX, TILE_PX)),
                             ((tx - x0) * TILE_PX, (ty - y0) * TILE_PX))
            mosaic = ctx.bundle.sr_level_unified(coarse, z + 1, t_start, overlap_px=overlap)
            cx0, cy0 = 2 * x0, 2 * y0          # child-grid origin
            nxt: Dict[Tile, Image.Image] = {}
            for t in cur:
                for c in children_of(t):
                    bx, by = (c[1] - cx0) * TILE_PX, (c[2] - cy0) * TILE_PX
                    nxt[c] = mosaic.crop((bx, by, bx + TILE_PX, by + TILE_PX))
            cur = nxt
            z += 1
        return cur

    def _sr_level(self, store: TileStore, sub: Subtree, ctx: GenContext,
                  z: int, t_start: float, overlap: int) -> None:
        """One SR pass: full level z → z+1 (seeds at the child level anchored)."""
        child_z = z + 1
        imgs = {t: store.image(t) for t in sub.at_zoom(z) if store.resolved(t)}
        if not imgs:
            return
        coarse = stitch_level(imgs, sub, z, tile_px=TILE_PX)
        seed_pils = {tile_pos(sub, t): store.image(t)
                     for t in sub.at_zoom(child_z) if store.is_seed(t)}
        mosaic = ctx.bundle.sr_level_unified(
            coarse, child_z, t_start, seeds=seed_pils or None, overlap_px=overlap)
        for t in sub.at_zoom(child_z):
            if store.is_seed(t):
                continue
            col, row = tile_pos(sub, t)
            box = (col * TILE_PX, row * TILE_PX, (col + 1) * TILE_PX, (row + 1) * TILE_PX)
            store.put(t, mosaic.crop(box), TileState.SR, source=f"sr(t={t_start:.2f})")
            ctx.log(f"   [unified] leaf tile {t[1]},{t[2]}")   # space-prefixed → live snapshot

    def _seam_fix_leaf(self, store: TileStore, sub: Subtree, ctx: GenContext) -> Set[Tile]:
        """Detect high-discontinuity tile boundaries in the leaf and OP-blend them.

        Windows are centered on the seam (free-floating, not tile-aligned); a band
        over the seam is the OP hole; the result is feathered back into the mosaic.
        Returns the set of leaf tiles whose pixels were modified by the seam-fix.
        """
        leaf_z = sub.leaf_zoom
        n = level_grid_n(sub, leaf_z)
        if n < 2:
            return set()
        thresh = float(ctx.params.get("seam_thresh", 14.0))
        band = int(ctx.params.get("seam_band", 64))
        max_fix = int(ctx.params.get("seam_max", 64))
        S = TILE_PX

        tiles = {tile_pos(sub, t): t for t in sub.at_zoom(leaf_z)}
        imgs = {t: store.image(t) for t in sub.at_zoom(leaf_z) if store.resolved(t)}
        if len(imgs) < n * n:
            return set()
        mosaic = stitch_level(imgs, sub, leaf_z, tile_px=S)
        arr = np.asarray(mosaic.convert("RGB"), dtype=np.float32)
        side = n * S

        # Score internal tile boundaries by mean abs discontinuity across them.
        seams: List[Tuple[float, str, int, int]] = []
        for c in range(1, n):
            x = c * S
            for r in range(n):
                y0 = r * S
                d = float(np.abs(arr[y0:y0 + S, x - 1, :] - arr[y0:y0 + S, x, :]).mean())
                if d > thresh:
                    seams.append((d, "v", x, y0 + S // 2))
        for r in range(1, n):
            y = r * S
            for c in range(n):
                x0 = c * S
                d = float(np.abs(arr[y - 1, x0:x0 + S, :] - arr[y, x0:x0 + S, :]).mean())
                if d > thresh:
                    seams.append((d, "h", x0 + S // 2, y))
        seams.sort(reverse=True)   # worst first
        op_batch = int(ctx.params.get("op_batch", 16))

        # Free-floating seam windows; clamp each to the mosaic. Two windows whose
        # S×S rectangles don't overlap are independent (disjoint pixels), so they
        # batch in one sampler pass. Overlapping windows must run in separate rounds
        # (a later feather reads the earlier paste). Worst-first order is preserved.
        pending: List[Tuple[int, int, str, int, int]] = []
        for _, kind, cx, cy in seams[:max_fix]:
            x0 = int(np.clip(cx - S // 2, 0, side - S))
            y0 = int(np.clip(cy - S // 2, 0, side - S))
            pending.append((x0, y0, kind, cx, cy))

        def _overlaps(a, b) -> bool:
            ax, ay, bx, by = a[0], a[1], b[0], b[1]
            return not (ax + S <= bx or bx + S <= ax or ay + S <= by or by + S <= ay)

        fixed: Set[Tile] = set()
        while pending:
            used: List[Tuple[int, int, str, int, int]] = []
            leftover: List[Tuple[int, int, str, int, int]] = []
            for sp in pending:
                if any(_overlaps(sp, u) for u in used):
                    leftover.append(sp)
                else:
                    used.append(sp)
            # Crop all windows up-front (disjoint → safe before any paste), batch-fill.
            items = []
            for (x0, y0, kind, cx, cy) in used:
                mask = np.zeros((S, S), np.uint8)
                if kind == "v":
                    mx = cx - x0
                    mask[:, max(0, mx - band // 2):min(S, mx + band // 2)] = 255
                else:
                    my = cy - y0
                    mask[max(0, my - band // 2):min(S, my + band // 2), :] = 255
                items.append((mosaic.crop((x0, y0, x0 + S, y0 + S)), Image.fromarray(mask, "L"), mask))
            # Independence guarantee: a seam batch is only pixel-disjoint windows, so
            # no fill depends on another's paste. Overlapping ones went to `leftover`
            # and run in a later (sequential) round.
            assert all(not _overlaps(used[i], used[j])
                       for i in range(len(used)) for j in range(i + 1, len(used))), \
                "seam-fix batch contains overlapping (dependent) windows"
            outs = ctx.bundle.op_fill_batch([(w, m) for (w, m, _) in items],
                                            tile_zoom=leaf_z, max_batch=op_batch)
            for (x0, y0, kind, cx, cy), (win, _mp, mask), out in zip(used, items, outs):
                # Feather the regenerated band back over the (frozen) known surroundings.
                soft = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(band / 6.0)),
                                  dtype=np.float32)[:, :, None] / 255.0
                blended = (soft * np.asarray(out.convert("RGB"), np.float32)
                           + (1 - soft) * np.asarray(win, np.float32)).clip(0, 255).astype(np.uint8)
                mosaic.paste(Image.fromarray(blended), (x0, y0))
                # Re-store the leaf tiles this window overlaps (live snapshot per fix).
                for col in range(x0 // S, (x0 + S - 1) // S + 1):
                    for row in range(y0 // S, (y0 + S - 1) // S + 1):
                        t = tiles.get((col, row))
                        if t is None or store.is_seed(t):
                            continue
                        box = (col * S, row * S, (col + 1) * S, (row + 1) * S)
                        store.put(t, mosaic.crop(box), TileState.OUTPAINT, source="seamfix")
                        fixed.add(t)
                ctx.log(f"   [unified] seam-fix {kind} @ ({x0},{y0})")
            pending = leftover
        return fixed
