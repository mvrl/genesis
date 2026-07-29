"""dense500 loader: site quadtrees, subtree carving, seed protocols.

The dense500 benchmark (``dense500/data/*.parquet``) stores one 256x256 tile
per row with absolute Web-Mercator coords (``zoom``, ``tile_x``, ``tile_y``) and
a ``site_id`` grouping each complete quadtree. When no local shards are present
they are auto-downloaded from the Hugging Face dataset ``dcher95/dense500``
(override with ``$GENESIS_DENSE500_REPO``). We:

1. read tile **metadata** with pyarrow (fast, no image decode) to build the
   per-site quadtree index, and
2. decode **images** lazily by row index through 🤗 ``datasets`` (row order is
   identical because both read the parquet shards in sorted filename order).

A ``Subtree`` is a complete depth-``D`` quadtree carved from a site. Seed
protocols choose which subtree tiles are revealed as known inputs.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .tiles import (
    Tile,
    ancestor_at,
    children_of,
    parent_of,
    subtree_tiles,
)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(_REPO, "dense500", "data")

# Published dense500 benchmark on the Hugging Face Hub. Used to auto-download the
# parquet shards when no local copy is present under DEFAULT_DATA_DIR.
DENSE500_HF_REPO = os.environ.get("GENESIS_DENSE500_REPO", "dcher95/dense500")


def _resolve_shards(data_dir: str) -> List[str]:
    """Return sorted dense500 parquet shard paths.

    Prefer local shards under ``data_dir`` (offline / custom builds). If none are
    present, auto-download the published benchmark from the Hugging Face dataset
    ``$GENESIS_DENSE500_REPO`` (default ``dcher95/dense500``) and use its cached
    ``data/*.parquet`` shards directly — nothing is copied into the repo.
    """
    local = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if local:
        return local
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FileNotFoundError(
            f"No parquet shards under {data_dir} and huggingface_hub is unavailable to "
            f"auto-download {DENSE500_HF_REPO}. Install huggingface_hub, or place the "
            f"dense500 parquet shards under {data_dir}."
        ) from e
    print(
        f"[dense500] no parquet under {data_dir} — downloading {DENSE500_HF_REPO} from "
        f"Hugging Face (override with $GENESIS_DENSE500_REPO)...",
        flush=True,
    )
    snap = snapshot_download(
        repo_id=DENSE500_HF_REPO, repo_type="dataset", allow_patterns=["data/*.parquet"]
    )
    shards = sorted(glob.glob(os.path.join(snap, "data", "*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"Downloaded {DENSE500_HF_REPO} but found no data/*.parquet under {snap}."
        )
    return shards

_META_COLS = [
    "location", "site_id", "tag", "continent",
    "zoom", "tile_x", "tile_y",
    "site_start_zoom", "site_leaf_zoom", "site_depth",
    "is_root", "is_leaf", "site_lat", "site_lon",
]


@dataclass
class Site:
    site_id: str
    tag: str
    continent: str
    start_zoom: int
    leaf_zoom: int
    depth: int
    lat: float
    lon: float
    root: Tile
    tiles: Dict[Tile, int] = field(default_factory=dict)  # (z,x,y) -> global row index


@dataclass
class Subtree:
    """A complete depth-``depth`` quadtree carved from ``site_id`` at ``root``."""
    site_id: str
    tag: str
    root: Tile
    depth: int
    tiles: List[Tile]          # all tiles, coarse→fine row-major
    row_index: Dict[Tile, int]  # (z,x,y) -> dataset row index (for image fetch)

    @property
    def leaf_zoom(self) -> int:
        return self.root[0] + self.depth - 1

    def leaves(self) -> List[Tile]:
        lz = self.leaf_zoom
        return [t for t in self.tiles if t[0] == lz]

    def at_zoom(self, z: int) -> List[Tile]:
        return [t for t in self.tiles if t[0] == z]


class GlobalDense:
    """Lazy reader over the global-dense parquet shards."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        # Local shards if present, else auto-download from Hugging Face.
        self.files = _resolve_shards(data_dir)
        self._ds = None  # lazy HF dataset for image decode
        self.sites: Dict[str, Site] = {}
        self._meta: Dict[str, list] = {}
        self._build_index()

    # ── metadata index (pyarrow, no image decode) ───────────────────────────
    def _build_index(self) -> None:
        import pyarrow.parquet as pq

        merged = {c: [] for c in _META_COLS}
        for f in self.files:
            t = pq.read_table(f, columns=_META_COLS).to_pydict()
            for c in _META_COLS:
                merged[c].extend(t[c])
        self._meta = merged
        n = len(merged["site_id"])

        for i in range(n):
            sid = merged["site_id"][i]
            z, x, y = int(merged["zoom"][i]), int(merged["tile_x"][i]), int(merged["tile_y"][i])
            s = self.sites.get(sid)
            if s is None:
                s = Site(
                    site_id=sid,
                    tag=merged["tag"][i],
                    continent=merged["continent"][i],
                    start_zoom=int(merged["site_start_zoom"][i]),
                    leaf_zoom=int(merged["site_leaf_zoom"][i]),
                    depth=int(merged["site_depth"][i]),
                    lat=float(merged["site_lat"][i]),
                    lon=float(merged["site_lon"][i]),
                    root=(0, 0, 0),
                )
                self.sites[sid] = s
            s.tiles[(z, x, y)] = i
            if bool(merged["is_root"][i]):
                s.root = (z, x, y)

    @property
    def dataset(self):
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset("parquet", data_files=self.files, split="train")
        return self._ds

    def image_by_row(self, row: int) -> Image.Image:
        return self.dataset[int(row)]["image"].convert("RGB")

    # ── verification ─────────────────────────────────────────────────────────
    def verify(self) -> Dict[str, object]:
        """Check every site is a complete quadtree with a unique root.

        Returns a report dict; ``ok`` is True iff no integrity problems found.
        """
        problems: List[str] = []
        n_tiles = sum(len(s.tiles) for s in self.sites.values())
        depth_hist: Dict[int, int] = {}
        tag_hist: Dict[str, int] = {}
        for sid, s in self.sites.items():
            depth_hist[s.depth] = depth_hist.get(s.depth, 0) + 1
            tag_hist[s.tag] = tag_hist.get(s.tag, 0) + 1
            present = set(s.tiles.keys())
            # root present and unique
            roots = [t for t in present if t[0] == s.start_zoom]
            if len(roots) != 1:
                problems.append(f"{sid}: expected 1 root at z={s.start_zoom}, found {len(roots)}")
            if s.root[0] != s.start_zoom:
                problems.append(f"{sid}: is_root flag at z={s.root[0]} != start_zoom {s.start_zoom}")
            # completeness: every internal node has all 4 children
            for (z, x, y) in present:
                if z < s.leaf_zoom:
                    for c in children_of((z, x, y)):
                        if c not in present:
                            problems.append(f"{sid}: missing child {c} of {(z, x, y)}")
            # zoom bounds
            zs = {t[0] for t in present}
            if min(zs) != s.start_zoom or max(zs) != s.leaf_zoom:
                problems.append(f"{sid}: zoom range {min(zs)}..{max(zs)} != {s.start_zoom}..{s.leaf_zoom}")
        return {
            "ok": len(problems) == 0,
            "n_sites": len(self.sites),
            "n_tiles": n_tiles,
            "depth_hist": dict(sorted(depth_hist.items())),
            "tag_hist": tag_hist,
            "n_problems": len(problems),
            "problems": problems[:50],
        }

    # ── subtree carving ───────────────────────────────────────────────────────
    def iter_subtrees(
        self,
        depth: Optional[int] = None,
        tags: Optional[Sequence[str]] = None,
        max_sites: Optional[int] = None,
        root_at_site_root: bool = True,
        seed: int = 0,
    ) -> List[Subtree]:
        """Carve one subtree per site.

        ``depth=None`` uses each site's full depth (whole site). Otherwise carve
        a depth-``depth`` subtree rooted at the site root (covering
        ``start_zoom .. start_zoom+depth-1``); sites shallower than ``depth`` are
        skipped. ``root_at_site_root`` is the only mode for now (deterministic,
        deepest coverage); kept as a flag for future random-root carving.
        """
        out: List[Subtree] = []
        sids = sorted(self.sites.keys())
        for sid in sids:
            s = self.sites[sid]
            if tags and s.tag not in tags:
                continue
            d = s.depth if depth is None else depth
            if s.depth < d:
                continue
            root = s.root
            tlist = subtree_tiles(root, d)
            # every carved tile must exist in the site (guaranteed for root-anchored
            # complete subtrees, but verify defensively)
            ridx = {}
            ok = True
            for t in tlist:
                ri = s.tiles.get(t)
                if ri is None:
                    ok = False
                    break
                ridx[t] = ri
            if not ok:
                continue
            out.append(Subtree(site_id=sid, tag=s.tag, root=root, depth=d, tiles=tlist, row_index=ridx))
            if max_sites and len(out) >= max_sites:
                break
        return out

    def load_gt(self, sub: Subtree) -> Dict[Tile, Image.Image]:
        """Decode all ground-truth images for a subtree, keyed by tile coord."""
        return {t: self.image_by_row(ri) for t, ri in sub.row_index.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Seed protocols — choose which subtree tiles are revealed as known inputs.
# Each protocol: (subtree, rng) -> list[Tile]. All returned tiles must be in the
# subtree. Generation must reproduce every non-seed tile.
# ─────────────────────────────────────────────────────────────────────────────

SeedProtocol = Callable[[Subtree, random.Random], List[Tile]]


def _center_leaf(sub: Subtree) -> Tile:
    leaves = sub.leaves()
    lz = sub.leaf_zoom
    n = 1 << (lz - sub.root[0])
    cx = sub.root[1] * n + n // 2
    cy = sub.root[2] * n + n // 2
    return min(leaves, key=lambda t: (t[1] - cx) ** 2 + (t[2] - cy) ** 2)


def proto_single_root(sub: Subtree, rng: random.Random) -> List[Tile]:
    return [sub.root]


def proto_single_leaf_center(sub: Subtree, rng: random.Random) -> List[Tile]:
    return [_center_leaf(sub)]


def proto_two_far_leaves(sub: Subtree, rng: random.Random) -> List[Tile]:
    """Two leaves at opposite corners — their LCA is the subtree root (max stress)."""
    leaves = sub.leaves()
    lz = sub.leaf_zoom
    n = 1 << (lz - sub.root[0])
    x0, y0 = sub.root[1] * n, sub.root[2] * n
    ul = (lz, x0, y0)
    lr = (lz, x0 + n - 1, y0 + n - 1)
    chosen = [t for t in (ul, lr) if t in set(leaves)]
    return chosen or [leaves[0], leaves[-1]]


def make_k_random_leaves(k: int) -> SeedProtocol:
    def _proto(sub: Subtree, rng: random.Random) -> List[Tile]:
        leaves = sub.leaves()
        k2 = min(k, len(leaves))
        return rng.sample(leaves, k2)
    return _proto


def proto_sparse(sub: Subtree, rng: random.Random) -> List[Tile]:
    """4 random leaf tiles — tests sparse fine-scale seeds."""
    leaves = sub.leaves()
    k = min(4, len(leaves))
    return rng.sample(leaves, k)


def proto_multiscale(sub: Subtree, rng: random.Random) -> List[Tile]:
    """One random seed at every zoom level *except* the root.

    Gives the algorithm multi-resolution hints without handing it the whole
    coarse image. Seeds at different levels are independently sampled, so they
    may or may not overlap spatially.
    """
    seeds: List[Tile] = []
    root_z = sub.root[0]
    for z in range(root_z + 1, sub.leaf_zoom + 1):
        tiles_z = sub.at_zoom(z)
        if tiles_z:
            seeds.append(rng.choice(tiles_z))
    return seeds


def proto_nested(sub: Subtree, rng: random.Random) -> List[Tile]:
    """One seed at every zoom level (except root), each spatially nested within the one above.

    Pick a random leaf, then its ancestor at each intermediate level is also a
    seed — a single column in the quadtree. Tests whether the algorithm can
    propagate a consistent multi-scale signal outward from a single spatial locus.
    """
    leaf = rng.choice(sub.leaves())
    root_z = sub.root[0]
    seeds: List[Tile] = []
    for z in range(root_z + 1, sub.leaf_zoom + 1):
        seeds.append(ancestor_at(leaf, z))
    return seeds


# ── Dense500 benchmark protocols (A / B / C) — see demos/evaluate_pyramid.md ──
# Non-root levels of a depth-4 subtree: D-2 = root+1 (2x2=4 tiles),
# D-1 = root+2 (4x4=16), D = root+3 (8x8=64, leaf). The per-subtree ``rng`` is
# seeded by the harness as Random(global_seed*1_000_003 + subtree_idx).


def proto_A_single_ratio(sub: Subtree, rng: random.Random) -> List[Tile]:
    """Protocol A — one seed, level sampled by tile-count ratio 1:4:8.

    Pick a level among {D-2, D-1, D} = {root+1, root+2, root+3} with weights
    1:4:8 (P = 1/13, 4/13, 8/13 — proportional to tile count), then one tile
    uniformly at that level. The realistic single-seed scenario.
    """
    rz = sub.root[0]
    z = rng.choices([rz + 1, rz + 2, rz + 3], weights=[1, 4, 8], k=1)[0]
    return [rng.choice(sub.at_zoom(z))]


def proto_B_triple_independent(sub: Subtree, rng: random.Random) -> List[Tile]:
    """Protocol B — three seeds, one per non-root level, none ancestor/descendant.

    s1 ∈ D-2 (4); s2 ∈ D-1 minus s1's 4 children (12); s3 ∈ D minus the 16
    descendants of s1 and the 4 children of s2 (44). Independent multi-scale
    hints at unrelated spatial loci.
    """
    rz = sub.root[0]
    z1, z2, z3 = rz + 1, rz + 2, rz + 3
    s1 = rng.choice(sub.at_zoom(z1))
    s2 = rng.choice([t for t in sub.at_zoom(z2) if ancestor_at(t, z1) != s1])
    s3 = rng.choice([t for t in sub.at_zoom(z3)
                     if ancestor_at(t, z1) != s1 and ancestor_at(t, z2) != s2])
    return [s1, s2, s3]


def proto_C_quad_mixed(sub: Subtree, rng: random.Random) -> List[Tile]:
    """Protocol C — four seeds: two at D, one at D-1, one at D-2, all independent.

    Sampled bottom-up: s1,s2 two distinct leaves; s3 ∈ D-1 minus the parents of
    s1,s2; s4 ∈ D-2 minus the ancestors of s1,s2,s3 (pool ≥ 1, always feasible).
    No pair shares an ancestor/descendant relationship.
    """
    rz = sub.root[0]
    z_d, z_dm1, z_dm2 = rz + 3, rz + 2, rz + 1
    leaves = sub.at_zoom(z_d)
    s1 = rng.choice(leaves)
    s2 = rng.choice([t for t in leaves if t != s1])
    excl1 = {ancestor_at(s1, z_dm1), ancestor_at(s2, z_dm1)}
    s3 = rng.choice([t for t in sub.at_zoom(z_dm1) if t not in excl1])
    excl2 = {ancestor_at(s1, z_dm2), ancestor_at(s2, z_dm2), ancestor_at(s3, z_dm2)}
    s4 = rng.choice([t for t in sub.at_zoom(z_dm2) if t not in excl2])
    return [s1, s2, s3, s4]


PROTOCOLS: Dict[str, SeedProtocol] = {
    "single_root": proto_single_root,
    "single_leaf_center": proto_single_leaf_center,
    "two_far_leaves": proto_two_far_leaves,
    "k2_leaves": make_k_random_leaves(2),
    "k4_leaves": make_k_random_leaves(4),
    "sparse": proto_sparse,
    "multiscale": proto_multiscale,
    "nested": proto_nested,
    # Dense500 benchmark protocols (evaluate_pyramid.md)
    "A": proto_A_single_ratio,
    "B": proto_B_triple_independent,
    "C": proto_C_quad_mixed,
    "protoA_single_ratio": proto_A_single_ratio,
    "protoB_triple_independent": proto_B_triple_independent,
    "protoC_quad_mixed": proto_C_quad_mixed,
}


def get_protocol(name: str) -> SeedProtocol:
    if name in PROTOCOLS:
        return PROTOCOLS[name]
    import re
    m = re.fullmatch(r"k(\d+)_leaves", name)
    if m:
        return make_k_random_leaves(int(m.group(1)))
    raise KeyError(f"Unknown seed protocol {name!r}. Known: {sorted(PROTOCOLS)}")
