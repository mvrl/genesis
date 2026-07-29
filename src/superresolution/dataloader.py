import json
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_loc(loc):
    """'z_x_y' string or (z,x,y) sequence → (z, x, y) ints."""
    if isinstance(loc, str):
        z, x, y = loc.split('_')
        return int(z), int(x), int(y)
    if isinstance(loc, (list, tuple, np.ndarray)):
        z, x, y = loc
        return int(z), int(x), int(y)
    raise ValueError(f"Unsupported loc type: {type(loc)}")


class Git10MSuperResHF:
    """Thin wrapper — iterates the underlying DataLoader unchanged."""
    def __init__(self, dataloader, split="train"):
        self.loader  = dataloader
        self.split   = split
        self.dataset = dataloader.dataset

    def __len__(self):  return len(self.loader)
    def __iter__(self): yield from self.loader


# ── dataset ───────────────────────────────────────────────────────────────────

class HFParentChildDataset(Dataset):
    """
    Full-tile SR dataset: parent tile (ZL) → 4 individual child tiles (ZL+1).

    require_complete_quad=True  (val / test)
        Only parents where all 4 children are present in the split.
        children_valid is all-True — every quadrant has ground truth.

    require_complete_quad=False  (train)
        Any parent with ≥ 1 child in the split.  Missing quadrant positions are
        zero-filled in children_rgb; children_valid=False there → the engine
        computes loss only over known children.  Missing children are always
        genuinely absent from Git-10M (never withheld for eval), because the
        split builder moves parent + all 4 children into eval splits together,
        and eval-tree descendants (tiles that would form pairs with eval parents)
        are labeled 'train_excl' and excluded from splits['train'].
        The parent lookup additionally enforces same-split parents at runtime
        (parent_idx ∈ split_idx_set) as a second leakage guard.

    Quadrant ordering (QUAD_ORDER): UL, UR, LL, LR
        Index 0: UL — (qx=0, qy=0) — top-left
        Index 1: UR — (qx=1, qy=0) — top-right
        Index 2: LL — (qx=0, qy=1) — bottom-left
        Index 3: LR — (qx=1, qy=1) — bottom-right

    Batch keys
        parent_rgb      (3, S, S)    parent tile, normalised to [-1, 1]
        children_rgb    (4, 3, S, S) four child tiles in QUAD_ORDER; zeros where absent
        children_valid  (4,)         bool — True where the child tile is present
        location        str          parent tile z_x_y (for logging)
        zoom_level      float scalar target zoom level = parent_zl + 1
    """

    # UL, UR, LL, LR — index = qy*2 + qx
    QUAD_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]

    def __init__(self, hindex_pkl, indices, index_to_meta,
                 hf_name="lcybuaa/Git-10M", cache_dir=None,
                 ensure_size=256, max_pairs=None,
                 require_complete_quad=True):

        print(f"Loading {hf_name} dataset...")
        self.ds = load_dataset(hf_name, split="train", cache_dir=cache_dir)

        print(f"Loading hierarchy from {hindex_pkl}...")
        with open(hindex_pkl, 'rb') as f:
            hierarchy = pickle.load(f)

        self.ensure_size = ensure_size
        self.pairs = []

        mode = "complete quads only" if require_complete_quad else "partial quads (≥1 child)"
        print(f"Building pairs from {len(indices):,} indices ({mode})...")

        # Full location → dataset index map covering all tiles (all splits).
        # Used to resolve parent images regardless of which split they belong to.
        all_loc_to_idx = {}
        for idx_str, meta in index_to_meta.items():
            loc = meta.get('location')
            if loc:
                all_loc_to_idx[loc] = int(idx_str)

        # The HF dataset index space is contiguous by zoom level: ZL=10 tiles occupy
        # indices 0–93k, Million-AID (no geo location) tiles occupy the last 3.6M.
        # Neither group can ever be a child tile — ZL=10 is the minimum zoom in the
        # hierarchy (no parent exists), and Million-AID tiles have no location at all.
        # Filter child_loc_to_idx to only tiles whose location exists in
        # hierarchy.location_to_parent so any contiguous index slice works correctly,
        # and the pair-building loop only touches the ~2.5M tiles that can form pairs
        # instead of all ~10M train indices.
        split_idx_set = set(indices)
        has_parent_in_hierarchy = hierarchy.location_to_parent  # loc → parent loc dict
        child_loc_to_idx = {
            loc: idx
            for loc, idx in all_loc_to_idx.items()
            if idx in split_idx_set and loc in has_parent_in_hierarchy
        }
        print(f"  {len(child_loc_to_idx):,} of {len(indices):,} indices have a parent "
              f"in the hierarchy (rest are ZL-root or non-geo tiles, skipped).")

        # Group children by immediate parent; compute quadrant (qx, qy)
        # Parent (z, x, y) → child (z+1, 2x+qx, 2y+qy), qx/qy ∈ {0, 1}
        # Cross-split leakage is prevented at the source: build_quad_splits_hf.py
        # labels eval-tree descendants as 'train_excl' and excludes them from
        # splits['train'], so no eval tile can appear as a training parent.
        parent_map = {}
        for child_loc, child_idx in child_loc_to_idx.items():
            parent_loc = hierarchy.get_parent(child_loc)
            if parent_loc is None:
                continue
            parent_dataset_idx = all_loc_to_idx.get(parent_loc)
            if parent_dataset_idx is None or parent_dataset_idx not in split_idx_set:
                continue
            zp, xp, yp = _parse_loc(parent_loc)
            zc, xc, yc = _parse_loc(child_loc)
            qx = xc - 2 * xp
            qy = yc - 2 * yp
            if not (0 <= qx <= 1 and 0 <= qy <= 1):
                continue
            if parent_loc not in parent_map:
                parent_map[parent_loc] = {'parent_idx': parent_dataset_idx, 'children': {}}
            parent_map[parent_loc]['children'][(qx, qy)] = child_idx

        # Filter and build self.pairs
        n_complete = n_partial = 0
        for parent_loc, data in parent_map.items():
            children    = data['children']
            is_complete = all((qx, qy) in children for qx in range(2) for qy in range(2))

            if require_complete_quad and not is_complete:
                continue

            zp, _, _ = _parse_loc(parent_loc)
            self.pairs.append({
                'parent_loc':  parent_loc,
                'parent_idx':  data['parent_idx'],
                'parent_zl':   zp,
                'children':    children,
                'is_complete': is_complete,
            })
            if is_complete:
                n_complete += 1
            else:
                n_partial += 1

            if max_pairs and len(self.pairs) >= max_pairs:
                break

        if require_complete_quad:
            print(f"✓ {len(self.pairs):,} complete 2×2 quad pairs.")
        else:
            print(f"✓ {len(self.pairs):,} train pairs — "
                  f"{n_complete:,} complete + {n_partial:,} partial.")

    # ── image processor ──────────────────────────────────────────────────────

    def _process_img(self, pil_img):
        arr = np.array(pil_img)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = t * 2.0 - 1.0
        if t.shape[1] != self.ensure_size or t.shape[2] != self.ensure_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(self.ensure_size, self.ensure_size),
                mode="bicubic", align_corners=False,
            ).squeeze(0)
        return t

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec       = self.pairs[idx]
        tile_size = self.ensure_size

        parent_rgb     = self._process_img(self.ds[rec['parent_idx']]['image'])
        children_rgb   = torch.zeros(4, 3, tile_size, tile_size)
        children_valid = torch.zeros(4, dtype=torch.bool)

        for qi, (qx, qy) in enumerate(self.QUAD_ORDER):
            if (qx, qy) in rec['children']:
                children_rgb[qi]   = self._process_img(self.ds[rec['children'][(qx, qy)]]['image'])
                children_valid[qi] = True

        return {
            'parent_rgb':      parent_rgb,
            'children_rgb':    children_rgb,
            'children_valid':  children_valid,
            'location':        rec['parent_loc'],
            'zoom_level':      torch.tensor(float(rec['parent_zl'] + 1)),
        }


# ── __main__ sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    meta_file   = "data/git10m_quad_meta.json"
    hindex_file = "data/hierarchy_index_quad.pkl"
    data_root   = "data/hf_cache"

    print("Loading metadata...")
    with open(meta_file) as f:
        meta = json.load(f)
    index_to_meta = meta['index_to_meta']

    # Use pre-shuffled split lists if available (written by build_quad_splits_hf.py).
    # These are seeded-deterministic and ZL-balanced, so any prefix slice is safe.
    if 'splits' in meta:
        train_indices = meta['splits']['train']
        val_indices   = meta['splits']['val_random']
    else:
        print("WARNING: no 'splits' key in metadata; falling back to index_to_meta scan.")
        train_indices = sorted(int(k) for k, v in index_to_meta.items() if v.get('split') == 'train')
        val_indices   = sorted(int(k) for k, v in index_to_meta.items() if v.get('split') == 'val_random')

    # NOTE: the 50k slice below is just a pipeline smoke-test (batch shapes, dtypes).
    # Because the same-split parent restriction is now enforced, a tiny slice finds
    # very few pairs (parent tiles at ZL=N-1 are spread across the full 10M index list
    # and rarely coincide with their ZL=N children in a 50k sample). The true full
    # training pair count is reported by the "Full train pair count" probe below.
    print("\n=== Training pipeline smoke-test (50k slice) ===")
    train_ds = HFParentChildDataset(
        hindex_pkl          = hindex_file,
        indices             = train_indices[:50_000],
        index_to_meta       = index_to_meta,
        hf_name             = "lcybuaa/Git-10M",
        cache_dir           = data_root,
        ensure_size         = 256,
        require_complete_quad = False,
    )
    loader = DataLoader(train_ds, batch_size=4, num_workers=2, shuffle=True)
    batch  = next(iter(Git10MSuperResHF(loader)))
    print("parent_rgb     :", batch["parent_rgb"].shape)
    print("children_rgb   :", batch["children_rgb"].shape)
    print("children_valid :", batch["children_valid"].shape,
          " frac =", batch["children_valid"].float().mean().item())
    print("zoom_level     :", batch["zoom_level"])

    print("\n=== Validation (complete quads only) ===")
    val_ds = HFParentChildDataset(
        hindex_pkl          = hindex_file,
        indices             = val_indices,
        index_to_meta       = index_to_meta,
        hf_name             = "lcybuaa/Git-10M",
        cache_dir           = data_root,
        ensure_size         = 256,
        require_complete_quad = True,
    )
    vloader = DataLoader(val_ds, batch_size=4, num_workers=2, shuffle=False)
    vbatch  = next(iter(Git10MSuperResHF(vloader)))
    print("children_valid :", vbatch["children_valid"].shape,
          " frac =", vbatch["children_valid"].float().mean().item(), "(should be 1.0)")

    print("\n=== Full train pair count (no HF images loaded, same-split parents only) ===")
    print("Building pairs from all train indices — this takes ~30s, skips HF image loading...")
    import pickle, time
    _t0 = time.time()
    with open(hindex_file, 'rb') as _f:
        _hier = pickle.load(_f)
    _all_loc_to_idx = {v['location']: int(k) for k, v in index_to_meta.items() if v.get('location')}
    _split_idx_set  = set(train_indices)
    _has_parent     = _hier.location_to_parent
    _child_locs     = {loc: idx for loc, idx in _all_loc_to_idx.items()
                       if idx in _split_idx_set and loc in _has_parent}
    _parent_map: dict = {}
    for _cloc, _cidx in _child_locs.items():
        _ploc = _hier.get_parent(_cloc)
        if _ploc is None or _ploc not in _all_loc_to_idx:
            continue
        _pidx = _all_loc_to_idx[_ploc]
        if _pidx not in _split_idx_set:      # same-split restriction
            continue
        _zp, _xp, _yp = int(_ploc.split('_')[0]), int(_ploc.split('_')[1]), int(_ploc.split('_')[2])
        _zc, _xc, _yc = int(_cloc.split('_')[0]), int(_cloc.split('_')[1]), int(_cloc.split('_')[2])
        _qx, _qy = _xc - 2 * _xp, _yc - 2 * _yp
        if not (0 <= _qx <= 1 and 0 <= _qy <= 1):
            continue
        if _ploc not in _parent_map:
            _parent_map[_ploc] = {}
        _parent_map[_ploc][(_qx, _qy)] = _cidx
    _n_complete = sum(1 for ch in _parent_map.values()
                      if all((qx, qy) in ch for qx in range(2) for qy in range(2)))
    _n_partial   = len(_parent_map) - _n_complete
    print(f"  Total train pairs : {len(_parent_map):,}  "
          f"({_n_complete:,} complete + {_n_partial:,} partial)  "
          f"[{time.time()-_t0:.1f}s]")

    print("\n✔ Sanity check completed")
