import json
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from typing import Dict, List


def load_all_splits(meta_file: str) -> Dict[str, List[int]]:
    """Return per-split HF index lists.

    Prefers the pre-shuffled 'splits' key written by build_quad_splits_hf.py
    (seeded-deterministic, ZL-balanced).  Falls back to scanning index_to_meta
    for older metadata files that pre-date the 'splits' key.
    """
    with open(meta_file, 'r') as f:
        data = json.load(f)
    if 'splits' in data:
        return {k: list(v) for k, v in data['splits'].items()}
    # Fallback: derive from index_to_meta (ZL-contiguous sorted order)
    print("WARNING: metadata has no 'splits' key; falling back to index_to_meta scan. "
          "Re-run build_quad_splits_hf.py to generate ZL-balanced shuffled split lists.")
    index_to_meta = data['index_to_meta']
    splits: Dict[str, List[int]] = {
        'train': [], 'val_random': [], 'val_spatial': [], 'random_test': [], 'spatial_test': []
    }
    for idx_str, meta in index_to_meta.items():
        s = meta.get('split')
        if s in splits:
            splits[s].append(int(idx_str))
    for s in splits:
        splits[s] = sorted(splits[s])
    return splits


class Git10MOutpaintingHF:
    """Thin wrapper — iterates the underlying dataloader unchanged."""
    def __init__(self, dataloader: DataLoader, split: str = "train"):
        self.loader = dataloader
        self.split = split
        self.dataset = dataloader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        yield from self.loader


_QUADS = ("UL", "UR", "LL", "LR")
# Adjacent pairs: top, bottom, left, right.
_ADJ_PAIRS = (("UL", "UR"), ("LL", "LR"), ("UL", "LL"), ("UR", "LR"))
# Diagonal pairs: main and anti.
_DIAG_PAIRS = (("UL", "LR"), ("UR", "LL"))


def _mark_known_quadrants(mask: torch.Tensor, known: tuple, H: int, W: int) -> None:
    """In-place: set each named quadrant in `known` to 0 (rest stays at 1 = hole)."""
    h2, w2 = H // 2, W // 2
    for q in known:
        if   q == "UL": mask[:, :h2, :w2] = 0.0
        elif q == "UR": mask[:, :h2, w2:] = 0.0
        elif q == "LL": mask[:, h2:, :w2] = 0.0
        else:           mask[:, h2:, w2:] = 0.0  # LR


def _sample_quad_regime(stage: int) -> str:
    """Return the regime name for the given curriculum stage."""
    if stage == 0:
        return "quad3"
    if stage == 1:
        return "quad2_adj" if torch.rand(1).item() < 0.5 else "quad2_diag"
    if stage == 2:
        return "quad1"
    # stage 3 — uniform mix across all 4 regimes (consolidation matches eval distribution)
    return ("quad1", "quad2_adj", "quad2_diag", "quad3")[torch.randint(0, 4, (1,)).item()]


def _sample_known_for_regime(regime: str) -> tuple:
    """Return the tuple of known quadrant names for a regime."""
    if regime == "quad1":
        return (_QUADS[torch.randint(0, 4, (1,)).item()],)
    if regime == "quad2_adj":
        return _ADJ_PAIRS[torch.randint(0, 4, (1,)).item()]
    if regime == "quad2_diag":
        return _DIAG_PAIRS[torch.randint(0, 2, (1,)).item()]
    if regime == "quad3":
        hole = _QUADS[torch.randint(0, 4, (1,)).item()]
        return tuple(q for q in _QUADS if q != hole)
    raise ValueError(f"unknown regime: {regime}")


class Git10MOutpaintingDataset(Dataset):
    """
    Dataset for Git10M outpainting. Loads images and generates quadrant-aligned
    masks on-the-fly. The curriculum is fully quadrant-aligned and pace easy →
    hard by known-count:

        0: quad3                                          (3 known, 25% hole)
        1: 50/50 quad2_adj / quad2_diag                   (2 known, 50% hole)
        2: quad1                                          (1 known, 75% hole)
        3: uniform across {quad1, quad2_adj, quad2_diag, quad3}

    Batch keys:
        image       : (3, H, W)  — normalised to [-1, 1]
        roi_mask    : (1, H, W)  — 1=hole, 0=known
        zoom_level  : scalar float tensor — tile zoom level from
                      ``index_to_meta['zoom']``; 0.0 when location could not be
                      parsed (rare) — used as GSD proxy. Same dataset split as
                      SR so zoom is available for all samples.
    """
    def __init__(
        self,
        hf_dataset,
        indices: List[int],
        index_to_meta: Dict[str, Dict],
        img_size: int = 256,
        split: str = "train",
        enable_curriculum: bool = False,
        shared_stage=None,
    ):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.index_to_meta = index_to_meta
        self.img_size = img_size
        self.split = split
        self.stage = 0
        self.enable_curriculum = enable_curriculum
        self.shared_stage = shared_stage

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        hf_idx = self.indices[idx]
        item = self.hf_dataset[hf_idx]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = self.transform(image)
        if not torch.isfinite(image).all():
            image = torch.zeros_like(image)

        C, H, W = image.shape
        mask = torch.zeros((1, H, W))

        # Quadrant-aligned curriculum — see class docstring for the stage map.
        # When curriculum is disabled (val/test loaders), we sample uniformly
        # from all 4 quadrant regimes (matches the eval-time mix).
        cur_stage = int(self.shared_stage.value) if self.shared_stage is not None else self.stage
        if not self.enable_curriculum:
            cur_stage = 3
        regime = _sample_quad_regime(cur_stage)
        known = _sample_known_for_regime(regime)
        mask[:] = 1.0
        _mark_known_quadrants(mask, known, H, W)

        meta = self.index_to_meta.get(str(hf_idx), {})
        zoom = float(meta.get('zoom', 0))   # 0 = unknown / location parse failure
        return {"image": image, "roi_mask": mask, "zoom_level": torch.tensor(zoom)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "image":      torch.stack([item["image"]      for item in batch]),
        "roi_mask":   torch.stack([item["roi_mask"]   for item in batch]),
        "zoom_level": torch.stack([item["zoom_level"] for item in batch]),
    }


def build_outpainting_dataloader(
    hf_dataset,
    indices: List[int],
    index_to_meta: Dict[str, Dict],
    img_size: int,
    batch_size: int,
    num_workers: int,
    split: str = "train",
    shuffle: bool = True,
    enable_curriculum: bool = False,
    shared_stage=None,
):
    dataset = Git10MOutpaintingDataset(
        hf_dataset=hf_dataset,
        indices=indices,
        index_to_meta=index_to_meta,
        img_size=img_size,
        split=split,
        enable_curriculum=enable_curriculum,
        shared_stage=shared_stage,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False,
    )


if __name__ == "__main__":
    data_root = "data/hf_cache"
    meta_file = "data/git10m_quad_meta.json"

    print("Loading HuggingFace dataset...")
    hf_dataset = load_dataset("lcybuaa/Git-10M", cache_dir=data_root, split="train")
    print(f"Dataset size: {len(hf_dataset):,}")

    with open(meta_file) as f:
        index_to_meta = json.load(f)['index_to_meta']

    splits = load_all_splits(meta_file)
    loader = build_outpainting_dataloader(
        hf_dataset=hf_dataset,
        indices=splits['train'][:100],
        index_to_meta=index_to_meta,
        img_size=256, batch_size=4, num_workers=2, split="train",
    )
    batch = next(iter(Git10MOutpaintingHF(loader)))
    print("Image:", batch["image"].shape)
    print("Mask: ", batch["roi_mask"].shape)
    print("\nSanity check completed ✔")
