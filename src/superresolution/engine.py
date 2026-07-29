import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
from pytorch_lightning.loggers import WandbLogger
from contextlib import contextmanager
import wandb
from model_sr import JiT_models, get_dino_embeddings
import copy


class DenoiserSR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = JiT_models[args.model](
            output_size=getattr(args, "output_size", 512),
            use_dino=getattr(args, "use_dino", True),
        )
        self.P_mean, self.P_std = args.P_mean, args.P_std
        self.t_eps, self.noise_scale = args.t_eps, args.noise_scale
        self.ema_decay1, self.ema_decay2 = args.ema_decay1, args.ema_decay2
        self.ema_params1, self.ema_params2 = None, None
        self.steps, self.cfg_scale = args.num_sampling_steps, args.cfg

        # Optional LPIPS perceptual loss (disabled when lpips_weight=0)
        self.lpips_weight = getattr(args, "lpips_weight", 0.0)
        if self.lpips_weight > 0:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex')
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
            except ImportError:
                raise ImportError(
                    "lpips package not found. Install with: pip install lpips"
                )
        else:
            self.lpips_fn = None

    def sample_t(self, n: int, device=None):
        return torch.sigmoid(torch.randn(n, device=device) * self.P_std + self.P_mean)

    def forward(self, batch):
        # children_rgb:   (B, 4, 3, S, S)  — individual child tiles; zeros where absent
        # children_valid: (B, 4)            — bool, True where the child tile is present
        # Quadrant order (QUAD_ORDER): UL(0,0), UR(1,0), LL(0,1), LR(1,1)
        children_rgb   = batch["children_rgb"]
        children_valid = batch["children_valid"]
        parent_img     = batch["parent_rgb"]

        B         = children_rgb.shape[0]
        S         = children_rgb.shape[-1]   # tile size (256)
        device    = children_rgb.device

        # Reconstruct full 512×512 x1 from the four child tiles (zeros for missing).
        # children_rgb[:, 0..3] maps to [(qx=0,qy=0),(qx=1,qy=0),(qx=0,qy=1),(qx=1,qy=1)]
        # Reshaping to (B, qy=2, qx=2, C, S, S) and permuting to (B, C, qy, S, qx, S)
        # then viewing as (B, C, 2S, 2S) correctly tiles the quadrants.
        x1 = (children_rgb
              .view(B, 2, 2, 3, S, S)
              .permute(0, 3, 1, 4, 2, 5)
              .contiguous()
              .view(B, 3, 2 * S, 2 * S))

        t  = self.sample_t(B, device=device).view(-1, 1, 1, 1)
        x0 = torch.randn_like(x1) * self.noise_scale
        zt = t * x1 + (1 - t) * x0

        # Pixel parent is always the base conditioning
        conditions = {"parent_img": parent_img}
        # DINO features are an optional additional signal
        if self.net.use_dino:
            with torch.no_grad():
                dino_feats = get_dino_embeddings(parent_img, weights_path=self.args.dino_weights)
            conditions["dino_feats"] = dino_feats
        # Zoom-level / GSD conditioning — 0 = unknown, passed through as-is
        if "zoom_level" in batch:
            conditions["zoom_level"] = batch["zoom_level"].float().to(device)
        x_pred = self.net(zt, t.flatten(), conditions=conditions)

        # Build per-pixel mask (B, 1, 2S, 2S) from per-quadrant validity flags.
        # Each flag broadcasts to cover its S×S region.
        v = children_valid.float().to(device)   # (B, 4)
        mask = torch.zeros(B, 1, 2 * S, 2 * S, device=device)
        mask[:, :, 0:S,    0:S   ] = v[:, 0].view(B, 1, 1, 1)   # UL
        mask[:, :, 0:S,    S:2*S ] = v[:, 1].view(B, 1, 1, 1)   # UR
        mask[:, :, S:2*S,  0:S   ] = v[:, 2].view(B, 1, 1, 1)   # LL
        mask[:, :, S:2*S,  S:2*S ] = v[:, 3].view(B, 1, 1, 1)   # LR

        denom  = (1.0 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - zt) / denom
        v_tgt  = (x1    - zt) / denom
        sq     = (v_pred - v_tgt) ** 2

        main_loss = (sq * mask).sum() / mask.sum().clamp_min(1.0)

        if self.lpips_fn is None:
            return main_loss, {}

        # LPIPS perceptual loss — crop each quadrant from x_pred and compare
        # directly with the individual child tile GT.  This avoids passing
        # zero-filled masked composites through AlexNet which corrupts the
        # perceptual similarity signal.  Invalid quadrants are excluded by
        # zeroing their weight; no gradient flows through those crops.
        # Note: lpips_fn is guaranteed to be on the correct device by on_fit_start;
        # no device check here to avoid torch.compile graph breaks.
        pred_crops = torch.stack([
            x_pred[:, :, 0:S,   0:S  ],   # UL
            x_pred[:, :, 0:S,   S:2*S],   # UR
            x_pred[:, :, S:2*S, 0:S  ],   # LL
            x_pred[:, :, S:2*S, S:2*S],   # LR
        ], dim=1)  # (B, 4, 3, S, S)

        pred_flat  = pred_crops.reshape(B * 4, 3, S, S)
        gt_flat    = children_rgb.to(device).reshape(B * 4, 3, S, S)
        valid_flat = v.reshape(B * 4)                                  # (B*4,)
        t_flat     = t.view(B).unsqueeze(1).expand(B, 4).reshape(B * 4)

        lpips_vals = self.lpips_fn(
            pred_flat.clamp(-1, 1), gt_flat.clamp(-1, 1)
        ).view(B * 4)

        # Weight = t × valid; invalid tiles contribute zero loss and zero gradient.
        weights    = t_flat * valid_flat
        lpips_loss = (weights * lpips_vals).sum() / weights.sum().clamp_min(1e-6)

        total_loss = main_loss + self.lpips_weight * lpips_loss
        return total_loss, {"lpips_loss": lpips_loss.detach(), "main_loss": main_loss.detach()}

    @torch.no_grad()
    def generate(self, batch):
        parent_img = batch["parent_rgb"]

        # Pixel parent is always the base conditioning
        conditions = {"parent_img": parent_img}
        # DINO features are an optional additional signal
        if self.net.use_dino:
            conditions["dino_feats"] = get_dino_embeddings(parent_img, weights_path=self.args.dino_weights)
        # Zoom-level / GSD conditioning — 0 = unknown
        if "zoom_level" in batch:
            conditions["zoom_level"] = batch["zoom_level"].float().to(parent_img.device)

        output_size = self.net.output_size  # 512
        t_start = 0.0

        if getattr(self.args, "bicubic_init", False):
            # SDEdit-style warm start: bicubic-upsample full parent 256×256 → 512×512
            t_start = float(getattr(self.args, "bicubic_init_t_start", 0.5))
            bicubic_up = F.interpolate(
                parent_img, size=(output_size, output_size), mode="bicubic", align_corners=False
            )
            noise = self.noise_scale * torch.randn_like(bicubic_up)
            z = t_start * bicubic_up + (1.0 - t_start) * noise
        else:
            z = self.noise_scale * torch.randn(
                parent_img.size(0), 3, output_size, output_size,
                device=parent_img.device,
            )
        timesteps = torch.linspace(t_start, 1.0, self.steps+1, device=parent_img.device).view(-1, 1, 1, 1)
        dt = (1.0 - t_start) / self.steps
        for i in range(self.steps):
            t = timesteps[i].expand(z.size(0), -1, -1, -1)
            
            # 1. Conditional Pass
            x_cond = self.net(z, t.flatten(), conditions=conditions)
            v_cond = (x_cond - z) / (1.0 - t + 1e-5)

            # 2. Unconditional Pass (CFG) — drop all conditions incl. zoom
            if self.cfg_scale > 1.0:
                 uncond_conditions = {"dino_feats": None, "parent_img": None, "zoom_level": None}
                 x_uncond = self.net(z, t.flatten(), conditions=uncond_conditions)
                 v_uncond = (x_uncond - z) / (1.0 - t + 1e-5)
                
                 v_pred = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                 v_pred = v_cond
            
            # Integration Step (Euler)
            z = z + dt * v_pred
        return z



    def update_ema(self):
        src = [p for p in self.parameters() if p.requires_grad]
        for t1, t2, s in zip(self.ema_params1, self.ema_params2, src):
            t1.detach().mul_(self.ema_decay1).add_(s, alpha=1 - self.ema_decay1)
            t2.detach().mul_(self.ema_decay2).add_(s, alpha=1 - self.ema_decay2)

class ZLCurriculumSampler(torch.utils.data.Sampler):
    """Step-aware ZL curriculum sampler for super-resolution.

    Reads shared_stage at the start of every epoch (__iter__) to compute
    fresh per-sample probabilities. shared_stage is updated every training
    step by LitDenoiserSR.on_train_batch_start — so transitions are
    accurate to within one epoch length (~1000 steps by default).

    Stage 0: 100% easy (ZL ≤ zl_easy_max)
    Stage 1:  50% easy / 50% hard
    Stage 2:  uniform ZL (natural dataset proportions, ~81% hard), all pairs
    Stage 3:  uniform ZL, complete quads only (all 4 children present)
    """
    def __init__(self, zls, shared_stage, zl_easy_max, num_samples, complete_flags=None):
        import numpy as np
        self.zls = np.asarray(zls)
        self.shared_stage = shared_stage
        self.zl_easy_max = zl_easy_max
        self.num_samples = num_samples
        if len(self.zls) == 0:
            raise ValueError(
                "ZLCurriculumSampler received an empty training set. "
                "Check that splits['train'] contains geo-tagged tiles with parents in train."
            )
        self.easy_idx = np.where(self.zls <= zl_easy_max)[0]
        self.hard_idx = np.where(self.zls >  zl_easy_max)[0]
        self.all_idx  = np.arange(len(self.zls))

        # Stage-3 pool: complete quads only, uniform ZL within that subset
        if complete_flags is not None:
            complete = np.asarray(complete_flags, dtype=bool)
        else:
            complete = np.ones(len(self.zls), dtype=bool)
        self.complete_idx = np.where(complete)[0]

        if len(self.easy_idx) == 0:
            print(f"[ZLCurriculumSampler] WARNING: no easy tiles (ZL ≤ {zl_easy_max}); "
                  f"Stage 0/1 will fall back to all_idx.")
        if len(self.complete_idx) == 0:
            print("[ZLCurriculumSampler] WARNING: no complete quads; "
                  "Stage 3 will fall back to all_idx.")

    def __iter__(self):
        import numpy as np
        stage = int(self.shared_stage.value)
        # Each DDP rank must produce different indices.  Lightning seeds all ranks
        # identically, so global np.random.choice would give every GPU the same
        # curriculum batch.  Instead build a local RNG seeded by torch.initial_seed()
        # (which Lightning advances per step) offset by the process rank.
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        rng = np.random.default_rng(seed=(int(torch.initial_seed()) + rank) & 0xFFFFFFFFFFFFFFFF)
        if stage >= 3:
            # Uniform ZL sampling restricted to complete quads (all 4 children present).
            # Fall back to all_idx if no complete quads are in the training set
            # (e.g. small debug samples where siblings land in different splits).
            pool = self.complete_idx if len(self.complete_idx) > 0 else self.all_idx
            indices = rng.choice(pool, size=self.num_samples, replace=True)
        elif stage == 2:
            indices = rng.choice(self.all_idx, size=self.num_samples, replace=True)
        elif stage == 1:
            n_easy = int(self.num_samples * 0.5)
            n_hard = self.num_samples - n_easy
            easy_pool = self.easy_idx if len(self.easy_idx) > 0 else self.all_idx
            hard_pool = self.hard_idx if len(self.hard_idx) > 0 else self.all_idx
            easy = rng.choice(easy_pool, size=n_easy, replace=True)
            hard = rng.choice(hard_pool, size=n_hard, replace=True)
            indices = np.concatenate([easy, hard])
            rng.shuffle(indices)
        else:  # stage 0
            pool = self.easy_idx if len(self.easy_idx) > 0 else self.all_idx
            indices = rng.choice(pool, size=self.num_samples, replace=True)
        yield from indices.tolist()

    def __len__(self):
        return self.num_samples


class LitDenoiserSR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = torch.compile(DenoiserSR(args))

        # EMA tracks only trainable parameters — excludes frozen LPIPS weights.
        # Must be assigned to _orig_mod (the inner DenoiserSR) so that
        # update_ema() can access them via `self.ema_params1/2` on its own instance.
        # Assigning to the compiled wrapper instead would leave the inner module
        # seeing None and crash on the first update_ema() call.
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.model._orig_mod.ema_params1 = copy.deepcopy(trainable)
        self.model._orig_mod.ema_params2 = copy.deepcopy(trainable)

        # Shared stage — updated every step in on_train_batch_start,
        # read by ZLCurriculumSampler at each epoch start
        import torch.multiprocessing as mp
        self.shared_stage = mp.Value('i', 0)

    def on_train_batch_start(self, batch, batch_idx):
        step = int(self.trainer.global_step)
        b1 = getattr(self.model.args, "stage_boundary1", 100_000)
        b2 = getattr(self.model.args, "stage_boundary2", 200_000)
        b3 = getattr(self.model.args, "stage_boundary3", 300_000)
        stage = 0 if step < b1 else (1 if step < b2 else (2 if step < b3 else 3))
        self.shared_stage.value = stage
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "shared_stage"):
            dm.shared_stage.value = stage

    def on_fit_start(self):
        device = next(self.model.parameters()).device
        orig = self.model._orig_mod
        # Move LPIPS to the correct device once at fit start (avoids graph breaks in compiled forward)
        if getattr(orig, 'lpips_fn', None) is not None:
            orig.lpips_fn.to(device)
        if hasattr(self, "_loaded_ema1") and self._loaded_ema1 is not None:
             orig.ema_params1 = [p.to(device) for p in self._loaded_ema1]
             if getattr(self, "_loaded_ema2", None) is not None:
                 orig.ema_params2 = [p.to(device) for p in self._loaded_ema2]
             else:
                 # Checkpoint only had ema_params1 — initialise ema_params2 from it
                 orig.ema_params2 = [p.detach().clone() for p in orig.ema_params1]
             del self._loaded_ema1
             if hasattr(self, "_loaded_ema2"):
                 del self._loaded_ema2
             print("Restored EMA from checkpoint")
        else:
             trainable = [p for p in self.model.parameters() if p.requires_grad]
             orig.ema_params1 = [p.detach().clone() for p in trainable]
             orig.ema_params2 = [p.detach().clone() for p in trainable]

    def on_save_checkpoint(self, checkpoint):
        orig = self.model._orig_mod
        if orig.ema_params1 is not None:
            checkpoint["ema_params1"] = [p.cpu() for p in orig.ema_params1]
        if orig.ema_params2 is not None:
            checkpoint["ema_params2"] = [p.cpu() for p in orig.ema_params2]

    def on_load_checkpoint(self, checkpoint):
        if "ema_params1" in checkpoint:
            self._loaded_ema1 = checkpoint["ema_params1"]
            self._loaded_ema2 = checkpoint.get("ema_params2")

    @contextmanager
    def ema_context(self, ema_idx: int = 1):
        """Temporarily swap EMA weights into the live model for eval/generation.

        Restores the original live weights on exit so training state is untouched.
        ema_idx=1 uses ema_decay1 (slower, more stable); ema_idx=2 uses ema_decay2.
        Falls through as a no-op if EMA params haven't been initialised yet.
        """
        orig = self.model._orig_mod
        ema_params = orig.ema_params1 if ema_idx == 1 else orig.ema_params2
        if ema_params is None:
            yield
            return
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        saved = [p.data.clone() for p in trainable]
        for p, ema_p in zip(trainable, ema_params):
            p.data.copy_(ema_p)
        try:
            yield
        finally:
            for p, saved_p in zip(trainable, saved):
                p.data.copy_(saved_p)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with self.ema_context(ema_idx=1):
            loss, _ = self.model(batch)
        split_name = "val_random" if dataloader_idx == 0 else "val_spatial"
        self.log(f"val/{split_name}_loss", loss, on_epoch=True, sync_dist=True, prog_bar=(dataloader_idx == 0), add_dataloader_idx=False)
        return loss

    def training_step(self, batch, batch_idx):
        loss, aux = self.model(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        if aux:
            self.log("train/main_loss",  aux["main_loss"],  on_step=True, on_epoch=False)
            self.log("train/lpips_loss", aux["lpips_loss"], on_step=True, on_epoch=False)
        if self.trainer.global_step % getattr(self.model.args, "log_freq", 1000) == 0:
            if isinstance(self.logger, WandbLogger):
                with torch.no_grad(), self.ema_context(ema_idx=1):
                    gen = self.model.generate(batch)
                def to01(x): return (x * 0.5 + 0.5).clamp(0, 1)
                parents  = to01(batch["parent_rgb"][:4])   # (≤4, 3, 256, 256)
                # Reconstruct stitched 512×512 from individual child tiles for display
                cr  = batch["children_rgb"][:4]            # (≤4, 4, 3, S, S)
                n   = cr.shape[0]
                S   = cr.shape[-1]
                children = to01(
                    cr.view(n, 2, 2, 3, S, S)
                      .permute(0, 3, 1, 4, 2, 5)
                      .contiguous()
                      .view(n, 3, 2 * S, 2 * S)
                )                                          # (≤4, 3, 512, 512)
                gen_viz  = to01(gen[:4])                   # (≤4, 3, 512, 512)

                # Log at native resolutions — do NOT upsample the parent
                # so the viewer sees the true 256 → 512 resolution jump.
                parent_grid = make_grid(parents,  nrow=4)  # 256×256 grid
                # Interleave GT and generated: [GT_0, Gen_0, GT_1, Gen_1, ...]
                sr_pairs = torch.stack([children, gen_viz], dim=1).flatten(0, 1)
                sr_grid  = make_grid(sr_pairs, nrow=4)     # 512×512 grid, pairs side-by-side

                self.logger.experiment.log({
                    "viz/input_256":    wandb.Image(parent_grid,
                                                    caption="Parent input (256×256)"),
                    "viz/sr_512":       wandb.Image(sr_grid,
                                                    caption="GT (odd cols) | Generated (even cols) — 512×512"),
                })
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx): self.model.update_ema()
    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.model.args.lr)


class SuperResDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        import torch.multiprocessing as mp
        self.shared_stage = mp.Value('i', 0)

    def setup(self, stage=None):
        import json
        from dataloader import HFParentChildDataset

        print(f"Loading metadata from {self.args.meta_file_path}...")
        with open(self.args.meta_file_path, 'r') as f:
            meta = json.load(f)

        index_to_meta = meta['index_to_meta']

        # Prefer the pre-shuffled split index lists written by build_quad_splits_hf.py.
        # These are seeded-deterministic and ZL-balanced (not ZL-contiguous), making
        # them safe to slice without risking an all-ZL=10-root prefix.
        # Fall back to deriving from index_to_meta for backwards compatibility with
        # older metadata files that don't carry the 'splits' key.
        if 'splits' in meta:
            splits = {k: meta['splits'][k]
                      for k in ('train', 'val_random', 'val_spatial')}
        else:
            print("WARNING: metadata has no 'splits' key; falling back to "
                  "index_to_meta scan.  Re-run build_quad_splits_hf.py to "
                  "generate ZL-balanced, shuffled split lists.")
            splits = {'train': [], 'val_random': [], 'val_spatial': []}
            for idx_str, m in index_to_meta.items():
                s = m.get('split')
                if s in splits:
                    splits[s].append(int(idx_str))

        if stage == "fit" or stage is None:
            # Training: partial quads allowed (≥1 child), masked loss in engine
            self.train_dataset = HFParentChildDataset(
                hindex_pkl            = self.args.hindex_pkl,
                indices               = splits['train'],
                index_to_meta         = index_to_meta,
                ensure_size           = self.args.img_size,
                cache_dir             = self.args.hf_data_cache,
                require_complete_quad = False,
            )
            # Validation: complete quads only (random split)
            self.val_dataset = HFParentChildDataset(
                hindex_pkl            = self.args.hindex_pkl,
                indices               = splits['val_random'],
                index_to_meta         = index_to_meta,
                ensure_size           = self.args.img_size,
                cache_dir             = self.args.hf_data_cache,
                require_complete_quad = True,
            )
            # Validation: complete quads only (spatial split — 50% urban / 50% non-urban)
            self.val_spatial_dataset = HFParentChildDataset(
                hindex_pkl            = self.args.hindex_pkl,
                indices               = splits['val_spatial'],
                index_to_meta         = index_to_meta,
                ensure_size           = self.args.img_size,
                cache_dir             = self.args.hf_data_cache,
                require_complete_quad = True,
            )

    def train_dataloader(self):
        from torch.utils.data import DataLoader

        zl_easy_max = getattr(self.args, "zl_easy_max", 15)
        zls            = [rec["parent_zl"]   for rec in self.train_dataset.pairs]
        complete_flags = [rec["is_complete"] for rec in self.train_dataset.pairs]

        # epoch length = effective_bs × steps_per_epoch (1000 steps)
        # so the sampler reads the updated stage every ~1000 optimizer steps
        effective_bs = (self.args.batch_size
                        * getattr(self.args, "num_devices", 1)
                        * getattr(self.args, "accumulate_grad_batches", 1))
        num_samples = effective_bs * 1000

        sampler = ZLCurriculumSampler(
            zls=zls,
            shared_stage=self.shared_stage,
            zl_easy_max=zl_easy_max,
            num_samples=num_samples,
            complete_flags=complete_flags,
        )

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sampler=sampler,
        )
        from dataloader import Git10MSuperResHF
        return Git10MSuperResHF(loader, split="train")

    def val_dataloader(self):
        from torch.utils.data import DataLoader
        return [
            DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            ),
            DataLoader(
                self.val_spatial_dataset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            ),
        ]
