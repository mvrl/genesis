import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid, draw_segmentation_masks
from pytorch_lightning.loggers import WandbLogger
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "superresolution")))
from model_op import JiT_models
from model_sr import get_dino_embeddings

class DenoiserOutpainting(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 1. Init Model
        self.net = JiT_models[args.model](
            input_size=args.img_size,
        )
        
        self.img_size = args.img_size
        self.P_mean, self.P_std = args.P_mean, args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale
        
        self.ema_decay1 = args.ema_decay1
        self.ema_decay2 = args.ema_decay2
        self.ema_params1 = None
        self.ema_params2 = None

        self.steps = args.num_sampling_steps
        self.cfg_scale = args.cfg
    
    @staticmethod
    def _get_known_crop(masked_image: torch.Tensor, mask: torch.Tensor, out_size: int = 224):
        """
        Extract the bounding box of known region (mask < 0.5), fill any zeroed pixels with mean color,
        and resize to square for DINO. Works for both training and inference, avoids GT leakage.
        Args:
            masked_image: [B,3,H,W], holes already zeroed
            mask: [B,1,H,W], 1=hole, 0=known
        Returns:
            crop_resized: [B,3,out_size,out_size]
        """
        B, C, H, W = masked_image.shape
        crops = []
        for b in range(B):
            known = (mask[b, 0] < 0.5)
            coords = known.nonzero(as_tuple=False)
            if coords.numel() == 0:
                crop = masked_image[b:b+1]
            else:
                y0, y1 = coords[:, 0].min(), coords[:, 0].max()
                x0, x1 = coords[:, 1].min(), coords[:, 1].max()
                crop = masked_image[b:b+1, :, y0:y1+1, x0:x1+1]
            # Fill zeroed pixels inside crop with mean color to reduce black-block bias
            zero_mask = (crop.abs().sum(dim=1, keepdim=True) == 0).float()
            if zero_mask.sum() > 0:
                mean_color = (crop * (1 - zero_mask)).sum(dim=[2, 3], keepdim=True) / \
                             (1 - zero_mask).sum(dim=[2, 3], keepdim=True).clamp_min(1.0)
                crop = crop * (1 - zero_mask) + mean_color * zero_mask
            crop = F.interpolate(crop, size=(out_size, out_size), mode="bicubic", align_corners=False)
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z).clamp(0.0, 1.0 - self.t_eps)  # prevent t=1 → denominator blowup

    def forward(self, batch):
        x1 = batch["image"]      # Target [-1, 1]
        mask = batch["roi_mask"] # 1 = hole, 0 = known

        # SD-inpainting style conditioning: concat(noised image, mask, masked image)
        masked_image = x1 * (1.0 - mask)

        B = x1.shape[0]
        t = self.sample_t(B, device=x1.device).view(-1, 1, 1, 1)
        x0 = torch.randn_like(x1) * self.noise_scale
        # Only add noise to hole region; known region remains clean. Condition no longer uses GT to avoid leakage.
        noisy_hole = t * x1 + (1 - t) * x0
        zt = masked_image + noisy_hole * mask
        inpaint_cond = torch.cat([zt, mask, masked_image], dim=1)

        # Predict
        conditions = {"inpaint_cond": inpaint_cond, "mask": mask}

        if getattr(self.args, "dino_weights", None):
            with torch.no_grad():
                dino_input = self._get_known_crop(masked_image, mask)
                conditions["dino_feats"] = get_dino_embeddings(dino_input, weights_path=self.args.dino_weights)
        # Zoom-level / GSD conditioning — 0 = unknown, passed through as-is
        if "zoom_level" in batch:
            conditions["zoom_level"] = batch["zoom_level"].float().to(x1.device)
        x_pred = self.net(zt, t.flatten(), conditions=conditions)

        # Velocity MSE loss
        denom = (1.0 - t).clamp_min(self.t_eps)
        v_pred = (x_pred - zt) / denom
        v_tgt = (x1 - zt) / denom
        loss_mse = F.mse_loss(v_pred, v_tgt, reduction='none')
        
        num = (loss_mse * mask).sum(dim=(1,2,3))             # [B]
        den = mask.sum(dim=(1,2,3)).clamp_min(1e-6)          # [B]
        loss_per = num / den                                 # [B]
        t_flat = t.view(-1)                                  # [B]
        if getattr(self.args, "enable_time_weight", True):
            w = (1.0 / (1.0 - t_flat).clamp_min(self.t_eps)).clamp(max=20.0)
            loss = (loss_per * w).mean()
        else:
            loss = loss_per.mean()
        
        # Safety guard: if loss is NaN/Inf, return a DDP-safe zero (connected to trainable params only)
        # Must use 0*p.sum() not a leaf tensor — leaf tensors give UNDEFINED grads in DDP
        if not torch.isfinite(loss):
            return sum(0.0 * p.sum() for p in self.parameters() if p.requires_grad)

        return loss

    @torch.no_grad()
    def generate(self, batch):
        x1, mask = batch["image"], batch["roi_mask"]
        masked_image = x1 * (1.0 - mask)  # zero in holes to avoid GT leakage

        x0 = torch.randn_like(x1) * self.noise_scale

        # Zoom-level / GSD — resolved once, reused every denoising step
        zoom_level = batch["zoom_level"].float().to(x1.device) if "zoom_level" in batch else None

        z = masked_image + x0 * mask
        timesteps = torch.linspace(0.0, 1.0, self.steps+1, device=x1.device)
        dt = 1.0 / self.steps

        for i in range(self.steps):
            t = timesteps[i].expand(z.shape[0])
            # diffusion only on the unknown areas
            inpaint_cond = torch.cat([z, mask, masked_image], dim=1)
            conditions = {"inpaint_cond": inpaint_cond, "mask": mask}
            if getattr(self.args, "dino_weights", None):
                with torch.no_grad():
                    # DINO only based on the known region.
                    dino_input = self._get_known_crop(masked_image, mask)
                    conditions["dino_feats"] = get_dino_embeddings(dino_input, weights_path=self.args.dino_weights)
            if zoom_level is not None:
                conditions["zoom_level"] = zoom_level

            # 1. Conditional Pass
            x_cond = self.net(z, t, conditions=conditions)
            v_cond = (x_cond - z) / (1.0 - t.view(-1,1,1,1)).clamp_min(self.t_eps)

            # 2. Unconditional Pass (CFG) — drop all conditions incl. zoom
            if self.cfg_scale > 1.0:
                uncond_conditions = conditions.copy()
                uncond_conditions["dino_feats"] = None
                uncond_conditions["inpaint_cond"] = None
                uncond_conditions["mask"] = None
                uncond_conditions["zoom_level"] = None

                x_uncond = self.net(z, t, conditions=uncond_conditions)
                v_uncond = (x_uncond - z) / (1.0 - t.view(-1,1,1,1)).clamp_min(self.t_eps)

                # Mix
                v_pred = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            z = z + dt * v_pred

            # Keep known region frozen every step to avoid drift
            z = z * mask + masked_image * (1 - mask)
            
        # Hard overwrite of known region to ensure identity
        return z * mask + masked_image * (1 - mask)

    def update_ema(self):
        source_params = [p for p in self.parameters() if p.requires_grad]

        if self.ema_params1 is None or len(self.ema_params1) != len(source_params):
            self.ema_params1 = [p.detach().clone() for p in source_params]
        if self.ema_params2 is None or len(self.ema_params2) != len(source_params):
            self.ema_params2 = [p.detach().clone() for p in source_params]
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)

class LitDenoiserOutpainting(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = torch.compile(DenoiserOutpainting(args))
        # EMA tracks only trainable parameters — excludes any frozen components
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.model.ema_params1 = copy.deepcopy(trainable)
        self.model.ema_params2 = copy.deepcopy(trainable)

    def on_train_batch_start(self, batch, batch_idx):
        # Update curriculum stage every batch (Lightning won't re-call train_dataloader by default)
        step = int(self.trainer.global_step)
        s1 = getattr(self.model.args, "stage_step1", 100_000)
        s2 = getattr(self.model.args, "stage_step2", 200_000)
        s3 = getattr(self.model.args, "stage_step3", 300_000)
        if step < s1:
            stage = 0
        elif step < s2:
            stage = 1
        elif step < s3:
            stage = 2
        else:
            stage = 3

        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None and getattr(dm, "train_loader", None) is not None:
            ds = dm.train_loader.dataset
            if hasattr(ds, "stage"):
                ds.stage = stage
            # Broadcast to workers via shared memory
            if getattr(dm, "shared_stage", None) is not None:
                dm.shared_stage.value = stage

    def on_fit_start(self):
        device = next(self.model.parameters()).device
        if hasattr(self, "_loaded_ema1") and self._loaded_ema1 is not None:
             self.model.ema_params1 = [p.to(device) for p in self._loaded_ema1]
             self.model.ema_params2 = [p.to(device) for p in self._loaded_ema2]
             del self._loaded_ema1, self._loaded_ema2
             print("Restored EMA from checkpoint")
        else:
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.model.ema_params1 = [p.detach().clone().to(device) for p in trainable]
            self.model.ema_params2 = [p.detach().clone().to(device) for p in trainable]

    def on_save_checkpoint(self, checkpoint):
        if self.model.ema_params1 is not None:
            checkpoint["ema_params1"] = [p.cpu() for p in self.model.ema_params1]
        if self.model.ema_params2 is not None:
            checkpoint["ema_params2"] = [p.cpu() for p in self.model.ema_params2]

    def on_load_checkpoint(self, checkpoint):
        if "ema_params1" in checkpoint and "ema_params2" in checkpoint:
            self._loaded_ema1 = checkpoint["ema_params1"]
            self._loaded_ema2 = checkpoint["ema_params2"]

    def _ddp_safe_zero_loss(self):
        """Returns a true-zero loss connected to all TRAINABLE params — safe for DDP all-reduce.
        Leaf tensors (requires_grad=True) give UNDEFINED gradients to params in DDP; this gives ZERO gradients.
        Frozen params (requires_grad=False) are excluded — DDP doesn't all-reduce them anyway.
        """
        return sum(0.0 * p.sum() for p in self.parameters() if p.requires_grad)

    def training_step(self, batch, batch_idx):
        # Sanity-check the batch for NaN/Inf inputs before forward pass
        if not torch.isfinite(batch["image"]).all():
            import warnings
            warnings.warn(f"[step {self.trainer.global_step}] NaN/Inf in batch images, skipping batch.")
            return self._ddp_safe_zero_loss()
        loss = self.model(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        
        if self.trainer.global_step % getattr(self.model.args, "log_freq", 500) == 0:
            if isinstance(self.logger, WandbLogger):
                with torch.no_grad():
                    gen = self.model.generate(batch)
                # Convert from [-1,1] to [0,1] for visualization
                def to01(x): return (x * 0.5 + 0.5).clamp(0, 1)

                imgs = batch["image"][:4]
                masks = batch["roi_mask"][:4]


                gt  = to01(imgs).nan_to_num(0.0)        # NaN-safe: clamp doesn't fix NaN
                out = to01(gen[:4]).nan_to_num(0.0)     # NaN in out would corrupt entire grid

                def draw_mask(img01, mask1):
                    hole = (mask1[0] > 0.5)
                    if hole.sum() == 0:
                        return img01
                    img_uint8 = (img01 * 255).clamp(0, 255).to(torch.uint8).cpu()
                    mask_bool = hole.cpu()
                    painted = draw_segmentation_masks(img_uint8, mask_bool, alpha=0.45, colors="red")
                    return painted.to(img01.device).float() / 255.0

                gt_painted = torch.stack([draw_mask(gt[i], masks[i]) for i in range(gt.shape[0])])

                # Log GT and prediction separately so one NaN run doesn't black out GT
                grid = make_grid(torch.cat([gt_painted, out], dim=0), nrow=4)
                self.logger.experiment.log({"train/viz": wandb.Image(grid)})
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.model(batch)
        split_name = "val_random" if dataloader_idx == 0 else "val_spatial"
        self.log(f"val/{split_name}_loss", loss, on_epoch=True, prog_bar=(dataloader_idx == 0), sync_dist=True, add_dataloader_idx=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.update_ema()

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.model.args.lr, betas=(0.9, 0.95))

class OutpaintingDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_loader = None
        self.val_loader = None
        self.val_spatial_loader = None
        self._train_loader_builds = 0  # unused, kept for potential future metrics
        
    def setup(self, stage=None):
        from dataloader import build_outpainting_dataloader
        from datasets import load_dataset
        import json
        import torch.multiprocessing as mp
        
        # 1. Load HF Dataset (once)
        print(f"Loading HF dataset from {self.args.hf_data_cache}...")
        hf_dataset = load_dataset(
            "lcybuaa/Git-10M",
            cache_dir=self.args.hf_data_cache,
            split="train"
        )
        
        # 2. Load Metadata (once)
        print(f"Loading metadata from {self.args.meta_file_path}...")
        with open(self.args.meta_file_path, 'r') as f:
            data = json.load(f)
        index_to_meta = data['index_to_meta']
        
        # 3. Get Splits — prefer the pre-shuffled lists from the 'splits' key
        # written by build_quad_splits_hf.py (seeded-deterministic, ZL-balanced).
        # Fall back to scanning index_to_meta for older metadata files.
        print("Extracting splits...")
        if 'splits' in data:
            splits = {k: list(v) for k, v in data['splits'].items()}
        else:
            print("WARNING: metadata has no 'splits' key; falling back to index_to_meta scan. "
                  "Re-run build_quad_splits_hf.py to generate ZL-balanced shuffled split lists.")
            splits = {'train': [], 'val_random': [], 'val_spatial': [], 'random_test': [], 'spatial_test': []}
            for idx_str, meta in index_to_meta.items():
                s = meta.get('split')
                if s in splits:
                    splits[s].append(int(idx_str))
            for s in splits:
                splits[s] = sorted(splits[s])

        # 4. Build Loaders
        # Shared stage keeps curriculum stage consistent across dataloader workers
        if not hasattr(self, "shared_stage"):
            self.shared_stage = mp.Value('i', 0)
        if stage == "fit" or stage is None:
            if 'train' in splits:
                self.train_loader = build_outpainting_dataloader(
                    hf_dataset=hf_dataset,
                    indices=splits['train'],
                    index_to_meta=index_to_meta,
                    img_size=self.args.img_size,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    split="train",
                    shuffle=True,
                    enable_curriculum=True,
                    shared_stage=self.shared_stage,
                )

            if 'val_random' in splits:
                self.val_loader = build_outpainting_dataloader(
                    hf_dataset=hf_dataset,
                    indices=splits['val_random'],
                    index_to_meta=index_to_meta,
                    img_size=self.args.img_size,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    split="val_random",
                    shuffle=False,
                    enable_curriculum=False,
                    shared_stage=self.shared_stage,
                )
            if 'val_spatial' in splits:
                self.val_spatial_loader = build_outpainting_dataloader(
                    hf_dataset=hf_dataset,
                    indices=splits['val_spatial'],
                    index_to_meta=index_to_meta,
                    img_size=self.args.img_size,
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    split="val_spatial",
                    shuffle=False,
                    enable_curriculum=False,
                    shared_stage=self.shared_stage,
                )

    def train_dataloader(self):
        # Stage is kept current by LitDenoiserOutpainting.on_train_batch_start every step.
        # Sync dataset.stage from shared_stage so the first epoch starts at the right stage.
        from dataloader import Git10MOutpaintingHF
        if self.train_loader is None:
            raise RuntimeError("train_loader is None — did setup() complete successfully?")
        stage = int(self.shared_stage.value) if hasattr(self, "shared_stage") and self.shared_stage is not None else 0
        if hasattr(self.train_loader.dataset, "stage"):
            self.train_loader.dataset.stage = stage
        return Git10MOutpaintingHF(self.train_loader, split="train")

    def val_dataloader(self):
        if self.val_loader is None:
            raise RuntimeError("val_loader is None — did setup() complete successfully?")
        loaders = [self.val_loader]
        if self.val_spatial_loader is not None:
            loaders.append(self.val_spatial_loader)
        return loaders
