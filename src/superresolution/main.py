import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from engine import LitDenoiserSR, SuperResDataModule
from utils import compute_stage_boundaries, StepCheckpointCallback

# Git-10M quad meta + hierarchy index live in the Google Drive release under
# ``shared/`` and cache under ``$GENESIS_CKPT_BASE`` (same base as the checkpoints).
# Pull them once (see src/dataprep/README.md) or point the flags elsewhere.
_CKPT_BASE = os.environ.get("GENESIS_CKPT_BASE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "ckpts")


def get_args_parser():
    parser = argparse.ArgumentParser("JiT SuperRes")
    parser.add_argument('--model', default='JiT-B/16', type=str)
    parser.add_argument('--img_size', default=256, type=int, help="Parent tile size (256)")
    parser.add_argument('--output_size', default=512, type=int, help="Child/output image size (2x img_size)")
    parser.add_argument('--use_dino', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Use DINO features as additional conditioning on top of pixel parent. "
                             "Pass --no-use_dino to ablate pixel-only conditioning.")
    parser.add_argument('--max_steps', default=800000, type=int,
                        help="Total training steps (paper: 800k). Stage boundaries auto-computed at 20/40/60%.")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=1e-5, type=float)
    parser.add_argument('--sampling_method', default='euler', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--lpips_weight', default=0.5, type=float,
                        help="Weight for the t-scaled LPIPS perceptual loss (paper main: 0.5; "
                             "0.0 = disabled, i.e. the *_nolpips ablation cells).")
    # ── Zoom-level curriculum ──────────────────────────────────────────────
    parser.add_argument('--zl_easy_max', default=15, type=int,
                        help="Parent ZL ≤ this = easy tiles (coarse, ≥4m/px). Default: 15")
    parser.add_argument('--hf_data_cache', default=os.environ.get("HF_DATASETS_CACHE") or None, type=str)
    parser.add_argument('--cfg', default=1.0, type=float, help="Classifier-free guidance scale")
    parser.add_argument('--meta_file_path', default=os.path.join(_CKPT_BASE, "shared", "git10m_quad_meta.json"), type=str)
    parser.add_argument('--hindex_pkl', default=os.path.join(_CKPT_BASE, "shared", "hierarchy_index_quad.pkl"), type=str)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int, help="Accumulate gradients over k batches before stepping")
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--output_dir', default="./output_sr")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--wandb_mode', default='disabled', type=str)
    parser.add_argument('--wandb_project', default='jit-superres', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help="W&B run name. If not set, W&B auto-generates one.")
    parser.add_argument('--dino_weights', default=os.environ.get("GENESIS_DINO_WEIGHTS", ""), type=str)
    parser.add_argument('--precision', default='bf16-mixed', type=str, help="Trainer precision (e.g., '32', '16-mixed', 'bf16-mixed')")
    parser.add_argument('--gradient_clip_val', default=1.0, type=float, help="Gradient clipping value to prevent NaNs")
    parser.add_argument('--ckpt_prefix', default='sr-full-tile', type=str, help="Prefix for checkpoint filenames")
    parser.add_argument('--devices', default="2", type=str, help="Number of GPUs (int) or list '0,1' (str)")
    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help="Resume full training state (model + optimizer + global_step) from this Lightning checkpoint.")
    parser.add_argument('--bicubic_init', default=False, action=argparse.BooleanOptionalAction,
                        help="If set, initialize diffusion from bicubic-upsampled parent instead of random noise (inference only).")
    parser.add_argument('--bicubic_init_t_start', default=0.5, type=float,
                        help="SDEdit start time in [0,1] when bicubic_init is on.")
    return parser

def main(args):
    args.zoom_diffs = [1]
    args.zoom_diff = 1

    # Parse devices
    if "," in str(args.devices):
        devices = [int(x) for x in str(args.devices).split(",")]
        num_devices = len(devices)
    else:
        try:
            devices = int(args.devices)
            num_devices = devices
        except ValueError:
            devices = "auto"
            num_devices = 1

    args.num_devices = num_devices

    # Auto-compute curriculum stage boundaries from max_steps
    b1, b2, b3 = compute_stage_boundaries(args.max_steps)
    args.stage_boundary1 = b1
    args.stage_boundary2 = b2
    args.stage_boundary3 = b3
    print(f"Curriculum: enabled")
    print(f"  stage_boundary1={args.stage_boundary1:,}  ({min(args.stage_boundary1, args.max_steps)/args.max_steps*100:.0f}%)  — 100% easy ZL → 50/50 easy/hard")
    print(f"  stage_boundary2={args.stage_boundary2:,}  ({min(args.stage_boundary2, args.max_steps)/args.max_steps*100:.0f}%)  — 50/50 easy/hard → uniform ZL all pairs")
    print(f"  stage_boundary3={args.stage_boundary3:,}  ({min(args.stage_boundary3, args.max_steps)/args.max_steps*100:.0f}%)  — uniform ZL, complete quads only")

    pl.seed_everything(args.seed)
    dm = SuperResDataModule(args)
    print(f"Initializing model: {args.model} (training from scratch)")
    model = LitDenoiserSR(args)

    # Resume full training state if a checkpoint is provided
    resume_ckpt = None
    if args.resume_ckpt:
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {args.resume_ckpt}")
        resume_ckpt = args.resume_ckpt
        print(f"Resuming training state from: {resume_ckpt}")

    logger = None
    if args.wandb_mode != 'disabled':
        logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name, mode=args.wandb_mode)

    os.makedirs(args.output_dir, exist_ok=True)
    fname_prefix = f"{args.ckpt_prefix}-" if args.ckpt_prefix else ""
    # Saves the top-3 checkpoints by val_random_loss — checked after every val epoch.
    # `every_n_train_steps=10000` keeps `last.ckpt` fresh independent of epoch length,
    # so a walltime/requeue mid-epoch loses at most ~10k steps of progress.
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename=f"{fname_prefix}step{{step}}-val{{val_val_random_loss:.4f}}",
        monitor="val/val_random_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=10000,
    )
    # Uniform + stage-boundary checkpoints (10 uniform + 4 stage ends, overlaps deduplicated)
    step_ckpt_cb = StepCheckpointCallback(
        max_steps=args.max_steps,
        stage_steps=(b1, b2, b3),
        output_dir=args.output_dir,
        fname_prefix=args.ckpt_prefix,
    )

    # Device auto-detection: GPU is strongly recommended for training. If no CUDA
    # device is present, fall back to CPU (single process, full precision) so the
    # run still starts instead of crashing on accelerator="gpu".
    if torch.cuda.is_available():
        accelerator = "gpu"
        precision = args.precision
    else:
        print("[info] No CUDA GPU detected — training on CPU (very slow; single process, precision=32).")
        accelerator = "cpu"
        devices = 1
        num_devices = 1
        args.num_devices = 1
        precision = "32"

    strategy = "auto"
    if num_devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        precision=precision,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb, step_ckpt_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
        default_root_dir=args.output_dir,
    )
    trainer.fit(model, dm, ckpt_path=resume_ckpt)

if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
