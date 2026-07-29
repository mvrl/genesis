import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from engine import LitDenoiserOutpainting, OutpaintingDataModule
from utils import compute_stage_boundaries, StepCheckpointCallback

# Git-10M quad meta lives in the Google Drive release under ``shared/`` and caches
# under ``$GENESIS_CKPT_BASE`` (same base as the checkpoints); pull it once
# (see src/dataprep/README.md) or point ``--meta_file_path`` elsewhere.
_CKPT_BASE = os.environ.get("GENESIS_CKPT_BASE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "ckpts")


def get_args_parser():
    parser = argparse.ArgumentParser("JiT Outpainting")
    parser.add_argument('--model', default='JiT-H/16', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--max_steps', default=800000, type=int,
                        help="Total training steps. Stage boundaries are auto-computed from this.")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=1e-5, type=float)
    parser.add_argument('--sampling_method', default='euler', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--hf_data_cache', default=os.environ.get("HF_DATASETS_CACHE") or None, type=str)
    parser.add_argument('--cfg', default=1.0, type=float, help="Classifier-free guidance scale")
    parser.add_argument('--meta_file_path', default=os.path.join(_CKPT_BASE, "shared", "git10m_quad_meta.json"), type=str)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--output_dir', default="./output_op_new")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_freq', default=1000, type=int)
    parser.add_argument('--wandb_mode', default='disabled', type=str)
    parser.add_argument('--wandb_project', default='jit-outpainting', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help="W&B run name. If not set, W&B auto-generates one.")
    parser.add_argument('--ckpt_prefix', default='op-new', type=str,
                        help="Prefix for checkpoint filenames")
    parser.add_argument('--resume_ckpt', default=None, type=str,
                        help="Resume full training state (model + optimizer + global_step) from this Lightning checkpoint.")
    parser.add_argument('--devices', default="2", type=str,
                        help="Number of GPUs (int) or list '0,1' (str)")
    parser.add_argument('--gradient_clip_val', default=0.5, type=float,
                        help="Clip gradient norm to avoid NaNs")
    parser.add_argument('--precision', default='bf16-mixed', type=str,
                        help="Trainer precision: '32', '16-mixed', 'bf16-mixed'")
    parser.add_argument('--dino_weights',
                        default=os.environ.get("GENESIS_DINO_WEIGHTS", ""),
                        type=str, help="Path to DINOv3 weights (empty string to disable)")
    parser.add_argument('--enable_time_weight', default=True,
                        action=argparse.BooleanOptionalAction,
                        help="Apply time weighting w=1/(1-t); disable for uniform loss")
    parser.add_argument('--accumulate_grad_batches', default=4, type=int,
                        help="Accumulate gradients over k batches before stepping")
    return parser


def main(args):
    # Auto-compute curriculum stage boundaries from max_steps
    s1, s2, s3 = compute_stage_boundaries(args.max_steps)
    args.stage_step1 = s1
    args.stage_step2 = s2
    args.stage_step3 = s3
    print(f"Curriculum boundaries (from max_steps={args.max_steps:,}):")
    print(f"  stage_step1={s1:,}  ({s1/args.max_steps*100:.0f}%)  — stage 0→1 (quad3 → mix of quad2_adj/quad2_diag)")
    print(f"  stage_step2={s2:,}  ({s2/args.max_steps*100:.0f}%)  — stage 1→2 (quad2_* → quad1)")
    print(f"  stage_step3={s3:,}  ({s3/args.max_steps*100:.0f}%)  — stage 2→3 (quad1 → uniform mix of all 4 regimes)")

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

    pl.seed_everything(args.seed)
    dm = OutpaintingDataModule(args)
    print(f"Initializing model: {args.model} (training from scratch)")
    model = LitDenoiserOutpainting(args)

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

    strategy = None
    if num_devices > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)

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
        stage_steps=(s1, s2, s3),
        output_dir=args.output_dir,
        fname_prefix=args.ckpt_prefix,
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=devices,
        strategy=strategy if strategy is not None else "auto",
        logger=logger,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_cb, step_ckpt_cb],
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
