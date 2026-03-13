"""
Minimal WandB wrapper for offline SLURM training.

Usage in training:
    from wandb_config import WandbLogger
    wb = WandbLogger(name="run_name", config={...})
    wb.log({'loss': 0.5}, step=global_step)
    wb.finish()

Offline sync from login node:
    bash scripts/wandb_sync.sh          # watches and syncs in real-time
    bash scripts/wandb_sync.sh once     # one-shot sync after job finishes
"""

import os
import wandb


class WandbLogger:
    """
    Thin wrapper around wandb for offline SLURM training.
    Sets WANDB_MODE=offline automatically so no internet is needed on compute nodes.
    All data is written to WANDB_DIR/wandb/offline-run-*/
    and can be synced to the website later from the login node.
    """

    def __init__(self, name, config=None, project="hypyolov2", dir="wandb_logs"):
        os.makedirs(dir, exist_ok=True)

        # Force offline — no network calls whatsoever during training
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DIR"] = os.path.abspath(dir)

        self.run = wandb.init(
            project=project,
            name=name,
            config=config or {},
            dir=dir,
            mode="offline",
        )
        print(f"[WandB] Offline run: {self.run.dir}")
        print(f"[WandB] To sync: bash scripts/wandb_sync.sh")

    def log(self, metrics: dict, step: int = None):
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def finish(self):
        if self.run is not None:
            self.run.finish()
            print(f"[WandB] Run finished. Sync with: bash scripts/wandb_sync.sh once")
