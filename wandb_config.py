"""WandB logger for offline SLURM training."""
import os
import wandb

class WandbLogger:
    def __init__(self, project="hypyolov2", name=None, config=None, offline=True, wandb_dir="./wandb_logs"):
        self.offline = offline
        if offline:
            os.environ["WANDB_MODE"] = "offline"
            os.environ["WANDB_DIR"] = wandb_dir
            print(f"WandB offline - logs: {wandb_dir}")
        
        self.run = wandb.init(project=project, name=name, config=config, dir=wandb_dir)
        print(f"WandB run: {self.run.name} ({self.run.id})")
    
    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)
    
    def finish(self):
        if self.offline:
            print(f"\nSync later: wandb sync {wandb.run.dir}")
        wandb.finish()

