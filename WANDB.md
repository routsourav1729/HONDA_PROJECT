# WandB Integration - Quick Guide

## Training with WandB (Offline Mode)

```bash
sbatch train_hyp.sbatch              # Enable WandB logging
```

## Sync Logs to Cloud

After training completes, run on login node (has internet):

```bash
./sync_wandb.sh
```

## View Metrics

Visit: https://wandb.ai

## Logged Metrics

- `cls`: Classification loss
- `bbox`: Bounding box loss  
- `horo`: Horospherical loss

## Loss Analysis from Your Training

**Epoch 0 → 47:**
- Classification: 51.2 → 23.6 (✓ converged)
- BBox: 39.8 → 35.8 (✓ improving)
- Horospherical: 2.2 → 0.41 (✓ 81% reduction)
- Proto bias: 0.0 → -0.268 (✓ learning hyperbolic structure)

**Status:** Training is stable and converging well!
