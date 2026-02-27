#!/bin/bash
#SBATCH --job-name=patch_ckpt
#SBATCH --output=slurm_logs/patch_ckpt_%j.out
#SBATCH --error=slurm_logs/patch_ckpt_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# ============================================================================
# Patch existing checkpoint with adaptive thresholds + hyp_config
# For checkpoints saved BEFORE calibration was integrated into training.
# ============================================================================

TASK="IDD_HYP/t1"
CKPT="IDD_HYP/t1/horospherical_v2/model_30.pth"
CLIP_R=3.0  # MUST match what was used during training!

echo "=========================================="
echo "PATCH CHECKPOINT WITH ADAPTIVE THRESHOLDS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "clip_r: $CLIP_R"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate ovow2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python patch_checkpoint.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --clip_r $CLIP_R

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Now run: sbatch test_hyp.sbatch"
echo "=========================================="
