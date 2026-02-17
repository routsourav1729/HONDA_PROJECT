#!/bin/bash
#SBATCH --job-name=adaptive_thr
#SBATCH --output=slurm_logs/adaptive_thr_%j.out
#SBATCH --error=slurm_logs/adaptive_thr_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# ============================================================================
# Adaptive Per-Prototype Threshold Analysis
# Phase 1: Calibrate from TRAIN data (compute per-class mean/std)
# Phase 2: Evaluate on TEST data with multiple alphas
# ============================================================================

BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/horospherical_v2/model_30.pth"

echo "=========================================="
echo "ADAPTIVE THRESHOLD ANALYSIS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH

# Configure offline CLIP cache
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Set offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

mkdir -p visualizations/adaptive

python debug/adaptive_threshold_analysis.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 1.0 \
    --hyp_dim 256 \
    --clip_r 0.95 \
    --output_dir visualizations/adaptive

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
