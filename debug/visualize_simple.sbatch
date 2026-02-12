#!/bin/bash
#SBATCH --job-name=vis_simple
#SBATCH --output=slurm_logs/vis_simple_%j.out
#SBATCH --error=slurm_logs/vis_simple_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# ============================================================================
# Simple Poincaré Visualization
# ============================================================================

# Base directory
BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/horospherical/model_latest.pth"

echo "=========================================="
echo "POINCARÉ VISUALIZATION"
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

mkdir -p visualizations

python debug/visualize_simple.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 1.0 \
    --hyp_dim 256 \
    --clip_r 0.95 \
    --output_dir visualizations \
    --num_batches 50 \
    --samples_per_class 50

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
