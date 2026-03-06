#!/bin/bash
#SBATCH --job-name=vis_hyp
#SBATCH --output=slurm_logs/vis_hyp_%j.out
#SBATCH --error=slurm_logs/vis_hyp_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:50:00

# ============================================================================
# Poincaré Ball Visualization
# ============================================================================

# Base directory
BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="$BASE_DIR/IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/hyp_v2/model_latest.pth"

echo "=========================================="
echo "POINCARÉ BALL VISUALIZATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

# Activate conda environment
source /home/agipml/sourav.rout/miniconda3/bin/activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH=$BASE_DIR/YOLO-World:$PYTHONPATH

# Change to working directory
cd $BASE_DIR

# Configure offline CLIP cache
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Set offline mode flags
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

mkdir -p visualizations

echo "=========================================="
echo "Starting VISUALIZATION"
echo "=========================================="

python debug/visualize_simple.py \
    --config-file configs/IDD_HYP/t1.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 1.0 \
    --hyp_dim 64 \
    --clip_r 1.0 \
    --output_dir visualizations \
    --num_batches 100 \
    --samples_per_class 50 \
    --projection hyperbolic_umap \
    --n_neighbors 15 \
    --min_dist 0.1

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Visualizations saved to: visualizations/"
echo "=========================================="
