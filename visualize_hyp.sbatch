#!/bin/bash
#SBATCH --job-name=vis_hyp
#SBATCH --output=slurm_logs/vis_hyp_%j.out
#SBATCH --error=slurm_logs/vis_hyp_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# ============================================================================
# Poincaré Ball Visualization
# ============================================================================

TASK="IDD/t1"
CKPT="IDD/t1/hyperbolic_sep/model_latest.pth"

echo "=========================================="
echo "POINCARÉ BALL VISUALIZATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH

# Configure offline CLIP cache
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Set offline mode flags
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

mkdir -p visualizations

echo "=========================================="
echo "Starting VISUALIZATION"
echo "=========================================="

python debug/visualize_simple.py \
    --config-file configs/IDD/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 0.1 \
    --hyp_dim 256 \
    --clip_r 2.3 \
    --output_dir visualizations \
    --num_batches 100 \
    --samples_per_class 50

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Visualizations saved to: visualizations/"
echo "=========================================="
