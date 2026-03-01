#!/bin/bash
#SBATCH --job-name=test_hyp
#SBATCH --output=slurm_logs/test_hyp_%j.out
#SBATCH --error=slurm_logs/test_hyp_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Hyperbolic YOLO-World Evaluation
# ============================================================================
# Everything is loaded from the checkpoint:
#   - hyp_config (curvature, embed_dim, clip_r)
#   - adaptive_stats (per-class thresholds)
# Just point CKPT to the model and run.
# ============================================================================

TASK="IDD_HYP/t1"
CKPT="IDD_HYP/t1/4gpu_1x1_40ep/model_final.pth"
# CKPT="IDD_HYP/t1/horo_2conv/model_final.pth"


# Optional: override alpha (default=0.75 from calibration)
# ALPHA_ARG="--alpha 1.0"
ALPHA_ARG=""

echo "=========================================="
echo "HYPERBOLIC YOLO-WORLD EVALUATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH to prioritize hypyolov2's YOLO-World
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH

# Configure offline CLIP cache (no internet on compute nodes)
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Set offline mode flags
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Verify environment
echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "=========================================="
echo "Starting EVALUATION"
echo "=========================================="

python test_hyp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    $ALPHA_ARG

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
