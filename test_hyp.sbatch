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

TASK="IDD/t1"
CKPT="IDD/t1/hyperbolic_sep/model_final.pth"

# Hyperbolic hyperparameters (must match training)
HYP_C=0.1
HYP_DIM=256
CLIP_R=2.3
# OOD threshold - hyperbolic distances are typically 2.0-3.0
# Higher threshold = fewer unknowns (more permissive)
# Lower threshold = more unknowns (more strict)
OOD_THRESHOLD=5

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
    --config-file configs/IDD/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c $HYP_C \
    --hyp_dim $HYP_DIM \
    --clip_r $CLIP_R \
    --ood_threshold $OOD_THRESHOLD

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
