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

TASK="IDD_HYP/t1"
CKPT="IDD_HYP/t1/horospherical_v2/model_30.pth"

# Hyperbolic hyperparameters (must match training)
# NOTE: For horospherical classifiers, c=1.0 (ball radius=1) is standard
HYP_C=1.0
HYP_DIM=256
CLIP_R=0.95

# ---- OOD Strategy (choose one) ----
# Option A: Global threshold (legacy)
#   OOD_THRESHOLD=0.0
#   ADAPTIVE_ARGS=""

# Option B: Adaptive per-prototype threshold (recommended)
#   Uses calibration stats from debug/adaptive_threshold_analysis.py
OOD_THRESHOLD=0.0
ADAPTIVE_JSON="visualizations/adaptive/adaptive_stats_model_30.json"
ALPHA=0.75
ADAPTIVE_ARGS="--adaptive_threshold $ADAPTIVE_JSON --alpha $ALPHA"

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
    --hyp_c $HYP_C \
    --hyp_dim $HYP_DIM \
    --clip_r $CLIP_R \
    --ood_threshold $OOD_THRESHOLD \
    $ADAPTIVE_ARGS

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
