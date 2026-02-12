#!/bin/bash
#SBATCH --job-name=hyp_v2
#SBATCH --output=slurm_logs/hyp_v2_%j.out
#SBATCH --error=slurm_logs/hyp_v2_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# Hyperbolic YOLO-World V2 Training - With Regularization Fixes
# ============================================================================
# Changes from V1:
#   1. clip_r=2.0 (was 0.95) → allows norm variation for OOD discrimination
#   2. bias_reg_weight=0.1   → L2 penalty on biases to prevent horosphere inflation
#   3. dispersion_weight=0.1 → push prototype directions apart
#   4. compactness_weight=0.05 → pull known embeddings into their horosphere
#
# These fixes address the analysis showing:
#   - All norms collapsed to 0.7398 (clip_r=0.95 → tanh(0.95) = 0.7398)
#   - Biases grew to [0.88, 0.93, ...] inflating horospheres
#   - Unknown scores (1.394) ≈ Known scores (1.357) → zero OOD discrimination
# ============================================================================

SPLIT=${1:-t1}
EXP_NAME=${2:-horospherical_v2}
TASK="IDD_HYP/$SPLIT"
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

echo "=========================================="
echo "HYPERBOLIC YOLO-WORLD V2 TRAINING"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Experiment: $EXP_NAME"
echo "  Key changes: clip_r=2.0, bias_reg=0.1, disp=0.1, compact=0.05"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH to prioritize hypyolov2's YOLO-World over installed version
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
echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
echo ""

# Create directories
mkdir -p slurm_logs
mkdir -p $TASK/$EXP_NAME

echo "=========================================="
echo "Starting V2 TRAINING (fresh start with regularization)"
echo "=========================================="

# Train from scratch with new regularization
# The config already has the updated values (clip_r=2.0, etc.)
# CLI overrides ensure the new params are used even if config cache is stale
python dev_hyp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $PRETRAINED_CKPT \
    --exp_name $EXP_NAME \
    --clip_r 2.0 \
    --dispersion_weight 0.1 \
    --bias_reg_weight 0.1 \
    --compactness_weight 0.05

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
