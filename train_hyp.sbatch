#!/bin/bash
#SBATCH --job-name=hyp_train
#SBATCH --output=slurm_logs/hyp_%j.out
#SBATCH --error=slurm_logs/hyp_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================
# Hyperbolic YOLO-World Training - SBATCH Script
# ============================================================================
# Usage:
#   sbatch train_hyp.sbatch [exp_name]
#
# Examples:
#   sbatch train_hyp.sbatch                    # saves to IDD/t1/hyperbolic
#   sbatch train_hyp.sbatch hyp_v2             # saves to IDD/t1/hyp_v2
#   sbatch train_hyp.sbatch hyp_c0.05_temp0.2  # saves to IDD/t1/hyp_c0.05_temp0.2
# ============================================================================

# Configuration - EDIT THESE
TASK="IDD/t1"
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

# Hyperbolic hyperparameters - EDIT THESE
HYP_C=0.1           # Poincaré ball curvature (ball radius = 1/sqrt(c) = 3.16)
HYP_DIM=256         # Embedding dimension
CLIP_R=2.3          # Clip radius
TEMPERATURE=0.1     # Contrastive loss temperature
HYP_LOSS_WEIGHT=1.0 # Weight for hyperbolic loss

# Prototype separation hyperparameters - keeps prototypes spread apart
# separation_weight: 0.1 is conservative, 0.5 is aggressive
# min_proto_dist: minimum pairwise distance between prototypes (in hyperbolic space)
SEPARATION_WEIGHT=0.1   # Weight for separation loss (0=disabled)
MIN_PROTO_DIST=1.5      # Minimum distance between prototypes

# Boundary push hyperparameters - pushes prototypes toward ball boundary
# boundary_weight: 0.1 is conservative
# target_norm: FRACTION of ball radius (0.9 = 90% toward boundary = 2.85 for c=0.1)
BOUNDARY_WEIGHT=0.1     # Weight for boundary push loss (0=disabled)
TARGET_NORM=0.9         # Target = 90% of ball radius

# Experiment name (from argument or default)
EXP_NAME=${1:-hyperbolic_sep}

echo "=========================================="
echo "HYPERBOLIC YOLO-WORLD TRAINING"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Experiment: $EXP_NAME"
echo "  Output dir: $TASK/$EXP_NAME"
echo ""
echo "Hyperbolic params:"
echo "  c=$HYP_C (ball_radius=$(python3 -c "import math; print(f'{1/math.sqrt($HYP_C):.2f}')"))"
echo "  dim=$HYP_DIM, clip_r=$CLIP_R"
echo "  temperature=$TEMPERATURE, loss_weight=$HYP_LOSS_WEIGHT"
echo "Separation params:"
echo "  separation_weight=$SEPARATION_WEIGHT, min_proto_dist=$MIN_PROTO_DIST"
echo "Boundary params:"
echo "  boundary_weight=$BOUNDARY_WEIGHT, target_norm=$TARGET_NORM (fraction of radius)"
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
echo "Starting TRAINING"
echo "=========================================="

# Run training
python dev_hyp.py \
    --config-file configs/IDD/base.yaml \
    --task $TASK \
    --ckpt $PRETRAINED_CKPT \
    --hyp_c $HYP_C \
    --hyp_dim $HYP_DIM \
    --clip_r $CLIP_R \
    --temperature $TEMPERATURE \
    --hyp_loss_weight $HYP_LOSS_WEIGHT \
    --separation_weight $SEPARATION_WEIGHT \
    --min_proto_dist $MIN_PROTO_DIST \
    --boundary_weight $BOUNDARY_WEIGHT \
    --target_norm $TARGET_NORM \
    --exp_name $EXP_NAME

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Models saved to: $TASK/$EXP_NAME"
echo "=========================================="
