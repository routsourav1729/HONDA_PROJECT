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
#   sbatch train_hyp.sbatch              # T1 training
#   sbatch train_hyp.sbatch t2           # T2 training
#   sbatch train_hyp.sbatch t1 my_exp    # Custom exp name
# ============================================================================

# Arguments
SPLIT=${1:-t1}
EXP_NAME=${2:-horospherical}

TASK="IDD_HYP/$SPLIT"
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

echo "=========================================="
echo "HYPERBOLIC YOLO-WORLD TRAINING"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Experiment: $EXP_NAME"
echo "  Output dir: $TASK/$EXP_NAME"
echo ""
echo "NOTE: All hyperbolic params are in configs/IDD_HYP/$SPLIT.py"
echo "      Edit hyp_config section to change: curvature, embed_dim, clip_r, etc."
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

python dev_hyp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $PRETRAINED_CKPT \
    --exp_name $EXP_NAME \
    --wandb

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "To sync WandB: ./sync_wandb.sh"
echo "=========================================="

