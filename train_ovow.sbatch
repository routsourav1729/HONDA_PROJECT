#!/bin/bash
#SBATCH --job-name=ovow_train
#SBATCH --output=slurm_logs/ovow_%j.out
#SBATCH --error=slurm_logs/ovow_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# ============================================================================
# OVOW Training/Evaluation - Unified SBATCH Script
# ============================================================================
# Usage:
#   Training: sbatch train_ovow.sbatch
#   Evaluation: sbatch train_ovow.sbatch eval
#
# To change training mode/task:
#   Edit train.sh and comment/uncomment the desired command
# To change evaluation mode/task:
#   Edit test_owod.sh and comment/uncomment the desired command
# ============================================================================

echo "=========================================="
echo "OVOW TRAINING / EVALUATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
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
python -c "import mmcv; print(f'  mmcv: {mmcv.__version__}')"
python -c "import detectron2; print(f'  detectron2: {detectron2.__version__}')"
echo ""

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Check if running evaluation or training
if [ "$1" = "eval" ]; then
    echo "=========================================="
    echo "Starting EVALUATION via test_owod.sh"
    echo "=========================================="
    bash test_owod.sh
else
    echo "=========================================="
    echo "Starting TRAINING via train.sh"
    echo "=========================================="
    bash train.sh
fi

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
