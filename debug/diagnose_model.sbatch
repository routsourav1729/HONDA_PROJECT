#!/bin/bash
#SBATCH --job-name=diag_hyp
#SBATCH --output=slurm_logs/diag_hyp_%j.out
#SBATCH --error=slurm_logs/diag_hyp_%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --exclude=cn15-dgx

# ============================================================================
# Hyperbolic Model Diagnostic Report
# ============================================================================

BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/hyp_v2/model_latest.pth"
OUTPUT_DIR="$BASE_DIR/diagnostics"

echo "=========================================="
echo "HYPERBOLIC MODEL DIAGNOSTICS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

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

echo "Python: $(which python)"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

mkdir -p "$OUTPUT_DIR"
mkdir -p slurm_logs

echo "=========================================="
echo "Starting DIAGNOSTICS"
echo "=========================================="

python debug/diagnose_model.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --output_dir "$OUTPUT_DIR" \
    --split test

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
