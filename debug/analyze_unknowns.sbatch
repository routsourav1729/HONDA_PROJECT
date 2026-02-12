#!/bin/bash
#SBATCH --job-name=analyze_unk
#SBATCH --output=slurm_logs/analyze_unk_%j.out
#SBATCH --error=slurm_logs/analyze_unk_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Unknown Embedding Analysis
# Projects all test embeddings (known + unknown) and computes stats
# ============================================================================

BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/horospherical/model_45.pth"

echo "=========================================="
echo "UNKNOWN EMBEDDING ANALYSIS"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH

# Configure offline CLIP cache
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Set offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

mkdir -p visualizations

python debug/analyze_unknowns.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 1.0 \
    --hyp_dim 256 \
    --clip_r 0.95 \
    --output_dir visualizations \
    --num_batches 150 \
    --samples_per_class 500

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
