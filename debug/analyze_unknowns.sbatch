#!/bin/bash
#SBATCH --job-name=analyze_unk
#SBATCH --output=slurm_logs/analyze_unk_%j.out
#SBATCH --error=slurm_logs/analyze_unk_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# ============================================================================
# Unknown Embedding Analysis (train split)
# Projects all train embeddings (known + unknown) and computes stats
# + 2D Poincare ball visualization with horospheres
# ============================================================================

BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

TASK="IDD_HYP/t1"
CKPT="$BASE_DIR/IDD_HYP/t1/my_experiment/model_final.pth"
# clip_r MUST match the value used during training
CLIP_R=3.0
SPLIT="train"

echo "=========================================="
echo "UNKNOWN EMBEDDING ANALYSIS ($SPLIT)"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "clip_r: $CLIP_R"
echo "Split: $SPLIT"
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
    --clip_r $CLIP_R \
    --split $SPLIT \
    --output_dir visualizations

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
