#!/bin/bash
#SBATCH --job-name=vis_qual
#SBATCH --output=slurm_logs/vis_qual_%j.out
#SBATCH --error=slurm_logs/vis_qual_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=cn15-dgx

# ============================================================================
# Qualitative Visualization: GT vs HypYOLO Predictions
# Runs inference on selected test images and generates side-by-side figures.
# ============================================================================

BASE_DIR="/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2"

# --- Configuration ---
TASK="IDD_HYP/t2"
T2_EXP="fewshotfinetunev3"
CKPT="$BASE_DIR/IDD_HYP/t2/${T2_EXP}/model_final.pth"
OUTPUT_DIR="$BASE_DIR/visualizations/qualitative_${T2_EXP}"

# Selected images: large animals (>100x100) + max novel/unknown diversity
# 080205: bus+truck+animal+concrete_mixer (Div=4, animal=333x376, 35 obj)
# 080221: bus+street_cart+animal+concrete_mixer (Div=4, animal=343x198, 60 obj)
# 075538: truck+animal+pull_cart (Div=3, animal=227x357, 28 obj)
# 076054: truck+animal+pull_cart (Div=3, animal=317x254, 17 obj, clean)
# 094255: street_cart+truck+animal (Div=3, animal=246x168, 9 obj, very clean)
# 110394: bus+truck+animal (Div=3, animal=209x251, 62 obj)
IMAGES="080205 080221 075538 076054 094255 110394"

echo "=========================================="
echo "QUALITATIVE VISUALIZATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Checkpoint: $CKPT"
echo "Images: $IMAGES"
echo "Output: $OUTPUT_DIR"
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

cd "$BASE_DIR"
mkdir -p slurm_logs
mkdir -p "$OUTPUT_DIR"

python debug/visualize_qualitative.py \
    --config-file configs/IDD_HYP/t2.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --images $IMAGES \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
