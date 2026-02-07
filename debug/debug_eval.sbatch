#!/bin/bash
#SBATCH --job-name=debug_eval
#SBATCH --output=slurm_logs/debug_eval_%j.out
#SBATCH --error=slurm_logs/debug_eval_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Debug Evaluation - Analyze hyperbolic distances
# ============================================================================

TASK="IDD/t1"
CKPT="IDD/t1/hyperbolic/model_0.pth"
# MODE: 'stats' (just distances), 'compare' (with/without OOD), 'full' (normal eval)
MODE=${1:-stats}

echo "=========================================="
echo "DEBUG EVALUATION"
echo "=========================================="
echo "Job started at: $(date)"
echo "Mode: $MODE"
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

python debug/eval_debug.py \
    --config-file configs/IDD/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --hyp_c 0.1 \
    --hyp_dim 256 \
    --clip_r 2.3 \
    --ood_threshold 2.5 \
    --mode $MODE

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
