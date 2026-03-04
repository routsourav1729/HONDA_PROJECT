#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --output=slurm_logs/diagnose_%j.out
#SBATCH --error=slurm_logs/diagnose_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Diagnostic: where do GT unknown objects end up?
# ============================================================================

TASK="IDD_HYP/t2"
CKPT="IDD_HYP/t2/fewshotfinetunev3/model_final.pth"
ALPHA_ARG=""
# Limit images for quick test (0 = all)
MAX_IMAGES=0

eval "$(conda shell.bash hook)"
conda activate ovow2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
echo "Task: $TASK"
echo "Checkpoint: $CKPT"

python diagnose.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $CKPT \
    --max_images $MAX_IMAGES \
    $ALPHA_ARG

echo "Job finished at: $(date)"
