#!/bin/bash
#SBATCH --job-name=gpm_bases
#SBATCH --output=slurm_logs/gpm_bases_%j.out
#SBATCH --error=slurm_logs/gpm_bases_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Usage:
#   sbatch scripts/compute_gpm_bases.sbatch
#   sbatch scripts/compute_gpm_bases.sbatch path/to/model_final.pth

CKPT=${1:-IDD_HYP/t1/vmf_v1/model_final.pth}

echo "=========================================="
echo "GPM Bases Computation"
echo "Job: $SLURM_JOB_ID | $(date)"
echo "Checkpoint: $CKPT"
echo "=========================================="

source /home/agipml/sourav.rout/miniconda3/bin/activate ovow2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

python compute_gpm_bases.py \
    --ckpt "$CKPT" \
    --threshold 0.97 \
    --max_batches 20

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
