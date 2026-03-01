#!/bin/bash
#SBATCH --job-name=init_protos
#SBATCH --output=slurm_logs/init_protos_%j.out
#SBATCH --error=slurm_logs/init_protos_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Initialize prototype directions for horospherical classification.
# Run ONCE before training whenever classes change.
#
# Usage:
#   sbatch init_protos.sbatch              # runs init_protos.sh as-is
#   sbatch init_protos.sbatch <t1_ckpt>    # pass T1 checkpoint path for T2
#
# Edit init_protos.sh to switch between T1 / T2 (comment/uncomment).
# ============================================================================

echo "=========================================="
echo "PROTOTYPE INITIALIZATION"
echo "Job: $SLURM_JOB_ID | $(date)"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate ovow2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

bash init_protos.sh "$@"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
