#!/bin/bash
#SBATCH --job-name=hyp_t2_fs
#SBATCH --output=slurm_logs/hyp_ddp_%j.out
#SBATCH --error=slurm_logs/hyp_ddp_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --exclude=cn15-dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00

# ============================================================================
# vMF Hyperspherical YOLO-World Training - DDP (single or multi-GPU)
# ============================================================================
# Usage:
#   sbatch train_hyp_ddp.sbatch                          # T2 few-shot (default)
#   sbatch train_hyp_ddp.sbatch t2 vmf_t2_fs 2 IDD_HYP/t1/vmf_v1/model_final.pth
# ============================================================================

# Arguments
SPLIT=${1:-t2}
EXP_NAME=${2:-vmf_t2_fs}
NUM_GPUS=${3:-1}
T2_CKPT=${4:-IDD_HYP/t1/vmf_v1/model_final.pth}

TASK="IDD_HYP/$SPLIT"

echo "=========================================="
echo "vMF HYPERSPHERICAL YOLO-WORLD DDP TRAINING"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Experiment: $EXP_NAME"
echo "  GPUs: $NUM_GPUS"
echo "  T2 init ckpt: $T2_CKPT"
echo "  Output dir: $TASK/$EXP_NAME"
echo "=========================================="

# Activate conda environment
source /home/agipml/sourav.rout/miniconda3/bin/activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Set PYTHONPATH
export PYTHONPATH=/home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2/YOLO-World:$PYTHONPATH

# CLIP/HuggingFace cache (offline on compute nodes)
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache

# Offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# DDP settings
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

echo ""
echo "Environment Information:"
echo "  Python: $(which python)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  MASTER_PORT: $MASTER_PORT"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')"
echo ""

# Create directories
mkdir -p slurm_logs
mkdir -p $TASK/$EXP_NAME

echo "=========================================="
echo "Starting DDP TRAINING with $NUM_GPUS GPUs"
echo "=========================================="

bash scripts/train_hyp.sh $SPLIT $NUM_GPUS $MASTER_PORT $EXP_NAME $T2_CKPT

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
