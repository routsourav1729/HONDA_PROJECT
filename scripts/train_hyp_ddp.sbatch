#!/bin/bash
#SBATCH --job-name=hyp_t1
#SBATCH --output=slurm_logs/hyp_ddp_%j.out
#SBATCH --error=slurm_logs/hyp_ddp_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --exclude=cn15-dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=30:00:00

# ============================================================================
# Hyperbolic YOLO-World Training - DDP (single or multi-GPU)
# ============================================================================
# Usage:
#   sbatch train_hyp_ddp.sbatch              # T1, 1 GPU on a40
#   sbatch train_hyp_ddp.sbatch t1 my_exp    # T1, custom exp name
# ============================================================================

# Arguments
SPLIT=${1:-t1}
EXP_NAME=${2:-hyp_v2}
NUM_GPUS=${3:-2}

TASK="IDD_HYP/$SPLIT"
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

echo "=========================================="
echo "HYPERBOLIC YOLO-WORLD DDP TRAINING"
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Experiment: $EXP_NAME"
echo "  GPUs: $NUM_GPUS"
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

bash scripts/train_hyp.sh $NUM_GPUS $MASTER_PORT $EXP_NAME

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
