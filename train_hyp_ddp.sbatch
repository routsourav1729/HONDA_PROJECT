#!/bin/bash
#SBATCH --job-name=hyp_ddp
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
#SBATCH --time=24:00:00

# ============================================================================
# Hyperbolic YOLO-World Training - Multi-GPU DDP
# ============================================================================
# Usage:
#   sbatch train_hyp_ddp.sbatch              # T1 with 2 GPUs
#   sbatch train_hyp_ddp.sbatch t1 my_exp 4  # T1, custom name, 4 GPUs
#
# To change GPU count, edit --gres=gpu:N and --cpus-per-task accordingly.
# Rule of thumb: cpus-per-task = 8 * num_gpus
# ============================================================================

# Arguments
SPLIT=${1:-t1}
EXP_NAME=${2:-horospherical}
# NUM_GPUS must match --gres=gpu:N in the SBATCH header above!
# If you change this default, also change --gres=gpu:N and --cpus-per-task=(8*N)
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
eval "$(conda shell.bash hook)"
conda activate ovow2

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
# Use a random port to avoid conflicts with other jobs
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT

# Set OMP_NUM_THREADS to suppress torchrun warning and tune CPU threading.
# Rule: total_cpus / num_gpus  (e.g. 16 cpus / 2 gpus = 8)
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

# Use torchrun for DDP launch
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    dev_hyp_ddp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $PRETRAINED_CKPT \
    --exp_name $EXP_NAME \
    --wandb

echo ""
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
