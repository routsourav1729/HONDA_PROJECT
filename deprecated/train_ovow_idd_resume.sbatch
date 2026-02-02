#!/bin/bash
#SBATCH --job-name=ovow_resume
#SBATCH --output=slurm_logs/ovow_idd_%j.out
#SBATCH --error=slurm_logs/ovow_idd_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Print job information
echo "=========================================="
echo "RESUMING TRAINING FROM EPOCH 35"
echo "=========================================="
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ovow2

# Export LD_LIBRARY_PATH for detectron2
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Configure offline CLIP cache (no internet on compute nodes)
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hyp_yolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hyp_yolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hyp_yolo/clip_cache

# Set offline mode flags
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
python -c "import detectron2; print(f'detectron2: {detectron2.__version__}')"

# Create slurm_logs directory if it doesn't exist
mkdir -p slurm_logs

# Training parameters
TASK="IDD/t1"  # Dataset task (IDD with t1 split)
CONFIG_FILE="configs/IDD/t1.yaml"  # Detectron2 config
CKPT="../HONDA/ovow/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"  # Initial pretrained checkpoint (not used when resuming)
RESUME_CKPT="IDD/t1/model_latest.pth"  # Checkpoint to resume from (epoch 35)

echo "Checkpoint to resume from: $RESUME_CKPT"
if [ -f "$RESUME_CKPT" ]; then
    echo "✓ Checkpoint file exists"
    python -c "import torch; ckpt=torch.load('$RESUME_CKPT', map_location='cpu'); print(f'  Resuming from epoch: {ckpt[\"epoch\"]}')"
else
    echo "✗ ERROR: Checkpoint file not found!"
    exit 1
fi

# Run training with resume
echo "=========================================="
echo "Starting OVOW training (RESUME MODE)..."
echo "Will train from epoch 36 to epoch 50"
echo "=========================================="
python dev.py \
    --config-file $CONFIG_FILE \
    --task $TASK \
    --ckpt $CKPT \
    --resume_from $RESUME_CKPT \
    --num-gpus 1

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
