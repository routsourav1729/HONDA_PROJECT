#!/bin/bash
#SBATCH --job-name=abl_B
#SBATCH --output=slurm_logs/abl_B_%j.out
#SBATCH --error=slurm_logs/abl_B_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ============================================================================
# Ablation B: CE + dispersion(0.1) + compactness(0.05) + bias_reg(0.1)
# Same as A but with bias L2 reg to constrain horosphere size.
# ============================================================================

SPLIT=${1:-t1}
TASK="IDD_HYP/$SPLIT"
EXP_NAME="abl_B_compact_biasreg"
NUM_GPUS=2
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

echo "=========================================="
echo "ABLATION B: CE + disp(0.1) + compact(0.05) + bias_reg(0.1)"
echo "Job: $SLURM_JOB_ID | $(date)"
echo "=========================================="

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
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
MASTER_PORT=$(( 29500 + RANDOM % 1000 ))
export MASTER_PORT
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / NUM_GPUS ))

mkdir -p slurm_logs $TASK/$EXP_NAME

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    dev_hyp_ddp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK \
    --ckpt $PRETRAINED_CKPT \
    --exp_name $EXP_NAME \
    --dispersion_weight 0.1 \
    --bias_reg_weight 0.1 \
    --compactness_weight 0.05 \
    --wandb

echo "=========================================="
echo "Ablation B finished at: $(date)"
echo "=========================================="
