#!/bin/bash
# ============================================================================
# Hyperbolic YOLO-World Training Script (DDP)
# ============================================================================
# Called by train_hyp_ddp.sbatch. Contains T1 and T2 training commands.
# Comment/uncomment sections as needed.
#
# Usage (called from sbatch, not directly):
#   bash train_hyp.sh <NUM_GPUS> <MASTER_PORT> <EXP_NAME>
# ============================================================================

NUM_GPUS=${1:-2}
MASTER_PORT=${2:-29500}
EXP_NAME=${3:-horospherical}
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

# ============================================================================
# T1: Base Training (8 known classes)
# ============================================================================
# Comment this section out when running T2.

# echo "=========================================="
# echo "T1: Base Training"
# echo "=========================================="
#
# TASK_T1="IDD_HYP/t1"
# mkdir -p $TASK_T1/$EXP_NAME
#
# torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     dev_hyp_ddp.py \
#     --config-file configs/IDD_HYP/base.yaml \
#     --task $TASK_T1 \
#     --ckpt $PRETRAINED_CKPT \
#     --exp_name $EXP_NAME \
#     --wandb

# ============================================================================
# T2: Few-Shot Fine-Tuning (8 base + 6 novel classes)
# ============================================================================
# Uncomment this section and comment T1 above when ready for T2.
# T1_EXP_NAME: the experiment folder that T1 was saved under (may differ from T2's EXP_NAME)
#
T1_EXP_NAME="horo_2conv"
T1_CKPT="IDD_HYP/t1/${T1_EXP_NAME}/model_final.pth"
T2_PROTOS="init_protos_t2.pt"

# --- Pre-flight checks ---
if [ ! -f "$T1_CKPT" ]; then
    echo "ERROR: T1 checkpoint not found: $T1_CKPT"
    exit 1
fi
if [ ! -f "$T2_PROTOS" ]; then
    echo "ERROR: T2 prototype file not found: $T2_PROTOS"
    exit 1
fi

echo "=========================================="
echo "T2: Few-Shot Fine-Tuning"
echo "  T1 checkpoint : $T1_CKPT"
echo "  Novel protos  : $T2_PROTOS"
echo "  T2 output     : IDD_HYP/t2/$EXP_NAME"
echo "=========================================="

TASK_T2="IDD_HYP/t2"
mkdir -p $TASK_T2/$EXP_NAME

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    dev_hyp_ddp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK_T2 \
    --ckpt $T1_CKPT \
    --init_protos $T2_PROTOS \
    --exp_name $EXP_NAME \
    --wandb

echo ""
echo "=========================================="
echo "Training complete at: $(date)"
echo "=========================================="
