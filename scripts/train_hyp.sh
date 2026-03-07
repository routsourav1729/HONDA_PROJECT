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
# T1: Base Training (8 known classes)  ← ACTIVE
# ============================================================================

echo "=========================================="
echo "T1: Base Training (V2)"
echo "  Pretrained   : $PRETRAINED_CKPT"
echo "  Output       : IDD_HYP/t1/$EXP_NAME"
echo "=========================================="

TASK_T1="IDD_HYP/t1"
mkdir -p $TASK_T1/$EXP_NAME

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    dev_hyp_ddp.py \
    --config-file configs/IDD_HYP/base.yaml \
    --task $TASK_T1 \
    --ckpt $PRETRAINED_CKPT \
    --exp_name $EXP_NAME \
    --wandb

# ============================================================================
# T2: Few-Shot Fine-Tuning  ← COMMENTED (run after T1 is done)
# ============================================================================
# Update T1_EXP_NAME to match whatever EXP_NAME was used for T1.
#
# T1_EXP_NAME="hyp_v2"
# T1_CKPT="IDD_HYP/t1/${T1_EXP_NAME}/model_final.pth"
# T2_PROTOS="datasets/prototype/init_protos_t2.pt"
#
# if [ ! -f "$T1_CKPT" ]; then echo "ERROR: $T1_CKPT not found"; exit 1; fi
# if [ ! -f "$T2_PROTOS" ]; then echo "ERROR: $T2_PROTOS not found"; exit 1; fi
#
# TASK_T2="IDD_HYP/t2"
# mkdir -p $TASK_T2/$EXP_NAME
#
# torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --master_port=$MASTER_PORT \
#     dev_hyp_ddp.py \
#     --config-file configs/IDD_HYP/base.yaml \
#     --task $TASK_T2 \
#     --ckpt $T1_CKPT \
#     --init_protos $T2_PROTOS \
#     --exp_name $EXP_NAME \
#     --wandb

echo ""
echo "=========================================="
echo "Training complete at: $(date)"
echo "=========================================="
