#!/bin/bash
# ============================================================================
# Hyperbolic YOLO-World Training Script (DDP)
# ============================================================================
# Called by train_hyp_ddp.sbatch.
# Active path is T2 few-shot fine-tuning.
# T1 block is intentionally disabled in this launcher.
#
# Usage (called from sbatch, not directly):
#   bash train_hyp.sh <SPLIT> <NUM_GPUS> <MASTER_PORT> <EXP_NAME> <T2_CKPT>
# ============================================================================

set -euo pipefail

SPLIT=${1:-t2}
NUM_GPUS=${2:-2}
MASTER_PORT=${3:-29500}
EXP_NAME=${4:-vmf_t2_fs}
T2_CKPT=${5:-IDD_HYP/t1/vmf_v1/model_final.pth}
PRETRAINED_CKPT="yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"

# ============================================================================
# T1: Base Training (8 known classes)
# ============================================================================
# Disabled for few-shot runs. Kept as reference only.

# TASK_T1="IDD_HYP/t1"
# mkdir -p $TASK_T1/$EXP_NAME
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
# T2: Few-Shot Fine-Tuning with GPM
# ============================================================================
# Requires: T1 checkpoint + T2 prototype init + GPM bases (auto-computed if missing)

T2_PROTOS="datasets/prototype/init_protos_t2.pt"

if [ "$SPLIT" != "t2" ]; then
    echo "ERROR: This launcher is configured for T2 few-shot only. Received SPLIT=$SPLIT"
    echo "Set SPLIT=t2, or run T1 from a dedicated script."
    exit 1
fi

if [ ! -f "$T2_CKPT" ]; then echo "ERROR: $T2_CKPT not found"; exit 1; fi
if [ ! -f "$T2_PROTOS" ]; then echo "ERROR: $T2_PROTOS not found"; exit 1; fi

# ============================================================================
# GPM Bases: compute from T1 model if not already present
# ============================================================================
GPM_BASES="$(dirname $T2_CKPT)/gpm_bases.pt"
if [ ! -f "$GPM_BASES" ]; then
    echo "=========================================="
    echo "GPM Bases not found — computing from T1 model"
    echo "  T1 ckpt : $T2_CKPT"
    echo "  Output  : $GPM_BASES"
    echo "=========================================="
    python compute_gpm_bases.py --ckpt "$T2_CKPT"
    if [ ! -f "$GPM_BASES" ]; then
        echo "ERROR: GPM bases computation failed"; exit 1
    fi
    echo "GPM bases ready: $GPM_BASES"
else
    echo "GPM bases already exist: $GPM_BASES"
fi

echo "=========================================="
echo "T2: Few-Shot Fine-Tuning with GPM"
echo "  T1 Checkpoint: $T2_CKPT"
echo "  T2 Prototypes: $T2_PROTOS"
echo "  GPM Bases    : $GPM_BASES"
echo "  Output       : IDD_HYP/t2/$EXP_NAME"
echo "=========================================="

TASK_T2="IDD_HYP/t2"
mkdir -p $TASK_T2/$EXP_NAME

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    dev_hyp_ddp.py \
    --config-file configs/IDD_HYP/t2.yaml \
    --task $TASK_T2 \
    --ckpt $T2_CKPT \
    --exp_name $EXP_NAME \
    --wandb

echo ""
echo "=========================================="
echo "Training complete at: $(date)"
echo "=========================================="
