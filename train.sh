#!/bin/bash

# ============================================================================
# OVOW Training Script
# ============================================================================
# All training configurations are managed here by commenting/uncommenting
# the desired command. The sbatch file (train_ovow.sbatch) calls this script.
# ============================================================================

# Disable tokenizers parallelism warning
export TOKENIZERS_PARALLELISM=false

# Dataset selection
BENCHMARK="IDD"  # Options: IDD, M-OWODB, S-OWODB, nu-OWODB

# ============================================================================
# TRAINING COMMANDS - Comment/Uncomment as needed
# ============================================================================

# ----------------------------------------------------------------------------
# T1: Base Training (from pretrained YOLO-World)
# ----------------------------------------------------------------------------
python dev.py \
    --task ${BENCHMARK}/t1 \
    --config-file configs/${BENCHMARK}/t1.yaml \
    --ckpt yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
    --num-gpus 1

# ----------------------------------------------------------------------------
# T1: Resume from checkpoint (if training was interrupted)
# ----------------------------------------------------------------------------
# python dev.py \
#     --task ${BENCHMARK}/t1 \
#     --config-file configs/${BENCHMARK}/t1.yaml \
#     --ckpt yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth \
#     --resume_from ${BENCHMARK}/t1/model_latest.pth \
#     --num-gpus 1

# ----------------------------------------------------------------------------
# T2: Few-shot Fine-tuning (BASE + NOVEL classes, from T1 checkpoint)
# ----------------------------------------------------------------------------
# python dev.py \
#     --task ${BENCHMARK}/t2 \
#     --config-file configs/${BENCHMARK}/t2.yaml \
#     --ckpt ${BENCHMARK}/t1/model_final.pth \
#     --num-gpus 1

# ----------------------------------------------------------------------------
# T2: Resume few-shot fine-tuning
# ----------------------------------------------------------------------------
# python dev.py \
#     --task ${BENCHMARK}/t2 \
#     --config-file configs/${BENCHMARK}/t2.yaml \
#     --ckpt ${BENCHMARK}/t1/model_final.pth \
#     --resume_from ${BENCHMARK}/t2/model_latest.pth \
#     --num-gpus 1

# ----------------------------------------------------------------------------
# T3: Continue incremental learning (if applicable)
# ----------------------------------------------------------------------------
# python dev.py \
#     --task ${BENCHMARK}/t3 \
#     --config-file configs/${BENCHMARK}/t3.yaml \
#     --ckpt ${BENCHMARK}/t2/model_final.pth \
#     --num-gpus 1

# ----------------------------------------------------------------------------
# T4: Final stage (if applicable)
# ----------------------------------------------------------------------------
# python dev.py \
#     --task ${BENCHMARK}/t4 \
#     --config-file configs/${BENCHMARK}/t4.yaml \
#     --ckpt ${BENCHMARK}/t3/model_final.pth \
#     --num-gpus 1

# ============================================================================
# NOTES:
# ============================================================================
# - T1 (COMPLETED): Base classes trained, model at IDD/t1/model_final.pth
# - T2 (ACTIVE): Few-shot fine-tuning with 18 classes (11 base + 7 novel)
#   - Using 10-shot by default (136 images, BASE+NOVEL classes)
#   - Data: datasets/FewShot_Annotations/10shot/
#   - Config: configs/IDD/t2.py (fewshot_k=10)
# - For T1 training: Use pretrained YOLO-World checkpoint
# - For resume: Use model_latest.pth from the same stage
# - Change BENCHMARK variable for different datasets
# - Adjust --num-gpus based on available GPUs
# ============================================================================


