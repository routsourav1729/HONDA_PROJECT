#!/bin/bash
# ============================================================================
# Hyperbolic YOLO-World Evaluation Script
# ============================================================================
# Called by eval_ovow.sbatch. Uses test_hyp.py (Busemann + adaptive thresholds).
# Comment/uncomment sections as needed.
#
# Usage:
#   bash test_owod.sh                           # defaults
#   BENCHMARK=IDD_HYP EXP=horo_2conv bash test_owod.sh  # override
# ============================================================================

BENCHMARK=${BENCHMARK:-"IDD_HYP"}
EXP=${EXP:-"horo_2conv"}   # experiment folder name under each task dir



# ============================================================================
# T1 Evaluation: 8 base classes + 1 unknown  (IDD)
# ============================================================================
# python test_hyp.py \
#     --task ${BENCHMARK}/t1 \
#     --config-file configs/${BENCHMARK}/t1.yaml \
#     --ckpt ${BENCHMARK}/t1/${EXP}/model_final.pth




# ============================================================================
# T2 Evaluation: 8 base + 6 novel = 14 known + 1 unknown  (IDD)
# Few-shot fine-tuned model with merged calibration (T1 base + T2 novel)
# ============================================================================
T2_EXP=${T2_EXP:-"fewshotfinetunev2"}
python test_hyp.py \
    --task ${BENCHMARK}/t2 \
    --config-file configs/${BENCHMARK}/t2.yaml \
    --ckpt ${BENCHMARK}/t2/${T2_EXP}/model_final.pth

