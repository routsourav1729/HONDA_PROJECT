#!/bin/bash

BENCHMARK=${BENCHMARK:-"IDD"}  # M-OWODB, S-OWODB or nu-OWODB or IDD

# ============================================================================
# T1 Evaluation: 11 known classes + 1 unknown
# ============================================================================
# python test.py --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --ckpt ${BENCHMARK}/t1/model_final.pth

# ============================================================================
# T2 Evaluation: 18 known classes (11 base + 7 novel) + 1 unknown
# Few-shot fine-tuned model
# ============================================================================
python test.py --task ${BENCHMARK}/t2 --config-file configs/${BENCHMARK}/t2.yaml --ckpt ${BENCHMARK}/t2/model_final.pth

# ============================================================================
# T3 Evaluation (if available)
# ============================================================================
# python test.py --task ${BENCHMARK}/t3 --config-file configs/${BENCHMARK}/t3.yaml --ckpt ${BENCHMARK}/t3/model_final.pth

# ============================================================================
# T4 Evaluation (if available)
# ============================================================================
# python test.py --task ${BENCHMARK}/t4 --config-file configs/${BENCHMARK}/t4.yaml --ckpt ${BENCHMARK}/t4/model_final.pth
