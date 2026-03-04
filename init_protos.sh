#!/bin/bash
# ============================================================================
# Prototype Initialization Script
# ============================================================================
# Called by init_protos.sbatch. Reads class names from text files.
# Comment/uncomment sections to run T1 or T2.
#
# Class files: datasets/ImageSets/Main/IDD/t1_classes.txt
#              datasets/ImageSets/Main/IDD/t2_classes.txt
# ============================================================================

CLASS_DIR="datasets/ImageSets/Main/IDD"
PROMPT="a photo of a {} on an Indian road"

# Helper: read class names from file, join with commas
read_classes() {
    paste -sd',' "$1" | sed 's/,$//'
}

# ============================================================================
# T1: Base prototypes (8 classes)  ← ACTIVE
# ============================================================================
# V2: out_dim=64 (reduced from 256)

T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
echo "=== T1 Prototypes: $T1_CLASSES ==="
python init_prototypes.py \
    --classes "$T1_CLASSES" \
    --out_dim 64 \
    --output init_protos_t1.pt \
    --prompt_template "$PROMPT" \
    --n_iters 3000 \
    --seed 42

# ============================================================================
# T2: Novel prototypes (6 classes, anchored to T1)  ← COMMENTED
# ============================================================================
# Uncomment when T1 is done. Update T1_CKPT to the new hyp_v2 checkpoint.

# T1_CKPT=${1:-"IDD_HYP/t1/hyp_v2/model_final.pth"}
#
# T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
# T2_CLASSES=$(read_classes "$CLASS_DIR/t2_classes.txt")
# ALL_CLASSES="${T1_CLASSES},${T2_CLASSES}"
#
# echo "=== T2 Prototypes ==="
# echo "  All classes (14): $ALL_CLASSES"
# echo "  Novel only (6):   $T2_CLASSES"
# echo "  T1 checkpoint:    $T1_CKPT"
# echo ""
#
# python init_prototypes.py \
#     --classes "$ALL_CLASSES" \
#     --out_dim 64 \
#     --output init_protos_t2.pt \
#     --prompt_template "$PROMPT" \
#     --n_iters 5000 \
#     --seed 42 \
#     --base_protos "$T1_CKPT" \
    --num_base 8

echo ""
echo "Done: $(date)"
