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
# T1: Base prototypes (8 classes)
# ============================================================================
# Uncomment to regenerate T1 prototypes (already done once)

# T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
# echo "=== T1 Prototypes: $T1_CLASSES ==="
# python init_prototypes.py \
#     --classes "$T1_CLASSES" \
#     --out_dim 256 \
#     --output init_protos_t1.pt \
#     --prompt_template "$PROMPT" \
#     --n_iters 3000 \
#     --seed 42

# ============================================================================
# T2: Novel prototypes (6 classes, anchored to T1)
# ============================================================================
# Generates directions for ALL 14 classes, then extracts only the 6 novel ones.
# Uses T1 checkpoint to ensure novel directions don't overlap with base ones.

T1_CKPT=${1:-"IDD_HYP/t1/horo_2conv/model_final.pth"}

T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
T2_CLASSES=$(read_classes "$CLASS_DIR/t2_classes.txt")
ALL_CLASSES="${T1_CLASSES},${T2_CLASSES}"

echo "=== T2 Prototypes ==="
echo "  All classes (14): $ALL_CLASSES"
echo "  Novel only (6):   $T2_CLASSES"
echo "  T1 checkpoint:    $T1_CKPT"
echo ""

python init_prototypes.py \
    --classes "$ALL_CLASSES" \
    --out_dim 256 \
    --output init_protos_t2.pt \
    --prompt_template "$PROMPT" \
    --n_iters 5000 \
    --seed 42 \
    --base_protos "$T1_CKPT" \
    --num_base 8

echo ""
echo "Done: $(date)"
