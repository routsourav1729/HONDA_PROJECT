#!/bin/bash
# ============================================================================
# Prototype Initialization Script
# ============================================================================
# Usage:
#   bash scripts/init_protos.sh t1                           # T1 base protos
#   bash scripts/init_protos.sh t2                           # T2 novel protos (default ckpt)
#   bash scripts/init_protos.sh t2 path/to/model_final.pth   # T2 with explicit ckpt
#   bash scripts/init_protos.sh all                          # both T1 then T2
# ============================================================================

set -euo pipefail

STAGE=${1:-t2}
T1_CKPT=${2:-"IDD_HYP/t1/vmf_v1/model_final.pth"}

CLASS_DIR="datasets/ImageSets/Main/IDD"
PROMPT="a photo of a {} on an Indian road"

read_classes() {
    paste -sd',' "$1" | sed 's/,$//'
}

run_t1() {
    T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
    echo "=== T1 Prototypes: $T1_CLASSES ==="
    python init_prototypes.py \
        --classes "$T1_CLASSES" \
        --out_dim 64 \
        --output datasets/prototype/init_protos_t1.pt \
        --prompt_template "$PROMPT" \
        --n_iters 3000 \
        --seed 42
}

run_t2() {
    if [ ! -f "$T1_CKPT" ]; then
        echo "ERROR: T1 checkpoint not found: $T1_CKPT"
        echo "  Run T1 training first, or pass checkpoint path as 2nd arg."
        exit 1
    fi

    T1_CLASSES=$(read_classes "$CLASS_DIR/t1_classes.txt")
    T2_CLASSES=$(read_classes "$CLASS_DIR/t2_classes.txt")
    ALL_CLASSES="${T1_CLASSES},${T2_CLASSES}"

    echo "=== T2 Prototypes (Procrustes-aligned) ==="
    echo "  All classes (14): $ALL_CLASSES"
    echo "  Novel only (6):   $T2_CLASSES"
    echo "  T1 checkpoint:    $T1_CKPT"
    echo ""

    python init_prototypes.py \
        --classes "$ALL_CLASSES" \
        --out_dim 64 \
        --output datasets/prototype/init_protos_t2.pt \
        --prompt_template "$PROMPT" \
        --n_iters 5000 \
        --seed 42 \
        --base_protos "$T1_CKPT" \
        --num_base 8
}

case "$STAGE" in
    t1)  run_t1 ;;
    t2)  run_t2 ;;
    all) run_t1; echo ""; run_t2 ;;
    *)   echo "Usage: $0 {t1|t2|all} [t1_checkpoint_path]"; exit 1 ;;
esac

echo ""
echo "Done: $(date)"
