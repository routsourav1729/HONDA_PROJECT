#!/bin/bash
# ============================================================================
# WandB Offline Sync Script
# Run this from the LOGIN NODE while training is running on compute node.
# It polls wandb_logs/wandb/ and syncs new data to wandb.ai every INTERVAL seconds.
#
# Usage:
#   bash scripts/wandb_sync.sh              # live sync (Ctrl+C to stop)
#   bash scripts/wandb_sync.sh once         # one-shot sync (use after job finishes)
#   bash scripts/wandb_sync.sh <JOB_ID>     # auto-stop when SLURM job finishes
#
# From hypyolov2/ directory (where wandb_logs/ lives).
# ============================================================================

WANDB_OFFLINE_DIR="wandb_logs/wandb"
INTERVAL=60   # seconds between syncs

MODE=${1:-""}   # "once", a job ID number, or empty (loop forever)

# Make sure wandb is available
if ! command -v wandb &>/dev/null; then
    source /home/agipml/sourav.rout/miniconda3/bin/activate ovow2
fi

sync_once() {
    echo "[$(date '+%H:%M:%S')] Syncing offline runs in $WANDB_OFFLINE_DIR ..."
    # --sync-all picks up all offline-run-* subdirs, including mid-run partial data
    wandb sync --sync-all "$WANDB_OFFLINE_DIR" 2>&1 | tail -5
}

if [[ "$MODE" == "once" ]]; then
    # ---- One-shot: sync and exit ----
    sync_once
    echo "Done."

elif [[ "$MODE" =~ ^[0-9]+$ ]]; then
    # ---- Auto-stop when SLURM job $MODE finishes ----
    JOB_ID="$MODE"
    echo "Watching SLURM job $JOB_ID. Will stop syncing when it ends."
    echo "Press Ctrl+C to stop manually."
    echo ""
    while squeue -j "$JOB_ID" -h &>/dev/null; do
        sync_once
        sleep "$INTERVAL"
    done
    echo "[$(date '+%H:%M:%S')] Job $JOB_ID finished. Final sync..."
    sync_once
    echo "All done."

else
    # ---- Loop forever until Ctrl+C ----
    echo "Live syncing every ${INTERVAL}s. Press Ctrl+C to stop."
    echo ""
    while true; do
        sync_once
        sleep "$INTERVAL"
    done
fi
