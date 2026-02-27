#!/bin/bash
#SBATCH --job-name=init_protos
#SBATCH --output=slurm_logs/init_protos_%j.out
#SBATCH --error=slurm_logs/init_protos_%j.err
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# ============================================================================
# Initialize prototype directions for horospherical classification.
# Run ONCE before training whenever classes change.
#
# Usage:
#   sbatch init_protos.sbatch t1        # generate T1 prototypes (8 classes)
#   sbatch init_protos.sbatch t2        # generate T2 novel-only prototypes
# ============================================================================

TASK=${1:-t1}

echo "=========================================="
echo "PROTOTYPE INITIALIZATION: $TASK"
echo "Job: $SLURM_JOB_ID | $(date)"
echo "=========================================="

eval "$(conda shell.bash hook)"
conda activate ovow2

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TRANSFORMERS_CACHE=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export TORCH_HOME=/home/agipml/sourav.rout/ALL_FILES/hypyolo/clip_cache
export PYTHONUNBUFFERED=1

mkdir -p slurm_logs

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

if [ "$TASK" = "t1" ]; then
    echo ""
    echo "=== T1: 8 base classes ==="
    echo "  car, motorcycle, rider, person, autorickshaw, bicycle, traffic sign, traffic light"
    echo ""
    python init_prototypes.py \
        --classes "car,motorcycle,rider,person,autorickshaw,bicycle,traffic sign,traffic light" \
        --out_dim 256 \
        --output init_protos_t1.pt \
        --prompt_template "a photo of a {} on an Indian road" \
        --n_iters 3000 \
        --seed 42

elif [ "$TASK" = "t2" ]; then
    echo ""
    echo "=== T2: novel-only (6 new classes), needs T1 checkpoint ==="
    echo "  bus, truck, tanker_vehicle, crane_truck, street_cart, excavator"
    echo ""
    # Uses T1 trained directions as anchor so novel dirs don't overlap
    T1_CKPT=${2:-"IDD_HYP/t1/horospherical/model_final.pth"}
    echo "  T1 checkpoint: $T1_CKPT"
    python init_prototypes.py \
        --classes "car,motorcycle,rider,person,autorickshaw,bicycle,traffic sign,traffic light,bus,truck,tanker_vehicle,crane_truck,street_cart,excavator" \
        --out_dim 256 \
        --output init_protos_t2.pt \
        --prompt_template "a photo of a {} on an Indian road" \
        --n_iters 5000 \
        --seed 42 \
        --base_protos "$T1_CKPT" \
        --num_base 8

else
    echo "ERROR: Unknown task '$TASK'. Use: t1 or t2"
    exit 1
fi

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
