#!/bin/bash
#SBATCH --job-name=diagnose_unk
#SBATCH --output=slurm_logs/diagnose_unk_%j.out
#SBATCH --error=slurm_logs/diagnose_unk_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "=========================================="
echo "DIAGNOSING UNKNOWN FALSE POSITIVES"
echo "=========================================="
echo "Job started at: $(date)"

eval "$(conda shell.bash hook)"
conda activate ovow2

cd /home/agipml/sourav.rout/ALL_FILES/hypyolo/hypyolov2

python debug/diagnose_unknown_fps.py

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
