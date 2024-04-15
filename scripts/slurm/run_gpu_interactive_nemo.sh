#!/bin/bash

echo "Using SLURM_ACCOUNT=$SLURM_ACCOUNT"
srun \
    -A $SLURM_ACCOUNT \
    --time=00:30:00 \
    --nodes=1 \
    --ntasks-per-node=16 \
    --tasks=1 \
    --cpus-per-task=16 \
    --partition=gpuA40x4 \
    --gpus=1 \
    --mem=64g \
    --pty scripts/singularity/run_nemo_interactive.sh
