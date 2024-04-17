#!/bin/bash -l

#SBATCH --job-name="nemo_finetune_4xA100x2"
#SBATCH --output="logs/%j.%N.nemo_finetune_4xA100x2.out"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch&projects"
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcbf-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 00:30:00

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=hsn
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=16

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0
# module load nccl/2.19.3-1.awsplugin
# module load cuda/12.4.0

WORK_DIR=/scratch/bcsi/xingyao6/nemo-util
SINGULARITY_IMAGE=/scratch/bcsi/xingyao6/nemo-util/nemo_main.sif
SCRIPT_TO_RUN=scripts/train/train_mistral.sh

echo "WORK_DIR=$WORK_DIR"
echo "SINGULARITY_IMAGE=$SINGULARITY_IMAGE"
echo "SCRIPT_TO_RUN=$SCRIPT_TO_RUN"

# check if the script to run exists
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "Script $SCRIPT_TO_RUN does not exist"
    exit 1
fi
# srun scripts/train/slurm/apptainer_wrapper.sh \
#     $SINGULARITY_IMAGE \
#     $WORK_DIR \
#     scripts/train/train_mistral.sh

# Testing NCCL
# cd $WORK_DIR/nccl-tests
# make CUDA_HOME=$CUDA_HOME NCCL_HOME=$NCCL_HOME
# srun ./build/all_reduce_perf -b 4 -e 8G -f 2 -g 1 -c 1 -n 10
# --env "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN" \


srun apptainer exec \
    --nv \
    --no-home \
    --no-mount bind-paths \
    --env "HF_HOME=/workspace/.cache/huggingface" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --writable-tmpfs \
    --bind $WORK_DIR:/workspace \
    $SINGULARITY_IMAGE \
    /bin/bash -c "cd /workspace && $SCRIPT_TO_RUN"
