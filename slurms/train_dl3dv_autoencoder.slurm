#!/bin/bash
#SBATCH --job-name=lvsm_ae_dl3dv
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH -o /u/nstjohn/LVSM/sout/%x-%j.out
#SBATCH -e /u/nstjohn/LVSM/sout/%x-%j.err

set -euo pipefail
mkdir -p /u/nstjohn/LVSM/sout

source /u/nstjohn/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/bdxf/nstjohn/conda_envs/LVSM

cd /u/nstjohn/LVSM

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"
export NCCL_DEBUG=WARN

# Optional: resume from latest checkpoint
# Add --config.training.resume_ckpt=experiments/checkpoints/AE_dl3dv/latest.pt
# to the torchrun command below if resuming.

torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29520 \
    /u/nstjohn/LVSM/scripts/train_ae.py \
    --config /u/nstjohn/LVSM/configs/autoencoder/dl3vd_autoencoder.yaml