#!/bin/bash
#SBATCH --job-name=lvsm_dl3dv_ae4
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o sout/%x-%j.out
#SBATCH -e sout/%x-%j.err

OUTDIR=/work/hdd/bdxf/nstjohn/LVSM/experiments/LVSMwAE8

set -euo pipefail
mkdir -p sout
mkdir -p /work/hdd/bdxf/nstjohn/LVSM/experiments/LVSMwAE8

source /u/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/bdxf/nstjohn/conda_envs/LVSM

cd /u/nstjohn/LVSM

echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python --version
nvidia-smi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=""

/work/nvme/bdxf/nstjohn/conda_envs/LVSM/bin/torchrun \
    --nproc_per_node 1 \
    --nnodes 1 \
    --rdzv_id 42001 \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29510 \
    /u/nstjohn/LVSM/scripts/train_LVSM_ae.py \
    --config /u/nstjohn/LVSM/configs/LVSM/LVSM_scene_encoder_decoder_ae.yaml \
    output_dir=$OUTDIR \
    training.checkpoint_dir=$OUTDIR/checkpoints

