#!/bin/bash
#SBATCH --job-name=test_lvsm
#SBATCH --account=bdxf-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o sout/%x-%j.out
#SBATCH -e sout/%x-%j.err

set -euo pipefail
mkdir -p sout_ins

source /u/$USER/miniconda3/etc/profile.d/conda.sh
conda activate /work/nvme/bdxf/nstjohn/conda_envs/LVSM

cd /u/$USER/LVSM

echo "Running on $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python --version
nvidia-smi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

# srun python -u torchrun \
#     --nproc_per_node 8 \
#     --nnodes 8 \
#     --rdzv_id 18635 \
#     --rdzv_backend c10d \
#     --rdzv_endpoint localhost:29502 \

# srun python -u train.py \
#     --config configs/LVSM_scene_decoder_only.yaml