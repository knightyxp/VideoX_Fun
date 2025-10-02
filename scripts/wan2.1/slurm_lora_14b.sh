#!/bin/bash -l
#SBATCH --job-name=lora_14b_swap
#SBATCH --account=OD-235404
#SBATCH --partition=h2gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH --output=./slurmlog/slurm-%j.out
#SBATCH --error=./slurmlog/slurm-%j.err

set -euo pipefail

# ===== Environment Setup =====
export CONDA_ROOT=/scratch3/yan204/env/miniconda3
export PATH="$CONDA_ROOT/bin:$PATH"
module load cuda/12.1.0 || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate videox-fun

PROJECT_DIR=/scratch3/yan204/yxp/VideoX_Fun
cd "$PROJECT_DIR"
mkdir -p slurmlog

# ===== Distributed Training Setup =====
# Use first node as master
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use a deterministic port shared across nodes (derived from SLURM_JOB_ID)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
# Total number of processes
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))  # 2 nodes * 4 GPUs = 8

# ===== Model and Dataset Paths =====
export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-14B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"

# ===== NCCL Configuration =====
# Basic NCCL settings
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
# InfiniBand settings (adjust based on your cluster)
export NCCL_IB_DISABLE=0  # Enable IB if available
# export NCCL_IB_HCA=  # Set specific IB device if needed
# export NCCL_SOCKET_IFNAME=  # Set network interface if IB disabled

# ===== Cache and other settings =====
export TRITON_CACHE_DIR=/scratch3/yan204/.triton_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== Print Debug Info =====
echo "========================================="
echo "START TIME: $(date)"
echo "========================================="
echo "Job Configuration:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  NODES: $SLURM_NNODES"
echo "  GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  Current node: $(hostname)"
echo "  SLURM_NODEID: $SLURM_NODEID"
echo "========================================="

# Network diagnostics (optional, comment out if not needed)
# echo "Network Status:"
# echo "  ibstatus: $(ibstatus 2>/dev/null || echo 'N/A')"
# echo "  ibdev2netdev: $(ibdev2netdev 2>/dev/null || echo 'N/A')"
# echo "========================================="

# ===== Launch Training =====
srun --label accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file config/zero_stage2_config.json \
    --mixed_precision bf16 \
    --num_machines $SLURM_NNODES \
    --num_processes $SLURM_GPUS_PER_NODE \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_backend c10d \
    scripts/wan2.1/train_lora.py \
        --config_path config/wan2.1/wan_civitai.yaml \
        --pretrained_model_name_or_path "$MODEL_NAME" \
        --train_data_dir "$DATASET_NAME" \
        --train_data_meta "$DATASET_META_NAME" \
        --video_sample_n_frames 65 \
        --rank 128 \
        --source_frames 33 \
        --edit_frames 32 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --dataloader_num_workers 2 \
        --num_train_epochs 2 \
        --checkpointing_steps 500 \
        --learning_rate 1e-04 \
        --seed 42 \
        --output_dir experiments/obj_swap_1w_14b_bz1_2epoch_zero2_slurm \
        --gradient_checkpointing \
        --adam_weight_decay 3e-2 \
        --adam_epsilon 1e-10 \
        --vae_mini_batch 1 \
        --max_grad_norm 0.05 \
        --random_hw_adapt \
        --enable_bucket \
        --uniform_sampling \
        --video_edit_loss_on_edited_frames_only \
        --use_deepspeed

echo "========================================="
echo "END TIME: $(date)"
echo "JOB COMPLETED"
echo "========================================="