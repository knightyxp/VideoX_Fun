#!/bin/bash -l
#SBATCH --job-name=lora_14b_swap
#SBATCH --account=OD-235404
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=2:00:00
#SBATCH --output=./slurmlog/slurm-%j.out
#SBATCH --error=./slurmlog/slurm-%j.err

set -euo pipefail

# 1) Conda & CUDA
export CONDA_ROOT=/scratch3/yan204/env/miniconda3
export PATH="$CONDA_ROOT/bin:$PATH"
module load cuda/12.1.0 || true

# 2) Activate env first
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate videox-fun || { echo 'Conda env videox-fun not found'; exit 1; }

# 3) Project paths
PROJECT_DIR=/scratch3/yan204/yxp/VideoX_Fun
cd "$PROJECT_DIR"

# 4) Setup distributed training parameters
GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# Master node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "START TIME: $(date)"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "NUM_PROCESSES=$NUM_PROCESSES"

# 5) Model and data paths
export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-14B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"

# 6) NCCL optimization
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # 强制在NCCL挂起时崩溃
export TRITON_CACHE_DIR=/scratch3/yan204/.triton_cache

# 如果有InfiniBand
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0

# 7) 构建launcher命令（注意转义 \$SLURM_PROCID）
export LAUNCHER="accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --mixed_precision bf16 \
  --num_machines $NNODES \
  --num_processes $NUM_PROCESSES \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT"

# 8) 构建训练命令
export PROGRAM="scripts/wan2.1/train_lora.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --train_data_meta $DATASET_META_NAME \
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
  --use_deepspeed"

# 9) 组合命令
export CMD="$LAUNCHER $PROGRAM"

# 10) 使用srun启动，确保在所有节点上运行
mkdir -p slurmlog
LOG_PATH="slurmlog/training_${SLURM_JOB_ID}.log"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
echo "JOB COMPLETED"