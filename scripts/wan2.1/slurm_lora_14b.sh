#!/bin/bash -l
#SBATCH --job-name=lora_14b_swap
#SBATCH --account=OD-235404
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=./slurmlog/slurm-%j.out
#SBATCH --error=./slurmlog/slurm-%j.err

set -euo pipefail

# 1) Conda & CUDA
export CONDA_ROOT=/scratch3/yan204/env/miniconda3
export PATH="$CONDA_ROOT/bin:$PATH"
module load cuda/12.1.0 || true

# 2) Rendezvous setup for DeepSpeed multi-GPU
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$SLURM_JOB_NUM_NODES"
echo "GPUS_PER_NODE=$SLURM_GPUS_PER_TASK"
echo

mkdir -p slurmlog

# 3) Activate env
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate videox-fun || { echo 'Conda env videox-fun not found'; exit 1; }

echo "WORKER $SLURM_PROCID on $(hostname)"
echo "which python: $(which python)"
echo "python -V: $(python -V)"
echo

# 5) Project & data/model paths
PROJECT_DIR=/scratch3/yan204/yxp/VideoX_Fun
cd "$PROJECT_DIR"

export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-14B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"

# 6) Launch training with torchrun (use train_lora.py with LoRA)
torchrun \
  --nnodes "$SLURM_JOB_NUM_NODES" \
  --nproc_per_node "$SLURM_GPUS_PER_TASK" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  --rdzv_id "$SLURM_JOB_ID" \
  scripts/wan2.1/train_lora.py \
    --config_path "config/wan2.1/wan_civitai.yaml" \
    --pretrained_model_name_or_path "$MODEL_NAME" \
    --train_data_dir "$DATASET_NAME" \
    --train_data_meta "$DATASET_META_NAME" \
    --video_sample_n_frames 65 \
    --rank 128 \
    --source_frames 33 \
    --edit_frames 32 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 0 \
    --num_train_epochs 2 \
    --checkpointing_steps 500 \
    --learning_rate 1e-04 \
    --seed 42 \
    --output_dir "experiments/lora_obj_swap_1w_14b_bz8_2epoch_zero2_slurm" \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --adam_weight_decay 3e-2 \
    --adam_epsilon 1e-10 \
    --vae_mini_batch 1 \
    --max_grad_norm 0.05 \
    --random_hw_adapt \
    --enable_bucket \
    --uniform_sampling \
    --enable_text_encoder_in_dataloader \
    --video_edit_loss_on_edited_frames_only

echo "JOB COMPLETED"
