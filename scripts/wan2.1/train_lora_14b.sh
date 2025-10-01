export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-14B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

# 放本地盘，别用 NFS
export TRITON_CACHE_DIR=/scratch3/yan204/.triton_cache
export HF_HOME=/scratch3/yan204/.hf
export TRANSFORMERS_CACHE=/scratch3/yan204/.hf/transformers
export TORCH_HOME=/scratch3/yan204/.torch

# 降并行与开销
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 显存碎片更友好
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --mixed_precision="bf16" \
  scripts/wan2.1/train_lora.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=65 \
  --rank=128 \
  --source_frames=33 \
  --edit_frames=32 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=0 \
  --num_train_epochs=2 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="experiments/obj_swap_1w_14b_bz1_2epoch_zero2" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --enable_bucket \
  --uniform_sampling \
  --video_edit_loss_on_edited_frames_only \
  --use_deepspeed