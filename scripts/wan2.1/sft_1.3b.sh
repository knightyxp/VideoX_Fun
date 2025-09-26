export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-1.3B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
NCCL_DEBUG=INFO

accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --mixed_precision="bf16" \
  scripts/wan2.1/train.py \
  --config_path="config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --video_sample_n_frames=65 \
  --source_frames=33 \
  --edit_frames=32 \
  --train_batch_size=8 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=2 \
  --num_train_epochs=2 \
  --checkpointing_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="experiments/sft_obj_swap_top1w_1.3b_bz32_lr_2e-05_zero2_2epoch" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=8 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --train_mode="normal" \
  --trainable_modules "." \
  --video_edit_loss_on_edited_frames_only \
  --use_deepspeed
  