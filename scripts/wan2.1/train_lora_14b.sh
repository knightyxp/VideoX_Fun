export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-14B"
export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4
NCCL_DEBUG=INFO

accelerate launch \
  --mixed_precision="bf16" \
  --use_fsdp \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap=WanAttentionBlock \
  --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_backward_prefetch "BACKWARD_PRE" \
  --fsdp_cpu_ram_efficient_loading False \
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
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=2 \
  --num_train_epochs=2 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 \
  --seed=42 \
  --output_dir="experiments/obj_swap_1w_14b_bz1_2epoch_fsdp" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --random_hw_adapt \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --video_edit_loss_on_edited_frames_only \
  --use_fsdp