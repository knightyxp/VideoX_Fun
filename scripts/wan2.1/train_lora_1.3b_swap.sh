# export MODEL_NAME="/scratch3/yan204/models/Wan2.1-T2V-1.3B"
# export DATASET_NAME="/scratch3/yan204/yxp/Senorita"
# export DATASET_META_NAME="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json"
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export OMP_NUM_THREADS=4
# NCCL_DEBUG=INFO

# accelerate launch \
#   --use_deepspeed \
#   --deepspeed_config_file config/zero_stage2_config.json \
#   --mixed_precision="bf16" \
#   scripts/wan2.1/train_lora.py \
#   --config_path="config/wan2.1/wan_civitai.yaml" \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATASET_NAME \
#   --train_data_meta=$DATASET_META_NAME \
#   --video_sample_n_frames=65 \
#   --rank=128 \
#   --source_frames=33 \
#   --edit_frames=32 \
#   --train_batch_size=8 \
#   --gradient_accumulation_steps=1 \
#   --dataloader_num_workers=2 \
#   --num_train_epochs=2 \
#   --checkpointing_steps=500 \
#   --learning_rate=1e-04 \
#   --seed=42 \
#   --output_dir="experiments/lora_obj_swap_bucket_only_336_592_1w_1.3b_bz32_2epoch_zero2" \
#   --gradient_checkpointing \
#   --mixed_precision="bf16" \
#   --adam_weight_decay=3e-2 \
#   --adam_epsilon=1e-10 \
#   --vae_mini_batch=8 \
#   --max_grad_norm=0.05 \
#   --random_hw_adapt \
#   --enable_bucket \
#   --uniform_sampling \
#   --low_vram \
#   --video_edit_loss_on_edited_frames_only \
#   --use_deepspeed

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_json.py \
  --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/senorita_obj_swap_test.json \
  --output_dir results/videox_fun_lora_obj_swap_1w_1.3b_bz32_2epoch_zero2_test \
  --seed 0 \
  --lora_path experiments/lora_obj_swap_bucket_only_336_592_1w_1.3b_bz32_2epoch_zero2/checkpoint-624.safetensors