export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_json.py \
  --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/test_multi_instance_removal_ground.json \
  --output_dir results/videox_fun_1.3b_bz32_2epoch_zero2_test_rem_ground \
  --seed 0 \
  --lora_path experiments/videox_fun_bucket_dynamic_resolution_1.3b_bz32_2epoch_right_init_latents_zero2/checkpoint-928.safetensors