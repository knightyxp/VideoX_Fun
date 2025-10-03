export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_json.py \
  --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/test_multi_instance_removal_ground.json \
  --output_dir results/chain_of_frames_ground_and_rem_test_in_domain \
  --seed 0 \
  --num_frames 69 \
  --source_frames 33 \
  --lora_path experiments/CoF_ground_then_removal_only_multi_instance_data_zero2/checkpoint-302.safetensors