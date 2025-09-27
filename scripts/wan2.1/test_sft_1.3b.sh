export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 examples/wan2.1/predict_sft_v2v.py \
  --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/test_multi_instance_removal_ground.json \
  --output_dir results/sft_rem_ground_1.3b_bz16_2epoch_test \
  --seed 0 \
  --transformer_path ./experiments/sft_rem_grounding_gray_1.3b_bz16_lr_2e-05_zero2_2epoch/checkpoint-1857/transformer/diffusion_pytorch_model.safetensors


# torchrun --nproc_per_node=4 examples/wan2.1/predict_v2v_json.py \
#   --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/senorita_obj_swap_test.json \
#   --output_dir results/sft_obj_swap_1w_1.3b_bz32_2epoch_test \
#   --seed 0 \
#   --transformer_path ./experiments/sft_obj_swap_top1w_1.3b_bz32_lr_2e-05_2epoch_slurm/checkpoint-624/transformer/diffusion_pytorch_model.safetensors