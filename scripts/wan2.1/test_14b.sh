export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 examples/wan2.1/predict_14b_v2v_json.py \
  --test_json  /scratch3/yan204/yxp/InContext-VideoEdit/data/test_json/senorita_obj_swap_test.json \
  --output_dir results/videox_fun_14b_bz2_2epoch_zero2_test_obj_swap \
  --seed 0 \
  --lora_path experiments/obj_swap_1w_14b_bz1_2epoch_zero2/checkpoint-10000.safetensors