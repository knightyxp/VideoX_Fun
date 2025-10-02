python scripts/convert_grounding_triplets.py \
  --in_json /scratch3/yan204/yxp/InContext-VideoEdit/data/json/grounding_multi_instance_gray.json \
  --out_json /scratch3/yan204/yxp/VideoX_Fun/data/json/reasoning_grournd_then_removal.json \
  --base_path /scratch3/yan204/yxp/Senorita\
  --src_frames 33 --grd_frames 4 --edt_frames 32 \
  --workers 16 --ffprobe ffprobe \
  --verify_decode