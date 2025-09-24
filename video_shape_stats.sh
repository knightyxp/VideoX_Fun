python scripts/utils/video_shape_stats.py \
    --ann_path /scratch3/yan204/yxp/InContext-VideoEdit/data/json/grounding_gray_and_obj_removal.json \
    --base_dir /scratch3/yan204/yxp/Senorita \
    --include_edited \
    --dedup \
    --num_workers 32 \
    --output video_shape_stats.json \
    --topk 50 