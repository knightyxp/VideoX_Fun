import os
import sys
import json
import argparse

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import torchvision
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer,
                              WanTransformer3DModel)
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8, replace_parameters_by_name,
                                              convert_weight_dtype_wrapper)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_to_video_latent, get_video_to_video_latent,
                                   save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import imageio
from typing import List, Optional, Tuple, Union
import cv2


def load_video_frames(
    video_path: str,
    height: int = 480,
    width: int = 832,
    num_frames: int = None,
    fps: float = None,
) -> List[Image.Image]:
    """
    Load video frames for temporal‐concatenation inference:
      - First half: input video frames sampled at `fps` (or native rate if fps=None)
      - Second half: duplicates of those input frames (to serve as “mask” frames)

    Args:
        video_path:        本地视频文件路径
        height, width:     所有帧的目标 H×W
        num_frames:        总帧数（input + mask）
        fps:               如果指定，将以此帧率对原视频做抽帧；否则用原视频的 FPS

    Returns:
        一共 num_frames 张 PIL.Image 列表
    """
    assert num_frames is not None, "请传入 num_frames"
    # 1) 计算 input / mask 数量
    input_frames_count = (num_frames + 1) // 2
    mask_frames_count  = num_frames - input_frames_count

    # 2) 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    skip     = 1 if fps is None else max(1, int(orig_fps // fps))

    frames = []
    frame_idx = 0
    while len(frames) < input_frames_count:
        ret, frame = cap.read()
        if not ret:
            break
        # 按 skip 抽帧
        if frame_idx % skip == 0:
            # Resize & BGR→RGB → PIL
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        frame_idx += 1
    cap.release()

    # 3) 如果不够 input_frames_count，则用最后一帧补齐
    while len(frames) < input_frames_count:
        frames.append(frames[-1].copy())

    # 4) 生成 mask 部分（重复前 input_frames_count 帧）
    for i in range(mask_frames_count):
        frames.append(frames[i].copy())

    assert len(frames) == num_frames
    print(f"Loaded {input_frames_count} input + {mask_frames_count} mask = {len(frames)} frames")

    input_video = torch.from_numpy(np.array(frames))[:video_length]
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    # 修复：使用与DiffSynth一致的归一化方式 [0,255] -> [-1,1]
    input_video = input_video * (2.0 / 255.0) - 1.0
    return input_video

# GPU memory mode, which can be choosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = "sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# TeaCache config
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "/scratch3/yan204/models/Wan2.1-T2V-1.3B"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow_Unipc"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = "exp/equivalent_diffsynth_625_steps_4card_global_bz=8/checkpoint-625.safetensors"

# Other params
sample_size         = [480, 832]
video_length        = 81
fps                 = 10

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16

def parse_args():
    parser = argparse.ArgumentParser(description="Video-to-video generation from JSON task list")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    return parser.parse_args()
weight_dtype        = torch.bfloat16
negative_prompt     = "blurry, low quality, distorted, artifacts"
guidance_scale      = 5.0
seed                = 0
num_inference_steps = 50
lora_weight         = 1.0
save_path           = "samples/bz2_4card_625step"

device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
)
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)



generator = torch.Generator(device=device).manual_seed(seed)

if lora_path is not None:
    pipeline = merge_lora(pipeline, lora_path, lora_weight, device=device)


def run_from_json(test_json_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(test_json_path, "r", encoding="utf-8") as f:
        items_list = json.load(f)

    # Convert to dict keyed by filename, following reference format
    eval_prompts = {}
    for item in items_list:
        fname = f"{item['task_type']}_{item['sample_id']}.mp4"
        eval_prompts[fname] = item

    for fname, item in eval_prompts.items():
        base = os.path.splitext(fname)[0]
        output_video_path = os.path.join(output_dir, f"gen_{base}.mp4")
        info_path         = os.path.join(output_dir, f"gen_{base}_info.txt")

        if os.path.exists(output_video_path):
            print(f"Output exists for {fname}, skip.")
            continue

        video_path = item["source_video_path"]

        prompt = (
            "A video sequence showing two parts: the first half shows the original scene, "
            f"and the second half shows the same scene but {item['qwen_vl_72b_refined_instruction']}"
        )

        # Prepare input video tensor [-1, 1]
        input_video = load_video_frames(
            video_path,
            height=sample_size[0],
            width=sample_size[1],
            num_frames=video_length,
        )

        # Align num_frames to VAE temporal compression
        num_frames_eff = (
            int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
            if video_length != 1 else 1
        )
        print('video_length', num_frames_eff)

        with torch.no_grad():
            sample = pipeline(
                input_video,
                prompt,
                num_frames=num_frames_eff,
                negative_prompt=negative_prompt,
                height=sample_size[0],
                width=sample_size[1],
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                shift=shift,
            ).videos

        print('output video tensor shape', sample.shape)
        print('input video tensor shape', input_video.shape)

        # Save outputs
        save_results(sample, output_video_path, fps)

        with open(info_path, "w", encoding="utf-8") as info_f:
            info_f.write(prompt)


if lora_path is not None:
    pipeline = unmerge_lora(pipeline, lora_path, lora_weight, device=device)

def save_results(tensor: torch.Tensor, file_path: str, fps: int = 16):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    B, C, T, H, W = tensor.shape
    arr = tensor[0].cpu().numpy()  # (C, T, H, W)
    if T == 1:
        img = arr[:, 0].transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(file_path)
    else:
        save_videos_grid(tensor, file_path, fps=fps)
    print(f"Saved video → {file_path}")

def save_input_video(tensor: torch.Tensor, file_path: str, fps: int = 16):
    """保存输入视频，处理[-1,1]范围的tensor"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # 将[-1,1]范围转换为[0,1]范围
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    save_videos_grid(tensor, file_path, fps=fps)
    print(f"Saved input video → {file_path}")



def save_pil_frames(frames, path, fps=16):
    writer = imageio.get_writer(path, fps=fps, codec="libx264")
    for img in frames:
        writer.append_data(np.array(img))
    writer.close()

def main():
    args = parse_args()
    # Override save_path by CLI output_dir for clarity
    out_dir = args.output_dir
    run_from_json(args.test_json, out_dir)

if __name__ == "__main__":
    main()
