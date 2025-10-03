import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
import torchvision
import imageio

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
from typing import List, Optional, Tuple, Union


def load_video_frames(
    video_path: str,
    num_frames: int = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Load video frames for temporal-concatenation inference using imageio:
      - First half: input video frames
      - Second half: duplicates of those input frames (to serve as "mask" frames)

    Args:
        video_path:        本地视频文件路径
        num_frames:        总帧数（input + mask）

    Returns:
        Tuple of (video_tensor, original_height, original_width)
        - video_tensor: torch.Tensor of shape [1, C, T, H, W] in range [-1, 1]
        - original_height: 原始视频高度
        - original_width: 原始视频宽度
    """
    assert num_frames is not None, "请传入 num_frames"
    
    # 1) 计算 input / mask 数量
    input_frames_count = (num_frames + 1) // 2
    mask_frames_count = num_frames - input_frames_count

    # 2) 使用 imageio 读取视频
    reader = imageio.get_reader(video_path)
    try:
        total_frames = reader.count_frames()
    except:
        # Some formats don't support count_frames, iterate to count
        total_frames = sum(1 for _ in reader)
        reader = imageio.get_reader(video_path)  # Re-open reader
    
    stride = max(1, total_frames // input_frames_count)
    start_frame = torch.randint(0, max(1, total_frames - stride * input_frames_count), (1,))[0].item()

    frames = []
    original_height, original_width = None, None
    
    for i in range(input_frames_count):
        idx = start_frame + i * stride
        if idx >= total_frames:
            break
        try:
            frame = reader.get_data(idx)
            pil_frame = Image.fromarray(frame)
            
            # Store original dimensions from first frame
            if original_height is None:
                original_width, original_height = pil_frame.size
                print(f"Original video dimensions: {original_width}x{original_height}")
            
            frames.append(pil_frame)
        except IndexError:
            break
    
    reader.close()

    # 3) 如果不够 input_frames_count，则用最后一帧补齐
    while len(frames) < input_frames_count:
        if frames:
            frames.append(frames[-1].copy())
        else:
            # Create a black frame if no frames loaded (use original dimensions if available)
            w, h = (original_width, original_height) if original_width else (832, 480)
            frames.append(Image.new('RGB', (w, h), (0, 0, 0)))

    # 4) 生成 mask 部分（重复前 input_frames_count 帧）
    for i in range(mask_frames_count):
        frames.append(frames[i].copy())

    assert len(frames) == num_frames
    print(f"Loaded {input_frames_count} input + {mask_frames_count} mask = {len(frames)} frames")

    # 5) Convert to tensor and normalize to [-1, 1]
    input_video = torch.from_numpy(np.array(frames))
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0).float()
    # 归一化 [0,255] -> [-1,1]
    input_video = input_video * (2.0 / 255.0) - 1.0
    
    return input_video, original_height, original_width


def parse_args():
    parser = argparse.ArgumentParser(description="Video-to-video generation from JSON task list with parallel inference")
    parser.add_argument("--test_json", type=str, required=True, help="Path to test JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated videos")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible generation")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--num_frames", type=int, default=65, help="Total number of frames (input + mask)")
    parser.add_argument("--source_frames", type=int, default=33, help="Number of source frames; default 33")
    args = parser.parse_args()
    return args


# Configuration parameters (keeping your original settings)
GPU_memory_mode = "sequential_cpu_offload"
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = True
compile_dit = False
enable_teacache = True
teacache_threshold = 0.10
num_skip_start_steps = 5
teacache_offload = False
cfg_skip_ratio = 0
enable_riflex = False
riflex_k = 6

config_path = "config/wan2.1/wan_civitai.yaml"
model_name = "/scratch3/yan204/models/Wan2.1-T2V-1.3B"
sampler_name = "Flow_Unipc"
shift = 3
transformer_path = None
vae_path = None

sample_size = None  # Will be set dynamically based on input video
fps = 10
weight_dtype = torch.bfloat16
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
guidance_scale = 5.0
num_inference_steps = 50
lora_weight = 1.0


def save_results(tensor: torch.Tensor, file_path: str, fps: int = 16):
    """Save output video tensor"""
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


def main():
    args = parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"Running parallel inference with {world_size} GPUs")
        print(f"Using seed: {args.seed}")
    
    # Load test entries from JSON
    with open(args.test_json, 'r', encoding='utf-8') as f:
        eval_prompts_list = json.load(f)
    
    # Convert list to dictionary format
    eval_prompts = {}
    for item in eval_prompts_list:
        fname = f"{item['task_type']}_{item['sample_id']}.mp4"
        eval_prompts[fname] = item
    
    # Distribute items across GPUs
    items = list(eval_prompts.items())
    items_per_gpu = len(items) // world_size
    start_idx = rank * items_per_gpu
    end_idx = (rank + 1) * items_per_gpu if rank != world_size - 1 else len(items)
    subset_items = items[start_idx:end_idx]
    
    if rank == 0:
        print(f"Total items: {len(items)}, items per GPU: ~{items_per_gpu}")
    print(f"[GPU {rank}] Processing {len(subset_items)} items")
    
    # Setup device for this rank
    device = torch.device(f"cuda:{rank}")
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Initialize transformer
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    if transformer_path is not None:
        print(f"[GPU {rank}] Loading transformer from checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"[GPU {rank}] Missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Initialize VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)
    
    if vae_path is not None:
        print(f"[GPU {rank}] Loading VAE from checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"[GPU {rank}] Missing keys: {len(m)}, unexpected keys: {len(u)}")
    
    # Initialize tokenizer and text encoder
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )
    
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    
    # Initialize scheduler
    Choosen_Scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    
    if sampler_name in ["Flow_Unipc", "Flow_DPM++"]:
        config['scheduler_kwargs']['shift'] = 1
    
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )
    
    # Create pipeline
    pipeline = WanPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    
    # Configure GPU memory mode
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
    
    # Load LoRA if provided
    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, lora_weight, device=device)
        print(f"[GPU {rank}] Loaded LoRA from {args.lora_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    generator = torch.Generator(device=device).manual_seed(args.seed + rank)
    
    # Inference loop
    for fname, item in subset_items:
        base = os.path.splitext(fname)[0]
        output_video_path = os.path.join(args.output_dir, f"gen_{base}.mp4")
        info_path = os.path.join(args.output_dir, f"gen_{base}_info.txt")
        
        if os.path.exists(output_video_path):
            print(f"[GPU {rank}] Output exists for {fname}, skipping...")
            continue
        
        print(f"[GPU {rank}] Processing {fname}...")
        
        video_path = item["source_video_path"]
        
        # Create prompt
        prompt = (
            "A video sequence showing two parts: the first half shows the original scene, "
            f"and the second half shows the same scene but {item['qwen_vl_72b_refined_instruction']}"
        )
        
        # Load video frames and get original dimensions
        input_video, video_height, video_width = load_video_frames(
            video_path,
            num_frames=args.num_frames,
        )
        
        # Align num_frames to VAE temporal compression
        num_frames = args.num_frames
        
        # Generate video
        with torch.no_grad():
            sample = pipeline(
                video=input_video,
                prompt=prompt,
                num_frames=num_frames,
                source_frames=args.source_frames,
                negative_prompt=negative_prompt,
                height=video_height,
                width=video_width,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                shift=shift,
            ).videos
        
        print(f'[GPU {rank}] Output video tensor shape: {sample.shape}')
        
        # Save results
        save_results(sample, output_video_path, fps)
        
        # Save prompt info
        with open(info_path, "w", encoding="utf-8") as info_f:
            info_f.write(prompt)
        
        print(f"[GPU {rank}] Completed {fname}")
    
    # Unmerge LoRA if it was loaded
    if args.lora_path is not None:
        pipeline = unmerge_lora(pipeline, args.lora_path, lora_weight, device=device)
    
    print(f"[GPU {rank}] Finished processing all assigned items")
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print("All GPUs finished processing")


if __name__ == "__main__":
    main()