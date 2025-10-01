import os
import json
import argparse
from typing import Dict, Any

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import distributed as dist

from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel, AutoTokenizer


def read_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def save_tensors(path: str, tensors: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensors, path)


def encode_text(tokenizer, text_encoder, texts: list, max_length: int = 226, device: torch.device = torch.device('cuda')) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    with torch.no_grad():
        context = text_encoder(input_ids, attention_mask=attention_mask)[0]
    return {
        'context': context,                # [B, L, C]
        'attention_mask': attention_mask,  # [B, L]
    }


def build_argparser():
    parser = argparse.ArgumentParser(description='Precompute VAE latents and text embeddings.')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Path to model root (contains tokenizer, text_encoder, vae).')
    parser.add_argument('--config_path', type=str, required=True, help='YAML config to read subpaths and kwargs.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Base path for dataset files.')
    parser.add_argument('--video_edit_metadata', type=str, required=True, help='JSON metadata with original/edited pairs and instruction text.')
    parser.add_argument('--tensor_folder', type=str, default='tensor', help='Folder under dataset_path to save tensors.')
    parser.add_argument('--source_frames', type=int, default=33)
    parser.add_argument('--edit_frames', type=int, default=32)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--max_pixels', type=int, default=1920*1080)
    parser.add_argument('--height_division_factor', type=int, default=16)
    parser.add_argument('--width_division_factor', type=int, default=16)
    parser.add_argument('--stripe', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bf16', action='store_true', help='Use bfloat16 for model weights and latents.')
    parser.add_argument('--tokenizer_max_length', type=int, default=226)
    return parser


def get_height_width_dynamic(image_w: int, image_h: int, max_pixels: int, h_factor: int, w_factor: int):
    if image_w * image_h > max_pixels:
        scale = (image_w * image_h / max_pixels) ** 0.5
        image_h = int(image_h / scale)
        image_w = int(image_w / scale)
    image_h = image_h // h_factor * h_factor
    image_w = image_w // w_factor * w_factor
    return image_h, image_w


def load_video_frames(video_path: str, max_frames: int, target_h: int, target_w: int, stripe: int):
    import imageio
    import torchvision
    from PIL import Image
    from torchvision.transforms import v2

    if not os.path.isabs(video_path):
        raise ValueError('Expected absolute video paths in metadata.')
    reader = imageio.get_reader(video_path)
    video_num_frames = reader.count_frames()
    if stripe * max_frames > video_num_frames:
        stripe = 1

    frame_process = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frames = []
    for i in range(max_frames):
        frame_idx = i * stripe
        if frame_idx >= video_num_frames:
            break
        frame = reader.get_data(frame_idx)
        frame = Image.fromarray(frame)
        frame = torchvision.transforms.functional.resize(frame, (target_h, target_w), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
        frame = torchvision.transforms.functional.center_crop(frame, (target_h, target_w))
        tensor_frame = frame_process(frame)
        frames.append(tensor_frame)
    reader.close()
    if len(frames) == 0:
        raise RuntimeError(f'No frames loaded from {video_path}')
    return torch.stack(frames, dim=0)  # [T, 3, H, W]


def init_distributed_if_needed():
    local_rank_env = os.environ.get('LOCAL_RANK', None)
    ddp = False
    rank = 0
    world_size = 1
    local_rank = 0
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        ddp = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    is_main = (rank == 0)
    return ddp, rank, world_size, local_rank, is_main


def main():
    args = build_argparser().parse_args()

    ddp, rank, world_size, local_rank, is_main = init_distributed_if_needed()

    if ddp and torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    weight_dtype = torch.bfloat16 if args.bf16 else torch.float32

    config = OmegaConf.load(args.config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(device)
    text_encoder.eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(device, dtype=weight_dtype)
    vae.eval()

    metadata = read_json(args.video_edit_metadata)

    tensor_dir = os.path.join(args.dataset_path, args.tensor_folder)
    os.makedirs(tensor_dir, exist_ok=True)

    ids = list(metadata.keys())
    # Shard work across ranks
    for idx, vid in enumerate(ids):
        if world_size > 1 and (idx % world_size) != rank:
            continue
        out_path = os.path.join(tensor_dir, f'{vid}.tensors.pth')
        if os.path.exists(out_path):
            continue

        info = metadata[vid]
        original_path = info['original_video'] if os.path.isabs(info['original_video']) else os.path.join(args.dataset_path, info['original_video'])
        edited_path = info['edited_video'] if os.path.isabs(info['edited_video']) else os.path.join(args.dataset_path, info['edited_video'])
        instruction = info['edit_instruction']

        # Probe first frame to compute dynamic size if needed
        import imageio
        reader = imageio.get_reader(original_path)
        first = reader.get_data(0)
        reader.close()
        h, w = first.shape[0], first.shape[1]
        if args.height is None and args.width is None:
            target_h, target_w = get_height_width_dynamic(w, h, args.max_pixels, args.height_division_factor, args.width_division_factor)
        else:
            target_h = args.height if args.height is not None else 480
            target_w = args.width if args.width is not None else 832

        # Load frames for both original and edited
        orig_frames = load_video_frames(original_path, args.source_frames, target_h, target_w, args.stripe)
        edit_frames = load_video_frames(edited_path, args.edit_frames, target_h, target_w, args.stripe)

        frames = torch.cat([orig_frames, edit_frames], dim=0)  # [T, 3, H, W]
        # Ensure 4k+1 frames
        assert frames.shape[0] % 4 == 1, f'Total frames {frames.shape[0]} should be 4k+1'

        # To [B, C, T, H, W]
        video = rearrange(frames, 't c h w -> 1 c t h w').to(device=device, dtype=weight_dtype)

        with torch.no_grad():
            # VAE encode in micro-batches along batch dimension if needed
            latents_dist = vae.encode(video)[0]
            latents = latents_dist.sample()  # [1, C_lat, T_lat, H_lat, W_lat]

        text_emb = encode_text(tokenizer, text_encoder, [instruction], max_length=args.tokenizer_max_length, device=device)
        # Squeeze batch dim for saving convenience
        text_emb = {k: v[0].cpu() for k, v in text_emb.items()}

        payload = {
            'latents': latents[0].cpu(),
            'prompt_emb': text_emb,
        }
        save_tensors(out_path, payload)

    if ddp:
        dist.barrier()
    if is_main:
        print('Precompute done.')
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

