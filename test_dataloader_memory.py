#!/usr/bin/env python
"""Test dataloader memory usage"""
import torch
import sys
import os
import psutil
import gc
from omegaconf import OmegaConf

# Add project paths
sys.path.insert(0, "/home/xianyang/Data/code/VideoX-Fun")

from videox_fun.data.dataset_image_video import VideoEditDataset
from videox_fun.data.bucket_sampler import AspectRatioBatchImageVideoSampler, RandomSampler, CUSTOM_ASPECT_RATIOS

def get_memory_info():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / 1024**3
    
    # Get system memory
    sys_mem = psutil.virtual_memory()
    sys_mem_gb = sys_mem.used / 1024**3
    sys_mem_total_gb = sys_mem.total / 1024**3
    
    return {
        'process_mem_gb': mem_gb,
        'system_mem_gb': sys_mem_gb,
        'system_mem_total_gb': sys_mem_total_gb,
        'system_mem_percent': sys_mem.percent
    }

def test_dataset_loading():
    print("="*60)
    print("Testing VideoEditDataset Memory Usage")
    print("="*60)
    
    # Initial memory
    mem = get_memory_info()
    print(f"\n[INITIAL] Process: {mem['process_mem_gb']:.2f}GB, System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")
    
    # Create dataset
    print("\n[1] Creating VideoEditDataset...")
    dataset = VideoEditDataset(
        ann_path="/scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json",
        data_root="/scratch3/yan204/yxp/Senorita",
        video_sample_stride=4,
        video_sample_n_frames=65,
        source_frames=33,
        edit_frames=32,
        text_drop_ratio=0.1,
        enable_bucket=True,
        enable_inpaint=True,
    )
    
    mem = get_memory_info()
    print(f"[AFTER DATASET] Process: {mem['process_mem_gb']:.2f}GB, System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")
    
    # Create sampler
    print("\n[2] Creating batch sampler...")
    batch_sampler = AspectRatioBatchImageVideoSampler(
        sampler=RandomSampler(dataset.dataset, generator=torch.Generator().manual_seed(42)),
        dataset=dataset.dataset,
        batch_size=1,
        train_folder="/scratch3/yan204/yxp/Senorita",
        drop_last=True,
        aspect_ratios=CUSTOM_ASPECT_RATIOS,
    )
    
    mem = get_memory_info()
    print(f"[AFTER SAMPLER] Process: {mem['process_mem_gb']:.2f}GB, System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")
    
    # Test different num_workers
    for num_workers in [0, 1, 2]:
        print(f"\n[3] Testing with num_workers={num_workers}")
        
        # Force garbage collection
        gc.collect()
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            persistent_workers=False,
        )
        
        mem = get_memory_info()
        print(f"  [AFTER DATALOADER] Process: {mem['process_mem_gb']:.2f}GB, System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")
        
        # Load a few batches
        print(f"  Loading 5 batches...")
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            
            pixel_values = batch['pixel_values']
            text = batch['text']
            
            mem = get_memory_info()
            print(f"    Batch {i}: shape={pixel_values.shape}, "
                  f"Process: {mem['process_mem_gb']:.2f}GB, "
                  f"System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")
            
            # Clean up
            del pixel_values, text, batch
            
        del dataloader
        gc.collect()
        
        mem = get_memory_info()
        print(f"  [AFTER CLEANUP] Process: {mem['process_mem_gb']:.2f}GB, System: {mem['system_mem_gb']:.2f}/{mem['system_mem_total_gb']:.2f}GB ({mem['system_mem_percent']:.1f}%)")

if __name__ == "__main__":
    test_dataset_loading()
