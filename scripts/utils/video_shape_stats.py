#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import imageio
import imageio_ffmpeg as iioff
from tqdm import tqdm


def read_json(ann_path: str) -> List[Dict]:
    with open(ann_path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # {id: {original_video, edited_video, ...}}
        return list(data.values())
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unsupported JSON format: expected list or dict at top-level.")


def probe_video_size(video_path: str) -> Optional[Tuple[int, int]]:
    # Prefer imageio (FFmpeg-backed), fallback to OpenCV
    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        try:
            meta = reader.get_meta_data()
            size = meta.get("size", None)  # (width, height)
            if size and len(size) == 2:
                width, height = int(size[0]), int(size[1])
                if width > 0 and height > 0:
                    return height, width
        except Exception:
            pass
        # Fallback to grabbing first frame
        try:
            frame = reader.get_data(0)
            height, width = int(frame.shape[0]), int(frame.shape[1])
            if width > 0 and height > 0:
                return height, width
        except Exception:
            pass
        finally:
            try:
                reader.close()
            except Exception:
                pass
    except Exception:
        # Try imageio-ffmpeg direct reader
        try:
            reader2 = iioff.get_reader(video_path)
            try:
                meta2 = reader2.get_meta_data()
                size2 = meta2.get("size", None)
                if size2 and len(size2) == 2:
                    width, height = int(size2[0]), int(size2[1])
                    if width > 0 and height > 0:
                        return height, width
                # try first frame
                frame2 = next(reader2)
                if frame2 is not None:
                    height, width = int(frame2.shape[0]), int(frame2.shape[1])
                    if width > 0 and height > 0:
                        return height, width
            finally:
                try:
                    reader2.close()
                except Exception:
                    pass
        except Exception:
            pass

    # OpenCV fallback if imageio fails
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width <= 0 or height <= 0:
            return None
        return height, width
    except Exception:
        return None


def collect_paths(sample: Dict, base_dir: str, include_edited: bool) -> List[str]:
    paths: List[str] = []
    # prefer keys used by VideoEditDataset
    if "original_video" in sample:
        paths.append(os.path.join(base_dir, sample["original_video"]))
    if include_edited and "edited_video" in sample:
        paths.append(os.path.join(base_dir, sample["edited_video"]))

    # fallbacks (support alternate schemas)
    if not paths and "file_path" in sample:
        paths.append(os.path.join(base_dir, sample["file_path"]))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Multithread video shape statistics from JSON dataset.")
    parser.add_argument("--ann_path", type=str, required=True, help="Path to JSON annotations.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory for video files.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker threads.")
    parser.add_argument("--include_edited", action="store_true", help="Also probe edited_video clips if present.")
    parser.add_argument("--dedup", action="store_true", help="Skip repeated paths (deduplicate).")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write stats JSON.")
    parser.add_argument("--topk", type=int, default=50, help="Print top-K most common resolutions.")
    args = parser.parse_args()

    samples = read_json(args.ann_path)

    # Build path list
    all_paths: List[str] = []
    for s in samples:
        all_paths.extend(collect_paths(s, args.base_dir, include_edited=args.include_edited))

    if args.dedup:
        # keep order while dedup
        seen = set()
        deduped = []
        for p in all_paths:
            if p not in seen:
                deduped.append(p)
                seen.add(p)
        all_paths = deduped

    total = len(all_paths)
    if total == 0:
        print("No video paths found. Check --base_dir and JSON keys.")
        sys.exit(1)

    counts: Counter = Counter()
    missing_paths: List[str] = []
    decode_failures: List[str] = []

    def task(path: str) -> Tuple[str, str, Optional[Tuple[int, int]]]:
        if not os.path.exists(path):
            return path, "missing", None
        size = probe_video_size(path)
        if size is None:
            return path, "decode_fail", None
        return path, "ok", size

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(task, p) for p in all_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Probing"):
            path, status, size = fut.result()
            if status == "ok" and size is not None:
                counts[size] += 1
            elif status == "missing":
                missing_paths.append(path)
            else:
                decode_failures.append(path)

    success = sum(counts.values())
    print(f"Total videos: {total}")
    print(f"Successful:   {success}")
    print(f"Missing:      {len(missing_paths)}")
    print(f"Decode fail:  {len(decode_failures)}")

    # Print top-K resolutions
    print("\nTop resolutions (H x W):")
    for (h, w), c in counts.most_common(args.topk):
        aspect = h / w if w != 0 else 0.0
        print(f"  {h}x{w}  count={c}  aspect={aspect:.4f}")

    # Aggregate by aspect ratio rounded to 4 decimals
    ratio_counts: Dict[str, int] = defaultdict(int)
    for (h, w), c in counts.items():
        if w > 0:
            key = f"{(h / w):.4f}"
        else:
            key = "inf"
        ratio_counts[key] += c

    # Prepare JSON output
    if args.output is not None:
        out = {
            "total": total,
            "successful": success,
            "missing": len(missing_paths),
            "decode_fail": len(decode_failures),
            "resolutions": {f"{h}x{w}": c for (h, w), c in counts.items()},
            "aspect_ratios": dict(sorted(ratio_counts.items(), key=lambda kv: kv[1], reverse=True)),
            "missing_paths": missing_paths,
            "decode_failed_paths": decode_failures,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nWrote stats to: {args.output}")


if __name__ == "__main__":
    main()


