#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
import multiprocessing as mp
from functools import partial
from typing import Tuple, Dict, Any

from tqdm import tqdm


def abspath(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(base, p)


def ffprobe_count_frames(path: str, ffprobe: str = "ffprobe") -> int:
    try:
        cmd1 = [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames",
                "-of", "default=nokey=1:noprint_wrappers=1", path]
        out = subprocess.check_output(cmd1, stderr=subprocess.STDOUT, text=True).strip()
        if out.isdigit():
            return int(out)
        cmd2 = [ffprobe, "-v", "error", "-count_packets", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=nokey=1:noprint_wrappers=1", path]
        out2 = subprocess.check_output(cmd2, stderr=subprocess.STDOUT, text=True).strip()
        return int(out2) if out2.isdigit() else 0
    except Exception:
        return 0


def quick_decode(path: str) -> bool:
    try:
        import imageio
        r = imageio.get_reader(path)
        _ = r.get_data(0)
        r.close()
        return True
    except Exception:
        return False


_GRAY_NUM_RE = re.compile(r"_gray_(\d+)\.mp4$")
_GRAY_RE = re.compile(r"_gray(\.mp4)$")


def transform_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    new_e = dict(entry)
    old_edited = entry.get("edited_video", "")
    new_e["grounded_video"] = old_edited

    def _replace_gray_to_rem(p: str) -> str:
        m = _GRAY_NUM_RE.search(p)
        if m is not None:
            return _GRAY_NUM_RE.sub(r"_rem_\1.mp4", p)
        m2 = _GRAY_RE.search(p)
        if m2 is not None:
            return _GRAY_RE.sub(r"_rem\1", p)
        return p.replace("gray", "rem")

    new_e["edited_video"] = _replace_gray_to_rem(old_edited)
    return new_e


def check_triplet(item: Tuple[str, Dict[str, Any]], base_path: str, src_need: int, grd_need: int, edt_need: int,
                  ffprobe_bin: str, verify_decode: bool) -> Tuple[str, str, str, Dict[str, Any]]:
    vid, info = item
    info = transform_entry(info)

    src_path = abspath(base_path, info["original_video"]) if "original_video" in info else None
    grd_path = abspath(base_path, info["grounded_video"]) if "grounded_video" in info else None
    edt_path = abspath(base_path, info["edited_video"]) if "edited_video" in info else None

    if not src_path or not os.path.exists(src_path):
        return ("drop", vid, "missing_src", info)
    if not grd_path or not os.path.exists(grd_path):
        return ("drop", vid, "missing_grd", info)
    if not edt_path or not os.path.exists(edt_path):
        return ("drop", vid, "missing_edt", info)

    src_n = ffprobe_count_frames(src_path, ffprobe=ffprobe_bin)
    if src_n < src_need:
        return ("drop", vid, f"src_short({src_n}<{src_need})", info)

    grd_n = ffprobe_count_frames(grd_path, ffprobe=ffprobe_bin)
    if grd_n < grd_need:
        return ("drop", vid, f"grd_short({grd_n}<{grd_need})", info)

    edt_n = ffprobe_count_frames(edt_path, ffprobe=ffprobe_bin)
    if edt_n < edt_need:
        return ("drop", vid, f"edt_short({edt_n}<{edt_need})", info)

    if verify_decode:
        if not quick_decode(src_path):
            return ("drop", vid, "src_decode_fail", info)
        if not quick_decode(grd_path):
            return ("drop", vid, "grd_decode_fail", info)
        if not quick_decode(edt_path):
            return ("drop", vid, "edt_decode_fail", info)

    return ("keep", vid, "ok", info)


def main():
    ap = argparse.ArgumentParser("Convert gray->rem, add grounded_video, and validate triplets")
    ap.add_argument("--in_json", required=True, help="输入原始 JSON（字典：{video_id: {...}}）")
    ap.add_argument("--out_json", required=True, help="输出的新 JSON 路径")
    ap.add_argument("--base_path", required=True, help="视频根目录（解析相对路径用）")
    ap.add_argument("--src_frames", type=int, default=33, help="original_video 最少帧数")
    ap.add_argument("--grd_frames", type=int, default=4, help="grounded_video 最少帧数")
    ap.add_argument("--edt_frames", type=int, default=32, help="edited_video 最少帧数")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8)//2), help="并行进程数")
    ap.add_argument("--ffprobe", default="ffprobe", help="ffprobe 可执行文件名或路径")
    ap.add_argument("--verify_decode", action="store_true", help="尝试解码第一帧以剔除假阳（更稳但更慢）")
    ap.add_argument("--drop_log", default=None, help="被删除项日志(.jsonl)，默认与 out_json 同名旁存")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError("metadata 必须是字典 {video_id: {...}} 结构")

    items = list(meta.items())
    fn = partial(
        check_triplet,
        base_path=args.base_path,
        src_need=args.src_frames,
        grd_need=args.grd_frames,
        edt_need=args.edt_frames,
        ffprobe_bin=args.ffprobe,
        verify_decode=args.verify_decode,
    )

    kept_ids = []
    kept_infos = {}
    dropped = []

    with mp.Pool(processes=args.workers) as pool:
        for tag, vid, reason, new_info in tqdm(pool.imap_unordered(fn, items, chunksize=64), total=len(items), desc="Converting+Checking"):
            if tag == "keep":
                kept_ids.append(vid)
                kept_infos[vid] = new_info
            else:
                dropped.append((vid, reason, new_info))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as wf:
        json.dump(kept_infos, wf, ensure_ascii=False, indent=2)

    drop_log = args.drop_log or (os.path.splitext(args.out_json)[0] + ".dropped.jsonl")
    with open(drop_log, "w", encoding="utf-8") as lf:
        for vid, reason, info in dropped:
            lf.write(json.dumps({"video_id": vid, "reason": reason, "info": info}, ensure_ascii=False) + "\n")

    print(f"\nInput total: {len(items)}")
    print(f"Kept: {len(kept_ids)}")
    print(f"Dropped: {len(dropped)}")
    if dropped:
        agg = {}
        for _, r, _ in dropped:
            agg[r] = agg.get(r, 0) + 1
        print("Drop reasons (top):")
        for k, v in sorted(agg.items(), key=lambda x: -x[1])[:20]:
            print(f"  {k}: {v}")
    print(f"\nWrote new JSON to: {args.out_json}")
    print(f"Wrote drop log to: {drop_log}")


if __name__ == "__main__":
    main()


