"""
Pre-extract start, middle, and end frames from SS-v2 webm videos.

Uniform 3-split: frame indices at 0%, 50%, 100% of video length.
These map to pi0-FAST camera slots:
  base_0_rgb  <- start frame
  base_1_rgb  <- middle frame
  wrist_0_rgb <- end frame

Output structure:
  datasets/20bn-something-something-v2-frames/
    {video_id}_start.jpg
    {video_id}_middle.jpg
    {video_id}_end.jpg
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys

import cv2

VIDEO_DIR = Path("/home/main/storage/sungjin/datasets/20bn-something-something-v2")
OUT_DIR = Path("/home/main/storage/sungjin/datasets/20bn-something-something-v2-frames")
import argparse

LABELS_PATH = Path(__file__).parent / "labels/filtered_train.json"
NUM_WORKERS = 16
JPEG_QUALITY = 95


def extract_frames(video_id: str) -> str | None:
    """Extract start, middle, end frames from a video. Returns error string or None on success."""
    video_path = VIDEO_DIR / f"{video_id}.webm"
    start_path = OUT_DIR / f"{video_id}_start.jpg"
    middle_path = OUT_DIR / f"{video_id}_middle.jpg"
    end_path = OUT_DIR / f"{video_id}_end.jpg"

    # Skip only if all three frames already exist
    if start_path.exists() and middle_path.exists() and end_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"cannot open {video_path}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    def read_at(frame_idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None

    # Start frame (index 0)
    if not start_path.exists():
        frame = read_at(0)
        if frame is None:
            cap.release()
            return f"cannot read start frame: {video_path}"
        cv2.imwrite(str(start_path), frame, encode_params)

    # Middle frame (index total // 2)
    # For a 1-frame video this equals the start frame — acceptable fallback
    if not middle_path.exists():
        frame = read_at(max(total // 2, 0))
        if frame is None:
            frame = read_at(0)  # fallback to start
        if frame is not None:
            cv2.imwrite(str(middle_path), frame, encode_params)

    # End frame (index total - 1)
    if not end_path.exists():
        frame = read_at(max(total - 1, 0))
        if frame is None:
            frame = read_at(0)  # fallback to start
        if frame is not None:
            cv2.imwrite(str(end_path), frame, encode_params)

    cap.release()
    return None


def worker(args):
    idx, video_id, total = args
    error = extract_frames(video_id)
    if idx % 1000 == 0:
        print(f"[{idx}/{total}] {video_id}", flush=True)
    return (video_id, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(LABELS_PATH))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.labels) as f:
        data = json.load(f)

    video_ids = [e["id"] for e in data]
    total = len(video_ids)
    print(f"Extracting frames for {total:,} videos with {NUM_WORKERS} workers...")

    args = [(i, vid, total) for i, vid in enumerate(video_ids)]

    errors = []
    with mp.Pool(NUM_WORKERS) as pool:
        for video_id, error in pool.imap_unordered(worker, args, chunksize=50):
            if error:
                errors.append(error)

    print(f"\nDone. Errors: {len(errors)}")
    if errors:
        error_path = Path(__file__).parent / "extract_errors.txt"
        error_path.write_text("\n".join(errors))
        print(f"Errors written to {error_path}")

    # Final count
    starts = len(list(OUT_DIR.glob("*_start.jpg")))
    middles = len(list(OUT_DIR.glob("*_middle.jpg")))
    ends = len(list(OUT_DIR.glob("*_end.jpg")))
    print(f"Start frames: {starts:,}  Middle frames: {middles:,}  End frames: {ends:,}")


if __name__ == "__main__":
    main()
