import cv2
import os
from pathlib import Path

# =========================
# CONFIG
# =========================
INPUT_DIR = Path(r"./videos")       # folder containing your source videos
OUTPUT_DIR = Path(r"./frames_out")  # where extracted images are saved
FRAMES_PER_VIDEO = 30               # number of frames to extract per video

# Optional resize (recommended): longest side in px. Set to None to disable resize.
RESIZE_LONG_SIDE = 1280

# Supported video extensions
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

# =========================
# UTILS
# =========================
def resize_keep_aspect(img, long_side: int):
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img
    if w >= h:
        new_w = long_side
        new_h = int(h * (long_side / w))
    else:
        new_h = long_side
        new_w = int(w * (long_side / h))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def safe_name(p: Path) -> str:
    # sanitized folder/file-friendly stem
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in p.stem)

# =========================
# MAIN
# =========================
def extract_frames_from_video(video_path: Path, out_dir: Path, frames_per_video: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path.name}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"[WARN] Frame count not available for: {video_path.name}")
        cap.release()
        return 0

    # Evenly distributed frame indexes.
    # Skip first and last frame to reduce blur/black frame risk.
    if total_frames < (frames_per_video + 2):
        indices = list(range(total_frames))
    else:
        indices = [
            int((i + 1) * (total_frames - 2) / (frames_per_video + 1))
            for i in range(frames_per_video)
        ]

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] Frame {frame_idx} could not be read from {video_path.name}")
            continue

        if RESIZE_LONG_SIDE is not None:
            frame = resize_keep_aspect(frame, RESIZE_LONG_SIDE)

        # output filename
        out_file = out_dir / f"{safe_name(video_path)}_{i:03d}_f{frame_idx}.jpg"
        ok = cv2.imwrite(str(out_file), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            saved += 1

    cap.release()
    return saved


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"[ERROR] No videos found in: {INPUT_DIR.resolve()}")
        print("Put videos in the 'videos' folder (or change INPUT_DIR).")
        return

    print(f"Found {len(videos)} videos.")
    total_saved = 0

    for v in sorted(videos):
        sub = OUTPUT_DIR / safe_name(v)
        saved = extract_frames_from_video(v, sub, FRAMES_PER_VIDEO)
        total_saved += saved
        print(f"[OK] {v.name}: saved {saved} images in {sub}")

    print(f"\nDone. Total images saved: {total_saved}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()