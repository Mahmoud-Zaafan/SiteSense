"""
Extract frames from YouTube videos for annotation.

Usage:
    1. Install: pip install yt-dlp
    2. Edit YOUTUBE_URLS below
    3. Run: python scripts/extract_youtube_frames.py
    4. Upload extracted frames to Roboflow for annotation
"""

import os
import subprocess
import sys


# === EDIT THESE ===
YOUTUBE_URLS = [
    # Add your YouTube URLs here — construction/mining equipment videos
    # Example: "https://www.youtube.com/watch?v=XXXXXXXXXXX",
]

OUTPUT_DIR = "data/youtube_frames"
FRAMES_PER_SECOND = 0.5  # 1 frame every 2 seconds
MAX_RESOLUTION = 720      # Download at 720p max (saves bandwidth)


def download_and_extract(url: str, index: int):
    """Download a YouTube video and extract frames."""
    video_dir = os.path.join(OUTPUT_DIR, f"video_{index:02d}")
    video_path = os.path.join(video_dir, "video.mp4")
    frames_dir = os.path.join(video_dir, "frames")

    os.makedirs(frames_dir, exist_ok=True)

    # Step 1: Download video
    print(f"\n[{index}] Downloading: {url}")
    dl_cmd = [
        "yt-dlp",
        "-f", f"best[height<={MAX_RESOLUTION}]",
        "-o", video_path,
        "--no-playlist",
        url
    ]
    subprocess.run(dl_cmd, check=True)

    # Step 2: Extract frames
    print(f"[{index}] Extracting frames (1 every {1/FRAMES_PER_SECOND:.0f}s)...")
    ff_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={FRAMES_PER_SECOND}",
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg")
    ]
    subprocess.run(ff_cmd, check=True)

    # Count frames
    n_frames = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    print(f"[{index}] Extracted {n_frames} frames → {frames_dir}")

    return n_frames


def main():
    if not YOUTUBE_URLS:
        print("No URLs configured!")
        print("Edit YOUTUBE_URLS in this script and add your construction video URLs.")
        sys.exit(1)

    # Check dependencies
    for dep in ["yt-dlp", "ffmpeg"]:
        if subprocess.run(["where", dep], capture_output=True).returncode != 0:
            print(f"ERROR: {dep} not found. Install it first.")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_frames = 0
    for i, url in enumerate(YOUTUBE_URLS, 1):
        try:
            n = download_and_extract(url, i)
            total_frames += n
        except Exception as e:
            print(f"[{i}] Failed: {e}")

    print(f"\n{'='*60}")
    print(f"  Done! {total_frames} frames extracted to {OUTPUT_DIR}")
    print(f"  Next: upload to Roboflow for annotation")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
