"""
Activity Clip Extractor
=======================
Runs the existing detection+tracking pipeline on a video and saves
16-frame cropped clips per tracked equipment for activity classification
training.

Usage:
  python scripts/extract_activity_clips.py \
    --video path/to/video.mp4 \
    --output clips/ \
    --model models/rfdetr_construction.pth

Output structure:
  clips/
    UNLABELED/          ← clips go here initially (no label yet)
      WL-001_f0120.mp4
      WL-001_f0136.mp4
      DT-001_f0120.mp4
      ...
    DIGGING/            ← you move clips here after labeling
    LOADING/
    DUMPING/
    WAITING/

Each clip is 16 frames of the cropped+resized equipment bbox at 224x224.
After extraction, manually sort clips into the 4 activity folders.
~500 clips per class is sufficient for fine-tuning.
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def extract_clips(video_path: str, output_dir: str, model_path: str,
                  clip_length: int = 16, clip_stride: int = 8,
                  crop_size: int = 224, conf_threshold: float = 0.50):
    """
    Extract equipment crops as short video clips for activity labeling.

    Args:
        video_path: Path to input video
        output_dir: Directory to save clips
        model_path: Path to RF-DETR weights
        clip_length: Frames per clip (default 16 — standard for C3D/X3D)
        clip_stride: Frames between clip starts (8 = 50% overlap)
        crop_size: Resize crops to this size (224 = standard ImageNet/Kinetics)
        conf_threshold: Minimum detection confidence
    """
    from ultralytics import RTDETR
    from boxmot import BoTSORT

    # Load detector
    detector = RTDETR(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"  FPS: {fps}, Frames: {total_frames}, Duration: {total_frames/fps:.1f}s")

    # Create output dirs
    unlabeled_dir = os.path.join(output_dir, 'UNLABELED')
    for d in ['UNLABELED', 'DIGGING', 'LOADING', 'DUMPING', 'WAITING']:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    # Class names (must match RF-DETR detector)
    CLASS_NAMES = {
        0: 'excavator', 1: 'dump_truck', 2: 'bulldozer', 3: 'wheel_loader',
        4: 'mobile_crane', 5: 'tower_crane', 6: 'roller_compactor', 7: 'cement_mixer',
    }
    CLASS_PREFIXES = {
        0: 'EX', 1: 'DT', 2: 'BD', 3: 'WL',
        4: 'MC', 5: 'TC', 6: 'RC', 7: 'CM',
    }

    # Track state: track_id -> {crop_buffer: list, equip_id: str, class_id: int}
    track_buffers = {}
    equip_counters = {}
    track_to_equip = {}
    clips_saved = 0
    frame_id = 0

    # Simple tracker for ID assignment
    tracker = BoTSORT(
        reid_weights=Path('models/osnet_x0_25_msmt17.pt'),
        device='0',
        half=True
    )

    print(f"\nExtracting {clip_length}-frame clips with stride {clip_stride}...")
    print(f"Crop size: {crop_size}x{crop_size}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % 100 == 0:
            print(f"  Frame {frame_id}/{total_frames} ({frame_id/total_frames*100:.0f}%) — {clips_saved} clips saved")

        # Detect
        results = detector(frame, conf=conf_threshold, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            continue

        detections = results[0].boxes.data.cpu().numpy()
        if len(detections) == 0:
            continue

        # Track
        tracked = tracker.update(detections, frame)
        if tracked is None or len(tracked) == 0:
            continue

        for det in tracked:
            if len(det) < 6:
                continue

            x1, y1, x2, y2 = det[:4].astype(int)
            track_id = int(det[4])
            class_id = int(det[5]) if len(det) > 5 else -1

            # Assign equipment ID
            if track_id not in track_to_equip:
                prefix = CLASS_PREFIXES.get(class_id, 'UN')
                equip_counters[prefix] = equip_counters.get(prefix, 0) + 1
                track_to_equip[track_id] = f"{prefix}-{equip_counters[prefix]:03d}"

            equip_id = track_to_equip[track_id]

            # Crop and resize
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            crop_resized = cv2.resize(crop, (crop_size, crop_size))

            # Buffer crops per track
            if track_id not in track_buffers:
                track_buffers[track_id] = {
                    'crops': [],
                    'equip_id': equip_id,
                    'class_id': class_id,
                    'start_frame': frame_id
                }

            buf = track_buffers[track_id]
            buf['crops'].append(crop_resized)

            # Save clip when buffer reaches clip_length
            if len(buf['crops']) >= clip_length:
                clip_frames = buf['crops'][:clip_length]

                # Save as mp4
                clip_name = f"{equip_id}_f{buf['start_frame']:06d}.mp4"
                clip_path = os.path.join(unlabeled_dir, clip_name)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(clip_path, fourcc, fps, (crop_size, crop_size))
                for cf in clip_frames:
                    writer.write(cf)
                writer.release()

                clips_saved += 1

                # Stride: keep last (clip_length - clip_stride) frames
                buf['crops'] = buf['crops'][clip_stride:]
                buf['start_frame'] = frame_id - len(buf['crops']) + 1

    cap.release()

    print(f"\nDone! Saved {clips_saved} clips to {unlabeled_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Open {unlabeled_dir}/ and watch each clip")
    print(f"  2. Move each clip to the correct folder:")
    print(f"     DIGGING/  — bucket low, scooping material")
    print(f"     LOADING/  — machine swinging/driving with load")
    print(f"     DUMPING/  — bucket elevated, releasing material")
    print(f"     WAITING/  — machine idle, no motion")
    print(f"  3. Aim for ~500 clips per class")
    print(f"  4. Run: python training/train_activity_classifier.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract activity clips for training')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', default='clips', help='Output directory')
    parser.add_argument('--model', required=True, help='RF-DETR model weights path')
    parser.add_argument('--clip-length', type=int, default=16, help='Frames per clip')
    parser.add_argument('--clip-stride', type=int, default=8, help='Stride between clips')
    parser.add_argument('--crop-size', type=int, default=224, help='Crop resize dimension')
    parser.add_argument('--conf', type=float, default=0.50, help='Detection confidence threshold')
    args = parser.parse_args()

    extract_clips(
        video_path=args.video,
        output_dir=args.output,
        model_path=args.model,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        crop_size=args.crop_size,
        conf_threshold=args.conf
    )
