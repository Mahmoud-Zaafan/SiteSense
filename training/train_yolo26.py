"""
YOLO26-L — 8-Class Construction Equipment Training Notebook
============================================================
Paste each CELL into a separate Kaggle notebook cell.

Mirrors `train_rfdetr_v3.py` cell-by-cell so the two runs are
directly comparable (same datasets, same splits, same oversampling).

Prerequisites:
  - H100 GPU (or any GPU with 40GB+ VRAM)
  - Input Dataset: MOCS (xiaopan9802/mocs-dataset)
  - Roboflow API key saved as Kaggle Secret: ROBOFLOW_API_KEY

Datasets:
  - MOCS (Kaggle: xiaopan9802/mocs-dataset) — 41,668 images, 13 classes
    Path: /kaggle/input/datasets/xiaopan9802/mocs-dataset
    Structure: instances_train/ + instances_train.json
               instances_val/   + instances_val.json
  - ACID v2 (Roboflow: test-blhxw/acid-dataset, version 2) — 23,801 images, 10 classes

Target Classes (8): excavator (0), dump_truck (1), bulldozer (2), wheel_loader (3),
                    mobile_crane (4), tower_crane (5), roller_compactor (6), cement_mixer (7)

Model: YOLO26-L (Ultralytics, Jan 2026)
  - 24.8M params, 55.0 COCO mAP50-95
  - STAL (Small-Target-Aware Label assignment) — ideal for aerial/drone footage
  - NMS-free end-to-end inference
  - ProgLoss for class imbalance
  - To switch to YOLO26-X: change MODEL_NAME = "yolo26x.pt"
"""

# ============================================================
# CELL 1 — Install Dependencies + GPU Profile
# ============================================================
!pip install -q ultralytics roboflow faster-coco-eval pyyaml

import os
import torch

print("=" * 60)
print("  GPU PROFILE")
print("=" * 60)

gpu_name = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()
print(f"  GPU:           {gpu_name}")

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
    print(f"  VRAM Total:    {total_mem:.1f} GB")
    print(f"  VRAM Free:     {free_mem:.1f} GB")
    print(f"  CUDA Version:  {torch.version.cuda}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  BF16 Support:  {torch.cuda.is_bf16_supported()}")
    print(f"  Compute Cap:   {torch.cuda.get_device_capability()}")
else:
    print("  WARNING: No GPU detected!")

# Verify ultralytics version supports YOLO26
import ultralytics
print(f"\n  Ultralytics:   {ultralytics.__version__}")
print(f"  Dependencies installed")

# ============================================================
# CELL 2 — Download ACID Dataset from Roboflow
# ============================================================
from roboflow import Roboflow

# Get API key from Kaggle Secrets (Add-ons → Secrets → ROBOFLOW_API_KEY)
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
api_key = secrets.get_secret("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=api_key)
project = rf.workspace("test-blhxw").project("acid-dataset")
version = project.version(2)  # v2 = 23,801 images (NOT v1 which is only 9,917)
dataset = version.download("coco", location="/kaggle/working/acid_raw")

print("ACID v2 dataset downloaded (23,801 images)")
print(f"   Location: /kaggle/working/acid_raw")

# ============================================================
# CELL 3 — Inspect Both Datasets + Class Balance Report
# ============================================================
import json
from collections import Counter
from pathlib import Path

def inspect_coco(name, json_path):
    """Print class distribution for a COCO dataset."""
    with open(json_path) as f:
        coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in coco['categories']}
    counts = Counter()
    for ann in coco['annotations']:
        counts[cat_map.get(ann['category_id'], f"id_{ann['category_id']}")] += 1

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Images: {len(coco['images']):,}")
    print(f"  Annotations: {len(coco['annotations']):,}")
    print(f"  Categories: {list(cat_map.values())}")
    print(f"\n  Class distribution:")
    max_count = counts.most_common(1)[0][1] if counts else 1
    for cls, count in counts.most_common():
        bar = '#' * int(40 * count / max_count)
        pct = 100 * count / sum(counts.values())
        print(f"    {cls:30s} {count:>6,} ({pct:5.1f}%) {bar}")
    return coco

# ── MOCS ──
MOCS_BASE = "/kaggle/input/datasets/xiaopan9802/mocs-dataset"
mocs_train_json = os.path.join(MOCS_BASE, "instances_train.json")
mocs_val_json = os.path.join(MOCS_BASE, "instances_val.json")

if os.path.exists(mocs_train_json):
    mocs_train_coco = inspect_coco("MOCS (train)", mocs_train_json)
if os.path.exists(mocs_val_json):
    mocs_val_coco = inspect_coco("MOCS (val)", mocs_val_json)

# ── ACID ──
acid_base = "/kaggle/working/acid_raw"
for split in ['train', 'valid', 'test']:
    acid_json = os.path.join(acid_base, split, "_annotations.coco.json")
    if os.path.exists(acid_json):
        inspect_coco(f"ACID ({split})", acid_json)

# ============================================================
# CELL 4 — Merge Datasets into 8-Class Unified COCO Dataset
# ============================================================
# IDENTICAL to train_rfdetr_v3.py CELL 4 — same classes, same mapping,
# same oversampling. This guarantees both runs train on identical data.

import json
import shutil
from pathlib import Path
from collections import Counter

# === 8 Target Classes ===
TARGET_CATEGORIES = [
    {"id": 0, "name": "excavator"},
    {"id": 1, "name": "dump_truck"},
    {"id": 2, "name": "bulldozer"},
    {"id": 3, "name": "wheel_loader"},
    {"id": 4, "name": "mobile_crane"},
    {"id": 5, "name": "tower_crane"},
    {"id": 6, "name": "roller_compactor"},
    {"id": 7, "name": "cement_mixer"},
]

# === Class Mapping: source name -> target ID ===
CLASS_MAPPING = {
    # CLASS 0: excavator
    "Excavator": 0, "excavator": 0,

    # CLASS 1: dump_truck (ONLY actual dump/haul trucks)
    "Truck": 1, "truck": 1, "dump_truck": 1,
    "Dump Truck": 1, "dump truck": 1, "Dump_Truck": 1,

    # CLASS 2: bulldozer (NOT compactors/rollers)
    "Bulldozer": 2, "bulldozer": 2,
    "dozer": 2, "Dozer": 2,

    # CLASS 3: wheel_loader
    "Loader": 3,
    "wheel_loader": 3,
    "Wheel Loader": 3,

    # CLASS 4: mobile_crane
    "mobile_crane": 4,
    "Crane": 4,

    # CLASS 5: tower_crane
    "tower_crane": 5,
    "Static crane": 5,

    # CLASS 6: roller_compactor
    "compactor": 6,
    "Roller": 6,

    # CLASS 7: cement_mixer
    "cement_truck": 7,
    "Concrete mixer": 7,

    # DROPPED — silently ignored
}

MOCS_BASE = "/kaggle/input/datasets/xiaopan9802/mocs-dataset"

MOCS_SPLITS = {
    "train": {
        "json": os.path.join(MOCS_BASE, "instances_train.json"),
        "img_dir": os.path.join(MOCS_BASE, "instances_train", "instances_train"),
    },
    "valid": {
        "json": os.path.join(MOCS_BASE, "instances_val.json"),
        "img_dir": os.path.join(MOCS_BASE, "instances_val", "instances_val"),
    },
}


def find_mocs_img_dir(base_dir, json_path):
    with open(json_path) as f:
        coco = json.load(f)
    if not coco['images']:
        return base_dir
    sample_fname = Path(coco['images'][0]['file_name']).name
    candidates = [
        Path(base_dir),
        Path(base_dir) / Path(json_path).stem,
        Path(base_dir).parent,
    ]
    for d in candidates:
        if (d / sample_fname).exists():
            return str(d)
    for root, dirs, files in os.walk(Path(base_dir).parent):
        if sample_fname in files:
            return root
    return base_dir


def process_mocs_split(mocs_json_path, mocs_img_dir, merged, img_id_counter, ann_id_counter, out_dir):
    if not os.path.exists(mocs_json_path):
        return img_id_counter, ann_id_counter, 0, 0

    mocs_img_dir = find_mocs_img_dir(mocs_img_dir, mocs_json_path)
    print(f"    MOCS image dir: {mocs_img_dir}")

    with open(mocs_json_path) as f:
        mocs_coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in mocs_coco['categories']}
    img_lookup = {img['id']: img for img in mocs_coco['images']}
    anns_by_img = {}
    for ann in mocs_coco['annotations']:
        anns_by_img.setdefault(ann['image_id'], []).append(ann)

    kept = 0
    skipped = 0

    for img_id, img_info in img_lookup.items():
        anns = anns_by_img.get(img_id, [])

        mapped_anns = []
        for ann in anns:
            src_name = cat_map.get(ann['category_id'], '')
            target_id = CLASS_MAPPING.get(src_name)
            if target_id is not None:
                mapped_anns.append((ann, target_id))

        if not mapped_anns:
            skipped += 1
            continue

        fname = Path(img_info['file_name']).name
        src_path = Path(mocs_img_dir) / fname
        if not src_path.exists():
            src_path = Path(mocs_img_dir) / img_info['file_name']
        if not src_path.exists():
            skipped += 1
            continue

        new_fname = f"mocs_{img_id_counter:06d}{src_path.suffix}"
        shutil.copy2(src_path, out_dir / new_fname)

        merged["images"].append({
            "id": img_id_counter,
            "file_name": new_fname,
            "width": img_info["width"],
            "height": img_info["height"]
        })

        for ann, target_id in mapped_anns:
            bbox = ann["bbox"]
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                merged["annotations"].append({
                    "id": ann_id_counter,
                    "image_id": img_id_counter,
                    "category_id": target_id,
                    "bbox": bbox,
                    "area": ann.get("area", bbox[2] * bbox[3]),
                    "iscrowd": 0
                })
                ann_id_counter += 1

        img_id_counter += 1
        kept += 1

    return img_id_counter, ann_id_counter, kept, skipped


def process_acid_split(acid_split_dir, merged, img_id_counter, ann_id_counter, out_dir):
    acid_ann_file = acid_split_dir / "_annotations.coco.json"
    if not acid_ann_file.exists():
        return img_id_counter, ann_id_counter, 0, 0, 0

    with open(acid_ann_file) as f:
        acid_coco = json.load(f)

    cat_map = {c['id']: c['name'] for c in acid_coco['categories']}
    acid_img_map = {}
    kept = 0
    skipped = 0

    for img_info in acid_coco['images']:
        src_path = acid_split_dir / img_info['file_name']
        if not src_path.exists():
            skipped += 1
            continue
        new_fname = f"acid_{img_id_counter:06d}{src_path.suffix}"
        shutil.copy2(src_path, out_dir / new_fname)
        acid_img_map[img_info['id']] = img_id_counter
        merged["images"].append({
            "id": img_id_counter,
            "file_name": new_fname,
            "width": img_info["width"],
            "height": img_info["height"]
        })
        img_id_counter += 1
        kept += 1

    ann_kept = 0
    for ann in acid_coco['annotations']:
        if ann['image_id'] not in acid_img_map:
            continue
        src_name = cat_map.get(ann['category_id'], '')
        target_id = CLASS_MAPPING.get(src_name)
        if target_id is None:
            continue
        bbox = ann["bbox"]
        if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            merged["annotations"].append({
                "id": ann_id_counter,
                "image_id": acid_img_map[ann['image_id']],
                "category_id": target_id,
                "bbox": bbox,
                "area": ann.get("area", bbox[2] * bbox[3]),
                "iscrowd": 0
            })
            ann_id_counter += 1
            ann_kept += 1

    return img_id_counter, ann_id_counter, kept, skipped, ann_kept


def merge_datasets(output_dir):
    acid_base = Path("/kaggle/working/acid_raw")

    for split in ['train', 'valid', 'test']:
        out_dir = Path(output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        merged = {
            "images": [],
            "annotations": [],
            "categories": TARGET_CATEGORIES
        }

        img_id_counter = 0
        ann_id_counter = 0

        # MOCS
        mocs_split = MOCS_SPLITS.get(split)
        if mocs_split:
            img_id_counter, ann_id_counter, mocs_kept, mocs_skipped = process_mocs_split(
                mocs_split["json"], mocs_split["img_dir"],
                merged, img_id_counter, ann_id_counter, out_dir
            )
            print(f"\n  [{split}] MOCS: {mocs_kept:,} images kept, {mocs_skipped:,} skipped")
        else:
            print(f"\n  [{split}] MOCS: no split (test is ACID-only)")

        # ACID
        acid_split_dir = acid_base / split
        img_id_counter, ann_id_counter, acid_kept, acid_skipped, acid_anns = process_acid_split(
            acid_split_dir, merged, img_id_counter, ann_id_counter, out_dir
        )
        if acid_kept > 0:
            print(f"  [{split}] ACID: {acid_kept:,} images, {acid_anns:,} annotations kept")

        # Oversample underrepresented classes on train split only
        if split == 'train' and merged["annotations"]:
            class_counts_pre = Counter(a["category_id"] for a in merged["annotations"])
            max_count = max(class_counts_pre.values())
            target_ratio = 0.40

            imgs_by_class = {}
            anns_by_img = {}
            for ann in merged["annotations"]:
                anns_by_img.setdefault(ann["image_id"], []).append(ann)
            for img in merged["images"]:
                img_classes = set(a["category_id"] for a in anns_by_img.get(img["id"], []))
                for cls in img_classes:
                    imgs_by_class.setdefault(cls, []).append(img)

            import random
            random.seed(42)  # Same seed as RF-DETR run for identical oversampling
            oversampled_imgs = 0

            for cls_id, count in class_counts_pre.items():
                target_count = int(max_count * target_ratio)
                if count >= target_count:
                    continue

                deficit = target_count - count
                source_imgs = imgs_by_class.get(cls_id, [])
                if not source_imgs:
                    continue

                def dominance(img):
                    img_anns = anns_by_img.get(img["id"], [])
                    if not img_anns:
                        return 0
                    return sum(1 for a in img_anns if a["category_id"] == cls_id) / len(img_anns)

                source_imgs_sorted = sorted(source_imgs, key=dominance, reverse=True)
                top_pool = source_imgs_sorted[:max(1, len(source_imgs_sorted) // 2)]

                added = 0
                while added < deficit:
                    src_img = random.choice(top_pool)
                    new_img_id = img_id_counter
                    img_id_counter += 1

                    merged["images"].append({
                        "id": new_img_id,
                        "file_name": src_img["file_name"],  # Same file, new ID
                        "width": src_img["width"],
                        "height": src_img["height"]
                    })

                    for ann in anns_by_img.get(src_img["id"], []):
                        merged["annotations"].append({
                            "id": ann_id_counter,
                            "image_id": new_img_id,
                            "category_id": ann["category_id"],
                            "bbox": ann["bbox"],
                            "area": ann["area"],
                            "iscrowd": 0
                        })
                        ann_id_counter += 1
                        if ann["category_id"] == cls_id:
                            added += 1

                    oversampled_imgs += 1

            if oversampled_imgs > 0:
                cat_names_tmp = {c["id"]: c["name"] for c in TARGET_CATEGORIES}
                print(f"  [{split}] OVERSAMPLED: +{oversampled_imgs:,} duplicate images")
                new_counts = Counter(a["category_id"] for a in merged["annotations"])
                for cid in sorted(new_counts.keys()):
                    old = class_counts_pre.get(cid, 0)
                    new = new_counts[cid]
                    if new > old:
                        print(f"    {cat_names_tmp[cid]:20s} {old:>6,} -> {new:>6,} (+{new-old:,})")

        # Write merged COCO JSON (source of truth — YOLO TXT generated in CELL 6)
        with open(out_dir / "_annotations.coco.json", "w") as f:
            json.dump(merged, f)

        # Print balance report
        class_counts = Counter(a["category_id"] for a in merged["annotations"])
        cat_names = {c["id"]: c["name"] for c in TARGET_CATEGORIES}
        total_imgs = len(merged["images"])
        total_anns = len(merged["annotations"])
        max_count = max(class_counts.values()) if class_counts else 1

        print(f"  [{split}] MERGED: {total_imgs:,} images, {total_anns:,} annotations")
        print(f"  {'Class':20s} {'Count':>8s}  {'%':>6s}  {'Ratio':>6s}  Balance")
        print(f"  {'-'*70}")
        for cid in sorted(class_counts.keys()):
            count = class_counts[cid]
            pct = 100 * count / total_anns if total_anns > 0 else 0
            ratio = count / max_count
            bar = '#' * int(30 * ratio)
            flag = " << UNDERREPRESENTED" if ratio < 0.10 else ""
            print(f"  {cat_names[cid]:20s} {count:>8,}  {pct:5.1f}%  {ratio:5.2f}x  {bar}{flag}")


OUTPUT_DIR = "/kaggle/working/construction_dataset_v3"
print("=" * 60)
print("  Merging MOCS + ACID -> 8-Class Dataset")
print("=" * 60)
merge_datasets(OUTPUT_DIR)
print(f"\nMerged dataset saved to: {OUTPUT_DIR}")


# ============================================================
# CELL 5 — Verify Dataset Structure
# ============================================================
dataset_dir = "/kaggle/working/construction_dataset_v3"

print("=" * 60)
print("  Dataset Verification")
print("=" * 60)

for split in ['train', 'valid', 'test']:
    split_dir = Path(dataset_dir) / split
    ann_file = split_dir / "_annotations.coco.json"

    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
        n_images = len(coco['images'])
        n_anns = len(coco['annotations'])
        sample_imgs = coco['images'][:5]
        all_exist = all((split_dir / img['file_name']).exists() for img in sample_imgs)
        status = "OK" if all_exist else "FILES MISSING"
        print(f"  [{split:5s}] {n_images:>6,} images, {n_anns:>7,} annotations — {status}")


# ============================================================
# CELL 6 — Convert COCO -> YOLO format + write dataset.yaml
# ============================================================
# YOLO expects:
#   dataset_root/
#     images/train/*.jpg
#     images/val/*.jpg
#     images/test/*.jpg
#     labels/train/*.txt   (one .txt per image: "class cx cy w h" normalized)
#     labels/val/*.txt
#     labels/test/*.txt
#   dataset.yaml

import yaml
from pathlib import Path
import shutil

YOLO_ROOT = Path("/kaggle/working/construction_yolo")
COCO_ROOT = Path(dataset_dir)

# Clean / create YOLO directory structure
if YOLO_ROOT.exists():
    shutil.rmtree(YOLO_ROOT)

for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    (YOLO_ROOT / sub).mkdir(parents=True, exist_ok=True)

# COCO split name -> YOLO split name
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

def coco_to_yolo(coco_split_dir, yolo_img_dir, yolo_lbl_dir):
    """
    Convert a COCO split directory (images + _annotations.coco.json)
    into YOLO format. Moves images (not copies — saves ~5GB of disk).
    """
    ann_file = coco_split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return 0, 0

    with open(ann_file) as f:
        coco = json.load(f)

    # Build image lookup: image_id -> (file_name, width, height)
    img_lookup = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in coco["images"]
    }

    # Group annotations per image
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # NOTE: oversampling in CELL 4 creates multiple image_ids pointing at the
    # same underlying file. YOLO looks up labels by filename (image stem), so we
    # need UNIQUE filenames per duplicate. We rename them here with the image_id.
    n_images = 0
    n_labels = 0
    for img_id, (fname, w, h) in img_lookup.items():
        src_img = coco_split_dir / fname
        if not src_img.exists():
            continue

        # Unique YOLO filename using the COCO image_id
        suffix = Path(fname).suffix
        yolo_stem = f"{Path(fname).stem}__id{img_id}"
        dst_img = yolo_img_dir / f"{yolo_stem}{suffix}"

        # Move (not copy) to save disk — but only the first time we see this
        # source file. For oversample duplicates we hardlink instead, which
        # is near-free on Linux and uses one inode per dup.
        if not dst_img.exists():
            try:
                os.link(src_img, dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

        n_images += 1

        # Write YOLO label file
        lbl_path = yolo_lbl_dir / f"{yolo_stem}.txt"
        anns = anns_by_img.get(img_id, [])
        with open(lbl_path, "w") as lf:
            for ann in anns:
                cls = ann["category_id"]
                x, y, bw, bh = ann["bbox"]
                # COCO is top-left xywh, absolute. YOLO is center xywh, normalized.
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0, 1] in case of slightly out-of-bounds boxes
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                nw = min(max(nw, 0.0), 1.0)
                nh = min(max(nh, 0.0), 1.0)
                if nw > 0 and nh > 0:
                    lf.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    n_labels += 1

    return n_images, n_labels


print("=" * 60)
print("  COCO -> YOLO format conversion")
print("=" * 60)

for coco_split, yolo_split in SPLIT_MAP.items():
    coco_split_dir = COCO_ROOT / coco_split
    if not coco_split_dir.exists():
        print(f"  [{coco_split}] skipped (no directory)")
        continue
    img_dir = YOLO_ROOT / "images" / yolo_split
    lbl_dir = YOLO_ROOT / "labels" / yolo_split
    n_imgs, n_lbls = coco_to_yolo(coco_split_dir, img_dir, lbl_dir)
    print(f"  [{coco_split:5s} -> {yolo_split:5s}] {n_imgs:,} images, {n_lbls:,} labels")

# Write dataset.yaml
yaml_path = YOLO_ROOT / "dataset.yaml"
yaml_content = {
    "path": str(YOLO_ROOT),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 8,
    "names": [c["name"] for c in TARGET_CATEGORIES],
}
with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"\n  dataset.yaml written: {yaml_path}")
with open(yaml_path) as f:
    print(f.read())

# Spot-check: print 3 random YOLO label files to confirm format
import random
random.seed(0)
sample_lbls = list((YOLO_ROOT / "labels/train").glob("*.txt"))[:3]
print(f"\n  Sample label files:")
for lbl in sample_lbls:
    print(f"  --- {lbl.name} ---")
    with open(lbl) as f:
        for line in f.read().strip().split("\n")[:5]:
            print(f"    {line}")


# ============================================================
# CELL 7 — Speed Optimizations
# ============================================================
import torch
import os

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Speed optimizations applied:")
print(f"  cuDNN benchmark:  {torch.backends.cudnn.benchmark}")
print(f"  TF32 matmul:      {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN:       {torch.backends.cudnn.allow_tf32}")
print(f"  CUDA alloc conf:  expandable_segments")


# ============================================================
# CELL 8 — Train YOLO26-L (8 Classes)
# ============================================================
from ultralytics import YOLO

# === Model choice ===
# YOLO26-L: 24.8M params, 55.0 COCO mAP — recommended for our use case
# Swap to "yolo26x.pt" for max accuracy (55.7M params, 57.5 mAP, ~2x slower)
MODEL_NAME = "yolo26l.pt"

# === Hyperparameters ===
# Matched to RF-DETR run where possible so the two checkpoints are comparable.
EPOCHS        = 40        # Same budget as RF-DETR v3
IMGSZ         = 800       # YOLO requires multiple of 32 (784 rounded up)
BATCH         = 32        # H100 80GB at 800px for YOLO26-L
WORKERS       = 16
PATIENCE      = 15        # Early stopping patience (matches RF-DETR)
WARMUP_EPOCHS = 2.0
CLOSE_MOSAIC  = 10        # Disable mosaic aug in last 10 epochs (standard practice)

print("=" * 60)
print("  YOLO26-L — 8-Class Construction Equipment")
print("=" * 60)
print(f"\n  Model:         {MODEL_NAME}")
print(f"  Dataset:       {yaml_path}")
print(f"  Epochs:        {EPOCHS}")
print(f"  Batch size:    {BATCH}")
print(f"  Image size:    {IMGSZ}px")
print(f"  Workers:       {WORKERS}")
print(f"  Patience:      {PATIENCE} epochs")
print(f"  Classes:       excavator, dump_truck, bulldozer, wheel_loader,")
print(f"                 mobile_crane, tower_crane, roller_compactor, cement_mixer\n")

model = YOLO(MODEL_NAME)

results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    device=0,
    # Let Ultralytics pick the optimizer (auto-tuned for dataset size)
    optimizer="auto",
    cos_lr=True,
    warmup_epochs=WARMUP_EPOCHS,
    close_mosaic=CLOSE_MOSAIC,
    patience=PATIENCE,
    # Aerial-friendly geometric augmentation
    degrees=15.0,       # ±15° rotation (drone arbitrary orientations)
    translate=0.1,
    scale=0.5,          # Equipment appears at highly variable scales
    shear=2.0,
    perspective=0.0005,
    fliplr=0.5,         # Horizontal flip OK
    flipud=0.0,         # NO vertical flip — orientation matters for equipment
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.1,
    # IO
    cache=False,        # Dataset is too big to cache in RAM
    amp=True,           # Mixed precision — free speedup on H100
    project="runs",
    name="yolo26l_v1",
    exist_ok=True,
    save_period=5,      # Checkpoint every 5 epochs
    plots=True,
    val=True,
)

print("\nTraining complete!")
print(f"   Best weights: runs/yolo26l_v1/weights/best.pt")


# ============================================================
# CELL 9 — Validate + Report Per-Class mAP
# ============================================================
# Load best checkpoint and run validation on the val split.
# Confusion matrix and PR curves are saved automatically by Ultralytics
# into runs/yolo26l_v1/val/ during training — this cell also reports
# per-class AP numbers so they can be pasted into the comparison table.

best_weights = "runs/yolo26l_v1/weights/best.pt"
print("=" * 60)
print(f"  Validating best checkpoint: {best_weights}")
print("=" * 60)

best_model = YOLO(best_weights)
metrics = best_model.val(
    data=str(yaml_path),
    imgsz=IMGSZ,
    batch=BATCH,
    split="val",
    device=0,
    plots=True,
    save_json=True,
)

# Overall metrics
print(f"\n  Overall mAP50:      {metrics.box.map50:.4f}")
print(f"  Overall mAP50-95:   {metrics.box.map:.4f}")
print(f"  Precision:          {metrics.box.mp:.4f}")
print(f"  Recall:             {metrics.box.mr:.4f}")

# Per-class AP
class_names = [c["name"] for c in TARGET_CATEGORIES]
print(f"\n  Per-class AP:")
print(f"  {'Class':20s} {'mAP50':>8s} {'mAP50-95':>10s}")
print(f"  {'-'*42}")
for i, name in enumerate(class_names):
    if i < len(metrics.box.maps):
        print(f"  {name:20s} {metrics.box.ap50[i]:>8.4f} {metrics.box.maps[i]:>10.4f}")


# ============================================================
# CELL 10 — Quick Inference Smoke Test
# ============================================================
# Run prediction on 3 random val images and save annotated PNGs to the
# Kaggle output so the user can eyeball the result without downloading the model.

import random
val_imgs = list((YOLO_ROOT / "images/val").glob("*"))
if val_imgs:
    random.seed(123)
    samples = random.sample(val_imgs, min(3, len(val_imgs)))
    print(f"\n  Running inference on {len(samples)} sample images...")
    results = best_model.predict(
        source=[str(p) for p in samples],
        imgsz=IMGSZ,
        conf=0.35,
        device=0,
        save=True,
        project="runs",
        name="yolo26l_v1_preview",
        exist_ok=True,
    )
    print(f"  Annotated previews saved to: runs/yolo26l_v1_preview/")


# ============================================================
# CELL 11 — Save Model as Kaggle Output
# ============================================================
import shutil

src = "runs/yolo26l_v1/weights/best.pt"
dst = "/kaggle/working/yolo26l_construction_v1.pt"

if os.path.exists(src):
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"Model saved: {dst} ({size_mb:.1f} MB)")

    # Also copy the confusion matrix and results plot for easy download
    for plot_name in ["confusion_matrix.png", "results.png", "PR_curve.png"]:
        plot_src = f"runs/yolo26l_v1/{plot_name}"
        if os.path.exists(plot_src):
            shutil.copy2(plot_src, f"/kaggle/working/yolo26l_{plot_name}")

    print(f"\n  Next steps:")
    print(f"  1. Download yolo26l_construction_v1.pt from Kaggle Output")
    print(f"  2. Compare val mAP50-95 against rfdetr_construction_v3.pth")
    print(f"  3. If YOLO26-L wins, wire it into services/cv-inference/main.py")
    print(f"     (replace RFDETRDetector with ultralytics YOLO inference)")
else:
    print(f"Training checkpoint not found at {src}")
    print("   Available files in runs/yolo26l_v1/weights/:")
    wdir = "runs/yolo26l_v1/weights"
    if os.path.exists(wdir):
        for f in os.listdir(wdir):
            print(f"     {f}")
