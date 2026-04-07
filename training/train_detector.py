"""
RF-DETR v3 — 8-Class Construction Equipment Detector Training
=============================================================
Trains RF-DETR Base on a merged MOCS + ACID dataset for 8-class
construction equipment detection.

Prerequisites:
  - GPU with 40GB+ VRAM
  - Input Dataset: MOCS (xiaopan9802/mocs-dataset)
  - Roboflow API key (set ROBOFLOW_API_KEY env var or modify below)

Datasets:
  - MOCS — 41,668 images, 13 classes
    Structure: instances_train/ + instances_train.json
               instances_val/   + instances_val.json
  - ACID v2 (Roboflow: test-blhxw/acid-dataset, version 2) — 23,801 images, 10 classes

Classes: excavator (0), dump_truck (1), bulldozer (2), wheel_loader (3),
         mobile_crane (4), tower_crane (5), roller_compactor (6), cement_mixer (7)
"""

# ============================================================
# CELL 1 — Install Dependencies + GPU Profile
# ============================================================
!pip install -q roboflow rfdetr faster-coco-eval

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

print(f"\n  Dependencies installed")

# ============================================================
# CELL 2 — Download ACID Dataset from Roboflow
# ============================================================
from roboflow import Roboflow

# Get API key from environment variable or secrets manager
api_key = os.environ.get("ROBOFLOW_API_KEY")
if not api_key:
    try:
        from kaggle_secrets import UserSecretsClient
        api_key = UserSecretsClient().get_secret("ROBOFLOW_API_KEY")
    except ImportError:
        raise RuntimeError("Set ROBOFLOW_API_KEY environment variable")

WORKING_DIR = os.environ.get("WORKING_DIR", "./working")

rf = Roboflow(api_key=api_key)
project = rf.workspace("test-blhxw").project("acid-dataset")
version = project.version(2)  # v2 = 23,801 images (NOT v1 which is only 9,917)
dataset = version.download("coco", location=os.path.join(WORKING_DIR, "acid_raw"))

print("ACID v2 dataset downloaded (23,801 images)")
print(f"   Location: {WORKING_DIR}/acid_raw")

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

# ── MOCS (has pre-split train + val) ──
MOCS_BASE = os.environ.get("MOCS_DATASET_PATH", "./datasets/mocs")

mocs_train_json = os.path.join(MOCS_BASE, "instances_train.json")
mocs_val_json = os.path.join(MOCS_BASE, "instances_val.json")

if os.path.exists(mocs_train_json):
    mocs_train_coco = inspect_coco("MOCS (train)", mocs_train_json)
else:
    print(f"  MOCS train JSON not found at: {mocs_train_json}")
    print(f"  Searching...")
    for root, dirs, files in os.walk(MOCS_BASE):
        for f in files:
            if f.endswith('.json'):
                print(f"    Found: {os.path.join(root, f)}")

if os.path.exists(mocs_val_json):
    mocs_val_coco = inspect_coco("MOCS (val)", mocs_val_json)

# ── ACID ──
acid_base = os.path.join(WORKING_DIR, "acid_raw")
for split in ['train', 'valid', 'test']:
    acid_json = os.path.join(acid_base, split, "_annotations.coco.json")
    if os.path.exists(acid_json):
        inspect_coco(f"ACID ({split})", acid_json)

# ============================================================
# CELL 4 — Merge Datasets into 8-Class Unified Dataset
# ============================================================
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

    # CLASS 2: bulldozer (ONLY dozers — NOT compactors/rollers)
    "Bulldozer": 2, "bulldozer": 2,
    "dozer": 2, "Dozer": 2,

    # CLASS 3: wheel_loader
    "Loader": 3,                     # MOCS class name
    "wheel_loader": 3,              # ACID Roboflow class name
    "Wheel Loader": 3,

    # CLASS 4: mobile_crane
    "mobile_crane": 4,              # ACID Roboflow class name
    "Crane": 4,                     # MOCS class name (verified from dataset)

    # CLASS 5: tower_crane
    "tower_crane": 5,               # ACID Roboflow class name
    "Static crane": 5,              # MOCS class name (verified from dataset)

    # CLASS 6: roller_compactor
    "compactor": 6,                 # ACID Roboflow class name
    "Roller": 6,                    # MOCS class name

    # CLASS 7: cement_mixer
    "cement_truck": 7,              # ACID Roboflow class name
    "Concrete mixer": 7,            # MOCS class name (verified from dataset)

    # DROPPED — not mapped, silently ignored:
    # "Pump truck" — rare, articulated boom adds noise
    # "backhoe_loader" — ambiguous overlap with excavator + wheel_loader
    # "grader" — uncommon, single dataset only (ACID)
    # "Worker", "Hanging head", "Pile driving", "Other vehicle"
}

MOCS_BASE = os.environ.get("MOCS_DATASET_PATH", "./datasets/mocs")

# MOCS sources: use the pre-existing train/val splits
MOCS_SPLITS = {
    "train": {
        "json": os.path.join(MOCS_BASE, "instances_train.json"),
        "img_dir": os.path.join(MOCS_BASE, "instances_train", "instances_train"),
    },
    "valid": {
        "json": os.path.join(MOCS_BASE, "instances_val.json"),
        "img_dir": os.path.join(MOCS_BASE, "instances_val", "instances_val"),
    },
    # No MOCS test split — test will be ACID-only
}

def find_mocs_img_dir(base_dir, json_path):
    """Auto-discover the actual MOCS image directory by checking a sample image."""
    with open(json_path) as f:
        coco = json.load(f)
    if not coco['images']:
        return base_dir

    sample_fname = Path(coco['images'][0]['file_name']).name
    # Try common MOCS directory structures
    candidates = [
        Path(base_dir),
        Path(base_dir) / Path(json_path).stem,  # e.g. instances_train/instances_train
        Path(base_dir).parent,
    ]
    for d in candidates:
        if (d / sample_fname).exists():
            return str(d)

    # Brute force: walk the base to find the image
    for root, dirs, files in os.walk(Path(base_dir).parent):
        if sample_fname in files:
            return root

    print(f"    WARNING: Could not find MOCS images! Sample: {sample_fname}")
    print(f"    Tried: {[str(c) for c in candidates]}")
    return base_dir


def process_mocs_split(mocs_json_path, mocs_img_dir, merged, img_id_counter, ann_id_counter, out_dir):
    """Add MOCS images/annotations for one split into merged dataset."""
    if not os.path.exists(mocs_json_path):
        print(f"    MOCS JSON not found: {mocs_json_path}")
        return img_id_counter, ann_id_counter, 0, 0

    # Auto-discover actual image directory
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
    skipped_no_ann = 0
    skipped_no_file = 0

    for img_id, img_info in img_lookup.items():
        anns = anns_by_img.get(img_id, [])

        # Only keep images that have at least one mapped annotation
        mapped_anns = []
        for ann in anns:
            src_name = cat_map.get(ann['category_id'], '')
            target_id = CLASS_MAPPING.get(src_name)
            if target_id is not None:
                mapped_anns.append((ann, target_id))

        if not mapped_anns:
            skipped += 1
            skipped_no_ann += 1
            continue

        # Find image file
        fname = Path(img_info['file_name']).name
        src_path = Path(mocs_img_dir) / fname
        if not src_path.exists():
            src_path = Path(mocs_img_dir) / img_info['file_name']
        if not src_path.exists():
            skipped += 1
            skipped_no_file += 1
            continue

        # Copy image
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

    if skipped_no_file > 0:
        print(f"    Skipped: {skipped_no_ann:,} no mapped annotations, {skipped_no_file:,} file not found")

    return img_id_counter, ann_id_counter, kept, skipped


def process_acid_split(acid_split_dir, merged, img_id_counter, ann_id_counter, out_dir):
    """Add ACID images/annotations for one split into merged dataset."""
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
    """Merge MOCS + ACID into a single 8-class COCO dataset."""
    acid_base = Path(WORKING_DIR) / "acid_raw"

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

        # ── Add MOCS (use pre-existing train/val splits) ──
        mocs_split = MOCS_SPLITS.get(split)
        if mocs_split:
            img_id_counter, ann_id_counter, mocs_kept, mocs_skipped = process_mocs_split(
                mocs_split["json"], mocs_split["img_dir"],
                merged, img_id_counter, ann_id_counter, out_dir
            )
            print(f"\n  [{split}] MOCS: {mocs_kept:,} images kept, {mocs_skipped:,} skipped")
        else:
            print(f"\n  [{split}] MOCS: no split (test is ACID-only)")

        # ── Add ACID ──
        acid_split_dir = acid_base / split
        img_id_counter, ann_id_counter, acid_kept, acid_skipped, acid_anns = process_acid_split(
            acid_split_dir, merged, img_id_counter, ann_id_counter, out_dir
        )
        if acid_kept > 0:
            print(f"  [{split}] ACID: {acid_kept:,} images, {acid_anns:,} annotations kept")
        else:
            print(f"  [{split}] ACID: no {split} split found, skipping")

        # ── Oversample underrepresented classes (train only) ──
        if split == 'train' and merged["annotations"]:
            class_counts_pre = Counter(a["category_id"] for a in merged["annotations"])
            max_count = max(class_counts_pre.values())
            target_ratio = 0.40  # Bring every class to at least 40% of max

            # Build index: class_id -> list of (image_id, [annotations])
            imgs_by_class = {}
            anns_by_img = {}
            for ann in merged["annotations"]:
                anns_by_img.setdefault(ann["image_id"], []).append(ann)
            for img in merged["images"]:
                img_classes = set(a["category_id"] for a in anns_by_img.get(img["id"], []))
                for cls in img_classes:
                    imgs_by_class.setdefault(cls, []).append(img)

            import random
            random.seed(42)
            oversampled_imgs = 0

            for cls_id, count in class_counts_pre.items():
                target_count = int(max_count * target_ratio)
                if count >= target_count:
                    continue

                # How many more annotations do we need?
                deficit = target_count - count
                source_imgs = imgs_by_class.get(cls_id, [])
                if not source_imgs:
                    continue

                # Sort source images by dominance: prefer images where this
                # class makes up the majority of annotations. This avoids
                # inflating already-dominant classes (e.g. excavator) when
                # oversampling rare ones (e.g. cement_mixer).
                def dominance(img):
                    img_anns = anns_by_img.get(img["id"], [])
                    if not img_anns:
                        return 0
                    return sum(1 for a in img_anns if a["category_id"] == cls_id) / len(img_anns)

                source_imgs_sorted = sorted(source_imgs, key=dominance, reverse=True)
                # Prefer top-dominant images (top 50%) for sampling
                top_pool = source_imgs_sorted[:max(1, len(source_imgs_sorted) // 2)]

                # Duplicate images containing this class
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
                print(f"  [{split}] OVERSAMPLED: +{oversampled_imgs:,} duplicate images to balance classes")
                new_counts = Counter(a["category_id"] for a in merged["annotations"])
                for cid in sorted(new_counts.keys()):
                    old = class_counts_pre.get(cid, 0)
                    new = new_counts[cid]
                    if new > old:
                        print(f"    {cat_names_tmp[cid]:20s} {old:>6,} -> {new:>6,} (+{new-old:,})")

        # Write merged JSON
        with open(out_dir / "_annotations.coco.json", "w") as f:
            json.dump(merged, f)

        # Print class distribution with balance indicators
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
            # Flag imbalance: if <10% of max class
            flag = " << UNDERREPRESENTED" if ratio < 0.10 else ""
            print(f"  {cat_names[cid]:20s} {count:>8,}  {pct:5.1f}%  {ratio:5.2f}x  {bar}{flag}")

        # Check for missing classes
        present_ids = set(class_counts.keys())
        expected_ids = {c["id"] for c in TARGET_CATEGORIES}
        missing = expected_ids - present_ids
        if missing:
            for mid in missing:
                print(f"  {cat_names[mid]:20s} {'0':>8s}  {'0.0%':>6s}  {'0.00x':>6s}  MISSING!")


# Run merge
OUTPUT_DIR = os.path.join(WORKING_DIR, "construction_dataset_v3")
print("=" * 60)
print("  Merging MOCS + ACID -> 8-Class Dataset")
print("=" * 60)
merge_datasets(OUTPUT_DIR)
print(f"\nMerged dataset saved to: {OUTPUT_DIR}")

# ============================================================
# CELL 5 — Verify Dataset Structure
# ============================================================
import os
from pathlib import Path

dataset_dir = os.path.join(WORKING_DIR, "construction_dataset_v3")

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

        # Check a few images actually exist
        sample_imgs = coco['images'][:5]
        all_exist = all((split_dir / img['file_name']).exists() for img in sample_imgs)

        status = "OK" if all_exist else "FILES MISSING"
        print(f"  [{split:5s}] {n_images:>6,} images, {n_anns:>7,} annotations — {status}")
    else:
        print(f"  [{split:5s}] No annotations file found")

# Verify categories
with open(Path(dataset_dir) / "train" / "_annotations.coco.json") as f:
    coco = json.load(f)
print(f"\n  Categories ({len(coco['categories'])}): {[c['name'] for c in coco['categories']]}")
print(f"  Category IDs: {[c['id'] for c in coco['categories']]}")

# ============================================================
# CELL 6 — Speed Optimizations
# ============================================================
import torch
import os

# cuDNN auto-tuner: finds the fastest convolution algorithms for your input sizes.
# Safe since our input size is fixed (784px).
torch.backends.cudnn.benchmark = True

# TF32 on Ampere+ GPUs: 3x faster matmuls with negligible precision loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Increase dataloader speed — pin memory for faster CPU→GPU transfer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Speed optimizations applied:")
print(f"  cuDNN benchmark:  {torch.backends.cudnn.benchmark}")
print(f"  TF32 matmul:      {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN:       {torch.backends.cudnn.allow_tf32}")
print(f"  CUDA alloc conf:  expandable_segments")

# ============================================================
# CELL 7 — Train RF-DETR (8 Classes)
# ============================================================
# Run Cell 6 (speed optimizations) before this cell
from rfdetr import RFDETRBase

dataset_dir = os.path.join(WORKING_DIR, "construction_dataset_v3")

# Training config — adjust batch_size based on your GPU VRAM
# NOTE: resolution must be passed to RFDETRBase() constructor, NOT to .train()
#       The .train(resolution=...) param is IGNORED. Resolution must be set at
#       model instantiation via RFDETRBase(resolution=X).
#       Resolution must be divisible by 56 (patch_size=14, num_windows=4).
epochs = 40                # ~15min/epoch on a high-end GPU
batch_size = 40           # Adjust based on VRAM (40 fits ~40GB; reduce for smaller GPUs)
resolution = 784          # Max res for aerial/drone small-object detection (divisible by 56)
lr = 1e-4 * (batch_size / 16)  # Linear LR scaling
lr_encoder = 1.5e-4 * (batch_size / 16)  # Backbone LR (default ratio from RF-DETR)
grad_accum_steps = 1
num_workers = 16          # Reduced from 24 — persistent_workers + prefetch needs headroom

# Scheduler & regularization
lr_scheduler = "cosine"   # Cosine annealing — smoother decay than step (default)
warmup_epochs = 2.0       # 2-epoch warmup avoids early instability with fresh head
drop_path = 0.1           # Stochastic depth — regularizes ViT backbone, reduces overfitting
lr_vit_layer_decay = 0.8  # Lower LR for earlier ViT layers (default, keep it)
weight_decay = 1e-4       # Default, prevents weight drift

# Early stopping — save GPU time if plateaued
early_stopping = True
early_stopping_patience = 15  # Stop if no improvement for 15 epochs
early_stopping_min_delta = 0.001

# Checkpointing
checkpoint_interval = 5   # Save every 5 epochs (default=10)

# Data loading
pin_memory = True         # Faster CPU→GPU transfer on single-GPU
persistent_workers = True # Keep workers alive between epochs (no respawn overhead)
prefetch_factor = 2       # Default — 4 was OOMing system RAM with 24 workers

effective_batch = batch_size * grad_accum_steps

print("=" * 60)
print("  RF-DETR v3 — 8-Class Construction Equipment")
print("=" * 60)
print(f"\n  Dataset:       {dataset_dir}")
print(f"  Epochs:        {epochs}")
print(f"  Batch size:    {batch_size} x {grad_accum_steps} = {effective_batch} effective")
print(f"  Resolution:    {resolution}px")
print(f"  LR:            {lr} (decoder) / {lr_encoder} (encoder)")
print(f"  Scheduler:     {lr_scheduler} + {warmup_epochs} epoch warmup")
print(f"  Drop path:     {drop_path}")
print(f"  Early stop:    patience={early_stopping_patience}")
print(f"  Num workers:   {num_workers}")
print(f"  Classes:       excavator, dump_truck, bulldozer, wheel_loader,")
print(f"                 mobile_crane, tower_crane, roller_compactor, cement_mixer\n")

# resolution goes in constructor, NOT in .train()
model = RFDETRBase(num_classes=8, resolution=resolution)

model.train(
    dataset_dir=dataset_dir,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    lr_encoder=lr_encoder,
    grad_accum_steps=grad_accum_steps,
    num_workers=num_workers,
    lr_scheduler=lr_scheduler,
    warmup_epochs=warmup_epochs,
    drop_path=drop_path,
    lr_vit_layer_decay=lr_vit_layer_decay,
    weight_decay=weight_decay,
    early_stopping=early_stopping,
    early_stopping_patience=early_stopping_patience,
    early_stopping_min_delta=early_stopping_min_delta,
    checkpoint_interval=checkpoint_interval,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)

print("\nTraining complete!")
print("   Best weights: output/checkpoint_best_regular.pth")

# ============================================================
# CELL 8 — Export Trained Model
# ============================================================
import shutil

# Copy best checkpoint to output
src = "output/checkpoint_best_regular.pth"
dst = os.path.join(WORKING_DIR, "rfdetr_construction_v3.pth")

if os.path.exists(src):
    shutil.copy2(src, dst)
    size_mb = os.path.getsize(dst) / (1024 * 1024)
    print(f"Model saved: {dst} ({size_mb:.1f} MB)")
    print(f"\n   Next steps:")
    print(f"   1. Copy rfdetr_construction_v3.pth to models/")
    print(f"   2. The pipeline will automatically use it on next run")
else:
    print("Training checkpoint not found!")
    print("   Available files in output/:")
    if os.path.exists("output"):
        for f in os.listdir("output"):
            print(f"     {f}")
