"""
DINOv3 Re-ID Projection Head — Contrastive Fine-Tuning (MOCS + ACID)
=====================================================================
Trains a projection head on top of frozen DINOv3 ViT-B/16 for
construction equipment re-identification.

Prerequisites:
  - GPU with 16GB+ VRAM
  - Input Dataset: MOCS (xiaopan9802/mocs-dataset)
  - Roboflow API key (set ROBOFLOW_API_KEY env var)
  - DINOv3 model: facebook/dinov3-vitb16-pretrain-lvd1689m

WHY THIS WORKS:
  DINOv3 out-of-the-box does CATEGORY recognition (all wheel loaders look similar).
  We need INSTANCE discrimination (this specific CAT 992C vs that Komatsu WA380).

  We use a self-supervised contrastive approach (SimCLR-style):
  For each equipment crop, we generate 2 HEAVILY augmented views with strong
  geometric transforms (perspective warp, rotation, aggressive scale) that
  simulate the viewpoint changes seen from aerial/drone footage.

  The NT-Xent loss pulls the 2 views of the same crop together while pushing
  apart views of different equipment. The projection head learns to produce
  embeddings that are INVARIANT to viewpoint changes.

Datasets:
  - MOCS — 41,668 images, 13 classes
  - ACID v2 (Roboflow: test-blhxw/acid-dataset, version 2) — 23,801 images, 10 classes

Classes: excavator (0), dump_truck (1), bulldozer (2), wheel_loader (3),
         mobile_crane (4), tower_crane (5), roller_compactor (6), cement_mixer (7)
"""

# ============================================================
# CELL 1 — Install Dependencies + GPU Profile
# ============================================================
!pip install -q pytorch-metric-learning roboflow

import os
import torch

print("=" * 60)
print("  GPU PROFILE")
print("=" * 60)

gpu_name = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()
print(f"  GPU:           {gpu_name}")

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  VRAM Total:    {total_mem:.1f} GB")
    print(f"  CUDA Version:  {torch.version.cuda}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  BF16 Support:  {torch.cuda.is_bf16_supported()}")

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
version = project.version(2)
dataset = version.download("coco", location=os.path.join(WORKING_DIR, "acid_raw"))

print("ACID v2 dataset downloaded (23,801 images)")

# ============================================================
# CELL 3 — Collect Equipment Crops from Both Datasets
# ============================================================
import json
from collections import Counter
from pathlib import Path
from PIL import Image

# === 8 Target Classes (same mapping as RF-DETR detector) ===
CLASS_MAPPING = {
    # 0: excavator
    "Excavator": 0, "excavator": 0,
    # 1: dump_truck
    "Truck": 1, "truck": 1, "dump_truck": 1,
    "Dump Truck": 1, "dump truck": 1, "Dump_Truck": 1,
    # 2: bulldozer
    "Bulldozer": 2, "bulldozer": 2, "dozer": 2, "Dozer": 2,
    # 3: wheel_loader
    "Loader": 3, "wheel_loader": 3, "Wheel Loader": 3,
    # 4: mobile_crane
    "mobile_crane": 4, "Crane": 4,
    # 5: tower_crane
    "tower_crane": 5, "Static crane": 5,
    # 6: roller_compactor
    "compactor": 6, "Roller": 6,
    # 7: cement_mixer
    "cement_truck": 7, "Concrete mixer": 7,
}

CLASS_NAMES = {
    0: "excavator", 1: "dump_truck", 2: "bulldozer", 3: "wheel_loader",
    4: "mobile_crane", 5: "tower_crane", 6: "roller_compactor", 7: "cement_mixer",
}

MIN_CROP_SIZE = 32  # Skip tiny annotations

def collect_crops_from_coco(name, json_path, img_dir):
    """Extract crop metadata from a COCO annotation file."""
    with open(json_path) as f:
        coco = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    img_map = {img['id']: img['file_name'] for img in coco['images']}

    crops = []
    skipped = Counter()
    for ann in coco['annotations']:
        cat_name = cat_id_to_name.get(ann['category_id'], '')
        if cat_name not in CLASS_MAPPING:
            skipped[cat_name] += 1
            continue

        x, y, w, h = ann['bbox']
        if w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
            skipped['too_small'] += 1
            continue

        fname = img_map.get(ann['image_id'], '')
        img_path = os.path.join(img_dir, Path(fname).name)
        if not os.path.exists(img_path):
            skipped['missing_file'] += 1
            continue

        crops.append({
            'img_path': img_path,
            'bbox': [x, y, w, h],
            'class_id': CLASS_MAPPING[cat_name],
        })

    print(f"  {name}: {len(crops)} crops collected")
    if skipped:
        print(f"    Skipped: {dict(skipped)}")
    return crops


# ── MOCS ──
MOCS_BASE = os.environ.get("MOCS_DATASET_PATH", "./datasets/mocs")

def find_mocs_img_dir(json_path):
    """Auto-discover MOCS image directory."""
    with open(json_path) as f:
        coco = json.load(f)
    if not coco['images']:
        return os.path.dirname(json_path)
    sample = Path(coco['images'][0]['file_name']).name
    candidates = [
        os.path.join(MOCS_BASE, "instances_train", "instances_train"),
        os.path.join(MOCS_BASE, "instances_train"),
        os.path.join(MOCS_BASE, "instances_val", "instances_val"),
        os.path.join(MOCS_BASE, "instances_val"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, sample)):
            return d
    # Brute force
    for root, dirs, files in os.walk(MOCS_BASE):
        if sample in files:
            return root
    return os.path.dirname(json_path)

all_crops = []

mocs_train_json = os.path.join(MOCS_BASE, "instances_train.json")
if os.path.exists(mocs_train_json):
    mocs_img_dir = find_mocs_img_dir(mocs_train_json)
    all_crops += collect_crops_from_coco("MOCS train", mocs_train_json, mocs_img_dir)

mocs_val_json = os.path.join(MOCS_BASE, "instances_val.json")
if os.path.exists(mocs_val_json):
    mocs_val_img_dir = find_mocs_img_dir(mocs_val_json)
    all_crops += collect_crops_from_coco("MOCS val", mocs_val_json, mocs_val_img_dir)

# ── ACID ──
acid_base = os.path.join(WORKING_DIR, "acid_raw")
for split in ['train', 'valid', 'test']:
    acid_json = os.path.join(acid_base, split, "_annotations.coco.json")
    acid_img_dir = os.path.join(acid_base, split)
    if os.path.exists(acid_json):
        all_crops += collect_crops_from_coco(f"ACID {split}", acid_json, acid_img_dir)

# Summary
print(f"\n  TOTAL: {len(all_crops)} crops")
dist = Counter(c['class_id'] for c in all_crops)
for cls_id in sorted(dist.keys()):
    print(f"    {CLASS_NAMES[cls_id]:25s}: {dist[cls_id]:6d}")

# ============================================================
# CELL 4 — Dataset with Dual-View Augmentation (SimCLR-style)
# ============================================================
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class DualViewCropDataset(Dataset):
    """
    For each equipment crop, returns TWO independently augmented views.
    Heavy geometric augmentation simulates aerial viewpoint changes.

    The key insight: if the model can match the same crop under
    perspective warp + rotation + scale change + color jitter,
    it can match the same real vehicle from different drone angles.
    """

    def __init__(self, crops):
        self.crops = crops

        # Strong augmentation pipeline simulating aerial viewpoint changes
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.6, 1.4)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),  # Drone can be tilted
            T.RandomApply([T.RandomRotation(degrees=45)], p=0.5),
            T.RandomApply([
                T.RandomPerspective(distortion_scale=0.4, p=1.0)
            ], p=0.6),  # Key: simulates viewing angle change
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15)
            ], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.3, scale=(0.02, 0.25)),  # Simulates partial occlusion
        ])

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop_meta = self.crops[idx]
        image = Image.open(crop_meta['img_path']).convert("RGB")

        x, y, w, h = [int(v) for v in crop_meta['bbox']]
        # Add 15% context padding (matches inference pipeline)
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.width, x + w + pad_x)
        y2 = min(image.height, y + h + pad_y)
        crop = image.crop((x1, y1, x2, y2))

        # Two independent augmented views of the same crop
        view1 = self.transform(crop)
        view2 = self.transform(crop)

        return view1, view2, idx  # idx serves as the identity label

dataset = DualViewCropDataset(all_crops)
print(f"Dataset ready: {len(dataset)} samples")

# ============================================================
# CELL 5 — Re-ID Model + NT-Xent Loss
# ============================================================
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DINOv3ReIDHead(nn.Module):
    """
    Frozen DINOv3 backbone + trainable projection head.

    Architecture:
      DINOv3 ViT-B/16 (frozen) -> CLS+mean(patches) [1536-d]
      -> Linear(1536, 768) -> BN -> GELU -> Dropout
      -> Linear(768, 256) -> BN -> GELU -> Dropout
      -> Linear(256, 128) -> L2-normalize

    Deeper head (3 layers vs 2) gives more capacity to learn
    the non-linear mapping from category features to instance features.
    """

    def __init__(self, base_model, input_dim=1536, hidden_dim=768, final_dim=128):
        super().__init__()
        self.dino = base_model

        for param in self.dino.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 3),  # 768 -> 256
            nn.BatchNorm1d(hidden_dim // 3),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 3, final_dim),    # 256 -> 128
        )

    def forward(self, x):
        with torch.no_grad():
            outputs = self.dino(pixel_values=x)
            hidden = outputs.last_hidden_state
            cls_token = hidden[:, 0, :]
            mean_patches = hidden[:, 5:, :].mean(dim=1)
            features = torch.cat([cls_token, mean_patches], dim=1)

        embeddings = self.projection(features)
        return F.normalize(embeddings, p=2, dim=1)


class NTXentLoss(nn.Module):
    """
    SimCLR contrastive loss. For a batch of N crops producing 2N views:
    - Each crop's 2 views are positive pairs
    - All other views in the batch are negatives

    With batch_size=128, each positive pair has 254 negatives.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        pos_idx = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        loss = F.cross_entropy(sim, pos_idx)
        return loss

print("Model + Loss created")

# ============================================================
# CELL 6 — Training Loop
# ============================================================
from tqdm.auto import tqdm

# Config
DINO_MODEL_NAME = os.environ.get("DINO_MODEL_PATH", "facebook/dinov3-vitb16-pretrain-lvd1689m")
BATCH_SIZE = 128
EPOCHS = 20
EMBEDDING_DIM = 128
LEARNING_RATE = 5e-4
TEMPERATURE = 0.07

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load DINOv3 in bfloat16
base_model = AutoModel.from_pretrained(DINO_MODEL_NAME, torch_dtype=torch.bfloat16)
model = DINOv3ReIDHead(base_model, input_dim=1536, final_dim=EMBEDDING_DIM).to(device)

max_cores = os.cpu_count() or 4
workers = min(24, max_cores)
print(f"  Workers: {workers}, Batch size: {BATCH_SIZE}")

dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=workers, pin_memory=True, drop_last=True,
    persistent_workers=True, prefetch_factor=2,
)

optimizer = torch.optim.AdamW(
    model.projection.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = NTXentLoss(temperature=TEMPERATURE)
scaler = torch.amp.GradScaler('cuda')

print(f"\nStarting contrastive training on MOCS + ACID...")
print(f"  Samples: {len(dataset)}, Batches/epoch: {len(dataloader)}")
print(f"  Negatives per pair: {2 * BATCH_SIZE - 2}")

best_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for view1, view2, labels in pbar:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1.float(), z2.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.projection.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})

    scheduler.step()
    avg_loss = total_loss / num_batches
    print(f"  Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        os.makedirs(os.path.join(WORKING_DIR, "output"), exist_ok=True)
        torch.save(model.projection.state_dict(), os.path.join(WORKING_DIR, "output/dinov3_reid_head_best.pth"))

os.makedirs(os.path.join(WORKING_DIR, "output"), exist_ok=True)
torch.save(model.projection.state_dict(), os.path.join(WORKING_DIR, "output/dinov3_reid_head.pth"))
print(f"\nTraining complete! Best loss: {best_loss:.4f}")

# ============================================================
# CELL 7 — Evaluation: Embedding Quality Check
# ============================================================
import numpy as np

model.eval()

# Collect embeddings from a subset
eval_embeddings = []
eval_classes = []

with torch.no_grad():
    for i, (view1, view2, labels) in enumerate(dataloader):
        if i >= 10:
            break
        view1 = view1.to(device)
        view2 = view2.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z1 = model(view1).float()
            z2 = model(view2).float()

        # Positive pair similarities (same crop, different augmentation)
        pos_sims = F.cosine_similarity(z1, z2, dim=1)
        eval_embeddings.append(z1.cpu().numpy())
        eval_embeddings.append(z2.cpu().numpy())

        if i == 0:
            print(f"\n  Sample positive pair similarities (same crop):")
            print(f"    Mean: {pos_sims.mean().item():.4f}")
            print(f"    Min:  {pos_sims.min().item():.4f}")
            print(f"    Max:  {pos_sims.max().item():.4f}")

embs = np.concatenate(eval_embeddings, axis=0)

# Sample negative pair similarities (different crops)
if len(embs) > 500:
    indices = np.random.choice(len(embs), 500, replace=False)
    embs_sample = embs[indices]
else:
    embs_sample = embs

sim_matrix = embs_sample @ embs_sample.T
np.fill_diagonal(sim_matrix, 0)

print(f"\n  Negative pair similarities (different crops):")
print(f"    Mean: {sim_matrix.mean():.4f}")
print(f"    Max:  {sim_matrix.max():.4f}")
print(f"    Std:  {sim_matrix.std():.4f}")
print(f"\n  Good if: positive sims >> 0.85, negative mean near 0.0")

# ============================================================
# CELL 8 — Download Instructions
# ============================================================
print("=" * 60)
print("  DEPLOY")
print("=" * 60)
print()
print("1. Copy trained weights:")
print("   - dinov3_reid_head.pth  (final epoch)")
print("   - dinov3_reid_head_best.pth  (best loss)")
print()
print("2. Place in your project:")
print("   models/dinov3_reid_head.pth")
print()
print("3. Enable in Docker (docker-compose.yml env var):")
print("   REID_USE_PROJECTION=1")
print()
print("4. Projection head architecture (must match main.py):")
print("   Linear(1536, 768) -> BN -> GELU -> Dropout(0.1)")
print("   Linear(768, 256) -> BN -> GELU -> Dropout(0.1)")
print("   Linear(256, 128)")
