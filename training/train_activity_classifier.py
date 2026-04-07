"""
Activity Classifier — X3D-S Fine-Tuning on Equipment Clips
===========================================================
Trains X3D-S on manually-labeled equipment activity clips for
4-class activity classification (DIGGING, LOADING, DUMPING, WAITING).

Prerequisites:
  - GPU with 16GB+ VRAM
  - Labeled clips extracted via scripts/extract_activity_clips.py

Expected dataset structure:
  activity-clips/
    DIGGING/
      WL-001_f000120.mp4
      EX-002_f001200.mp4
      ...
    LOADING/
      WL-001_f000200.mp4
      ...
    DUMPING/
      WL-001_f000350.mp4
      ...
    WAITING/
      WL-001_f002400.mp4
      ...

Model: X3D-S (Kinetics-400 pretrained) — 3.8M params, real-time capable
  - Input: 13 frames at 160x160 (native) — we override to 16 frames at 224x224
  - Alternative: SlowFast-R50 if accuracy is more important than speed

Output: activity_classifier_x3d_s.pt
"""

# ============================================================
# CELL 1 — Install Dependencies + GPU Profile
# ============================================================
!pip install -q pytorchvideo av

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
else:
    print("  WARNING: No GPU detected!")

print(f"\n  Dependencies installed")


# ============================================================
# CELL 2 — Dataset Inspection + Split
# ============================================================
import json
import random
from pathlib import Path
from collections import Counter

CLIPS_ROOT = Path(os.environ.get("CLIPS_PATH", "./activity-clips"))
WORKING_DIR = os.environ.get("WORKING_DIR", "./working")

# Activity classes — order matters (maps to model output indices)
ACTIVITY_CLASSES = ['DIGGING', 'LOADING', 'DUMPING', 'WAITING']
NUM_CLASSES = len(ACTIVITY_CLASSES)

# Gather all clip paths + labels
all_clips = []
for idx, cls_name in enumerate(ACTIVITY_CLASSES):
    cls_dir = CLIPS_ROOT / cls_name
    if not cls_dir.exists():
        print(f"  WARNING: {cls_dir} not found!")
        continue
    clips = list(cls_dir.glob("*.mp4"))
    print(f"  {cls_name:12s}: {len(clips):5d} clips")
    for c in clips:
        all_clips.append((str(c), idx))

print(f"\n  Total clips: {len(all_clips)}")

# Stratified train/val split (80/20)
random.seed(42)
random.shuffle(all_clips)

# Group by class for stratified split
by_class = {i: [] for i in range(NUM_CLASSES)}
for path, label in all_clips:
    by_class[label].append((path, label))

train_clips = []
val_clips = []

for cls_idx in range(NUM_CLASSES):
    cls_data = by_class[cls_idx]
    split_idx = int(len(cls_data) * 0.8)
    train_clips.extend(cls_data[:split_idx])
    val_clips.extend(cls_data[split_idx:])

random.shuffle(train_clips)
random.shuffle(val_clips)

print(f"\n  Train: {len(train_clips)}, Val: {len(val_clips)}")

# Class balance
train_counts = Counter(label for _, label in train_clips)
val_counts = Counter(label for _, label in val_clips)
print("\n  Train distribution:")
for i, name in enumerate(ACTIVITY_CLASSES):
    print(f"    {name:12s}: {train_counts.get(i, 0):5d}")
print("  Val distribution:")
for i, name in enumerate(ACTIVITY_CLASSES):
    print(f"    {name:12s}: {val_counts.get(i, 0):5d}")


# ============================================================
# CELL 3 — Dataset + DataLoader
# ============================================================
import av
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class ActivityClipDataset(Dataset):
    """
    Loads 16-frame video clips and returns (clip_tensor, label).
    Clip tensor shape: (C, T, H, W) = (3, 16, 224, 224) — PyTorchVideo convention.
    """

    def __init__(self, clip_list, num_frames=16, crop_size=224, is_train=True):
        """
        Args:
            clip_list: List of (video_path, label_idx) tuples
            num_frames: Frames to sample per clip
            crop_size: Spatial resolution
            is_train: If True, apply augmentations
        """
        self.clips = clip_list
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.is_train = is_train

        # Spatial transforms
        if is_train:
            self.spatial_transform = T.Compose([
                T.RandomResizedCrop(crop_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            ])
        else:
            self.spatial_transform = T.Compose([
                T.Resize(crop_size),
                T.CenterCrop(crop_size),
            ])

    def __len__(self):
        return len(self.clips)

    def _decode_video(self, path):
        """Decode video frames using PyAV."""
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        container.close()
        return frames

    def _sample_frames(self, frames):
        """Uniformly sample num_frames from the video."""
        total = len(frames)
        if total >= self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        else:
            # Loop/pad if too short
            indices = list(range(total))
            while len(indices) < self.num_frames:
                indices.append(indices[-1])  # Repeat last frame
            indices = indices[:self.num_frames]
        return [frames[i] for i in indices]

    def __getitem__(self, idx):
        path, label = self.clips[idx]

        # Decode
        frames = self._decode_video(path)
        if len(frames) == 0:
            # Fallback: return zeros
            clip = torch.zeros(3, self.num_frames, self.crop_size, self.crop_size)
            return clip, label

        # Sample 16 frames
        frames = self._sample_frames(frames)

        # Convert to tensor: list of (H, W, 3) uint8 → (T, C, H, W) float
        tensors = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            tensors.append(t)
        clip = torch.stack(tensors, dim=0)  # (T, C, H, W)

        # Apply spatial transforms (per-frame)
        transformed = []
        for t in range(clip.shape[0]):
            frame_pil = T.ToPILImage()(clip[t])
            frame_aug = self.spatial_transform(frame_pil)
            transformed.append(T.ToTensor()(frame_aug))
        clip = torch.stack(transformed, dim=0)  # (T, C, H, W)

        # Normalize with Kinetics mean/std
        mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)
        clip = (clip - mean) / std

        # Reshape to PyTorchVideo format: (C, T, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)

        return clip, label


# Create datasets
train_dataset = ActivityClipDataset(train_clips, is_train=True)
val_dataset = ActivityClipDataset(val_clips, is_train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Quick sanity check
clip, label = train_dataset[0]
print(f"Clip shape: {clip.shape}, Label: {label} ({ACTIVITY_CLASSES[label]})")


# ============================================================
# CELL 4 — Load Pretrained X3D-S + Replace Head
# ============================================================
import torch.nn as nn
from pytorchvideo.models.hub import x3d_s

# Load Kinetics-400 pretrained X3D-S
model = x3d_s(pretrained=True)

# X3D-S head structure:
#   model.blocks[5].proj = nn.Linear(2048, 400)
# We replace the final projection with our 4 classes
in_features = model.blocks[5].proj.in_features
model.blocks[5].proj = nn.Linear(in_features, NUM_CLASSES)

# Freeze backbone for first N epochs (transfer learning)
# Unfreeze blocks[4] and blocks[5] (last two stages)
for name, param in model.named_parameters():
    if 'blocks.4' in name or 'blocks.5' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Model: X3D-S")
print(f"  Total params:     {total:,}")
print(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
print(f"  Output classes:   {NUM_CLASSES}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# ============================================================
# CELL 5 — Training Loop
# ============================================================
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Hyperparameters
EPOCHS_FROZEN = 10     # Train only head first
EPOCHS_FINETUNE = 30   # Then unfreeze everything
TOTAL_EPOCHS = EPOCHS_FROZEN + EPOCHS_FINETUNE
LR_FROZEN = 1e-3       # Higher LR for head-only
LR_FINETUNE = 1e-4     # Lower LR for full fine-tune

# Compute class weights for imbalanced data
train_labels = [label for _, label in train_clips]
class_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = torch.tensor([
    total_samples / (NUM_CLASSES * class_counts.get(i, 1))
    for i in range(NUM_CLASSES)
], dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.cpu().tolist()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Phase 1: Frozen backbone — only train head
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_FROZEN,
    weight_decay=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FROZEN)

best_val_acc = 0.0
best_model_state = None
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clips.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * clips.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, 100.0 * correct / total, all_preds, all_labels


print(f"\n{'='*60}")
print(f"  Phase 1: Frozen backbone ({EPOCHS_FROZEN} epochs, lr={LR_FROZEN})")
print(f"{'='*60}")

for epoch in range(EPOCHS_FROZEN):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  Epoch {epoch+1:2d}/{EPOCHS_FROZEN} — "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%"
          f"{'  ★' if val_acc >= best_val_acc else ''}")


# Phase 2: Unfreeze all layers + fine-tune
print(f"\n{'='*60}")
print(f"  Phase 2: Full fine-tune ({EPOCHS_FINETUNE} epochs, lr={LR_FINETUNE})")
print(f"{'='*60}")

# Unfreeze everything
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINETUNE)

for epoch in range(EPOCHS_FINETUNE):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"  Epoch {epoch+1:2d}/{EPOCHS_FINETUNE} — "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%"
          f"{'  ★' if val_acc >= best_val_acc else ''}")

print(f"\n  Best Val Accuracy: {best_val_acc:.1f}%")


# ============================================================
# CELL 6 — Evaluation + Confusion Matrix
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load best model
model.load_state_dict(best_model_state)
val_loss, val_acc, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

print(f"\n  Best Model — Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%\n")

# Classification report
print(classification_report(
    all_labels, all_preds,
    target_names=ACTIVITY_CLASSES,
    digits=3
))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('Activity Classifier — Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(ACTIVITY_CLASSES, rotation=45, ha='right')
ax.set_yticklabels(ACTIVITY_CLASSES)

# Annotate cells
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')

fig.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'], label='Val')
ax1.axvline(x=EPOCHS_FROZEN, color='gray', linestyle='--', label='Unfreeze')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curves')
ax1.legend()

ax2.plot(history['val_acc'])
ax2.axvline(x=EPOCHS_FROZEN, color='gray', linestyle='--', label='Unfreeze')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Validation Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(WORKING_DIR, 'training_curves.png'), dpi=150)
plt.show()


# ============================================================
# CELL 7 — Export Model for Inference
# ============================================================
# Save in a format that the cv-inference pipeline can load

export_path = os.path.join(WORKING_DIR, 'activity_classifier_x3d_s.pt')

# Save full checkpoint with metadata
checkpoint = {
    'model_state_dict': best_model_state,
    'classes': ACTIVITY_CLASSES,
    'num_classes': NUM_CLASSES,
    'model_arch': 'x3d_s',
    'input_frames': 16,
    'input_size': 224,
    'mean': [0.45, 0.45, 0.45],
    'std': [0.225, 0.225, 0.225],
    'best_val_acc': best_val_acc,
    'total_epochs': TOTAL_EPOCHS,
}

torch.save(checkpoint, export_path)
print(f"\n  Model saved to: {export_path}")
print(f"  File size: {os.path.getsize(export_path) / (1024**2):.1f} MB")

# Also save a lightweight ONNX version for potential TensorRT deployment
try:
    dummy_input = torch.randn(1, 3, 16, 224, 224).to(device)
    onnx_path = os.path.join(WORKING_DIR, 'activity_classifier_x3d_s.onnx')
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['clip'],
        output_names=['activity'],
        dynamic_axes={'clip': {0: 'batch'}, 'activity': {0: 'batch'}},
        opset_version=17,
    )
    print(f"  ONNX exported to: {onnx_path} ({os.path.getsize(onnx_path) / (1024**2):.1f} MB)")
except Exception as e:
    print(f"  ONNX export failed (not critical): {e}")


# ============================================================
# CELL 8 — Quick Inference Test
# ============================================================
# Test the exported model on a few val clips to verify it works

model.load_state_dict(best_model_state)
model.eval()

print("\n  Quick inference test on 5 val clips:")
print(f"  {'Path':60s} {'True':10s} {'Pred':10s} {'Conf':>6s}")
print("  " + "-" * 90)

for i in range(min(5, len(val_clips))):
    clip_tensor, true_label = val_dataset[i]
    clip_tensor = clip_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_label = probs.argmax(1).item()
        confidence = probs[0, pred_label].item()

    path = val_clips[i][0]
    short_path = '/'.join(path.split('/')[-2:])
    match = "OK" if pred_label == true_label else "WRONG"
    print(f"  {short_path:60s} {ACTIVITY_CLASSES[true_label]:10s} "
          f"{ACTIVITY_CLASSES[pred_label]:10s} {confidence:.3f}  {match}")


print(f"\n{'='*60}")
print(f"  DONE — Copy activity_classifier_x3d_s.pt to models/")
print(f"  The cv-inference pipeline will use it automatically")
print(f"{'='*60}")
