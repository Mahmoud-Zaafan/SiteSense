"""
CV Inference Service (Main Entry Point)
=======================================================
Orchestrates the full detection → tracking → motion → activity pipeline.
Produces equipment telemetry events to Kafka.

Pipeline per frame:
  1. Video Ingestion → FramePacket
  2. (Optional) Zero-DCE low-light enhancement
  3. RF-DETR Detection → raw detections [x1,y1,x2,y2,conf,cls]
  4. TAI Heuristic Filter → clean detections (suppress fragments)
  5. BoT-SORT Tracking → tracked objects with persistent IDs
  6. DINOv3 Re-ID Gallery → re-identify lost tracks
  7. Grid Decomposition + Farneback → articulated motion detection
  8. Rule-based Activity Classifier → DIGGING/LOADING/DUMPING/WAITING
  9. Kafka Producer → equipment.telemetry.raw
"""

import os
import sys
import json
import time
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


# ── Model weight resolver ────────────────────────────────────────────────
HF_REPO_ID = "Zaafan/sitesense-weights"

def resolve_weights(filename: str, local_dir: str = "/models") -> str:
    """Return the path to a model weight file.
    
    Priority:
      1. Local file at <local_dir>/<filename>  (fastest — no network)
      2. Auto-download from Hugging Face Hub    (fallback)
    """
    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        logger.info(f"Found local weights: {local_path}")
        return local_path

    logger.warning(f"Weights not found at {local_path} — downloading from HuggingFace ({HF_REPO_ID})...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=local_dir,
        )
        logger.info(f"Downloaded weights to: {downloaded}")
        return downloaded
    except Exception as e:
        logger.error(f"Failed to download {filename} from HuggingFace: {e}")
        raise FileNotFoundError(
            f"Model weights '{filename}' not found locally at {local_path} "
            f"and could not be downloaded from {HF_REPO_ID}. "
            f"Run: huggingface-cli download {HF_REPO_ID} --local-dir {local_dir}"
        )
from collections import deque
from PIL import Image  # Moved to top level (audit #11)

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'video-ingestion'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('cv-inference')

# Class mapping matching our trained model (8-class v3)
CLASS_NAMES = {
    0: 'excavator', 1: 'dump_truck', 2: 'bulldozer', 3: 'wheel_loader',
    4: 'mobile_crane', 5: 'tower_crane', 6: 'roller_compactor', 7: 'cement_mixer',
}
CLASS_PREFIXES = {
    0: 'EX', 1: 'DT', 2: 'BD', 3: 'WL',
    4: 'MC', 5: 'TC', 6: 'RC', 7: 'CM',
}


# =============================================================================
# RF-DETR Detector Wrapper
# =============================================================================
class RFDETRDetector:
    """
    Wraps the fine-tuned RF-DETR model for inference.
    Loads checkpoint_best_regular.pth trained on our 8-class dataset.
    """

    def __init__(self, weights_path: str, num_classes: int = 8,
                 confidence_threshold: float = 0.35, device: str = 'cuda'):
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.weights_path = weights_path

    def load(self):
        """Load the fine-tuned RF-DETR model."""
        from rfdetr import RFDETRBase
        # In rfdetr, the weights path is passed as `pretrain_weights`
        self.model = RFDETRBase(num_classes=8, pretrain_weights=self.weights_path)
        # Optimize model for faster inference (torch.compile / fused ops)
        if hasattr(self.model, 'optimize_for_inference'):
            self.model.optimize_for_inference()
            logger.info("RF-DETR inference optimization applied")
        logger.info(f"RF-DETR loaded from {self.weights_path} on {self.device}")

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            (N, 6) array of [x1, y1, x2, y2, confidence, class_id]
        """
        if self.model is None:
            return np.empty((0, 6))

        detections = self.model.predict(frame, threshold=self.confidence_threshold)

        # RF-DETR returns a Supervision Detections object
        # with .xyxy (N,4), .confidence (N,), .class_id (N,)
        if detections is None or len(detections) == 0:
            return np.empty((0, 6))

        boxes = detections.xyxy                    # (N, 4)
        confs = detections.confidence.reshape(-1, 1)  # (N, 1)
        clses = detections.class_id.reshape(-1, 1)    # (N, 1)

        return np.hstack([boxes, confs, clses]).astype(np.float32)


# =============================================================================
# DINOv3 Re-ID Gallery (Persistent Identity Across Occlusions)
# =============================================================================
class DINOv3ReIDGallery:
    """
    Uses DINOv3 ViT-B/16 to extract visual embeddings from equipment crops.
    Maintains a gallery of lost tracks to re-identify them when they reappear.

    This solves the critical problem: when two identical yellow haul trucks
    drive behind a dirt mound, the tracker loses their IDs. This gallery
    matches them back to their original identity using visual fingerprints.
    """

    def __init__(self, similarity_threshold: float = 0.95,
                 gallery_ttl_frames: int = 1800, device: str = 'cuda',
                 color_threshold: float = 0.40,
                 spatial_gate_ratio: float = 0.40):
        self.color_threshold = color_threshold
        self.spatial_gate_ratio = spatial_gate_ratio
        self.similarity_threshold = similarity_threshold
        self.gallery_ttl_frames = gallery_ttl_frames
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model = None
        self.processor = None
        self.projection = None

        # Gallery: track_id -> {embedding, equipment_id, class_id, last_frame, bbox}
        self._gallery = {}
        # Active tracks: track_id -> embedding (updated every N frames)
        self._active_embeddings = {}

    def load(self):
        """Load DINOv3 ViT-B/16 from HuggingFace."""
        from transformers import AutoImageProcessor, AutoModel

        # Path to the mounted container volume pointing to P:\Projects\Eagle Vision Project\models\dinov3-vitb16-pretrain-lvd1689m
        model_id = "/models/dinov3-vitb16-pretrain-lvd1689m"
        logger.info(f"Loading DINOv3 Re-ID model locally from: {model_id}")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Load contrastive-trained projection head if available
        # Architecture must match training/train_reid.py
        import torch.nn as nn
        self.projection = nn.Sequential(
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        ).to(self.device)

        # Projection head disabled by default until trained with contrastive loss.
        # To enable: set REID_USE_PROJECTION=1
        use_projection = os.getenv('REID_USE_PROJECTION', '0') == '1'
        if use_projection:
            try:
                proj_path = resolve_weights('dinov3_reid_head.pth', local_dir=os.getenv('MODEL_PATH', '/models'))
                self.projection.load_state_dict(torch.load(proj_path, map_location=self.device))
                self.projection.eval()
                logger.info("Loaded contrastive Re-ID Projection Head (128-dim)")
            except FileNotFoundError:
                self.projection = None
                logger.warning("Re-ID projection head not found — falling back to base DINOv3 1536-dim embeddings")
        else:
            self.projection = None
            logger.info("Using base DINOv3 1536-dim embeddings (no projection head)")

        logger.info(f"DINOv3 Re-ID backbone loaded on {self.device}")

    def _crop_with_context(self, frame: np.ndarray, bbox: np.ndarray,
                           context_ratio: float = 0.15) -> np.ndarray:
        """
        Crop equipment with context padding and square normalization.
        Adding 15% context around the bbox captures surrounding details
        (shadow pattern, ground type) that help stabilize embeddings.
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        w, h = x2 - x1, y2 - y1

        # Add context padding
        pad_x = int(w * context_ratio)
        pad_y = int(h * context_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return None

        # Pad to square to avoid aspect-ratio distortion
        h, w = crop.shape[:2]
        if h != w:
            size = max(h, w)
            padded = np.zeros((size, size, 3), dtype=crop.dtype)
            y_off = (size - h) // 2
            x_off = (size - w) // 2
            padded[y_off:y_off + h, x_off:x_off + w] = crop
            crop = padded

        return crop

    @torch.inference_mode()
    def extract_embedding(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract a 1536-d embedding from an equipment crop.
        Uses CLS + mean(patch_tokens) with 15% context padding.

        Args:
            frame: Full BGR frame (H, W, 3)
            bbox: [x1, y1, x2, y2]

        Returns:
            Normalized embedding vector, or None
        """
        crop = self._crop_with_context(frame, bbox)
        if crop is None:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # CLS + mean(patch_tokens) — DINOv3 ViT-B/16: 1 CLS + 4 register + 196 patch
        hidden = outputs.last_hidden_state[0]
        cls_token = hidden[0]
        patch_tokens = hidden[5:]
        mean_patches = patch_tokens.mean(dim=0)
        embedding = torch.cat([cls_token, mean_patches])

        if self.projection is not None:
            import torch.nn.functional as F
            features = embedding.unsqueeze(0).to(self.device)
            projected = self.projection(features)
            projected = F.normalize(projected, p=2, dim=1)
            embedding = projected[0].cpu().numpy()
        else:
            embedding = embedding.cpu().numpy()
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding

    def _extract_color_histogram(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract an HSV color histogram from a crop. This is a hard physical
        discriminator: a blue truck and a white truck will NEVER match,
        regardless of what the neural network thinks.
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 2D histogram on Hue (color) and Saturation (vividness)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def update_active(self, track_id: int, embedding: np.ndarray,
                      equipment_id: str, class_id: int, bbox: np.ndarray,
                      frame_id: int):
        """Store embedding for an active track. Keeps up to 5 diverse views."""
        max_embeddings = 5
        if track_id not in self._active_embeddings:
            self._active_embeddings[track_id] = {
                'embeddings': [embedding],  # list of diverse views
                'equipment_id': equipment_id,
                'class_id': class_id,
                'bbox': bbox.copy(),
                'first_frame': frame_id,
                'last_frame': frame_id,
                'color_hist': None  # populated by process_frame
            }
        else:
            entry = self._active_embeddings[track_id]
            entry['equipment_id'] = equipment_id
            entry['bbox'] = bbox.copy()
            entry['last_frame'] = frame_id
            # Only add if sufficiently different from existing embeddings
            existing = entry['embeddings']
            max_sim = max(float(np.dot(embedding, e)) for e in existing)
            if max_sim < 0.85 and len(existing) < max_embeddings:
                existing.append(embedding)  # New diverse view
            elif len(existing) > 0:
                existing[-1] = embedding  # Refresh the most recent one

    def move_to_gallery(self, track_id: int, frame_id: int):
        """
        Move a lost track to the gallery for future re-identification.
        Called when BoT-SORT drops a track.
        Only moves tracks that have an equipment_id (confirmed tracks).
        Pending tracks (no equipment_id yet) are silently dropped.
        """
        if track_id in self._active_embeddings:
            entry = self._active_embeddings.pop(track_id)
            # Only gallery tracks that have a confirmed equipment ID
            if entry.get('equipment_id') is None:
                return  # pending track — never got an ID, just drop it
            entry['lost_frame'] = frame_id
            self._gallery[track_id] = entry
            logger.info(f"Track {track_id} ({entry['equipment_id']}) moved to Re-ID gallery (gallery size: {len(self._gallery)})")

    def _compute_spatial_similarity(self, query_bbox, gallery_bbox, frame_width, frame_height):
        """
        Compute spatial similarity between query and gallery bounding boxes.
        Uses both center distance and IoU-like size similarity.
        Returns 0.0 (far apart) to 1.0 (same position).
        """
        if query_bbox is None or gallery_bbox is None:
            return 0.0

        # Center distance (normalized by frame diagonal)
        lost_cx = (gallery_bbox[0] + gallery_bbox[2]) / 2
        lost_cy = (gallery_bbox[1] + gallery_bbox[3]) / 2
        new_cx = (query_bbox[0] + query_bbox[2]) / 2
        new_cy = (query_bbox[1] + query_bbox[3]) / 2

        diag = np.sqrt(frame_width**2 + frame_height**2)
        dist = np.sqrt((new_cx - lost_cx)**2 + (new_cy - lost_cy)**2)
        spatial_sim = max(0.0, 1.0 - dist / (diag * 0.5))  # 0 at half-diagonal away

        return spatial_sim

    def _get_alpha(self, frames_lost, fps=9.0):
        """
        Time-decaying alpha for combining visual + spatial similarity.
        For aerial drone footage, equipment barely moves during short track losses,
        so spatial proximity is the dominant signal. Visual appearance changes
        drastically with drone angle/lighting (same vehicle can score 0.40-0.80).
        """
        seconds_lost = frames_lost / fps
        if seconds_lost < 3.0:
            return 0.3   # 30% visual, 70% spatial — equipment hasn't moved
        elif seconds_lost < 10.0:
            return 0.5   # Equal weight — vehicle may have moved somewhat
        else:
            return 0.8   # Mostly visual — vehicle could be anywhere

    def query(self, embedding: np.ndarray, class_id: int,
              frame_id: int, query_bbox: np.ndarray = None,
              frame_width: int = 1920, frame_height: int = 1080,
              query_color_hist: np.ndarray = None) -> tuple:
        """
        Search the gallery for a matching lost track.
        Combines visual embedding similarity with spatial proximity,
        weighted by how long the track has been lost (time-decaying alpha).

        final_score = α * visual_sim + (1-α) * spatial_sim
        α increases with time lost (position becomes less reliable).

        Args:
            embedding: Query embedding from new detection
            class_id: Class of the new detection
            frame_id: Current frame number
            query_bbox: [x1,y1,x2,y2] of the new detection
            frame_width: Width of the video frame in pixels
            frame_height: Height of the video frame in pixels
            query_color_hist: HSV color histogram (unused, kept for API compat)

        Returns:
            (matched_equipment_id, final_score) or (None, 0.0)
        """
        if embedding is None or len(self._gallery) == 0:
            return None, 0.0

        best_match_id = None
        best_score = 0.0
        second_best_score = 0.0
        best_equip_id = None
        best_visual = 0.0
        best_spatial = 0.0

        expired = []
        for gal_track_id, entry in self._gallery.items():
            # Check TTL
            frames_lost = frame_id - entry.get('lost_frame', 0)
            if frames_lost > self.gallery_ttl_frames:
                expired.append(gal_track_id)
                continue

            # Only match same class
            if entry['class_id'] != class_id:
                continue

            # ── Visual Similarity (best of stored embeddings) ──
            visual_sim = max(
                float(np.dot(embedding, e)) for e in entry['embeddings']
            )

            # ── Spatial Similarity ──
            spatial_sim = self._compute_spatial_similarity(
                query_bbox, entry.get('bbox'), frame_width, frame_height
            )

            # ── Combined Score (time-decaying alpha) ──
            # BoT-SORT retains lost tracks for max_time_lost=1200 frames (20s at 60fps).
            # Gallery entries are for tracks BoT-SORT has already dropped.
            alpha = self._get_alpha(frames_lost)
            final_score = alpha * visual_sim + (1 - alpha) * spatial_sim

            # Hard floor: visual similarity must be at least 0.50
            # Prevents position-dominant matches between different same-class vehicles
            if visual_sim < 0.50:
                continue

            if final_score > best_score:
                # Track second-best for margin-based disambiguation
                second_best_score = best_score
                best_score = final_score
                best_match_id = gal_track_id
                best_equip_id = entry['equipment_id']
                best_visual = visual_sim
                best_spatial = spatial_sim
            elif final_score > second_best_score:
                second_best_score = final_score

        # Clean expired entries
        for tid in expired:
            del self._gallery[tid]

        # ── Threshold decision ──
        # BoT-SORT handles identity for up to 20s (max_time_lost=1200 at 60fps).
        # Re-ID only fires for tracks lost longer than that — be strict to avoid
        # false positives (different same-class vehicles getting the same ID).
        if best_match_id is not None:
            if best_score < self.similarity_threshold:
                best_match_id = None  # reject — below threshold

        # Remove matched entry from gallery
        if best_match_id is not None:
            del self._gallery[best_match_id]
            logger.info(
                f"Re-ID match: {best_equip_id} "
                f"(combined={best_score:.3f}, visual={best_visual:.3f}, spatial={best_spatial:.3f})"
            )
        else:
            # Log why we didn't match
            if len(self._gallery) > 0:
                all_sims = []
                for gal_entry in self._gallery.values():
                    if gal_entry['class_id'] == class_id:
                        v_sim = max(float(np.dot(embedding, emb)) for emb in gal_entry['embeddings'])
                        s_sim = self._compute_spatial_similarity(
                            query_bbox, gal_entry.get('bbox'), frame_width, frame_height
                        )
                        fl = frame_id - gal_entry.get('lost_frame', 0)
                        a = self._get_alpha(fl)
                        combined = a * v_sim + (1 - a) * s_sim
                        all_sims.append((gal_entry['equipment_id'], combined, v_sim, s_sim))
                if all_sims:
                    best_below = max(all_sims, key=lambda x: x[1])
                    logger.info(
                        f"Re-ID no match: best={best_below[0]} "
                        f"combined={best_below[1]:.3f} (visual={best_below[2]:.3f}, spatial={best_below[3]:.3f}), "
                        f"threshold={self.similarity_threshold:.2f}"
                    )

        return best_equip_id, best_score

    def absorb_near_misses(self, track_id: int, embedding: np.ndarray,
                           class_id: int, frame_id: int,
                           query_bbox: np.ndarray = None,
                           frame_width: int = 1920, frame_height: int = 1080,
                           id_generator=None):
        """
        When a new ID is minted (Re-ID missed), check if there are gallery entries
        of the same class that scored close to the threshold. If a high-confidence
        match is found (>0.85), retroactively reassign the new track to the old
        equipment ID — fixing phantom IDs (WL-003 → WL-002).
        For lower scores (>0.70), just absorb embeddings to help future matching.
        """
        near_miss_floor = self.similarity_threshold - 0.05
        candidates = []

        for gal_track_id, entry in list(self._gallery.items()):
            if entry['class_id'] != class_id:
                continue

            visual_sim = max(float(np.dot(embedding, e)) for e in entry['embeddings'])
            if visual_sim < 0.35:
                continue

            spatial_sim = self._compute_spatial_similarity(
                query_bbox, entry.get('bbox'), frame_width, frame_height
            )
            frames_lost = frame_id - entry.get('lost_frame', 0)
            alpha = self._get_alpha(frames_lost)
            combined = alpha * visual_sim + (1 - alpha) * spatial_sim

            if combined >= near_miss_floor:
                candidates.append((gal_track_id, entry, combined))

        if not candidates:
            return None

        # Sort by score — best match first
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_gal_id, best_entry, best_score = candidates[0]

        # High-confidence near-miss: retroactively reassign ID
        # This fixes phantom IDs (e.g., WL-003 → WL-002 when score=0.975)
        reassigned_id = None
        if best_score >= 0.85 and id_generator is not None:
            old_equip_id = best_entry['equipment_id']
            current_equip_id = id_generator.get_existing(track_id)
            if current_equip_id and current_equip_id != old_equip_id:
                id_generator.reassign(track_id, old_equip_id)
                # Recycle the phantom ID number so next new vehicle reuses it
                id_generator.recycle(current_equip_id)
                # Update active embeddings with the corrected equipment ID
                if track_id in self._active_embeddings:
                    self._active_embeddings[track_id]['equipment_id'] = old_equip_id
                reassigned_id = old_equip_id
                logger.info(
                    f"Re-ID retroactive fix: {current_equip_id} → {old_equip_id} "
                    f"(score={best_score:.3f}) — phantom ID corrected, "
                    f"{current_equip_id} recycled"
                )

        # Absorb embeddings from all near-misses
        for gal_track_id, entry, combined in candidates:
            if track_id in self._active_embeddings:
                active = self._active_embeddings[track_id]
                for old_emb in entry['embeddings']:
                    max_sim = max(float(np.dot(old_emb, e)) for e in active['embeddings'])
                    if max_sim < 0.90 and len(active['embeddings']) < 8:
                        active['embeddings'].append(old_emb)
            del self._gallery[gal_track_id]

        return reassigned_id

    def cleanup(self, frame_id: int, id_generator=None):
        """Remove expired gallery entries and recycle their equipment IDs.
        Self-healing: phantom IDs from false positives are recycled after TTL,
        making their numbers available for real equipment."""
        expired = [
            tid for tid, entry in self._gallery.items()
            if frame_id - entry['lost_frame'] > self.gallery_ttl_frames
        ]
        for tid in expired:
            entry = self._gallery[tid]
            equip_id = entry.get('equipment_id')
            if equip_id and id_generator:
                id_generator.recycle(equip_id)
                logger.info(f"Gallery TTL expired: {equip_id} recycled after {self.gallery_ttl_frames} frames")
            del self._gallery[tid]


# =============================================================================
# TAI Heuristic (Track-Aware Initialization from TrackTrack, CVPR 2025)
# =============================================================================
def apply_tai_heuristic(detections: np.ndarray,
                        tracked_boxes: np.ndarray,
                        ioa_threshold: float = 0.4) -> np.ndarray:
    """
    TAI port from TrackTrack (CVPR 2025).
    Filters detections that are spatially contained inside existing tracks,
    preventing fragmentary IDs (e.g., excavator bucket detected as separate equipment).

    Args:
        detections: (N, 6) array of [x1, y1, x2, y2, conf, class_id]
        tracked_boxes: (M, 4) array of [x1, y1, x2, y2] from active tracks
        ioa_threshold: Intersection-over-Area threshold (0.4 = 40% contained)

    Returns:
        Filtered detections (K, 6) where K <= N
    """
    if len(tracked_boxes) == 0 or len(detections) == 0:
        return detections

    filtered = []
    for det in detections:
        det_box = det[:4]

        # Calculate IoA: intersection / area_of_detection
        # (How much of the detection is inside an existing track?)
        det_area = max((det_box[2] - det_box[0]) * (det_box[3] - det_box[1]), 1e-6)

        is_fragment = False
        for track_box in tracked_boxes:
            track_area = max(
                (track_box[2] - track_box[0]) * (track_box[3] - track_box[1]), 1e-6
            )

            # Only consider fragment filtering if the detection is
            # significantly SMALLER than the track (< 50% of track area).
            # Full-size re-detections of the same equipment should pass through.
            if det_area > 0.5 * track_area:
                continue

            # Intersection
            x1 = max(det_box[0], track_box[0])
            y1 = max(det_box[1], track_box[1])
            x2 = min(det_box[2], track_box[2])
            y2 = min(det_box[3], track_box[3])

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            ioa = inter_area / det_area

            if ioa > ioa_threshold:
                is_fragment = True
                break

        if not is_fragment:
            filtered.append(det)

    if len(filtered) == 0:
        return np.empty((0, 6))
    return np.array(filtered)


# =============================================================================
# Grid-Based Articulated Motion Detection (Farneback Optical Flow)
# =============================================================================
class ArticulatedMotionDetector:
    """
    Detects motion within equipment bounding boxes using
    Grid Decomposition (3x3) + Farneback Dense Optical Flow.

    Equipment-agnostic: works for excavators (arm-only motion),
    wheel loaders (bucket pivot), dump trucks (driving), etc.
    No assumptions about which grid region corresponds to which part.
    """

    def __init__(self, grid_size: int = 3, motion_threshold: float = 0.15):
        self.grid_size = grid_size
        self.motion_threshold = motion_threshold
        self._prev_crops = {}  # track_id -> previous grayscale crop
        self._centroid_history = {}  # track_id -> deque of (cx, cy)
        self._centroid_window = 30  # 0.5s at 60fps
        self._active_history = {}  # track_id -> deque of recent is_active bools
        self._active_smoothing = 60  # 1 second at 60fps — bridges intra-cycle pauses

    def detect(self, frame_gray: np.ndarray, track_id: int,
               bbox: np.ndarray) -> dict:
        """
        Analyze motion within a bounding box.

        Returns:
            {
                'is_active': bool,
                'motion_source': str ('partial', 'whole_body', 'none'),
                'cell_motions': list[list[float]],  # 3x3 magnitude grid
                'cell_directions': list[list[list[float]]],  # 3x3x2 flow direction grid
                'max_flow': float,
                'active_cell_ratio': float  # fraction of cells with motion
            }
        """
        x1, y1, x2, y2 = bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame_gray.shape[1], x2)
        y2 = min(frame_gray.shape[0], y2)

        # --- Centroid displacement over rolling window ---
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        centroid_moving = False
        if track_id not in self._centroid_history:
            self._centroid_history[track_id] = deque(maxlen=self._centroid_window + 1)
        self._centroid_history[track_id].append((cx, cy))

        history = self._centroid_history[track_id]
        if len(history) >= 2:
            old_cx, old_cy = history[0]
            displacement = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)
            if displacement > 15.0:
                centroid_moving = True

        crop = frame_gray[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return self._empty_result(centroid_moving)

        # Resize crop to standard size — this inherently normalizes for scale.
        # Large boxes (close equipment) and small boxes (distant) produce
        # comparable flow magnitudes after resize. No additional normalization needed.
        crop_resized = self._resize_crop(crop, 300)

        if track_id not in self._prev_crops:
            self._prev_crops[track_id] = crop_resized
            return self._empty_result(centroid_moving)

        prev_crop = self._prev_crops[track_id]
        self._prev_crops[track_id] = crop_resized

        if prev_crop.shape != crop_resized.shape:
            return self._empty_result(centroid_moving)

        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_crop, crop_resized,
            None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )

        # ── Global motion detection (ACTIVE/INACTIVE) ──
        # Dual-gate approach: detect motion via EITHER of two independent signals.
        # This prevents the failure mode where one metric collapses while
        # the equipment is clearly active.
        all_magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        top_flow = float(np.percentile(all_magnitudes, 95))
        median_flow = float(np.median(all_magnitudes))
        localized_motion = top_flow - median_flow

        # Gate 1: Localized motion (top5% - median > threshold)
        # Catches articulated motion (arm/bucket) when background is stable.
        is_localized = localized_motion > self.motion_threshold

        # Gate 2: Absolute flow magnitude (top5% > 0.35)
        # Catches ALL significant motion regardless of median elevation.
        # Parked equipment: top5% ≈ 0.14-0.29 (max observed 0.29).
        # Bucket-only tip-in-place: top5% ≈ 0.35-0.60.
        # Active equipment: top5% ≈ 0.5-1.2.
        # 0.35 gives 0.06 margin above highest parked value while catching
        # the tip-in-place phase that was draining the counter.
        is_strong_flow = top_flow > 0.35

        is_active_flow = is_localized or is_strong_flow

        # ── Grid decomposition (for activity classification) ──
        h, w = crop_resized.shape[:2]
        cell_h, cell_w = h // self.grid_size, w // self.grid_size

        cell_motions = []
        cell_directions = []

        for row in range(self.grid_size):
            motion_row = []
            direction_row = []
            for col in range(self.grid_size):
                cy1 = row * cell_h
                cy2 = (row + 1) * cell_h if row < self.grid_size - 1 else h
                cx1 = col * cell_w
                cx2 = (col + 1) * cell_w if col < self.grid_size - 1 else w

                cell_flow = flow[cy1:cy2, cx1:cx2]
                magnitudes = np.sqrt(
                    cell_flow[..., 0] ** 2 + cell_flow[..., 1] ** 2
                )

                p95_mag = float(np.percentile(magnitudes, 95))
                motion_row.append(p95_mag)

                avg_dx = float(np.mean(cell_flow[..., 0]))
                avg_dy = float(np.mean(cell_flow[..., 1]))
                direction_row.append([avg_dx, avg_dy])

            cell_motions.append(motion_row)
            cell_directions.append(direction_row)

        flat = np.array(cell_motions)
        active_cells = flat > self.motion_threshold
        num_active = int(np.sum(active_cells))
        total_cells = self.grid_size * self.grid_size
        active_ratio = num_active / total_cells

        raw_active = is_active_flow or centroid_moving

        # ── Asymmetric hysteresis ──
        # Construction equipment works in cycles (scoop-lift-dump-return).
        # Brief pauses between phases should NOT flip state to INACTIVE.
        #   INACTIVE → ACTIVE: need 3/15 recent frames active (20%) — respond quickly
        #   ACTIVE → INACTIVE: need 12/15 recent frames inactive (80%) — hold through pauses
        if track_id not in self._active_history:
            self._active_history[track_id] = deque(maxlen=self._active_smoothing)
        self._active_history[track_id].append(raw_active)
        active_votes = sum(self._active_history[track_id])
        total_votes = len(self._active_history[track_id])

        # Asymmetric thresholds based on current state
        if not hasattr(self, '_prev_smoothed'):
            self._prev_smoothed = {}
        prev_active = self._prev_smoothed.get(track_id, False)
        if prev_active:
            # Currently ACTIVE — slow decay (stay active if ≥8/60 recent frames active)
            is_active = active_votes >= 8
        else:
            # Currently INACTIVE — fast attack (activate if ≥3/15 recent frames active)
            recent_votes = sum(list(self._active_history[track_id])[-15:])
            is_active = recent_votes >= 3
        self._prev_smoothed[track_id] = is_active

        # Motion source classification
        if centroid_moving or active_ratio > 0.5:
            motion_source = 'whole_body'
        elif is_active:
            motion_source = 'partial'
        else:
            motion_source = 'none'

        # Debug logging
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:
            gate = "LOC" if is_localized else ("ABS" if is_strong_flow else ("CEN" if centroid_moving else "---"))
            logger.info(
                f"Motion[T{track_id}]: top5%={top_flow:.3f}, localized={localized_motion:.3f}, "
                f"gate={gate}, raw={raw_active}, smoothed={is_active} ({active_votes}/{total_votes}), "
                f"crop={crop_resized.shape}"
            )

        return {
            'is_active': is_active,
            'motion_source': motion_source,
            'cell_motions': cell_motions,
            'cell_directions': cell_directions,
            'max_flow': top_flow,
            'localized_motion': localized_motion,
            'active_cell_ratio': active_ratio
        }

    def remove_track(self, track_id: int):
        self._prev_crops.pop(track_id, None)
        self._centroid_history.pop(track_id, None)
        self._active_history.pop(track_id, None)
        if hasattr(self, '_prev_smoothed'):
            self._prev_smoothed.pop(track_id, None)

    def _resize_crop(self, crop: np.ndarray, target_size: int) -> np.ndarray:
        return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    def _empty_result(self, centroid_moving: bool = False) -> dict:
        empty_grid = [[0.0] * self.grid_size for _ in range(self.grid_size)]
        empty_dirs = [[[0.0, 0.0]] * self.grid_size for _ in range(self.grid_size)]
        return {
            'is_active': centroid_moving,
            'motion_source': 'whole_body' if centroid_moving else 'none',
            'cell_motions': empty_grid,
            'cell_directions': empty_dirs,
            'max_flow': 0.0,
            'localized_motion': 0.0,
            'active_cell_ratio': 0.0
        }


# =============================================================================
# Rule-Based Activity Classifier
# =============================================================================
class RuleBasedActivityClassifier:
    """
    Per-track classifier: only determines ACTIVE/INACTIVE → WAITING.

    Activity sub-labels (DIGGING, LOADING, DUMPING) are assigned in Phase 10
    using spatial context — specifically, the equipment's proximity to dump
    trucks. The flow ratio cannot distinguish DIGGING from DUMPING because
    both produce identical signal topology (concentrated bucket motion).
    Spatial context resolves this: at the pile = DIGGING, at the truck = DUMPING.
    """

    def __init__(self):
        pass

    def classify_smoothed(self, track_id: int, motion_result: dict,
                          equipment_class: str = '') -> str:
        """Return WAITING if inactive, ACTIVE placeholder otherwise.
        Phase 10 will assign the real activity sub-label."""
        if not motion_result['is_active']:
            return 'WAITING'
        # Placeholder — Phase 10 spatial context replaces this
        return 'LOADING'

    def remove_track(self, track_id: int):
        pass


# =============================================================================
# Video Activity Classifier (Option B — X3D-S)
# =============================================================================
class VideoActivityClassifier:
    """
    Classifies equipment activity from 16-frame crop buffers using a
    fine-tuned X3D-S model (Kinetics-400 pretrained → 4 activity classes).

    Maintains a sliding window of cropped frames per track. When the buffer
    reaches `clip_length` frames, runs inference and returns the predicted
    activity label. Falls back to rule-based Phase 10 if model is not loaded.

    Activities: DIGGING, LOADING, DUMPING, WAITING
    """

    ACTIVITY_CLASSES = ['DIGGING', 'LOADING', 'DUMPING', 'WAITING']

    def __init__(self, weights_path: str = None, clip_length: int = 16,
                 clip_stride: int = 8, crop_size: int = 224,
                 confidence_threshold: float = 0.5, device: str = 'cuda'):
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.crop_size = crop_size
        self.confidence_threshold = confidence_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.weights_path = weights_path

        # Per-track crop buffers: track_id -> list of (H,W,3) uint8 crops
        self._buffers = {}
        # Per-track last prediction: track_id -> (activity, confidence)
        self._last_pred = {}

        # Normalization (Kinetics standard)
        self._mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

    def load(self):
        """Load the fine-tuned X3D-S model."""
        if not self.weights_path or not os.path.exists(self.weights_path):
            logger.warning(f"Activity classifier weights not found: {self.weights_path}")
            logger.warning("Falling back to rule-based Phase 10 classification")
            return False

        try:
            from pytorchvideo.models.hub import x3d_s
            import torch.nn as nn

            checkpoint = torch.load(self.weights_path, map_location=self.device)

            self.model = x3d_s(pretrained=False)
            in_features = self.model.blocks[5].proj.in_features
            self.model.blocks[5].proj = nn.Linear(in_features, len(self.ACTIVITY_CLASSES))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()

            # Move normalization tensors to device
            self._mean = self._mean.to(self.device)
            self._std = self._std.to(self.device)

            val_acc = checkpoint.get('best_val_acc', 'unknown')
            logger.info(f"Activity classifier loaded: X3D-S, val_acc={val_acc}%")
            return True
        except Exception as e:
            logger.error(f"Failed to load activity classifier: {e}")
            self.model = None
            return False

    @property
    def is_loaded(self):
        return self.model is not None

    def feed_crop(self, track_id: int, crop: np.ndarray):
        """
        Add a cropped frame to the track's buffer.
        Args:
            track_id: Tracker-assigned ID
            crop: BGR crop from cv2, any size (will be resized)
        """
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return

        # Resize to standard input size
        resized = cv2.resize(crop, (self.crop_size, self.crop_size))
        # BGR → RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if track_id not in self._buffers:
            self._buffers[track_id] = []
        self._buffers[track_id].append(resized)

    def classify(self, track_id: int) -> tuple:
        """
        Classify the track's activity if enough frames are buffered.

        Returns:
            (activity_label, confidence) or (None, 0.0) if not enough frames.
            activity_label is one of: DIGGING, LOADING, DUMPING, WAITING
        """
        if not self.is_loaded:
            return None, 0.0

        buf = self._buffers.get(track_id, [])
        if len(buf) < self.clip_length:
            # Not enough frames yet — return last prediction if available
            return self._last_pred.get(track_id, (None, 0.0))

        # Take the most recent clip_length frames
        clip_frames = buf[-self.clip_length:]

        # Convert to tensor: (T, H, W, 3) uint8 → (1, C, T, H, W) float
        frames_np = np.stack(clip_frames, axis=0)  # (T, H, W, 3)
        clip_tensor = torch.from_numpy(frames_np).float() / 255.0  # (T, H, W, C)
        clip_tensor = clip_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        # Normalize
        clip_tensor = (clip_tensor - self._mean.cpu()) / self._std.cpu()
        clip_tensor = clip_tensor.unsqueeze(0).to(self.device)  # (1, C, T, H, W)

        # Inference
        with torch.no_grad():
            logits = self.model(clip_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()
            confidence = probs[0, pred_idx].item()

        activity = self.ACTIVITY_CLASSES[pred_idx]

        # Only trust prediction if confidence is above threshold
        if confidence < self.confidence_threshold:
            # Low confidence — keep previous prediction or default
            prev = self._last_pred.get(track_id, (None, 0.0))
            if prev[0] is not None:
                activity, confidence = prev

        self._last_pred[track_id] = (activity, confidence)

        # Slide the buffer forward
        self._buffers[track_id] = buf[self.clip_stride:]

        return activity, confidence

    def remove_track(self, track_id: int):
        """Clean up state for a lost track."""
        self._buffers.pop(track_id, None)
        self._last_pred.pop(track_id, None)


# =============================================================================
# Equipment ID Generator
# =============================================================================
class EquipmentIDGenerator:
    """Generates class-prefixed equipment IDs: EX-001, DT-001, BD-001 (per-class counters)."""

    # Minimum confidence to hard-lock class at creation. Below this,
    # class can be corrected by a majority vote over the first N frames.
    CLASS_LOCK_CONF_THRESHOLD = 0.55

    def __init__(self):
        # Audit #7: per-class counters instead of global — avoids inflated IDs
        self._counters = {}  # prefix -> int
        self._track_to_equip = {}  # tracker_id -> equipment_id
        self._equip_to_class = {}  # equipment_id -> class_id (locked at creation)
        self._class_locked = {}   # equipment_id -> bool (True = hard-locked)
        self._class_votes = {}    # equipment_id -> list of (class_id, conf) for soft-lock
        self._recycled = {}  # prefix -> sorted list of recycled numbers

    def get_existing(self, track_id: int) -> str | None:
        """Get existing equipment ID for this track, or None if new."""
        return self._track_to_equip.get(track_id)

    def get_or_create(self, track_id: int, class_id: int, conf: float = 1.0) -> str:
        """Get existing equipment ID or create a new one for this track."""
        if track_id in self._track_to_equip:
            return self._track_to_equip[track_id]

        prefix = CLASS_PREFIXES.get(int(class_id), 'UK')
        self._counters.setdefault(prefix, 0)

        # Use recycled number if available, otherwise increment counter
        if prefix in self._recycled and self._recycled[prefix]:
            num = self._recycled[prefix].pop(0)
        else:
            self._counters[prefix] += 1
            num = self._counters[prefix]

        equip_id = f"{prefix}-{num:03d}"
        self._track_to_equip[track_id] = equip_id
        self._equip_to_class[equip_id] = class_id

        # Hard-lock class only if initial detection confidence is high enough
        if conf >= self.CLASS_LOCK_CONF_THRESHOLD:
            self._class_locked[equip_id] = True
        else:
            self._class_locked[equip_id] = False
            self._class_votes[equip_id] = [(class_id, conf)]
            logger.info(f"{equip_id} class soft-locked (conf={conf:.2f} < {self.CLASS_LOCK_CONF_THRESHOLD}), may be corrected")

        return equip_id

    def reassign(self, new_track_id: int, equip_id: str):
        """Reassign an equipment ID to a new track (after Re-ID match)."""
        self._track_to_equip[new_track_id] = equip_id

    def recycle(self, equip_id: str):
        """Recycle a phantom equipment ID so its number can be reused.
        Called when absorb_near_misses reassigns a phantom to an older ID."""
        parts = equip_id.split('-')
        if len(parts) == 2:
            prefix = parts[0]
            num = int(parts[1])
            self._recycled.setdefault(prefix, [])
            if num not in self._recycled[prefix]:
                self._recycled[prefix].append(num)
                self._recycled[prefix].sort()
            # Clean up the phantom from equip_to_class
            self._equip_to_class.pop(equip_id, None)
            logger.info(f"Recycled phantom ID {equip_id} — number available for reuse")

    def get_locked_class(self, equip_id: str) -> int | None:
        """Get the class_id that was locked when this equipment ID was created."""
        return self._equip_to_class.get(equip_id)

    def is_class_locked(self, equip_id: str) -> bool:
        """True if this equipment's class is hard-locked (high initial confidence)."""
        return self._class_locked.get(equip_id, True)

    def vote_class(self, track_id: int, equip_id: str, class_id: int, conf: float):
        """For soft-locked equipment, accumulate class votes over first 10 detections.
        After 10 votes, pick the weighted majority class and hard-lock it.
        If the majority differs from the initial class, reassign the equipment ID."""
        if self._class_locked.get(equip_id, True):
            return  # already hard-locked

        votes = self._class_votes.get(equip_id, [])
        votes.append((class_id, conf))
        self._class_votes[equip_id] = votes

        # After 10 observations, hard-lock to weighted majority class
        if len(votes) >= 10:
            # Weighted vote: sum confidence per class
            class_weights = {}
            for cid, c in votes:
                class_weights[cid] = class_weights.get(cid, 0.0) + c
            winner = max(class_weights, key=class_weights.get)
            current_class = self._equip_to_class.get(equip_id)

            if winner != current_class:
                # Class correction needed — change the equipment ID prefix
                old_prefix = CLASS_PREFIXES.get(int(current_class), 'UK')
                new_prefix = CLASS_PREFIXES.get(int(winner), 'UK')

                # Recycle the old number
                old_parts = equip_id.split('-')
                old_num = int(old_parts[1])
                self._recycled.setdefault(old_prefix, [])
                if old_num not in self._recycled[old_prefix]:
                    self._recycled[old_prefix].append(old_num)
                    self._recycled[old_prefix].sort()

                # Allocate new number with correct prefix
                self._counters.setdefault(new_prefix, 0)
                if new_prefix in self._recycled and self._recycled[new_prefix]:
                    new_num = self._recycled[new_prefix].pop(0)
                else:
                    self._counters[new_prefix] += 1
                    new_num = self._counters[new_prefix]

                new_equip_id = f"{new_prefix}-{new_num:03d}"
                self._track_to_equip[track_id] = new_equip_id
                self._equip_to_class[new_equip_id] = winner
                self._class_locked[new_equip_id] = True
                # Clean up old
                self._equip_to_class.pop(equip_id, None)
                self._class_locked.pop(equip_id, None)
                self._class_votes.pop(equip_id, None)
                logger.info(
                    f"Class correction: {equip_id} ({CLASS_NAMES.get(current_class)}) → "
                    f"{new_equip_id} ({CLASS_NAMES.get(winner)}) after {len(votes)} votes"
                )
            else:
                # Same class confirmed — hard-lock it
                self._class_locked[equip_id] = True
                self._class_votes.pop(equip_id, None)
                logger.info(f"{equip_id} class confirmed and hard-locked after {len(votes)} votes")

    def is_active(self, equip_id: str, exclude_track: int = None,
                  active_track_ids: set = None) -> bool:
        """Check if an equipment ID is currently assigned to any active track.
        active_track_ids: set of track IDs present in the current frame.
        Only these are truly 'active' — gallery tracks are not."""
        if active_track_ids is None:
            return False  # Can't determine without active set
        for tid, eid in self._track_to_equip.items():
            if eid == equip_id and tid != exclude_track and tid in active_track_ids:
                return True
        return False

    def remove(self, track_id: int):
        self._track_to_equip.pop(track_id, None)


# =============================================================================
# Kafka Telemetry Producer
# =============================================================================
class TelemetryProducer:
    """Produces equipment telemetry events to Kafka."""

    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        self._producer = None
        self._bootstrap_servers = bootstrap_servers

    def connect(self):
        from confluent_kafka import Producer
        self._producer = Producer({
            'bootstrap.servers': self._bootstrap_servers,
            'enable.idempotence': True,
            'acks': 'all',
            'linger.ms': 5,
            'batch.num.messages': 100,
        })
        logger.info(f"Kafka producer connected: {self._bootstrap_servers}")

    def produce(self, equipment_id: str, payload: dict):
        """Produce a single telemetry event, keyed by equipment_id."""
        if self._producer is None:
            return

        self._producer.produce(
            topic=self.topic,
            key=equipment_id.encode('utf-8'),
            value=json.dumps(payload).encode('utf-8'),
            callback=self._delivery_callback
        )
        self._producer.poll(0)

    def flush(self):
        if self._producer:
            self._producer.flush(timeout=5.0)

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            logger.error(f"Kafka delivery failed: {err}")


# =============================================================================
# Main Pipeline Orchestrator
# =============================================================================
class InferencePipeline:
    """
    Main pipeline class that orchestrates all CV components.

    Flow: Frame → Detect → TAI Filter → Track → Re-ID → Motion → Activity → Kafka
    """

    def __init__(self, config: dict):
        self.config = config

        # Core ML components (initialized in .initialize())
        self.detector = None
        self.tracker = None
        self.reid_gallery = None

        # Analysis components
        self.motion_detector = ArticulatedMotionDetector(
            grid_size=3,
            motion_threshold=config.get('motion_threshold', 0.5)  # Audit #3: match class default
        )
        # Activity classification is handled by Phase 10 (spatial context).
        # RuleBasedActivityClassifier kept for remove_track cleanup only.
        self.activity_classifier = RuleBasedActivityClassifier()

        # Option B: Video-based activity classifier (X3D-S)
        # Falls back to rule-based Phase 10 if weights not available.
        self.video_classifier = VideoActivityClassifier(
            weights_path=config.get('activity_classifier_weights', 'weights/activity_classifier_x3d_s.pt'),
            clip_length=16,
            clip_stride=8,
            crop_size=224,
            confidence_threshold=0.5,
            device='cuda',
        )

        self.id_generator = EquipmentIDGenerator()
        self.telemetry_producer = TelemetryProducer(
            bootstrap_servers=config.get('kafka_servers', 'kafka:9092'),
            topic=config.get('kafka_topic', 'equipment.telemetry.raw')
        )

        # Frame counter for skip strategies
        self._frame_count = 0
        # Track IDs from previous frame (for detecting lost tracks)
        self._prev_track_ids = set()
        # Track first-seen frame for deferred ID creation.
        # Tracks must persist for MIN_TRACK_FRAMES before getting an equipment ID.
        # This prevents transient false detections from consuming ID numbers.
        self._track_first_seen = {}  # track_id -> frame_id
        self.MIN_TRACK_FRAMES = 30  # 0.5s at 60fps
        # Activity smoothing
        self._activity_history = {}  # equip_id -> deque(...)

    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing inference pipeline...")
        
        import shutil
        model_dir = os.getenv('MODEL_PATH', '/models')

        # Phase 3: RF-DETR Detector
        weights_path = resolve_weights('rfdetr_construction.pth', local_dir=model_dir)
        
        # Prevent base model redownloads by copying the user mapped file
        user_model = os.path.join(model_dir, "rf-detr-base.pth")
        if os.path.exists(user_model) and not os.path.exists("rf-detr-base.pth"):
            try:
                shutil.copy2(user_model, "rf-detr-base.pth")
                logger.info("Found user's manually downloaded rf-detr-base.pth. Bypassing download!")
            except Exception as e:
                pass
        self.detector = RFDETRDetector(
            weights_path=weights_path,
            confidence_threshold=self.config.get('confidence_threshold', 0.35),
            device=self.config.get('device', 'cuda')
        )
        self.detector.load()

        # Phase 4: BoT-SORT Tracker
        osnet_path = resolve_weights('osnet_x0_25_msmt17.pt', local_dir=model_dir)
        
        # Copy to boxmot's expected location to avoid re-downloading
        reid_dst = "/usr/local/lib/python3.10/dist-packages/models/osnet_x0_25_msmt17.pt"
        if os.path.exists(osnet_path) and not os.path.exists(reid_dst):
            os.makedirs(os.path.dirname(reid_dst), exist_ok=True)
            shutil.copy2(osnet_path, reid_dst)
            logger.info("Copied cached ReID weights — skipping download")

        from boxmot import create_tracker
        botsort_config = Path("/configs/botsort.yaml")
        self.tracker = create_tracker(
            'botsort',
            tracker_config=botsort_config,
            reid_weights=Path(reid_dst if os.path.exists(reid_dst) else osnet_path),
            device='0',
            half=True
        )
        # Log actual tracker params to verify config was parsed correctly
        max_time_lost = getattr(self.tracker, 'max_time_lost', '?')
        buffer_size = getattr(self.tracker, 'buffer_size', '?')
        logger.info(
            f"BoT-SORT initialized: max_time_lost={max_time_lost}, "
            f"buffer_size={buffer_size}"
        )

        # Phase 5: DINOv3 Re-ID Gallery
        self.reid_gallery = DINOv3ReIDGallery(
            similarity_threshold=self.config.get('reid_threshold', 0.75),
            gallery_ttl_frames=self.config.get('reid_gallery_ttl', 1800),
            device=self.config.get('device', 'cuda'),
            spatial_gate_ratio=0.60,
        )
        self.reid_gallery.load()

        # Option B: Video activity classifier (X3D-S)
        if self.video_classifier.load():
            logger.info("Video activity classifier loaded — will override Phase 10 rules")
        else:
            logger.info("Video activity classifier not available — using rule-based Phase 10")

        # Kafka producer
        self.telemetry_producer.connect()

        logger.info("Pipeline initialized — all components ready.")

    def process_frame(self, frame: np.ndarray, frame_id: int,
                      timestamp: float, source_id: str = "cam_01",
                      fps: float = 30.0):
        """
        Process a single frame through the full pipeline.

        Args:
            frame: BGR image (H, W, 3)
            frame_id: Frame counter

        Returns:
            List of annotation dicts for visualization (empty if no tracks)
            timestamp: Seconds from video start
            source_id: Camera identifier
        """
        self._frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----- Phase 3: Detection -----
        raw_detections = self.detector.predict(frame)

        if len(raw_detections) == 0:
            return []

        # ----- Phase 4: TAI Filter -----
        # Audit #8: wrap in try/except — active_tracks API varies across BoxMOT versions
        tracked_boxes = np.empty((0, 4))
        try:
            if hasattr(self.tracker, 'active_tracks') and len(self.tracker.active_tracks) > 0:
                tracked_boxes = np.array([t.xyxy for t in self.tracker.active_tracks])
        except (AttributeError, TypeError):
            pass  # Graceful fallback — TAI filter runs without tracked boxes

        clean_detections = apply_tai_heuristic(
            raw_detections, tracked_boxes, ioa_threshold=0.4
        )

        # ----- Min-area filter: remove tiny background objects -----
        # Construction equipment from aerial/drone footage has a minimum
        # pixel footprint. Distant vehicles (like the background blue truck)
        # are too small to be real tracked equipment.
        if len(clean_detections) > 0:
            frame_area = frame.shape[0] * frame.shape[1]
            min_area_ratio = 0.003  # 0.3% of frame area (~6200px at 1080p)
            areas = (clean_detections[:, 2] - clean_detections[:, 0]) * \
                    (clean_detections[:, 3] - clean_detections[:, 1])
            area_mask = areas >= (frame_area * min_area_ratio)
            clean_detections = clean_detections[area_mask]

        # ----- Phase 4: BoT-SORT Tracking -----
        tracked_objects = self.tracker.update(clean_detections, frame)

        if self._frame_count % 30 == 0:
            logger.info(f"Frame {frame_id}: Dets={len(raw_detections)}, Clean={len(clean_detections)}, Tracked={len(tracked_objects)}")

        # ----- Phase 5: Re-ID — detect lost tracks -----
        current_track_ids = set()
        if len(tracked_objects) > 0:
            current_track_ids = {int(t[4]) for t in tracked_objects}

        lost_ids = self._prev_track_ids - current_track_ids
        new_track_ids = current_track_ids - self._prev_track_ids

        for lost_id in lost_ids:
            self.reid_gallery.move_to_gallery(lost_id, frame_id)
            self.motion_detector.remove_track(f"pending_{lost_id}")
            self.activity_classifier.remove_track(f"pending_{lost_id}")
            self.video_classifier.remove_track(f"pending_{lost_id}")
            self._track_first_seen.pop(lost_id, None)

        # ----- Process each tracked object -----
        annotations = []
        for track in tracked_objects:
            bbox = track[:4]
            track_id = int(track[4])
            conf = float(track[5])
            class_id = int(track[6])

            # Record first-seen frame for deferred ID creation
            if track_id not in self._track_first_seen:
                self._track_first_seen[track_id] = frame_id

            track_age = frame_id - self._track_first_seen[track_id]
            equip_id = self.id_generator.get_existing(track_id)

            # ----- Phase 5: Re-ID + Equipment ID assignment -----
            if track_id in new_track_ids or self._frame_count <= 1:
                # New track — attempt Re-ID immediately
                embedding = self.reid_gallery.extract_embedding(frame, bbox)
                color_hist = self.reid_gallery._extract_color_histogram(frame, bbox)
                matched_id = None

                if equip_id is None and embedding is not None:
                    matched_id, sim = self.reid_gallery.query(
                        embedding, class_id, frame_id,
                        query_bbox=bbox,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        query_color_hist=color_hist
                    )
                    if matched_id is not None:
                        if self.id_generator.is_active(matched_id, exclude_track=track_id, active_track_ids=current_track_ids):
                            logger.info(f"Re-ID rejected: {matched_id} already active on another track")
                            matched_id = None
                        else:
                            equip_id = matched_id
                            self.id_generator.reassign(track_id, equip_id)
                            locked = self.id_generator.get_locked_class(equip_id)
                            if locked is not None:
                                class_id = locked

                if embedding is not None:
                    self.reid_gallery.update_active(
                        track_id, embedding, equip_id, class_id,
                        bbox, frame_id
                    )
                    if color_hist is not None and track_id in self.reid_gallery._active_embeddings:
                        self.reid_gallery._active_embeddings[track_id]['color_hist'] = color_hist

                if equip_id is not None:
                    locked_class = self.id_generator.get_locked_class(equip_id)
                    if locked_class is not None:
                        class_id = locked_class
                else:
                    continue  # Pending — skip annotation

            elif equip_id is None:
                # Existing track, no equipment ID yet — check if mature enough
                if track_age >= self.MIN_TRACK_FRAMES:
                    embedding = self.reid_gallery.extract_embedding(frame, bbox)
                    color_hist = self.reid_gallery._extract_color_histogram(frame, bbox)
                    matched_id = None

                    if embedding is not None:
                        matched_id, sim = self.reid_gallery.query(
                            embedding, class_id, frame_id,
                            query_bbox=bbox,
                            frame_width=frame.shape[1],
                            frame_height=frame.shape[0],
                            query_color_hist=color_hist
                        )
                        if matched_id is not None:
                            if self.id_generator.is_active(matched_id, exclude_track=track_id, active_track_ids=current_track_ids):
                                logger.info(f"Re-ID rejected: {matched_id} already active on another track")
                                matched_id = None
                            else:
                                equip_id = matched_id
                                self.id_generator.reassign(track_id, equip_id)

                    if equip_id is None:
                        equip_id = self.id_generator.get_or_create(track_id, class_id, conf=conf)
                        logger.info(f"Track {track_id} confirmed after {track_age} frames → {equip_id}")

                    locked_class = self.id_generator.get_locked_class(equip_id)
                    if locked_class is not None:
                        class_id = locked_class

                    if embedding is not None:
                        self.reid_gallery.update_active(
                            track_id, embedding, equip_id, class_id,
                            bbox, frame_id
                        )
                        if color_hist is not None and track_id in self.reid_gallery._active_embeddings:
                            self.reid_gallery._active_embeddings[track_id]['color_hist'] = color_hist
                else:
                    if self._frame_count % 10 == 0:
                        embedding = self.reid_gallery.extract_embedding(frame, bbox)
                        if embedding is not None:
                            self.reid_gallery.update_active(
                                track_id, embedding, None, class_id,
                                bbox, frame_id
                            )
                    continue  # Still pending

            else:
                # Existing track WITH equipment ID — normal path
                if not self.id_generator.is_class_locked(equip_id):
                    self.id_generator.vote_class(track_id, equip_id, class_id, conf)
                    equip_id = self.id_generator.get_existing(track_id)
                locked_class = self.id_generator.get_locked_class(equip_id)
                if locked_class is not None:
                    class_id = locked_class
                if self._frame_count % 30 == 0:
                    embedding = self.reid_gallery.extract_embedding(frame, bbox)
                    if embedding is not None:
                        self.reid_gallery.update_active(
                            track_id, embedding, equip_id, class_id,
                            bbox, frame_id
                        )
                        color_hist = self.reid_gallery._extract_color_histogram(frame, bbox)
                        if color_hist is not None and track_id in self.reid_gallery._active_embeddings:
                            self.reid_gallery._active_embeddings[track_id]['color_hist'] = color_hist

            stable_id = equip_id if equip_id is not None else f"pending_{track_id}"

            # ----- Phase 6: Articulated Motion -----
            motion_result = self.motion_detector.detect(
                frame_gray, stable_id, bbox
            )

            # ----- Phase 7: Pass through raw motion state -----
            # Phase 10 owns ALL activity classification using spatial context.
            # Phase 7 only records the motion detector's ACTIVE/INACTIVE decision
            # and motion_source — no activity sub-labels assigned here.
            equip_class = self._class_name(class_id)
            is_active = motion_result['is_active']

            # Collect annotation — state/activity assigned by Phase 10
            annotations.append({
                'bbox': bbox,
                'equip_id': equip_id,
                'stable_id': stable_id,
                'class_name': equip_class,
                'conf': conf,
                'is_motion_active': is_active,  # raw from motion detector
                'motion_source': motion_result['motion_source'],
                'cell_motions': motion_result['cell_motions'],  # 3x3 grid for bucket height
                'state': '',      # Phase 10 assigns
                'activity': '',   # Phase 10 assigns
                'frame_id': frame_id,
                'timestamp': f"{int(timestamp // 3600):02d}:{int((timestamp % 3600) // 60):02d}:{timestamp % 60:06.3f}",
                'source_id': source_id,
                'track_id': track_id,  # needed for video classifier lookup
            })

            # Feed crop to video activity classifier buffer
            if self.video_classifier.is_loaded:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                self.video_classifier.feed_crop(stable_id, crop)

        # ----- Phase 10: Activity Classification -----
        # Single source of truth for ALL state + activity labels.
        #
        # Uses two signals from the 3x3 optical flow grid:
        #   1. Bucket height: flow concentrated in TOP rows = bucket elevated (DUMPING)
        #                     flow concentrated in BOTTOM rows = bucket low (DIGGING)
        #   2. Motion source: whole_body = machine translating (LOADING/swinging)
        #                     partial = only part moving (bucket work)
        #
        # Decision table for articulated equipment:
        #   INACTIVE                                → WAITING
        #   ACTIVE + whole_body                     → LOADING (swinging/translating)
        #   ACTIVE + partial + flow in top rows     → DUMPING (bucket elevated)
        #   ACTIVE + partial + flow in bottom rows  → DIGGING (bucket low, scooping)
        #
        # Dump trucks: classified via adjacent active articulated equipment.

        ARTICULATED = {'excavator', 'wheel_loader', 'bulldozer'}

        def bucket_height_score(cell_motions):
            """
            Compute where the flow is concentrated vertically in the 3x3 grid.
            Returns a score: positive = top rows (bucket high), negative = bottom rows (bucket low).

            Grid layout (3x3):
              row 0 = top of bbox    (bucket elevated / dumping position)
              row 1 = middle
              row 2 = bottom of bbox (bucket low / digging position)
            """
            if not cell_motions or len(cell_motions) < 3:
                return 0.0
            # Sum flow magnitude across each row (all 3 columns)
            top_flow = sum(cell_motions[0])      # row 0
            mid_flow = sum(cell_motions[1])       # row 1
            bottom_flow = sum(cell_motions[2])    # row 2
            total = top_flow + mid_flow + bottom_flow
            if total < 0.01:
                return 0.0
            # Score: +1 = all flow in top, -1 = all flow in bottom
            return (top_flow - bottom_flow) / total

        def bboxes_overlap(a, b):
            x_overlap = min(a[2], b[2]) - max(a[0], b[0])
            y_overlap = min(a[3], b[3]) - max(a[1], b[1])
            return x_overlap > 0 and y_overlap > 0

        # --- Pass 1: Classify articulated equipment ---
        # If X3D-S video classifier is loaded, use it (Option B).
        # Otherwise, fall back to bucket height heuristic (Option A).
        use_video_classifier = self.video_classifier.is_loaded
        
        # Calculate smoothing window dynamically (0.50 seconds to reduce lag)
        window_size = max(10, int(fps * 0.50))

        for ann in annotations:
            if ann is None or ann['class_name'] not in ARTICULATED:
                continue

            # Get video classifier output if available and motion is active
            if use_video_classifier and ann['is_motion_active']:
                activity, conf = self.video_classifier.classify(ann['stable_id'])
                if activity is not None:
                    ann['state'] = 'ACTIVE'
                    ann['activity'] = activity
                    ann['activity_conf'] = conf
                    continue

            # Fallback: rule-based Option A
            raw_activity = 'WAITING'
            if ann['is_motion_active']:
                score = bucket_height_score(ann['cell_motions'])
                if ann['motion_source'] == 'whole_body':
                    if score < -0.15:
                        raw_activity = 'DIGGING'
                    else:
                        raw_activity = 'LOADING'
                else:
                    if score > 0.1:
                        raw_activity = 'DUMPING'
                    else:
                        raw_activity = 'DIGGING'

            # Apply temporal smoothing for Option A using equip_id
            eid = ann['equip_id']
            if eid not in self._activity_history:
                self._activity_history[eid] = deque(maxlen=window_size)
            
            if self._activity_history[eid].maxlen != window_size:
                old_items = list(self._activity_history[eid])
                self._activity_history[eid] = deque(old_items[-window_size:], maxlen=window_size)
                
            self._activity_history[eid].append(raw_activity)
            
            # Majority vote (protect against plurality vote splitting active states)
            from collections import Counter
            counts = Counter(self._activity_history[eid])
            waiting_votes = counts.get('WAITING', 0)
            active_votes = sum(c for s, c in counts.items() if s != 'WAITING')
            
            if waiting_votes > active_votes:
                ann['activity'] = 'WAITING'
                ann['state'] = 'INACTIVE'
            else:
                active_only = {s: c for s, c in counts.items() if s != 'WAITING'}
                ann['activity'] = max(active_only.items(), key=lambda x: x[1])[0] if active_only else 'LOADING'
                ann['state'] = 'ACTIVE'

        # --- Pass 2: Classify dump trucks via adjacent active articulated ---
        dt_annotations = [a for a in annotations if a is not None and a['class_name'] == 'dump_truck']
        active_articulateds = [
            a for a in annotations
            if a is not None
            and a['state'] == 'ACTIVE'
            and a['class_name'] in ARTICULATED
        ]
        for dt in dt_annotations:
            d_bbox = dt['bbox'][:4]
            near_loader = False
            for loader in active_articulateds:
                if bboxes_overlap(d_bbox, loader['bbox'][:4]):
                    near_loader = True
                    break

            if near_loader:
                dt['state'] = 'ACTIVE'
                dt['activity'] = 'LOADING'
            elif dt['is_motion_active']:
                # Truck is moving — map to default ACTIVE (LOADING) since HAULING is not in the 4-activity schema!
                dt['state'] = 'ACTIVE'
                dt['activity'] = 'LOADING'
            else:
                dt['state'] = 'INACTIVE'
                dt['activity'] = 'WAITING'

        # --- Pass 3: Default for any remaining equipment (cranes, etc.) ---
        for ann in annotations:
            if ann is None or ann['state'] != '':
                continue
            if ann['is_motion_active']:
                ann['state'] = 'ACTIVE'
                ann['activity'] = 'LOADING'
            else:
                ann['state'] = 'INACTIVE'
                ann['activity'] = 'WAITING'

        # Debug: log Phase 10 decisions once per second
        if frame_id % 60 == 0:
            for ann in annotations:
                if ann is None:
                    continue
                bh = bucket_height_score(ann.get('cell_motions', []))
                src = 'x3d' if ann.get('activity_conf') else 'rules'
                conf_str = f" ({ann['activity_conf']:.2f})" if ann.get('activity_conf') else ''
                logger.info(
                    f"Phase10[{ann['equip_id']}]: class={ann['class_name']}, "
                    f"motion_active={ann['is_motion_active']}, "
                    f"motion_src={ann['motion_source']}, bucket_h={bh:+.2f}, "
                    f"→ {ann['state']} | {ann['activity']}{conf_str} [{src}]"
                )

        # ----- Deduplicate equipment IDs -----
        # If two tracks share the same equipment ID (e.g., WL-002 on both the
        # real wheel loader and a false detection), keep only the higher-confidence
        # annotation. The lower-confidence duplicate is dropped from display.
        seen_ids = {}
        for i, ann in enumerate(annotations):
            eid = ann['equip_id']
            if eid in seen_ids:
                prev_idx = seen_ids[eid]
                if ann['conf'] > annotations[prev_idx]['conf']:
                    annotations[prev_idx] = None  # mark old for removal
                    seen_ids[eid] = i
                else:
                    annotations[i] = None  # mark new for removal
            else:
                seen_ids[eid] = i
        annotations = [a for a in annotations if a is not None]

        # ----- Phase 11: Produce to Kafka (after spatial context + dedup) -----
        for ann in annotations:
            payload = {
                'frame_id': ann['frame_id'],
                'equipment_id': ann['equip_id'],
                'equipment_class': ann['class_name'],
                'timestamp': ann['timestamp'],
                'bbox': {
                    'x1': float(ann['bbox'][0]), 'y1': float(ann['bbox'][1]),
                    'x2': float(ann['bbox'][2]), 'y2': float(ann['bbox'][3])
                },
                'detection_confidence': round(ann['conf'], 3),
                'utilization': {
                    'current_state': ann['state'],
                    'current_activity': ann['activity'],
                    'motion_source': ann['motion_source']
                },
                'source_id': ann['source_id'],
                'pipeline_timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.telemetry_producer.produce(ann['equip_id'], payload)

        # Periodic flush
        if self._frame_count % 30 == 0:
            self.telemetry_producer.flush()

        # Periodic gallery cleanup — recycle expired phantom IDs (self-healing)
        if self._frame_count % 300 == 0:  # Every 5 seconds at 60fps
            self.reid_gallery.cleanup(frame_id, id_generator=self.id_generator)
            
            # Cleanup stale activity history
            active_known_eids = set(self.id_generator._equip_to_class.keys())
            stale_hist = [eid for eid in self._activity_history if eid not in active_known_eids]
            for eid in stale_hist:
                del self._activity_history[eid]
                
            # Cleanup stale stable_id memories inside motion detector
            if hasattr(self.motion_detector, '_prev_crops'):
                stale_md = [eid for eid in self.motion_detector._prev_crops if not str(eid).startswith('pending_') and eid not in active_known_eids]
                for eid in stale_md:
                    self.motion_detector.remove_track(eid)

            # Cleanup stale stable_id memories inside video classifier
            if hasattr(self.video_classifier, '_crop_buffer'):
                stale_vc = [eid for eid in self.video_classifier._crop_buffer if not str(eid).startswith('pending_') and eid not in active_known_eids]
                for eid in stale_vc:
                    self.video_classifier.remove_track(eid)

        # Audit #2: update _prev_track_ids AFTER the loop so new_track_ids works next frame
        self._prev_track_ids = current_track_ids

        return annotations


    @staticmethod
    def _class_name(class_id: int) -> str:
        return CLASS_NAMES.get(int(class_id), 'unknown')

    @staticmethod
    def draw_annotations(frame: np.ndarray, annotations: list,
                         frame_id: int, timestamp: float = 0.0,
                         fps: float = 0.0) -> np.ndarray:
        """
        Draw bounding boxes, labels, and overlay info on the frame.
        Returns annotated frame copy.
        """
        vis = frame.copy()

        # Color palette per class
        COLORS = {
            'excavator':       (0, 200, 255),   # Orange
            'dump_truck':      (255, 180, 0),    # Cyan-blue
            'bulldozer':       (0, 255, 100),    # Green
            'wheel_loader':    (0, 255, 255),    # Yellow
            'mobile_crane':    (255, 0, 150),    # Magenta
            'tower_crane':     (150, 100, 255),  # Purple
            'roller_compactor':(255, 100, 50),   # Red-orange
            'cement_mixer':    (100, 255, 200),  # Teal
            'unknown':         (200, 200, 200),  # Gray
        }

        for ann in annotations:
            x1, y1, x2, y2 = [int(v) for v in ann['bbox'][:4]]
            color = COLORS.get(ann['class_name'], (200, 200, 200))
            state_color = (0, 255, 0) if ann['state'] == 'ACTIVE' else (0, 0, 255)

            # Draw bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # State indicator bar (top of bbox)
            cv2.rectangle(vis, (x1, y1 - 4), (x2, y1), state_color, -1)

            # Label background
            label = f"{ann['equip_id']} | {ann['class_name']} | {ann['conf']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - 24), (x1 + tw + 6, y1 - 4), color, -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Activity label below bbox
            act_label = f"{ann['state']} | {ann['activity']}"
            cv2.putText(vis, act_label, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, state_color, 1, cv2.LINE_AA)

        # HUD overlay with video timestamp
        mins = int(timestamp // 60)
        secs = timestamp % 60
        fps_str = f"{fps:.1f}" if fps >= 0 else "..."
        hud = f"{mins:02d}:{secs:05.2f} | Tracked: {len(annotations)} | FPS: {fps_str}"
        cv2.putText(vis, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def shutdown(self):
        """Clean shutdown."""
        self.telemetry_producer.flush()
        logger.info("Pipeline shutdown complete.")


# =============================================================================
# Entry Point
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Construction Equipment Monitor - CV Inference")
    logger.info("=" * 60)

    # cv2 is now imported at module level

    config = {
        'kafka_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
        'kafka_topic': os.getenv('KAFKA_TOPIC_RAW', 'equipment.telemetry.raw'),
        'video_source': os.getenv('VIDEO_SOURCE', '/data/test_video.mp4'),
        'device': os.getenv('DEVICE', 'cuda'),
        'motion_threshold': float(os.getenv('MOTION_THRESHOLD', '0.5')),  # Audit #3: was 2.0
        'frame_skip': int(os.getenv('FRAME_SKIP', '1')),
        'target_width': int(os.getenv('TARGET_WIDTH', '960')),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', '0.50')),
        'reid_threshold': float(os.getenv('REID_THRESHOLD', '0.85')),  # Conservative: Re-ID only fires after BoT-SORT drops track (>20s lost)
        'reid_gallery_ttl': int(os.getenv('REID_GALLERY_TTL', '18000')),  # 5min at 60fps — vehicles can leave and return
        'weights_path': os.path.join(os.getenv('MODEL_PATH', '/models'), 'rfdetr_construction.pth'),
    }

    # Initialize pipeline
    pipeline = InferencePipeline(config)
    pipeline.initialize()

    # Initialize video ingestion
    from ingestion import VideoIngestionService
    ingestion = VideoIngestionService(
        source=config['video_source'],
        frame_skip=config['frame_skip'],
        target_width=config['target_width'],
        queue_maxsize=30
    )

    # Graceful shutdown handler
    shutdown_flag = False
    def signal_handler(sig, frame):
        nonlocal shutdown_flag
        logger.info("Shutdown signal received...")
        shutdown_flag = True
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize video writer for annotated output
    save_video = os.getenv('SAVE_ANNOTATED', 'true').lower() == 'true'
    video_writer = None
    output_path = '/data/output_annotated.mp4'

    # Start processing
    ingestion.start()
    logger.info("Processing started. Press Ctrl+C to stop.")

    fps_counter = 0
    fps_start = time.time()
    current_fps = -1.0  # Sentinel: -1 means "not yet computed"

    while not shutdown_flag:
        packet = ingestion.get_frame(timeout=2.0)
        if packet is None:
            logger.info("End of video stream.")
            break

        annotations = pipeline.process_frame(
            frame=packet.frame,
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            source_id=packet.source_id,
            fps=getattr(ingestion, '_src_fps', 30.0)
        )

        # Draw annotations and write to output video
        if save_video:
            if annotations:
                vis_frame = InferencePipeline.draw_annotations(
                    packet.frame, annotations, packet.frame_id,
                    timestamp=packet.timestamp, fps=current_fps
                )
            else:
                # Still write unannotated frames for smooth playback
                vis_frame = packet.frame.copy()
                mins = int(packet.timestamp // 60)
                secs = packet.timestamp % 60
                fps_str = f"{current_fps:.1f}" if current_fps >= 0 else "..."
                hud = f"{mins:02d}:{secs:05.2f} | Tracked: 0 | FPS: {fps_str}"
                cv2.putText(vis_frame, hud, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if video_writer is None:
                h, w = vis_frame.shape[:2]
                src_fps = ingestion._src_fps if hasattr(ingestion, '_src_fps') else 30.0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))
                logger.info(f"Saving annotated video to {output_path} @ {src_fps} FPS")
            video_writer.write(vis_frame)

        # FPS tracking
        fps_counter += 1
        elapsed = time.time() - fps_start
        # Update current_fps every frame so HUD never shows 0.0
        # Require at least 2 frames to avoid misleading spikes on first frame
        if elapsed > 0 and fps_counter >= 2:
            current_fps = fps_counter / elapsed
        if elapsed >= 5.0:
            logger.info(
                f"Pipeline FPS: {current_fps:.1f} | "
                f"Frames: {packet.frame_id} | "
                f"Dropped: {ingestion.frames_dropped}"
            )
            fps_counter = 0
            fps_start = time.time()

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        logger.info(f"Annotated video saved to {output_path}")
    ingestion.stop()
    pipeline.shutdown()
    logger.info("CV Inference service stopped.")


if __name__ == '__main__':
    main()
