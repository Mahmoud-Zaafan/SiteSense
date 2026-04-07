"""
Video Ingestion Service — threaded frame reader for the CV inference pipeline.

Reads frames from a video file (or RTSP stream), applies frame skipping and
optional resizing, then drops them into a bounded queue for the main loop to
consume via `get_frame()`.
"""

import cv2
import logging
import threading
import time
from dataclasses import dataclass
from queue import Queue, Full, Empty
from typing import Optional

import numpy as np

logger = logging.getLogger("cv-inference")


# ---------------------------------------------------------------------------
# Data container returned by the ingestion service
# ---------------------------------------------------------------------------
@dataclass
class FramePacket:
    """Single frame + metadata handed to the pipeline."""
    frame: np.ndarray          # BGR image (H, W, 3)
    frame_id: int              # monotonically increasing counter
    timestamp: float           # epoch seconds when frame was grabbed
    source_id: str             # identifier for the video source


# ---------------------------------------------------------------------------
# VideoIngestionService
# ---------------------------------------------------------------------------
class VideoIngestionService:
    """
    Background-threaded video reader.

    * Supports local files and RTSP/HTTP streams.
    * Applies frame skipping (`frame_skip`) so downstream only processes
      every N-th frame — essential when GPU utilisation is the bottleneck.
    * Resizes frames to `target_width` while preserving aspect ratio.
    * Uses a bounded queue so the reader never races ahead of inference.
    """

    def __init__(
        self,
        source: str,
        frame_skip: int = 1,
        target_width: int = 640,
        queue_maxsize: int = 30,
    ):
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.target_width = target_width

        self._queue: Queue[Optional[FramePacket]] = Queue(maxsize=queue_maxsize)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._frame_id = 0
        self.frames_dropped = 0
        self._source_id = source.split("/")[-1] if "/" in source else source

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self):
        """Spin up the background reader thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(f"Ingestion started — source={self.source}, skip={self.frame_skip}")

    def stop(self):
        """Signal the reader to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Ingestion stopped.")

    def get_frame(self, timeout: float = 2.0) -> Optional[FramePacket]:
        """
        Block until a frame is available or *timeout* seconds elapse.

        Returns ``None`` when the stream has ended (sentinel) or on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    # ------------------------------------------------------------------
    # Internal reader loop (runs on its own thread)
    # ------------------------------------------------------------------
    def _reader_loop(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.source}")
            self._queue.put(None)  # sentinel
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._src_fps = src_fps  # Expose to main loop for video writer
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        logger.info(
            f"Video opened: {self.source} | "
            f"FPS={src_fps:.1f} | Total frames={total_frames}"
        )

        # Detect if source is a file (block) or live stream (drop)
        is_file = not self.source.startswith(("rtsp://", "http://", "https://"))

        raw_idx = 0

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break  # end of file / stream error

            raw_idx += 1

            # Frame skipping
            if raw_idx % self.frame_skip != 0:
                continue

            # Resize to target width (preserve aspect ratio)
            if self.target_width and frame.shape[1] != self.target_width:
                scale = self.target_width / frame.shape[1]
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (self.target_width, new_h))

            # Use video timestamp (seconds from start) for files
            video_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            packet = FramePacket(
                frame=frame,
                frame_id=self._frame_id,
                timestamp=video_ts if is_file else time.time(),
                source_id=self._source_id,
            )
            self._frame_id += 1

            if is_file:
                # For files: block until pipeline consumes — no frames dropped
                self._queue.put(packet, block=True)
            else:
                # For live streams: drop oldest to stay real-time
                try:
                    self._queue.put_nowait(packet)
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                    self._queue.put_nowait(packet)
                    self.frames_dropped += 1

        cap.release()
        self._queue.put(None)  # sentinel → tells main loop the stream ended
        logger.info(f"Reader finished — read {raw_idx} raw frames, delivered {self._frame_id}")
