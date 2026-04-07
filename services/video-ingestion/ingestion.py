"""
Video Ingestion Service
========================================
Reads video files (or RTSP streams) and delivers decoded frames
to the CV inference service via a thread-safe shared queue.

Architecture:
  - Producer thread: reads frames from video source (OpenCV)
  - Consumer: CV inference service pulls frames from queue
  - Backpressure: queue has maxsize; if full, oldest frames are dropped

Supports:
  - Local video files (.mp4, .avi, .mkv)
  - RTSP streams (rtsp://...)
  - Configurable frame skip (process every Nth frame)
  - Configurable resolution downscaling
"""

import cv2
import time
import logging
import threading
from queue import Queue, Full
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger('video-ingestion')


@dataclass
class FramePacket:
    """Immutable frame container passed through the pipeline."""
    frame: np.ndarray            # BGR image (H, W, 3)
    frame_id: int                # Monotonically increasing frame counter
    timestamp: float             # Seconds from video start (computed from FPS)
    source_id: str               # Camera/source identifier
    original_size: Tuple[int, int]  # (width, height) before any resize
    fps: float                   # Source video FPS


class VideoIngestionService:
    """
    Threaded video reader with backpressure-aware frame delivery.

    Usage:
        ingestion = VideoIngestionService(
            source="/data/test_video.mp4",
            queue_maxsize=30,
            frame_skip=1,
            target_width=560
        )
        ingestion.start()

        while True:
            packet = ingestion.get_frame(timeout=5.0)
            if packet is None:
                break  # End of stream
            process(packet)

        ingestion.stop()
    """

    def __init__(
        self,
        source: str,
        source_id: str = "cam_01",
        queue_maxsize: int = 30,
        frame_skip: int = 1,
        target_width: Optional[int] = 560,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 5.0,
    ):
        self.source = source
        self.source_id = source_id
        self.queue_maxsize = queue_maxsize
        self.frame_skip = max(1, frame_skip)
        self.target_width = target_width
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay

        # Internal state
        self._queue: Queue = Queue(maxsize=queue_maxsize)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_stream = source.startswith("rtsp://") or source.startswith("http://")

        # Metrics
        self._frames_read = 0
        self._frames_dropped = 0
        self._fps = 0.0
        self._total_frames = 0

    def start(self):
        """Start the video ingestion thread."""
        if not self._open_source():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="video-ingestion-thread",
            daemon=True
        )
        self._thread.start()
        logger.info(
            f"Video ingestion started: {self.source} "
            f"({self._fps:.1f} FPS, {self._total_frames} total frames, "
            f"skip={self.frame_skip}, target_width={self.target_width})"
        )

    def stop(self):
        """Signal the ingestion thread to stop and wait for cleanup."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info(
            f"Video ingestion stopped. "
            f"Read: {self._frames_read}, Dropped: {self._frames_dropped}"
        )

    def get_frame(self, timeout: float = 5.0) -> Optional[FramePacket]:
        """
        Get the next frame from the queue.
        Returns None if the stream has ended or timeout expires.
        """
        try:
            packet = self._queue.get(timeout=timeout)
            if packet is None:  # Sentinel value = end of stream
                return None
            return packet
        except Exception:
            return None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _open_source(self) -> bool:
        """Open the video source with retry logic for streams."""
        for attempt in range(self.reconnect_attempts):
            self._cap = cv2.VideoCapture(self.source)
            if self._cap.isOpened():
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                return True
            logger.warning(
                f"Failed to open source (attempt {attempt + 1}/{self.reconnect_attempts}): "
                f"{self.source}"
            )
            if attempt < self.reconnect_attempts - 1:
                time.sleep(self.reconnect_delay)

        return False

    def _read_loop(self):
        """
        Producer thread: reads frames, applies skip/resize, pushes to queue.
        Drops frames (backpressure) if queue is full — never blocks the reader.
        """
        raw_frame_count = 0

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()

            if not ret:
                if self._is_stream:
                    # RTSP: attempt reconnect
                    logger.warning("Stream read failure, attempting reconnect...")
                    self._cap.release()
                    if not self._open_source():
                        logger.error("Reconnect failed. Stopping ingestion.")
                        break
                    continue
                else:
                    # File: end of video
                    logger.info("End of video file reached.")
                    break

            raw_frame_count += 1

            # Frame skip: only process every Nth frame
            if raw_frame_count % self.frame_skip != 0:
                continue

            self._frames_read += 1

            # Compute timestamp from frame position
            timestamp = raw_frame_count / self._fps

            # Store original size before resize
            original_h, original_w = frame.shape[:2]

            # Resize if target width is set
            if self.target_width and original_w != self.target_width:
                scale = self.target_width / original_w
                new_h = int(original_h * scale)
                frame = cv2.resize(frame, (self.target_width, new_h),
                                   interpolation=cv2.INTER_LINEAR)

            # Build packet
            packet = FramePacket(
                frame=frame,
                frame_id=self._frames_read,
                timestamp=timestamp,
                source_id=self.source_id,
                original_size=(original_w, original_h),
                fps=self._fps
            )

            # Push to queue with backpressure (drop if full)
            try:
                self._queue.put_nowait(packet)
            except Full:
                self._frames_dropped += 1
                # Drop the oldest frame and push the new one
                try:
                    self._queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._queue.put_nowait(packet)
                except Full:
                    pass

        # Send sentinel to signal end of stream
        try:
            self._queue.put(None, timeout=2.0)
        except Full:
            pass

        if self._cap and self._cap.isOpened():
            self._cap.release()

        logger.info(
            f"Read loop finished. Frames read: {self._frames_read}, "
            f"Dropped: {self._frames_dropped}"
        )
