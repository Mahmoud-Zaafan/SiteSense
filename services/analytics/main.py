"""
Analytics Service (Main Entry Point)
=====================================================
Consumes raw telemetry from Kafka, runs the dwell-time state machine
per equipment ID, and produces aggregated state change events.

State Machine:
  ACTIVE ↔ INACTIVE (with 10-frame hysteresis)
  ANY → SUSPENDED (track lost, timers continue on last known state)
  SUSPENDED → ACTIVE/INACTIVE (Re-ID recovery)
"""

import os
import sys
import json
import time
import logging
import signal
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('analytics')


# =============================================================================
# Equipment State Machine
# =============================================================================
class EquipmentState(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


@dataclass
class EquipmentTracker:
    """
    Stateful tracker per equipment ID.

    Implements:
    - 10-frame hysteresis buffer to prevent state flickering
    - SUSPENDED state for track loss (timers continue)
    - Accurate active/idle time accumulation
    - State change event emission
    """
    equipment_id: str
    equipment_class: str

    # Current confirmed state
    current_state: EquipmentState = EquipmentState.UNKNOWN
    current_activity: str = "UNKNOWN"

    # Hysteresis buffer
    pending_state: Optional[EquipmentState] = None
    state_change_buffer: int = 0
    hysteresis_threshold: int = 10  # Frames needed to confirm transition

    # Time tracking
    total_active_seconds: float = 0.0
    total_idle_seconds: float = 0.0
    total_tracked_seconds: float = 0.0
    last_timestamp: float = 0.0
    first_seen_timestamp: float = 0.0

    # Session tracking
    current_session_start: float = 0.0
    idle_sessions: int = 0

    # Suspension tracking
    suspended_at: float = 0.0
    frames_since_last_seen: int = 0
    max_suspend_frames: int = 1800  # 120s at 15fps

    # State change log
    state_changes: list = field(default_factory=list)

    def update(self, raw_state: str, activity: str, timestamp: float,
               motion_source: str = 'none') -> Optional[dict]:
        """
        Update the state machine with a new observation.

        Args:
            raw_state: 'ACTIVE' or 'INACTIVE' from CV service
            activity: Activity label from classifier
            timestamp: Seconds from video start
            motion_source: Where motion was detected

        Returns:
            State change event dict if a confirmed transition occurred, else None
        """
        self.frames_since_last_seen = 0

        # Initialize on first observation
        if self.current_state == EquipmentState.UNKNOWN:
            self.current_state = EquipmentState(raw_state)
            self.current_activity = activity
            self.last_timestamp = timestamp
            self.first_seen_timestamp = timestamp
            self.current_session_start = timestamp
            return None

        # Accumulate time since last update
        dt = max(0, timestamp - self.last_timestamp)
        self.total_tracked_seconds += dt

        if self.current_state == EquipmentState.ACTIVE:
            self.total_active_seconds += dt
        elif self.current_state == EquipmentState.INACTIVE:
            self.total_idle_seconds += dt
        elif self.current_state == EquipmentState.SUSPENDED:
            # During suspension, continue timing based on last known state
            # (This prevents timer loss during occlusion)
            if self.pending_state == EquipmentState.ACTIVE:
                self.total_active_seconds += dt
            else:
                self.total_idle_seconds += dt

        self.last_timestamp = timestamp
        self.current_activity = activity

        # Recovery from SUSPENDED
        if self.current_state == EquipmentState.SUSPENDED:
            new_state = EquipmentState(raw_state)
            prev = self.current_state
            self.current_state = new_state
            self.pending_state = None
            self.state_change_buffer = 0
            logger.info(
                f"[{self.equipment_id}] Recovered from SUSPENDED → {new_state.value}"
            )
            return self._emit_state_change(prev, new_state, timestamp)

        # Hysteresis logic for state transitions
        incoming = EquipmentState(raw_state)

        if incoming != self.current_state:
            # State differs from current confirmed state
            if incoming == self.pending_state:
                # Same pending state — increment buffer
                self.state_change_buffer += 1
            else:
                # New pending state — reset buffer
                self.pending_state = incoming
                self.state_change_buffer = 1

            # Check if hysteresis threshold reached
            if self.state_change_buffer >= self.hysteresis_threshold:
                prev = self.current_state
                self.current_state = incoming
                self.pending_state = None
                self.state_change_buffer = 0

                # Track idle sessions
                if incoming == EquipmentState.INACTIVE:
                    self.idle_sessions += 1
                    self.current_session_start = timestamp

                logger.info(
                    f"[{self.equipment_id}] State: {prev.value} → {incoming.value} "
                    f"(confirmed after {self.hysteresis_threshold} frames)"
                )
                return self._emit_state_change(prev, incoming, timestamp)
        else:
            # State matches current — reset any pending transition
            self.pending_state = None
            self.state_change_buffer = 0

        return None

    def mark_suspended(self, timestamp: float) -> Optional[dict]:
        """
        Mark equipment as SUSPENDED (track lost).
        Timers continue based on last known state.
        """
        if self.current_state == EquipmentState.SUSPENDED:
            return None

        prev = self.current_state
        self.suspended_at = timestamp
        # Store last active state as pending (for timer continuation)
        self.pending_state = self.current_state
        self.current_state = EquipmentState.SUSPENDED

        logger.info(
            f"[{self.equipment_id}] Entered SUSPENDED (last: {prev.value})"
        )
        return self._emit_state_change(prev, EquipmentState.SUSPENDED, timestamp)

    def tick_suspend(self):
        """Increment suspend frame counter. Returns True if TTL exceeded."""
        self.frames_since_last_seen += 1
        return self.frames_since_last_seen >= self.max_suspend_frames

    @property
    def utilization_percent(self) -> float:
        if self.total_tracked_seconds <= 0:
            return 0.0
        return (self.total_active_seconds / self.total_tracked_seconds) * 100.0

    def to_summary(self) -> dict:
        return {
            'equipment_id': self.equipment_id,
            'equipment_class': self.equipment_class,
            'current_state': self.current_state.value,
            'current_activity': self.current_activity,
            'total_tracked_seconds': round(self.total_tracked_seconds, 1),
            'total_active_seconds': round(self.total_active_seconds, 1),
            'total_idle_seconds': round(self.total_idle_seconds, 1),
            'utilization_percent': round(self.utilization_percent, 1),
            'idle_sessions': self.idle_sessions,
        }

    def _emit_state_change(self, prev: EquipmentState,
                           new: EquipmentState, timestamp: float) -> dict:
        """Build a state change event for Kafka emission."""
        duration = timestamp - self.current_session_start
        event = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equipment_id': self.equipment_id,
            'equipment_class': self.equipment_class,
            'previous_state': prev.value,
            'new_state': new.value,
            'current_activity': self.current_activity,
            'duration_in_previous_state_seconds': round(duration, 1),
            'total_active_seconds': round(self.total_active_seconds, 1),
            'total_idle_seconds': round(self.total_idle_seconds, 1),
            'utilization_percent': round(self.utilization_percent, 1),
        }
        self.state_changes.append(event)
        self.current_session_start = timestamp
        return event


# =============================================================================
# Analytics Service
# =============================================================================
class AnalyticsService:
    """
    Kafka consumer that maintains per-equipment state machines.
    Consumes raw telemetry → updates EquipmentTracker → produces state changes.
    """

    def __init__(self, config: dict):
        self.config = config
        self._trackers: Dict[str, EquipmentTracker] = {}
        self._consumer = None
        self._producer = None
        self._running = False

    def start(self):
        """Initialize Kafka connections and start processing."""
        from confluent_kafka import Consumer, Producer

        self._consumer = Consumer({
            'bootstrap.servers': self.config['kafka_servers'],
            'group.id': self.config['consumer_group'],
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
            'isolation.level': 'read_committed',
        })
        self._consumer.subscribe([self.config['topic_raw']])

        self._producer = Producer({
            'bootstrap.servers': self.config['kafka_servers'],
            'enable.idempotence': True,
            'acks': 'all',
        })

        self._running = True
        logger.info("Analytics service started. Consuming telemetry events.")

        self._process_loop()

    def stop(self):
        self._running = False
        if self._consumer:
            self._consumer.close()
        if self._producer:
            self._producer.flush(timeout=5.0)
        logger.info("Analytics service stopped.")

    def _process_loop(self):
        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                # Tick suspended trackers
                self._tick_suspensions()
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode('utf-8'))
                self._process_event(payload)
                self._consumer.commit(asynchronous=False)
            except Exception as e:
                logger.error(f"Error processing event: {e}", exc_info=True)

    def _process_event(self, payload: dict):
        """Process a single telemetry event through the state machine."""
        equip_id = payload['equipment_id']
        equip_class = payload.get('equipment_class', 'unknown')
        raw_state = payload.get('utilization', {}).get('current_state', 'INACTIVE')
        activity = payload.get('utilization', {}).get('current_activity', 'WAITING')
        motion_source = payload.get('utilization', {}).get('motion_source', 'none')

        # Parse timestamp (HH:MM:SS.mmm -> seconds)
        ts_str = payload.get('timestamp', '00:00:00.000')
        timestamp = self._parse_timestamp(ts_str)

        # Get or create tracker
        if equip_id not in self._trackers:
            self._trackers[equip_id] = EquipmentTracker(
                equipment_id=equip_id,
                equipment_class=equip_class
            )
            logger.info(f"New equipment tracked: {equip_id} ({equip_class})")

        tracker = self._trackers[equip_id]

        # Update state machine
        state_change = tracker.update(raw_state, activity, timestamp, motion_source)

        # If confirmed state change occurred, emit to Kafka
        if state_change:
            self._emit_state_change(state_change)

    def _tick_suspensions(self):
        """Check all trackers for suspension TTL expiry."""
        expired = []
        for equip_id, tracker in self._trackers.items():
            if tracker.current_state == EquipmentState.SUSPENDED:
                if tracker.tick_suspend():
                    logger.warning(
                        f"[{equip_id}] SUSPENDED TTL expired. "
                        f"Final util: {tracker.utilization_percent:.1f}%"
                    )
                    expired.append(equip_id)

        for equip_id in expired:
            # Don't delete — keep for reporting. Just log it.
            pass

    def _emit_state_change(self, event: dict):
        """Produce state change event to Kafka aggregated topic."""
        if self._producer is None:
            return

        self._producer.produce(
            topic=self.config['topic_aggregated'],
            key=event['equipment_id'].encode('utf-8'),
            value=json.dumps(event).encode('utf-8'),
        )
        self._producer.poll(0)
        logger.info(
            f"State change emitted: {event['equipment_id']} "
            f"{event['previous_state']} → {event['new_state']}"
        )

    @staticmethod
    def _parse_timestamp(ts_str: str) -> float:
        """Parse HH:MM:SS.mmm to seconds."""
        try:
            parts = ts_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (IndexError, ValueError):
            return 0.0


# =============================================================================
# Entry Point
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Construction Equipment Monitor - Analytics")
    logger.info("=" * 60)

    config = {
        'kafka_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
        'topic_raw': os.getenv('KAFKA_TOPIC_RAW', 'equipment.telemetry.raw'),
        'topic_aggregated': os.getenv('KAFKA_TOPIC_AGGREGATED', 'equipment.state.aggregated'),
        'consumer_group': os.getenv('KAFKA_CONSUMER_GROUP', 'analytics-service'),
    }

    service = AnalyticsService(config)

    # Graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received...")
        service.stop()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        service.start()
    except KeyboardInterrupt:
        service.stop()


if __name__ == '__main__':
    main()
