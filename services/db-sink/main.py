"""
DB Sink Service
================================
Consumes telemetry and state change events from Kafka
and batch-inserts into TimescaleDB.

Uses upsert (ON CONFLICT) for idempotency — replaying
Kafka messages will not create duplicates.
"""

import os
import sys
import json
import time
import logging
import signal
from typing import List
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger('db-sink')


# =============================================================================
# TimescaleDB Writer
# =============================================================================
class TimescaleDBWriter:
    """
    Batch writer for TimescaleDB with upsert idempotency.

    Batches rows in memory and flushes when either:
    - Batch reaches max_batch_size (default: 100)
    - Time since last flush exceeds max_batch_interval (default: 1.0s)
    """

    TELEMETRY_UPSERT_SQL = """
        INSERT INTO equipment_telemetry (
            time, frame_id, equipment_id, equipment_class,
            bbox_x1, bbox_y1, bbox_x2, bbox_y2,
            detection_confidence, current_state, current_activity,
            motion_source, total_tracked_seconds, total_active_seconds,
            total_idle_seconds, utilization_percent, source_id
        ) VALUES (
            %(time)s, %(frame_id)s, %(equipment_id)s, %(equipment_class)s,
            %(bbox_x1)s, %(bbox_y1)s, %(bbox_x2)s, %(bbox_y2)s,
            %(detection_confidence)s, %(current_state)s, %(current_activity)s,
            %(motion_source)s, %(total_tracked_seconds)s, %(total_active_seconds)s,
            %(total_idle_seconds)s, %(utilization_percent)s, %(source_id)s
        )
        ON CONFLICT (time, equipment_id)
        DO UPDATE SET
            current_state = EXCLUDED.current_state,
            current_activity = EXCLUDED.current_activity,
            total_active_seconds = EXCLUDED.total_active_seconds,
            total_idle_seconds = EXCLUDED.total_idle_seconds,
            utilization_percent = EXCLUDED.utilization_percent
    """

    STATE_CHANGE_SQL = """
        INSERT INTO equipment_state_changes (
            time, equipment_id, equipment_class,
            previous_state, new_state, previous_activity, new_activity,
            duration_in_previous_state_seconds,
            total_active_seconds, total_idle_seconds, utilization_percent
        ) VALUES (
            %(time)s, %(equipment_id)s, %(equipment_class)s,
            %(previous_state)s, %(new_state)s, %(previous_activity)s, %(new_activity)s,
            %(duration_in_previous_state_seconds)s,
            %(total_active_seconds)s, %(total_idle_seconds)s, %(utilization_percent)s
        )
    """

    def __init__(self, db_config: dict, max_batch_size: int = 100,
                 max_batch_interval: float = 1.0):
        self.db_config = db_config
        self.max_batch_size = max_batch_size
        self.max_batch_interval = max_batch_interval

        self._conn = None
        self._telemetry_batch: List[dict] = []
        self._state_change_batch: List[dict] = []
        self._last_flush = time.time()

    def connect(self):
        import psycopg2
        self._conn = psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            dbname=self.db_config['dbname'],
        )
        self._conn.autocommit = False
        logger.info(
            f"Connected to TimescaleDB: {self.db_config['host']}/"
            f"{self.db_config['dbname']}"
        )

    def add_telemetry(self, payload: dict):
        """Add a telemetry event to the batch."""
        row = {
            'time': payload.get('pipeline_timestamp',
                                datetime.now(timezone.utc).isoformat()),
            'frame_id': payload.get('frame_id', 0),
            'equipment_id': payload['equipment_id'],
            'equipment_class': payload.get('equipment_class', 'unknown'),
            'bbox_x1': payload.get('bbox', {}).get('x1'),
            'bbox_y1': payload.get('bbox', {}).get('y1'),
            'bbox_x2': payload.get('bbox', {}).get('x2'),
            'bbox_y2': payload.get('bbox', {}).get('y2'),
            'detection_confidence': payload.get('detection_confidence'),
            'current_state': payload.get('utilization', {}).get('current_state', 'UNKNOWN'),
            'current_activity': payload.get('utilization', {}).get('current_activity', 'UNKNOWN'),
            'motion_source': payload.get('utilization', {}).get('motion_source', 'none'),
            'total_tracked_seconds': payload.get('time_analytics', {}).get('total_tracked_seconds', 0),
            'total_active_seconds': payload.get('time_analytics', {}).get('total_active_seconds', 0),
            'total_idle_seconds': payload.get('time_analytics', {}).get('total_idle_seconds', 0),
            'utilization_percent': payload.get('time_analytics', {}).get('utilization_percent', 0),
            'source_id': payload.get('source_id', 'cam_01'),
        }
        self._telemetry_batch.append(row)
        self._maybe_flush()

    def add_state_change(self, event: dict):
        """Add a state change event to the batch."""
        row = {
            'time': event.get('timestamp',
                              datetime.now(timezone.utc).isoformat()),
            'equipment_id': event['equipment_id'],
            'equipment_class': event.get('equipment_class', 'unknown'),
            'previous_state': event.get('previous_state'),
            'new_state': event.get('new_state'),
            'previous_activity': event.get('previous_activity'),
            'new_activity': event.get('current_activity'),
            'duration_in_previous_state_seconds': event.get(
                'duration_in_previous_state_seconds', 0),
            'total_active_seconds': event.get('total_active_seconds', 0),
            'total_idle_seconds': event.get('total_idle_seconds', 0),
            'utilization_percent': event.get('utilization_percent', 0),
        }
        self._state_change_batch.append(row)
        self._maybe_flush()

    def flush(self):
        """Flush all pending batches to TimescaleDB."""
        if not self._conn:
            return

        cursor = self._conn.cursor()
        try:
            # Flush telemetry batch
            if self._telemetry_batch:
                for row in self._telemetry_batch:
                    cursor.execute(self.TELEMETRY_UPSERT_SQL, row)
                count = len(self._telemetry_batch)
                self._telemetry_batch.clear()

            # Flush state change batch
            if self._state_change_batch:
                for row in self._state_change_batch:
                    cursor.execute(self.STATE_CHANGE_SQL, row)
                self._state_change_batch.clear()

            self._conn.commit()
            self._last_flush = time.time()

        except Exception as e:
            logger.error(f"DB flush error: {e}", exc_info=True)
            self._conn.rollback()
        finally:
            cursor.close()

    def _maybe_flush(self):
        total = len(self._telemetry_batch) + len(self._state_change_batch)
        elapsed = time.time() - self._last_flush
        if total >= self.max_batch_size or elapsed >= self.max_batch_interval:
            self.flush()

    def close(self):
        self.flush()
        if self._conn:
            self._conn.close()


# =============================================================================
# DB Sink Service
# =============================================================================
class DBSinkService:
    """
    Consumes both Kafka topics and writes to TimescaleDB in batches.
    """

    def __init__(self, config: dict):
        self.config = config
        self._consumer = None
        self._writer = None
        self._running = False

    def start(self):
        from confluent_kafka import Consumer

        # Initialize DB writer
        self._writer = TimescaleDBWriter(
            db_config={
                'host': self.config['db_host'],
                'port': self.config['db_port'],
                'user': self.config['db_user'],
                'password': self.config['db_password'],
                'dbname': self.config['db_name'],
            }
        )
        self._writer.connect()

        # Initialize Kafka consumer
        self._consumer = Consumer({
            'bootstrap.servers': self.config['kafka_servers'],
            'group.id': self.config['consumer_group'],
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
        })
        self._consumer.subscribe([
            self.config['topic_raw'],
            self.config['topic_aggregated']
        ])

        self._running = True
        logger.info("DB Sink service started.")
        self._process_loop()

    def stop(self):
        self._running = False
        if self._consumer:
            self._consumer.close()
        if self._writer:
            self._writer.close()
        logger.info("DB Sink service stopped.")

    def _process_loop(self):
        while self._running:
            msg = self._consumer.poll(timeout=1.0)
            if msg is None:
                # Periodic flush
                self._writer.flush()
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode('utf-8'))
                topic = msg.topic()

                if topic == self.config['topic_raw']:
                    self._writer.add_telemetry(payload)
                elif topic == self.config['topic_aggregated']:
                    self._writer.add_state_change(payload)

                self._consumer.commit(asynchronous=False)

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)


# =============================================================================
# Entry Point
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Construction Equipment Monitor - DB Sink")
    logger.info("=" * 60)

    config = {
        'kafka_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
        'topic_raw': os.getenv('KAFKA_TOPIC_RAW', 'equipment.telemetry.raw'),
        'topic_aggregated': os.getenv('KAFKA_TOPIC_AGGREGATED', 'equipment.state.aggregated'),
        'consumer_group': os.getenv('KAFKA_CONSUMER_GROUP', 'db-sink-service'),
        'db_host': os.getenv('DB_HOST', 'timescaledb'),
        'db_port': int(os.getenv('DB_PORT', '5432')),
        'db_user': os.getenv('DB_USER', 'cem_admin'),
        'db_password': os.getenv('DB_PASSWORD', 'cem_secure_2026'),
        'db_name': os.getenv('DB_NAME', 'construction_monitor'),
    }

    service = DBSinkService(config)

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
