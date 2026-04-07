-- =============================================================================
-- Construction Equipment Monitor - TimescaleDB Schema Initialization
-- =============================================================================
-- This script runs automatically on first container start via
-- docker-entrypoint-initdb.d mount.

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- Core Telemetry Table (Raw frame-level events from CV service)
-- =============================================================================
CREATE TABLE IF NOT EXISTS equipment_telemetry (
    time                    TIMESTAMPTZ         NOT NULL,
    frame_id                INTEGER             NOT NULL,
    equipment_id            TEXT                NOT NULL,
    equipment_class         TEXT                NOT NULL,
    -- Bounding box (x1, y1, x2, y2)
    bbox_x1                 REAL,
    bbox_y1                 REAL,
    bbox_x2                 REAL,
    bbox_y2                 REAL,
    detection_confidence    REAL,
    -- Utilization state
    current_state           TEXT                NOT NULL DEFAULT 'UNKNOWN',
    current_activity        TEXT                DEFAULT 'UNKNOWN',
    motion_source           TEXT                DEFAULT 'none',
    -- Time analytics (computed by analytics service)
    total_tracked_seconds   REAL                DEFAULT 0.0,
    total_active_seconds    REAL                DEFAULT 0.0,
    total_idle_seconds      REAL                DEFAULT 0.0,
    utilization_percent     REAL                DEFAULT 0.0,
    -- Source metadata
    source_id               TEXT                DEFAULT 'cam_01',
    -- Unique constraint for upsert idempotency
    UNIQUE (time, equipment_id)
);

-- Convert to hypertable with daily chunk intervals
SELECT create_hypertable(
    'equipment_telemetry',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =============================================================================
-- Indexes for common query patterns
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_telemetry_equipment_id
    ON equipment_telemetry (equipment_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_telemetry_state
    ON equipment_telemetry (current_state, time DESC);

CREATE INDEX IF NOT EXISTS idx_telemetry_class
    ON equipment_telemetry (equipment_class, time DESC);

-- =============================================================================
-- Continuous Aggregate: 1-minute utilization rollups
-- =============================================================================
-- This materialized view auto-computes per-minute stats, making dashboard
-- queries orders of magnitude faster than scanning the raw hypertable.
CREATE MATERIALIZED VIEW IF NOT EXISTS equipment_utilization_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time)    AS bucket,
    equipment_id,
    equipment_class,
    -- State distribution within the minute
    COUNT(*) FILTER (WHERE current_state = 'ACTIVE')    AS active_frame_count,
    COUNT(*) FILTER (WHERE current_state = 'INACTIVE')  AS inactive_frame_count,
    COUNT(*) FILTER (WHERE current_state = 'SUSPENDED') AS suspended_frame_count,
    COUNT(*)                                             AS total_frame_count,
    -- Activity distribution
    COUNT(*) FILTER (WHERE current_activity = 'DIGGING')   AS digging_count,
    COUNT(*) FILTER (WHERE current_activity = 'SWINGING')  AS swinging_count,
    COUNT(*) FILTER (WHERE current_activity = 'DUMPING')   AS dumping_count,
    COUNT(*) FILTER (WHERE current_activity = 'WAITING')   AS waiting_count,
    -- Aggregate metrics
    AVG(detection_confidence)        AS avg_confidence,
    MAX(utilization_percent)         AS max_utilization_percent,
    -- Spatial tracking (average center point for heatmap)
    AVG((bbox_x1 + bbox_x2) / 2.0)  AS avg_center_x,
    AVG((bbox_y1 + bbox_y2) / 2.0)  AS avg_center_y
FROM equipment_telemetry
GROUP BY bucket, equipment_id, equipment_class
WITH NO DATA;

-- =============================================================================
-- Continuous Aggregate Policy: Auto-refresh every minute
-- =============================================================================
SELECT add_continuous_aggregate_policy(
    'equipment_utilization_1min',
    start_offset    => INTERVAL '10 minutes',
    end_offset      => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists   => TRUE
);

-- =============================================================================
-- Data Retention Policy: Keep raw data for 30 days
-- =============================================================================
SELECT add_retention_policy(
    'equipment_telemetry',
    drop_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

-- =============================================================================
-- State Change Log (populated by analytics service via Kafka)
-- =============================================================================
CREATE TABLE IF NOT EXISTS equipment_state_changes (
    time                TIMESTAMPTZ     NOT NULL,
    equipment_id        TEXT            NOT NULL,
    equipment_class     TEXT            NOT NULL,
    previous_state      TEXT,
    new_state           TEXT            NOT NULL,
    previous_activity   TEXT,
    new_activity        TEXT,
    duration_in_previous_state_seconds REAL,
    total_active_seconds    REAL        DEFAULT 0.0,
    total_idle_seconds      REAL        DEFAULT 0.0,
    utilization_percent     REAL        DEFAULT 0.0
);

SELECT create_hypertable(
    'equipment_state_changes',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_state_changes_equipment
    ON equipment_state_changes (equipment_id, time DESC);

-- =============================================================================
-- Verification: Confirm schema created successfully
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Schema initialized successfully.';
    RAISE NOTICE '   - equipment_telemetry (hypertable)';
    RAISE NOTICE '   - equipment_utilization_1min (continuous aggregate)';
    RAISE NOTICE '   - equipment_state_changes (hypertable)';
    RAISE NOTICE '   - Retention policy: 30 days';
    RAISE NOTICE '   - Aggregate refresh: every 1 minute';
END $$;
