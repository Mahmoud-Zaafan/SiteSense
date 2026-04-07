"""
Dashboard - Equipment Utilization Monitor (Streamlit)
======================================================
Real-time monitoring dashboard showing:
  - Live equipment status cards (ACTIVE/INACTIVE/SUSPENDED)
  - Utilization time-series charts
  - Activity timeline per equipment
  - Anomaly alerts (>15 min idle)
  - Shift report summary

Queries TimescaleDB continuous aggregates for performance.
"""

import os
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Database Connection
# =============================================================================
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'user': os.getenv('DB_USER', 'cem_admin'),
    'password': os.getenv('DB_PASSWORD', 'cem_secure_2026'),
    'dbname': os.getenv('DB_NAME', 'construction_monitor'),
}


@st.cache_resource
def get_db_connection():
    """Create a persistent database connection."""
    import psycopg2
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def execute_query(query: str, params=None) -> pd.DataFrame:
    """Execute a SQL query and return results as DataFrame."""
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception as e:
        st.error(f"Query error: {e}")
        # Reconnect on failure
        try:
            conn.close()
        except Exception:
            pass
        st.cache_resource.clear()
        return pd.DataFrame()


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Equipment Utilization Monitor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }

    /* Status cards */
    .status-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border-left: 4px solid #4a90d9;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .status-card.active {
        border-left-color: #00c853;
    }
    .status-card.inactive {
        border-left-color: #ff5252;
    }
    .status-card.suspended {
        border-left-color: #ffc107;
    }

    .equip-id {
        font-size: 1.4em;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 4px;
    }
    .equip-class {
        font-size: 0.85em;
        color: #90a4ae;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .state-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin-top: 8px;
    }
    .state-active { background: #1b5e20; color: #69f0ae; }
    .state-inactive { background: #b71c1c; color: #ff8a80; }
    .state-suspended { background: #f57f17; color: #fff176; }

    .util-bar {
        height: 8px;
        border-radius: 4px;
        background: #263238;
        margin-top: 12px;
        overflow: hidden;
    }
    .util-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .metric-value {
        font-size: 2em;
        font-weight: 700;
        color: #e0e0e0;
    }
    .metric-label {
        font-size: 0.85em;
        color: #78909c;
    }

    /* Alert card */
    .alert-card {
        background: linear-gradient(135deg, #3e1a1a 0%, #4a1f1f 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #ff5252;
        margin: 8px 0;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
        padding: 24px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(13, 71, 161, 0.3);
    }
    .main-title {
        font-size: 2em;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    .main-subtitle {
        font-size: 1em;
        color: #90caf9;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Header
# =============================================================================
st.markdown("""
<div class="main-header">
    <p class="main-title">🏗️ Equipment Utilization Monitor</p>
    <p class="main-subtitle">Real-time construction equipment activity tracking & utilization analytics</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_rate = st.slider("Auto-refresh (seconds)", 5, 60, 10)
    time_range = st.selectbox(
        "Time Range",
        ["Last 15 minutes", "Last 1 hour", "Last 4 hours", "Last 8 hours (shift)", "All time"],
        index=1
    )

    time_map = {
        "Last 15 minutes": "15 minutes",
        "Last 1 hour": "1 hour",
        "Last 4 hours": "4 hours",
        "Last 8 hours (shift)": "8 hours",
        "All time": "30 days"
    }
    interval = time_map[time_range]

    st.markdown("---")
    st.header("📊 Quick Stats")

    # Fleet overview query
    fleet_df = execute_query("""
        SELECT
            equipment_id,
            equipment_class,
            current_state,
            current_activity,
            utilization_percent,
            total_active_seconds,
            total_idle_seconds
        FROM equipment_telemetry
        WHERE time > NOW() - INTERVAL '5 minutes'
        ORDER BY time DESC
    """)

    if not fleet_df.empty:
        fleet_latest = fleet_df.drop_duplicates(subset='equipment_id', keep='first')
        total_equipment = len(fleet_latest)
        active_count = len(fleet_latest[fleet_latest['current_state'] == 'ACTIVE'])
        avg_util = fleet_latest['utilization_percent'].mean()

        st.metric("Total Equipment", total_equipment)
        st.metric("Currently Active", f"{active_count}/{total_equipment}")
        st.metric("Avg Utilization", f"{avg_util:.1f}%")
    else:
        st.info("No data yet. Start the CV pipeline to see stats.")


# =============================================================================
# Main Content
# =============================================================================

# --- Equipment Status Cards ---
st.subheader("📡 Live Equipment Status")

latest_df = execute_query(f"""
    SELECT DISTINCT ON (equipment_id)
        equipment_id,
        equipment_class,
        current_state,
        current_activity,
        motion_source,
        detection_confidence,
        utilization_percent,
        total_active_seconds,
        total_idle_seconds,
        total_tracked_seconds,
        time
    FROM equipment_telemetry
    WHERE time > NOW() - INTERVAL '{interval}'
    ORDER BY equipment_id, time DESC
""")

if not latest_df.empty:
    cols = st.columns(min(len(latest_df), 4))

    for idx, (_, row) in enumerate(latest_df.iterrows()):
        col = cols[idx % len(cols)]
        state = row['current_state'].lower()
        state_class = state if state in ('active', 'inactive', 'suspended') else 'inactive'
        util_pct = row.get('utilization_percent', 0) or 0
        util_color = '#00c853' if util_pct >= 70 else '#ffc107' if util_pct >= 40 else '#ff5252'

        with col:
            st.markdown(f"""
            <div class="status-card {state_class}">
                <div class="equip-id">{row['equipment_id']}</div>
                <div class="equip-class">{row['equipment_class']}</div>
                <span class="state-badge state-{state_class}">
                    {'⚡' if state == 'active' else '⏸️' if state == 'inactive' else '🔄'} {row['current_state']}
                </span>
                <div style="margin-top: 8px; color: #b0bec5;">
                    Activity: <strong>{row['current_activity']}</strong>
                </div>
                <div style="color: #78909c; font-size: 0.85em;">
                    Motion: {row.get('motion_source', 'N/A')} |
                    Conf: {row.get('detection_confidence', 0):.2f}
                </div>
                <div class="util-bar">
                    <div class="util-fill" style="width: {util_pct}%; background: {util_color};"></div>
                </div>
                <div style="margin-top: 4px; font-size: 0.85em; color: #b0bec5;">
                    Utilization: <strong>{util_pct:.1f}%</strong> |
                    Active: {row.get('total_active_seconds', 0):.0f}s |
                    Idle: {row.get('total_idle_seconds', 0):.0f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("🔌 No equipment data available. Start the CV inference pipeline to begin tracking.")

st.markdown("---")

# --- Utilization Chart ---
st.subheader("📈 Utilization Over Time")

util_df = execute_query(f"""
    SELECT
        bucket,
        equipment_id,
        equipment_class,
        active_frame_count,
        inactive_frame_count,
        total_frame_count,
        avg_confidence,
        CASE WHEN total_frame_count > 0
            THEN (active_frame_count::float / total_frame_count * 100)
            ELSE 0
        END as utilization_pct
    FROM equipment_utilization_1min
    WHERE bucket > NOW() - INTERVAL '{interval}'
    ORDER BY bucket ASC
""")

if not util_df.empty:
    fig = px.line(
        util_df,
        x='bucket',
        y='utilization_pct',
        color='equipment_id',
        title='Equipment Utilization (%) - 1 Minute Intervals',
        labels={'bucket': 'Time', 'utilization_pct': 'Utilization %', 'equipment_id': 'Equipment'},
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis_range=[0, 105],
        height=400,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ffc107",
                  annotation_text="Target (70%)", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No utilization data yet. Continuous aggregates will populate after data flows through the pipeline.")

# --- Activity Timeline ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔨 Activity Distribution")

    activity_df = execute_query(f"""
        SELECT
            equipment_id,
            SUM(digging_count) as digging,
            SUM(swinging_count) as swinging,
            SUM(dumping_count) as dumping,
            SUM(waiting_count) as waiting
        FROM equipment_utilization_1min
        WHERE bucket > NOW() - INTERVAL '{interval}'
        GROUP BY equipment_id
    """)

    if not activity_df.empty:
        activity_melted = activity_df.melt(
            id_vars=['equipment_id'],
            value_vars=['digging', 'swinging', 'dumping', 'waiting'],
            var_name='activity',
            value_name='frame_count'
        )
        fig2 = px.bar(
            activity_melted,
            x='equipment_id',
            y='frame_count',
            color='activity',
            barmode='stack',
            title='Activity Breakdown by Equipment',
            template='plotly_dark',
            color_discrete_map={
                'digging': '#ff7043',
                'swinging': '#42a5f5',
                'dumping': '#66bb6a',
                'waiting': '#78909c'
            }
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No activity data available yet.")

with col_right:
    st.subheader("⚠️ Anomaly Alerts")

    # Check for equipment idle > 15 minutes
    alert_df = execute_query(f"""
        SELECT DISTINCT ON (equipment_id)
            equipment_id,
            equipment_class,
            current_state,
            total_idle_seconds,
            time
        FROM equipment_telemetry
        WHERE time > NOW() - INTERVAL '{interval}'
          AND current_state = 'INACTIVE'
        ORDER BY equipment_id, time DESC
    """)

    if not alert_df.empty:
        alerts_found = False
        for _, row in alert_df.iterrows():
            idle_mins = (row.get('total_idle_seconds', 0) or 0) / 60
            if idle_mins > 15:
                alerts_found = True
                st.markdown(f"""
                <div class="alert-card">
                    <strong>🚨 {row['equipment_id']}</strong> ({row['equipment_class']})<br>
                    Idle for <strong>{idle_mins:.0f} minutes</strong><br>
                    <span style="color: #ff8a80; font-size: 0.85em;">
                        Last update: {row['time']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

        if not alerts_found:
            st.success("✅ No anomalies detected. All equipment within normal idle thresholds.")
    else:
        st.success("✅ No idle equipment detected.")

# --- State Change Log ---
st.markdown("---")
st.subheader("📋 Recent State Changes")

changes_df = execute_query(f"""
    SELECT
        time,
        equipment_id,
        equipment_class,
        previous_state,
        new_state,
        new_activity,
        duration_in_previous_state_seconds,
        utilization_percent
    FROM equipment_state_changes
    WHERE time > NOW() - INTERVAL '{interval}'
    ORDER BY time DESC
    LIMIT 20
""")

if not changes_df.empty:
    st.dataframe(
        changes_df.style.format({
            'duration_in_previous_state_seconds': '{:.1f}s',
            'utilization_percent': '{:.1f}%'
        }),
        use_container_width=True,
        height=300
    )
else:
    st.info("No state changes recorded yet.")


# --- Auto-refresh ---
st.markdown("---")
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: {refresh_rate}s")
time.sleep(refresh_rate)
st.rerun()
