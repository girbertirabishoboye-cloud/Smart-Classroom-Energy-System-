
import streamlit as st
import pandas as pd
from datetime import datetime

# =========================================================
# SMART CLASSROOM ENERGY SYSTEM
# Automatic Lighting Decision Dashboard
# =========================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

# =========================================================
# DASHBOARD STYLE
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}

h1, h2, h3 {
    color: #F1F5F9;
}

[data-testid="stMetric"] {
    background-color: #1C2940;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# PROJECT LIGHTING CONDITIONS
# =========================================================

LUX_THRESHOLD = 300
START_HOUR = 7
END_HOUR = 18

# =========================================================
# HEADER
# =========================================================

st.title("💡 Smart Classroom Energy System")

st.subheader("Automatic Intelligent Lighting Dashboard")

st.write("""
The system automatically determines whether classroom lamps should be ON
or OFF according to ambient light, motion detection, and time.
""")

st.divider()

# =========================================================
# SENSOR INPUTS
# These are for simulation/testing.
# Replace with live ESP8266 sensor readings for real operation.
# =========================================================

st.sidebar.header("📡 Sensor Monitoring")

lux = st.sidebar.slider(
    "Ambient Light Level (Lux)",
    min_value=0,
    max_value=1000,
    value=150
)

motion = st.sidebar.selectbox(
    "Motion Sensor",
    options=[1, 0],
    format_func=lambda value:
        "Motion Detected" if value == 1 else "No Motion"
)

hour = st.sidebar.slider(
    "Current Hour",
    min_value=0,
    max_value=23,
    value=datetime.now().hour
)

# =========================================================
# AUTOMATIC LIGHTING DECISION
# No manual ON/OFF selector
# =========================================================

if (
    motion == 1
    and START_HOUR <= hour <= END_HOUR
    and lux < LUX_THRESHOLD
):
    light_status = "ON"
else:
    light_status = "OFF"

# =========================================================
# CONDITION ANALYSIS
# =========================================================

motion_condition = motion == 1
time_condition = START_HOUR <= hour <= END_HOUR
darkness_condition = lux < LUX_THRESHOLD

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Ambient Light",
        f"{lux} Lux"
    )

with col2:
    st.metric(
        "Motion Detection",
        "DETECTED" if motion_condition else "NOT DETECTED"
    )

with col3:
    st.metric(
        "Time",
        f"{hour:02d}:00"
    )

st.divider()

# =========================================================
# AUTOMATIC LIGHT STATUS DISPLAY
# =========================================================

if light_status == "ON":

    st.markdown("""
    <div style="
        background-color:#064E3B;
        border:3px solid #10B981;
        border-radius:20px;
        padding:35px;
        text-align:center;
    ">
        <h1 style="color:#6EE7B7;font-size:55px;">
            💡 LIGHT ON
        </h1>
        <h3 style="color:white;">
            Lighting conditions require the classroom lamps to be ON.
        </h3>
        <h4 style="color:#A7F3D0;">
            Automatic Decision: ON
        </h4>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div style="
        background-color:#7F1D1D;
        border:3px solid #EF4444;
        border-radius:20px;
        padding:35px;
        text-align:center;
    ">
        <h1 style="color:#FCA5A5;font-size:55px;">
            🌙 LIGHT OFF
        </h1>
        <h3 style="color:white;">
            Lighting conditions do not require the classroom lamps to be ON.
        </h3>
        <h4 style="color:#FECACA;">
            Automatic Decision: OFF
        </h4>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# CONDITIONS STATUS
# =========================================================

st.subheader("🔍 Automatic Decision Conditions")

condition_data = pd.DataFrame({
    "Condition": [
        "Motion detected",
        "Hour between 07:00 and 18:00",
        "Ambient light below 300 lux"
    ],
    "Current Result": [
        "SATISFIED" if motion_condition else "NOT SATISFIED",
        "SATISFIED" if time_condition else "NOT SATISFIED",
        "SATISFIED" if darkness_condition else "NOT SATISFIED"
    ]
})

st.dataframe(
    condition_data,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# PREDICTION HISTORY
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Save Current Sensor Record"):

    st.session_state.history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Lux": lux,
        "Motion": motion,
        "Hour": hour,
        "Automatic Lighting Status": light_status
    })

st.subheader("📊 Lighting Decision History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download History CSV",
        data=csv,
        file_name="classroom_lighting_history.csv",
        mime="text/csv"
    )

else:
    st.info("No records saved yet.")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("""
Smart Classroom Energy System |
Automatic Lighting Decision |
IoT and Machine Learning
""")
