
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# =========================================================
# SMART CLASSROOM ENERGY SYSTEM
# Dataset-Based Decision Tree + Automatic Lighting
# =========================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

# =========================================================
# PROFESSIONAL DARK THEME
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
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL FILE
# =========================================================

MODEL_PATH = "smart_classroom_decision_tree_kaggle(2).joblib"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    features = saved["features"]

    return model, features, saved

loaded_model = load_model()

# =========================================================
# HEADER
# =========================================================

st.title("💡 Smart Classroom Energy System")

st.subheader("Intelligent Classroom Monitoring and Lighting")

st.markdown("""
The system monitors classroom environmental conditions and determines
the appropriate lighting status automatically.
""")

st.divider()

# =========================================================
# SENSOR INPUTS
# These inputs simulate sensor readings for dashboard testing.
# In physical operation, these values should come from ESP8266.
# =========================================================

st.sidebar.header("📡 Classroom Sensor Data")

st.sidebar.subheader("Environmental Sensors")

ambient_light_lux = st.sidebar.slider(
    "Ambient Light (Lux)",
    min_value=0,
    max_value=1000,
    value=150
)

motion_detected = st.sidebar.selectbox(
    "Motion Detection",
    [1, 0],
    format_func=lambda x: "Motion Detected" if x == 1 else "No Motion"
)

temperature_celsius = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

occupancy_count = st.sidebar.slider(
    "Occupancy Count",
    min_value=0,
    max_value=100,
    value=10
)

st.sidebar.subheader("Time Information")

hour_of_day = st.sidebar.slider(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=datetime.now().hour
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

if 5 <= hour_of_day < 12:
    time_of_day = "Morning"
elif 12 <= hour_of_day < 17:
    time_of_day = "Afternoon"
elif 17 <= hour_of_day < 21:
    time_of_day = "Evening"
else:
    time_of_day = "Night"

st.sidebar.info(f"Detected Time Period: {time_of_day}")

st.sidebar.subheader("Additional Dataset Conditions")

weather_condition = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Cloudy", "Rainy", "Stormy", "Other"]
)

special_event_flag = st.sidebar.selectbox(
    "Special Event",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

traffic_density = st.sidebar.slider(
    "Traffic Density",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

avg_pedestrian_speed = st.sidebar.slider(
    "Average Pedestrian Speed (m/s)",
    min_value=0.0,
    max_value=5.0,
    value=1.2
)

# =========================================================
# AUTOMATIC LIGHTING CONDITIONS
# Project rule:
# ON if motion detected, hour 7-18 inclusive,
# and ambient light below 300 lux.
# Otherwise OFF.
# =========================================================

motion_condition = motion_detected == 1

time_condition = 7 <= hour_of_day <= 18

darkness_condition = ambient_light_lux < 300

if (
    motion_condition
    and time_condition
    and darkness_condition
):
    automatic_light_status = "ON"
else:
    automatic_light_status = "OFF"

# =========================================================
# DECISION TREE INPUT DATA
# =========================================================

input_values = {
    "ambient_light_lux": ambient_light_lux,
    "motion_detected": motion_detected,
    "temperature_celsius": temperature_celsius,
    "occupancy_count": occupancy_count,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "time_of_day": time_of_day,
    "weather_condition": weather_condition,
    "special_event_flag": special_event_flag,
    "traffic_density": traffic_density,
    "avg_pedestrian_speed": avg_pedestrian_speed
}

# =========================================================
# MAIN DASHBOARD
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Ambient Light",
    f"{ambient_light_lux} Lux"
)

col2.metric(
    "Motion",
    "DETECTED" if motion_detected == 1 else "NOT DETECTED"
)

col3.metric(
    "Temperature",
    f"{temperature_celsius:.1f} °C"
)

col4.metric(
    "Occupancy",
    occupancy_count
)

st.divider()

# =========================================================
# AUTOMATIC LIGHT STATUS
# =========================================================

if automatic_light_status == "ON":

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
            The classroom lighting conditions require the lamps to be ON.
        </h3>
        <h4 style="color:#A7F3D0;">
            AUTOMATIC LIGHTING DECISION
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
            The classroom lighting conditions do not require the lamps to be ON.
        </h3>
        <h4 style="color:#FECACA;">
            AUTOMATIC LIGHTING DECISION
        </h4>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =========================================================
# CONDITION ANALYSIS
# =========================================================

st.subheader("🔍 Lighting Condition Analysis")

conditions_df = pd.DataFrame({
    "Condition": [
        "Motion Detected",
        "Operating Hour (07:00–18:00)",
        "Ambient Light Below 300 Lux"
    ],
    "Current Result": [
        "SATISFIED" if motion_condition else "NOT SATISFIED",
        "SATISFIED" if time_condition else "NOT SATISFIED",
        "SATISFIED" if darkness_condition else "NOT SATISFIED"
    ]
})

st.dataframe(
    conditions_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# DECISION TREE PREDICTION
# =========================================================

st.divider()

st.subheader("🌳 Dataset Decision Tree Prediction")

if loaded_model is not None:

    model, features, saved_details = loaded_model

    input_df = pd.DataFrame([input_values])

    input_df = input_df[features]

    tree_prediction = model.predict(input_df)[0]

    st.metric(
        "Original Dataset Predicted Class",
        str(tree_prediction)
    )

    st.caption(
        "This is the original classification output of the trained "
        "Decision Tree. The dataset class meanings 0, 1, and 2 must "
        "be verified before interpreting them as lighting actions."
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        confidence = max(probabilities) * 100

        st.write("Decision Tree Prediction Confidence")

        st.progress(int(confidence))

        st.write(f"{confidence:.2f}%")

else:

    st.error(
        "Model file not found. Place "
        "smart_classroom_decision_tree_kaggle(2).joblib "
        "in the same folder as app.py."
    )

# =========================================================
# PREDICTION HISTORY
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

st.divider()

st.subheader("📊 Classroom Monitoring History")

if st.button("Save Current Monitoring Record"):

    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Ambient Light Lux": ambient_light_lux,
        "Motion": motion_detected,
        "Temperature Celsius": temperature_celsius,
        "Occupancy": occupancy_count,
        "Hour": hour_of_day,
        "Day": day_of_week,
        "Time Period": time_of_day,
        "Weather": weather_condition,
        "Special Event": special_event_flag,
        "Traffic Density": traffic_density,
        "Pedestrian Speed": avg_pedestrian_speed,
        "Automatic Light Status": automatic_light_status
    }

    st.session_state.history.append(record)

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Monitoring History",
        data=csv_data,
        file_name="smart_classroom_monitoring_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info("Monitoring records will appear here.")

# =========================================================
# MODEL INFORMATION
# =========================================================

with st.expander("📘 Model Information"):

    st.write("Algorithm: Decision Tree Classifier")

    if loaded_model is not None:

        model, features, saved_details = loaded_model

        st.write("Model Features:")

        st.write(features)

        st.write("Target Variable:")

        st.write(saved_details.get("target"))

        st.write("Dataset Class Codes:")

        st.write(saved_details.get("class_codes"))

        st.write("Dataset Note:")

        st.write(saved_details.get("dataset_note"))

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption("""
Smart Classroom Energy System |
Machine Learning-Based Intelligent Lighting |
Decision Tree Classification
""")
