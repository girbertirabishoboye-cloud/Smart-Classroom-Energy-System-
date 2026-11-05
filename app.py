
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Try the expected model first, then known alternative filenames.
MODEL_FILES = [
    "smart_classroom_decision_tree_kaggle.joblib",
    "smart_classroom_decision_tree.joblib",
    "smart_classroom_final_model.joblib"
]

MODEL_PATH = next(
    (MODEL_DIR / name for name in MODEL_FILES
     if (MODEL_DIR / name).is_file()),
    None
)

# --------------------------------------------------
# DASHBOARD THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
[data-testid="stMetric"] {
    background: #1C2940;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
}
h1, h2, h3 {
    color: #F1F5F9;
}
[data-testid="stSidebar"] {
    background-color: #172338;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MACHINE LEARNING MODEL
# --------------------------------------------------
@st.cache_resource
def load_artifact(model_path, modified_time):
    return joblib.load(model_path)

if MODEL_PATH is None:
    st.error("Trained model file not found.")

    st.warning(
        "Upload your actual trained model into the GitHub "
        "repository's models folder."
    )

    st.code(
        "models/\n"
        "    smart_classroom_decision_tree_kaggle.joblib"
    )

    st.write("Files checked:")
    for filename in MODEL_FILES:
        st.write(f"- models/{filename}")

    st.stop()

try:
    artifact = load_artifact(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )

    if not isinstance(artifact, dict):
        st.error("The model artifact must be a dictionary.")
        st.stop()

    model = artifact["model"]
    features = artifact["features"]
    classes = artifact.get(
        "class_codes",
        list(model.classes_) if hasattr(model, "classes_") else []
    )

except Exception as error:
    st.error("Unable to load the trained machine learning model.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------
st.title("💡 Smart Classroom Energy System")

st.subheader(
    "Machine Learning-Based Classroom Monitoring "
    "and Automatic Lighting Recommendation"
)

st.write(
    "This dashboard uses a trained Decision Tree model "
    "to produce a dataset-based prediction. The lighting "
    "recommendation is calculated separately using the "
    "documented classroom conditions. Sensor inputs are "
    "simulated until the ESP8266 hardware is integrated."
)

st.divider()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("Classroom Sensor Inputs")

lux = st.sidebar.slider(
    "Ambient light (lux)", 0, 1000, 150
)

motion = st.sidebar.selectbox(
    "Motion detected",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

temp = st.sidebar.slider(
    "Temperature (°C)", 0.0, 50.0, 25.0
)

occupancy = st.sidebar.slider(
    "Occupancy count", 0, 100, 10
)

hour = st.sidebar.slider(
    "Hour of day", 0, 23, datetime.now().hour
)

day = st.sidebar.selectbox(
    "Day of week",
    [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
)

if 5 <= hour < 12:
    period = "Morning"
elif 12 <= hour < 17:
    period = "Afternoon"
elif 17 <= hour < 21:
    period = "Evening"
else:
    period = "Night"

weather = st.sidebar.selectbox(
    "Weather",
    ["Clear", "Cloudy", "Rainy", "Stormy", "Other"]
)

event = st.sidebar.selectbox(
    "Special event flag",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

traffic = st.sidebar.slider(
    "Traffic density", 0.0, 100.0, 20.0
)

speed = st.sidebar.slider(
    "Average pedestrian speed (m/s)", 0.0, 5.0, 1.2
)

# --------------------------------------------------
# AUTOMATIC LIGHTING RECOMMENDATION
# --------------------------------------------------
is_motion = motion == 1
in_window = 7 <= hour <= 18
dark = lux < 300

lamp_status = (
    "ON" if is_motion and in_window and dark else "OFF"
)

# --------------------------------------------------
# KEY METRICS
# --------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric("Ambient Light", f"{lux} lux")
m2.metric(
    "Motion",
    "DETECTED" if is_motion else "NOT DETECTED"
)
m3.metric("Temperature", f"{temp:.1f} °C")
m4.metric("Occupancy", occupancy)

# --------------------------------------------------
# LIGHTING STATUS
# --------------------------------------------------
if lamp_status == "ON":
    st.markdown("""
    <div style="
        background:#064E3B;
        border:3px solid #10B981;
        border-radius:18px;
        padding:28px;
        text-align:center;">
        <h1 style="color:#6EE7B7;font-size:48px;">
            💡 LIGHT ON
        </h1>
        <h3 style="color:white;">
            Automatic conditions recommend switching the light ON
        </h3>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="
        background:#7F1D1D;
        border:3px solid #EF4444;
        border-radius:18px;
        padding:28px;
        text-align:center;">
        <h1 style="color:#FCA5A5;font-size:48px;">
            🌙 LIGHT OFF
        </h1>
        <h3 style="color:white;">
            Automatic conditions recommend switching the light OFF
        </h3>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# CONDITION CHECK
# --------------------------------------------------
st.subheader("Automatic Lighting Condition Check")

conditions = pd.DataFrame({
    "Condition": [
        "Motion detected",
        "Hour is within 07:00–18:00",
        "Ambient light is below 300 lux"
    ],
    "Result": [
        "PASS" if is_motion else "FAIL",
        "PASS" if in_window else "FAIL",
        "PASS" if dark else "FAIL"
    ]
})

st.dataframe(
    conditions,
    use_container_width=True,
    hide_index=True
)

st.divider()

# --------------------------------------------------
# MACHINE LEARNING PREDICTION
# --------------------------------------------------
st.subheader("Decision Tree Model Prediction")

# These column names must match the training dataset.
input_row = {
    "ambient_light_lux": lux,
    "motion_detected": motion,
    "temperature_celsius": temp,
    "occupancy_count": occupancy,
    "hour_of_day": hour,
    "day_of_week": day,
    "time_of_day": period,
    "weather_condition": weather,
    "special_event_flag": event,
    "traffic_density": traffic,
    "avg_pedestrian_speed": speed
}

missing_features = [
    feature for feature in features
    if feature not in input_row
]

if missing_features:
    st.error(
        "The dashboard inputs do not match the trained model. "
        f"Missing features: {missing_features}"
    )
    st.stop()

X = pd.DataFrame(
    [{feature: input_row[feature] for feature in features}],
    columns=features
)

try:
    prediction = model.predict(X)[0]

    st.metric(
        "Original Model Target Prediction",
        str(prediction)
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        confidence = max(probabilities) * 100

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

    st.caption(
        "The model prediction is displayed using its original "
        "target class. It is not automatically relabeled as "
        "LIGHT ON or LIGHT OFF because a verified class-to-lamp "
        "mapping has not been established."
    )

except Exception as error:
    st.error("Prediction failed. Check the model's expected features.")
    st.exception(error)
    st.stop()

# --------------------------------------------------
# MONITORING HISTORY
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.divider()
st.subheader("Monitoring History")

if st.button("Record Current Monitoring Row"):
    st.session_state.history.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **input_row,
        "automatic_lamp_recommendation": lamp_status,
        "model_class_prediction": str(prediction)
    })

    st.success("Monitoring record added.")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Monitoring History (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="smart_classroom_history.csv",
        mime="text/csv"
    )
else:
    st.info("No monitoring records have been saved in this session.")

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
with st.expander("Model Details"):
    st.write("Algorithm: Decision Tree Classifier")
    st.write("Model features:", features)
    st.write("Target field:", artifact.get("target", "Not specified"))
    st.write("Original classes:", classes)
    st.write("Loaded model file:", MODEL_PATH.name)

st.divider()

st.caption(
    "Smart Classroom Energy System | Secondary Dataset | "
    "Simulated Sensor Inputs | No direct relay actuation"
)
