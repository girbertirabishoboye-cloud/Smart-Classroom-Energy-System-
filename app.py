
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "smart_classroom_final_model.joblib"


# ==========================================================
# 2. DASHBOARD DESIGN
# ==========================================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}

[data-testid="stMetric"] {
    background-color: #1C2940;
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

.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 18px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================================

@st.cache_resource
def load_artifact(path, modified_time):
    return joblib.load(path)


if not MODEL_PATH.is_file():

    st.error("Trained model file not found.")

    st.write("Expected model location:")

    st.code(
        "models/smart_classroom_final_model.joblib"
    )

    st.info(
        "Upload the actual trained model into the models "
        "folder of your GitHub repository."
    )

    st.stop()


try:

    artifact = load_artifact(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )

    if not isinstance(artifact, dict):
        raise ValueError("Unexpected model artifact format.")

    model = artifact["model"]

    features = list(artifact["features"])

    threshold = float(
        artifact.get("threshold", 0.5)
    )

    model_name = artifact.get(
        "model_name", "Trained classifier"
    )

    target = artifact.get(
        "target", "motion_detected"
    )

    classes = artifact.get(
        "class_codes",
        list(model.classes_)
        if hasattr(model, "classes_") else []
    )

except Exception as error:

    st.error("Unable to load the trained model.")

    st.exception(error)

    st.stop()


# ==========================================================
# 4. VERIFY MODEL FEATURES
# ==========================================================

expected_features = [
    "light_level",
    "temperature",
    "humidity",
    "co2",
    "hour"
]

if features != expected_features:

    st.error("Model feature mismatch.")

    st.write("Expected features:", expected_features)

    st.write("Model features:", features)

    st.stop()


# ==========================================================
# 5. DASHBOARD HEADER
# ==========================================================

st.title("💡 Smart Classroom Energy System")

st.subheader(
    "Machine Learning-Based Classroom Monitoring "
    "and Intelligent Lighting Recommendation"
)

st.write(
    "This dashboard uses a trained machine learning model "
    "to predict classroom motion activity using environmental "
    "sensor readings. The lighting recommendation is calculated "
    "separately using the project conditions."
)

st.caption(
    "Current operation: Simulated sensor inputs. "
    "ESP8266 live sensor integration and physical relay "
    "control are not yet connected."
)

st.divider()


# ==========================================================
# 6. SENSOR INPUTS
# ==========================================================

st.sidebar.header("Classroom Sensor Monitoring")

st.sidebar.caption(
    "Adjust the values to simulate classroom conditions."
)

lux = st.sidebar.slider(
    "Light Level (lux)",
    0, 1000, 250
)

temp = st.sidebar.slider(
    "Temperature (°C)",
    0.0, 50.0, 25.0
)

humidity = st.sidebar.slider(
    "Relative Humidity (%)",
    0.0, 100.0, 55.0
)

co2 = st.sidebar.slider(
    "CO₂ Concentration (ppm)",
    300, 5000, 700
)

hour = st.sidebar.slider(
    "Hour of Day",
    0, 23, datetime.now().hour
)


# ==========================================================
# 7. PREPARE MODEL INPUT
# ==========================================================

input_data = {
    "light_level": lux,
    "temperature": temp,
    "humidity": humidity,
    "co2": co2,
    "hour": hour
}

X = pd.DataFrame(
    [[input_data[feature] for feature in features]],
    columns=features
)


# ==========================================================
# 8. MACHINE LEARNING PREDICTION
# ==========================================================

try:

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(X)[0]

        model_classes = list(model.classes_)

        if 1 in model_classes:

            positive_index = model_classes.index(1)

            motion_probability = float(
                probabilities[positive_index]
            )

            prediction = int(
                motion_probability >= threshold
            )

        else:

            prediction = model.predict(X)[0]

            motion_probability = None

    else:

        prediction = model.predict(X)[0]

        motion_probability = None

except Exception as error:

    st.error("Prediction failed.")

    st.exception(error)

    st.stop()


# ==========================================================
# 9. SENSOR METRICS
# ==========================================================

st.subheader("Live Classroom Monitoring (Simulated)")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Light Level",
    f"{lux} lux"
)

m2.metric(
    "Temperature",
    f"{temp:.1f} °C"
)

m3.metric(
    "Humidity",
    f"{humidity:.0f}%"
)

m4.metric(
    "CO₂ Level",
    f"{co2:,} ppm"
)


# ==========================================================
# 10. MODEL PREDICTION DISPLAY
# ==========================================================

st.divider()

st.subheader("Machine Learning Prediction")

if int(prediction) == 1:

    st.success(
        "MOTION DETECTED — Model predicts class 1"
    )

else:

    st.info(
        "NO MOTION DETECTED — Model predicts class 0"
    )


if motion_probability is not None:

    col1, col2 = st.columns(2)

    col1.metric(
        "Motion Probability",
        f"{motion_probability * 100:.2f}%"
    )

    col2.metric(
        "Classification Threshold",
        f"{threshold:.4f}"
    )

    st.progress(
        min(max(motion_probability, 0.0), 1.0)
    )


st.caption(
    "The prediction is based on the trained Logistic Regression "
    "model and its saved classification threshold."
)


# ==========================================================
# 11. AUTOMATIC LIGHTING RECOMMENDATION
# ==========================================================

st.divider()

st.subheader("Automatic Lighting Recommendation")

is_motion = int(prediction) == 1

in_window = 7 <= hour <= 18

dark = lux < 300

lamp_status = (
    "ON"
    if is_motion and in_window and dark
    else "OFF"
)


if lamp_status == "ON":

    st.markdown("""
    <div style="
        background:#064E3B;
        border:3px solid #10B981;
        border-radius:18px;
        padding:28px;
        text-align:center;
    ">
        <h1 style="color:#6EE7B7;font-size:45px;">
            💡 LIGHT ON
        </h1>
        <h3 style="color:white;">
            Lighting conditions recommend switching ON
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
        text-align:center;
    ">
        <h1 style="color:#FCA5A5;font-size:45px;">
            🌙 LIGHT OFF
        </h1>
        <h3 style="color:white;">
            Lighting conditions recommend switching OFF
        </h3>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 12. CONDITION CHECK
# ==========================================================

st.subheader("Automatic Condition Check")

conditions = pd.DataFrame({
    "Condition": [
        "Model predicts motion",
        "Hour is within 07:00–18:00",
        "Light level is below 300 lux"
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

st.warning(
    "The lighting status is a software recommendation. "
    "It does not directly switch a physical lamp or relay."
)


# ==========================================================
# 13. MONITORING HISTORY
# ==========================================================

st.divider()

st.subheader("Classroom Monitoring History")

if "history" not in st.session_state:

    st.session_state.history = []


if st.button("Record Current Monitoring Row"):

    record = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),

        "light_level": lux,

        "temperature": temp,

        "humidity": humidity,

        "co2": co2,

        "hour": hour,

        "predicted_motion": int(prediction),

        "motion_probability": motion_probability,

        "classification_threshold": threshold,

        "lighting_recommendation": lamp_status
    }

    st.session_state.history.append(record)

    st.success("Monitoring record saved.")


if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Monitoring History CSV",

        history_df.to_csv(
            index=False
        ).encode("utf-8"),

        file_name="smart_classroom_history.csv",

        mime="text/csv"
    )

else:

    st.info(
        "Click Record Current Monitoring Row "
        "to save a monitoring record."
    )


# ==========================================================
# 14. MODEL INFORMATION
# ==========================================================

with st.expander("Machine Learning Model Details"):

    st.write("Algorithm:", model_name)

    st.write("Target variable:", target)

    st.write("Features:", features)

    st.write("Classification threshold:", threshold)

    st.write("Original classes:", classes)

    st.write("Model file:", MODEL_PATH.name)


# ==========================================================
# 15. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Smart Classroom Energy System | Secondary Dataset "
    "Machine Learning | Logistic Regression | "
    "Simulated Sensor Inputs | No Direct Relay Actuation"
)
