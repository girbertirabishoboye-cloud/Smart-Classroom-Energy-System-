
# ============================================================
# SMART CLASSROOM ENERGY SYSTEM
# Real KETI Room 413 Dataset
# Logistic Regression Motion Detection Dashboard
# ============================================================

from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# 2. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "smart_classroom_final_model.joblib"
)


# ------------------------------------------------------------
# 3. LOAD TRAINED MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "features",
        "threshold",
        "model_name"
    ]

    for key in required_keys:
        if key not in package:
            raise ValueError(
                f"The saved model is missing: {key}"
            )

    return package


try:
    model_package = load_model()

    model = model_package["model"]
    FEATURES = model_package["features"]
    THRESHOLD = float(model_package["threshold"])
    MODEL_NAME = model_package["model_name"]

except Exception as error:

    st.error("Unable to load the trained machine learning model.")
    st.code(str(error))

    st.info(
        "Check that smart_classroom_final_model.joblib "
        "is inside the models folder."
    )

    st.stop()


# Check model feature compatibility
EXPECTED_FEATURES = [
    "light_level",
    "temperature",
    "humidity",
    "co2",
    "hour"
]

if FEATURES != EXPECTED_FEATURES:

    st.error(
        "The model features do not match the dashboard inputs."
    )

    st.write("Expected:", EXPECTED_FEATURES)
    st.write("Loaded:", FEATURES)

    st.stop()


# ------------------------------------------------------------
# 4. DASHBOARD THEMES
# ------------------------------------------------------------

THEMES = {

    "Emerald Forest": {
        "background": "#071C18",
        "card": "#102D25",
        "primary": "#10B981",
        "secondary": "#34D399",
        "text": "#ECFDF5",
        "muted": "#A7C9BA"
    },

    "Cyber Blue": {
        "background": "#071426",
        "card": "#10243D",
        "primary": "#3B82F6",
        "secondary": "#38BDF8",
        "text": "#EFF6FF",
        "muted": "#A9C5E8"
    },

    "Midnight": {
        "background": "#101827",
        "card": "#1C2940",
        "primary": "#8B5CF6",
        "secondary": "#A78BFA",
        "text": "#F1F5F9",
        "muted": "#CBD5E1"
    }
}


# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.title("💡 Smart Classroom")

    st.caption("Energy Monitoring & Motion Prediction")

    st.divider()

    selected_theme = st.selectbox(
        "🎨 Dashboard Theme",
        list(THEMES.keys()),
        index=0
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("**Algorithm:**", MODEL_NAME)

    st.write("**Dataset:** KETI Room 413")

    st.write("**Target:** PIR Motion Activity")

    st.write(
        "**Threshold:**",
        round(THRESHOLD, 6)
    )

    st.divider()

    st.caption(
        "This dashboard uses the trained machine learning "
        "model. Current sensor values are entered manually."
    )


# ------------------------------------------------------------
# 6. APPLY THEME
# ------------------------------------------------------------

theme = THEMES[selected_theme]

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}

    [data-testid="stHeader"] {{
        background-color: {theme["background"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {theme["card"]};
    }}

    [data-testid="stMetric"] {{
        background-color: {theme["card"]};
        border: 1px solid {theme["primary"]};
        padding: 18px;
        border-radius: 12px;
    }}

    [data-testid="stMetricValue"] {{
        color: {theme["secondary"]};
    }}

    [data-testid="stMetricLabel"] {{
        color: {theme["text"]};
    }}

    div.stButton > button {{
        background-color: {theme["primary"]};
        color: white;
        border-radius: 9px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }}

    div.stButton > button:hover {{
        background-color: {theme["secondary"]};
        color: #071426;
    }}

    .main-title {{
        font-size: 32px;
        font-weight: 800;
        color: {theme["secondary"]};
    }}

    .subtitle {{
        color: {theme["muted"]};
        font-size: 15px;
    }}

    .info-card {{
        background-color: {theme["card"]};
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid {theme["primary"]};
        margin-bottom: 12px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# 7. SESSION STATE
# ------------------------------------------------------------

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ------------------------------------------------------------
# 8. DASHBOARD HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '💡 SMART CLASSROOM ENERGY SYSTEM'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning-Based Classroom Environmental '
    'Monitoring and Motion Activity Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# Status information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Machine Learning Algorithm",
        MODEL_NAME
    )

with col2:
    st.metric(
        "Test Accuracy",
        "89.83%"
    )

with col3:
    st.metric(
        "Motion Recall",
        "62.71%"
    )

st.divider()


# ------------------------------------------------------------
# 9. SENSOR INPUT SECTION
# ------------------------------------------------------------

st.subheader("🌡️ Classroom Sensor Readings")

st.caption(
    "Enter the current environmental readings. "
    "The model will estimate whether PIR motion activity "
    "is present."
)

with st.form("sensor_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        light_level = st.number_input(
            "💡 Light Level",
            min_value=0.0,
            max_value=100000.0,
            value=55.0,
            step=1.0,
            help="Light sensor measurement in the dataset."
        )

        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=-50.0,
            max_value=100.0,
            value=24.0,
            step=0.5
        )

        humidity = st.number_input(
            "💧 Relative Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0
        )

    with col2:

        co2 = st.number_input(
            "🌫️ CO₂ Level (ppm)",
            min_value=0.0,
            max_value=100000.0,
            value=475.0,
            step=10.0
        )

        hour = st.slider(
            "🕒 Hour of the Day",
            min_value=0,
            max_value=23,
            value=10
        )

        st.info(
            "The model uses these five inputs. "
            "The PIR motion label is the prediction target, "
            "not a direct input."
        )

    predict_button = st.form_submit_button(
        "🔍 PREDICT MOTION ACTIVITY",
        use_container_width=True
    )


# ------------------------------------------------------------
# 10. MACHINE LEARNING PREDICTION
# ------------------------------------------------------------

if predict_button:

    input_data = pd.DataFrame(
        [[
            light_level,
            temperature,
            humidity,
            co2,
            hour
        ]],
        columns=FEATURES
    )

    try:

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        prediction = int(
            probability >= THRESHOLD
        )

        if prediction == 1:

            status = "MOTION DETECTED"
            explanation = (
                "The model predicts PIR motion activity."
            )

        else:

            status = "NO MOTION DETECTED"
            explanation = (
                "The model predicts no PIR motion activity."
            )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Save prediction
        record = {
            "Timestamp": timestamp,
            "Light Level": light_level,
            "Temperature": temperature,
            "Humidity": humidity,
            "CO2": co2,
            "Hour": hour,
            "Motion Probability (%)": round(
                probability * 100, 2
            ),
            "Threshold (%)": round(
                THRESHOLD * 100, 2
            ),
            "Prediction": status
        }

        st.session_state.prediction_history.append(
            record
        )

        st.session_state.latest_prediction = {
            "status": status,
            "probability": probability,
            "explanation": explanation
        }

    except Exception as error:

        st.error("Prediction failed.")
        st.exception(error)


# ------------------------------------------------------------
# 11. DISPLAY LATEST PREDICTION
# ------------------------------------------------------------

if "latest_prediction" in st.session_state:

    result = st.session_state.latest_prediction

    st.divider()

    st.subheader("📊 Machine Learning Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if result["status"] == "MOTION DETECTED":

            st.error("🚶 MOTION DETECTED")

        else:

            st.success("🟢 NO MOTION DETECTED")

        st.write(result["explanation"])

    with result_col2:

        st.metric(
            "Motion Probability",
            f'{result["probability"] * 100:.2f}%'
        )

        st.metric(
            "Selected Threshold",
            f"{THRESHOLD * 100:.2f}%"
        )

    st.progress(
        min(max(result["probability"], 0.0), 1.0)
    )

    st.caption(
        "The probability is the model's estimated probability "
        "of motion activity. It is not a guarantee of actual "
        "human presence."
    )


# ------------------------------------------------------------
# 12. MODEL PERFORMANCE
# ------------------------------------------------------------

st.divider()

st.subheader("📈 Model Performance – Final Test Dataset")

st.caption(
    "These are the recorded results from the held-out "
    "KETI Room 413 test period, not live dashboard accuracy."
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Accuracy", "89.83%")
metric2.metric("Precision", "15.55%")
metric3.metric("Recall", "62.71%")
metric4.metric("F1-score", "0.2492")

st.write("**Final Test Confusion Matrix**")

confusion_data = pd.DataFrame(
    [
        [11592, 1206],
        [132, 222]
    ],
    index=["Actual No Motion", "Actual Motion"],
    columns=["Predicted No Motion", "Predicted Motion"]
)

st.dataframe(
    confusion_data,
    use_container_width=True
)

st.warning(
    "The model detects 62.71% of actual motion cases, "
    "but its 15.55% precision indicates many false alarms. "
    "Further improvement is needed before automatic control."
)


# ------------------------------------------------------------
# 13. PREDICTION HISTORY
# ------------------------------------------------------------

st.divider()

st.subheader("📋 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Prediction count
    total_predictions = len(history_df)

    motion_count = (
        history_df["Prediction"] == "MOTION DETECTED"
    ).sum()

    no_motion_count = (
        history_df["Prediction"] == "NO MOTION DETECTED"
    ).sum()

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Total Predictions",
        total_predictions
    )

    h2.metric(
        "Motion Predictions",
        int(motion_count)
    )

    h3.metric(
        "No Motion Predictions",
        int(no_motion_count)
    )

    # History chart
    chart_data = pd.DataFrame({
        "Prediction": [
            "Motion Detected",
            "No Motion"
        ],
        "Count": [
            int(motion_count),
            int(no_motion_count)
        ]
    })

    st.write("**Prediction Distribution**")

    st.bar_chart(
        chart_data.set_index("Prediction")
    )

    # CSV download
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=csv_data,
        file_name="smart_classroom_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        if "latest_prediction" in st.session_state:
            del st.session_state.latest_prediction

        st.rerun()

else:

    st.info(
        "No predictions yet. Enter sensor readings "
        "and click Predict Motion Activity."
    )


# ------------------------------------------------------------
# 14. PROJECT INFORMATION
# ------------------------------------------------------------

st.divider()

st.subheader("ℹ️ About This Project")

st.markdown(
    """
    **Project:** Smart Classroom Energy System

    **Dataset:** KETI Smart Building System – Room 413

    **Machine Learning Algorithm:** Logistic Regression

    **Prediction Target:** PIR motion activity

    **Input Features:**
    - Light level
    - Temperature
    - Relative humidity
    - CO₂ concentration
    - Hour of the day

    **Application:** Classroom environmental monitoring
    and motion activity prediction.
    """
)

st.info(
    "The dashboard currently supports manual sensor input. "
    "ESP8266 live monitoring and physical relay control "
    "are separate integration tasks. This model does not "
    "directly predict actual light-switch ON/OFF status."
)


# ------------------------------------------------------------
# 15. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Smart Classroom Energy System | "
    "KETI Room 413 Dataset | "
    "Machine Learning Dashboard"
)
