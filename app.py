
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import joblib


# ============================================================
# SMART CLASSROOM ENERGY SYSTEM
# Machine Learning-Based PIR Motion Prediction Dashboard
# ============================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 1. FILE PATHS AND MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "smart_classroom_final_model.joblib"
)


@st.cache_resource
def load_trained_model(model_path):

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    data = joblib.load(model_path)

    required_keys = [
        "model",
        "features",
        "threshold",
        "model_name"
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Missing '{key}' in the saved model file."
            )

    return data


# Load model safely
try:
    model_data = load_trained_model(MODEL_PATH)

    model = model_data["model"]
    features = model_data["features"]
    threshold = float(model_data["threshold"])
    model_name = model_data["model_name"]

except Exception as error:
    st.error("Unable to load the trained machine learning model.")

    st.code(str(error))

    st.markdown(
        """
        **Check your GitHub repository structure:**

        ```text
        Smart-Classroom-Energy-System-/
        │
        ├── app.py
        ├── requirements.txt
        │
        └── models/
            └── smart_classroom_final_model.joblib
        ```
        """
    )

    st.stop()


# Expected feature order
EXPECTED_FEATURES = [
    "light_level",
    "temperature",
    "humidity",
    "co2",
    "hour"
]

if list(features) != EXPECTED_FEATURES:
    st.error(
        "The model feature configuration does not match "
        "the dashboard input configuration."
    )
    st.write("Expected:", EXPECTED_FEATURES)
    st.write("Model features:", features)
    st.stop()


# ============================================================
# 2. THEME CONFIGURATION
# ============================================================

THEMES = {
    "Emerald Forest": {
        "background": "#071A17",
        "sidebar": "#0D2923",
        "card": "#12382E",
        "primary": "#10B981",
        "secondary": "#34D399",
        "text": "#F0FDF4",
        "muted": "#A7C8B8",
        "border": "#245746"
    },

    "Cyber Blue": {
        "background": "#081525",
        "sidebar": "#10243D",
        "card": "#172F4D",
        "primary": "#3B82F6",
        "secondary": "#60A5FA",
        "text": "#EFF6FF",
        "muted": "#B6C8DF",
        "border": "#294968"
    },

    "Midnight": {
        "background": "#111827",
        "sidebar": "#1F2937",
        "card": "#273449",
        "primary": "#8B5CF6",
        "secondary": "#A78BFA",
        "text": "#F9FAFB",
        "muted": "#CBD5E1",
        "border": "#3B4960"
    },

    "Ocean Teal": {
        "background": "#071C24",
        "sidebar": "#10313A",
        "card": "#174451",
        "primary": "#14B8A6",
        "secondary": "#2DD4BF",
        "text": "#F0FDFA",
        "muted": "#B2D9D5",
        "border": "#28606A"
    },

    "Dark Graphite": {
        "background": "#151719",
        "sidebar": "#202428",
        "card": "#2B3035",
        "primary": "#F59E0B",
        "secondary": "#FBBF24",
        "text": "#F8FAFC",
        "muted": "#CBD5E1",
        "border": "#454B51"
    }
}


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.markdown("# 💡 Smart Classroom")
st.sidebar.caption("Energy Monitoring & Motion Prediction")

st.sidebar.divider()

selected_theme = st.sidebar.selectbox(
    "🎨 Select Dashboard Theme",
    list(THEMES.keys()),
    index=1
)

theme = THEMES[selected_theme]


# Apply theme CSS
st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}

    [data-testid="stSidebar"] {{
        background-color: {theme["sidebar"]};
    }}

    [data-testid="stSidebar"] * {{
        color: {theme["text"]};
    }}

    h1, h2, h3, h4 {{
        color: {theme["text"]} !important;
    }}

    p, label, span {{
        color: {theme["text"]};
    }}

    div[data-testid="stMetric"] {{
        background-color: {theme["card"]};
        border: 1px solid {theme["border"]};
        padding: 18px;
        border-radius: 12px;
    }}

    div[data-testid="stMetricValue"] {{
        color: {theme["secondary"]};
    }}

    div[data-testid="stMetricLabel"] {{
        color: {theme["muted"]};
    }}

    div.stButton > button {{
        background-color: {theme["primary"]};
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: bold;
        min-height: 42px;
    }}

    div.stButton > button:hover {{
        background-color: {theme["secondary"]};
        color: #071525;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid {theme["border"]};
        border-radius: 10px;
    }}

    .dashboard-card {{
        background-color: {theme["card"]};
        border: 1px solid {theme["border"]};
        border-radius: 13px;
        padding: 20px;
        margin-bottom: 12px;
    }}

    .small-text {{
        color: {theme["muted"]};
        font-size: 13px;
    }}

    .status-on {{
        color: #34D399;
        font-weight: bold;
        font-size: 24px;
    }}

    .status-off {{
        color: #F87171;
        font-weight: bold;
        font-size: 24px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.markdown("### System Information")

st.sidebar.success("Machine Learning Model Loaded")

st.sidebar.write(f"**Model:** {model_name}")
st.sidebar.write(f"**Threshold:** {threshold:.4f}")
st.sidebar.write("**Prediction Target:** PIR Motion")

st.sidebar.divider()

st.sidebar.caption(
    "Smart Classroom Energy System | 2026"
)


# ============================================================
# 4. DASHBOARD HEADER
# ============================================================

st.title("💡 Smart Classroom Energy System")

st.markdown(
    """
    An intelligent monitoring dashboard using machine learning
    to predict classroom motion activity from environmental
    and time-based sensor measurements.
    """
)

st.caption(
    f"Model: {model_name} | "
    f"Last dashboard update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

st.divider()


# ============================================================
# 5. MODEL PERFORMANCE
# ============================================================

st.subheader("📊 Model Performance")

st.markdown(
    "Performance results from the held-out chronological test dataset."
)

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Accuracy",
    "89.83%"
)

metric2.metric(
    "Precision",
    "15.55%"
)

metric3.metric(
    "Recall",
    "62.71%"
)

metric4.metric(
    "F1 Score",
    "0.2492"
)

st.info(
    "The model detects approximately 62.71% of actual motion "
    "observations. Its low precision means it also produces "
    "many false motion alarms. Accuracy alone should not be "
    "used to judge this imbalanced dataset."
)


# ============================================================
# 6. CONFUSION MATRIX
# ============================================================

with st.expander("View Test Confusion Matrix and Evaluation"):

    confusion_matrix = pd.DataFrame(
        [
            [11592, 1206],
            [132, 222]
        ],
        index=["Actual: No Motion", "Actual: Motion"],
        columns=["Predicted: No Motion", "Predicted: Motion"]
    )

    st.write("Confusion Matrix")
    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.markdown(
        """
        **Interpretation:**

        - True negatives: 11,592
        - False positives: 1,206
        - False negatives: 132
        - True positives: 222

        The model correctly identified 222 motion observations
        out of 354 actual motion observations in the test set.
        """
    )


st.divider()


# ============================================================
# 7. SENSOR INPUTS
# ============================================================

st.subheader("🎛️ Classroom Sensor Measurements")

st.markdown(
    """
    Enter sensor readings below to test the trained model.
    These are manual inputs, not live sensor readings.
    """
)

col1, col2 = st.columns(2)

with col1:

    light_level = st.number_input(
        "💡 Light Level",
        min_value=0.0,
        value=300.0,
        step=10.0,
        help="Classroom light sensor reading."
    )

    temperature = st.number_input(
        "🌡️ Temperature (°C)",
        value=25.0,
        step=0.5
    )

    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0
    )

with col2:

    co2 = st.number_input(
        "🌫️ CO₂ Level",
        min_value=0.0,
        value=450.0,
        step=10.0,
        help="CO2 sensor measurement in the training dataset's units."
    )

    hour = st.slider(
        "🕒 Hour of the Day",
        min_value=0,
        max_value=23,
        value=10,
        help="Use 24-hour time format."
    )


st.divider()


# ============================================================
# 8. PREDICTION
# ============================================================

st.subheader("🤖 Machine Learning Prediction")

if st.button(
    "🔍 Predict Classroom Motion",
    use_container_width=True
):

    try:

        # Arrange inputs in the exact training feature order
        input_data = pd.DataFrame(
            [[
                light_level,
                temperature,
                humidity,
                co2,
                hour
            ]],
            columns=features
        )

        # Get probability for the positive class (motion = 1)
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            if 1 not in classes:
                raise ValueError(
                    "The trained model does not contain motion class 1."
                )

            motion_index = classes.index(1)

            motion_probability = float(
                probabilities[motion_index]
            )

            prediction = int(
                motion_probability >= threshold
            )

        else:

            prediction = int(model.predict(input_data)[0])
            motion_probability = None

        # Display results
        st.markdown("### Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.markdown(
                    '<div class="dashboard-card">'
                    '<div class="status-on">🟢 MOTION DETECTED</div>'
                    '<p class="small-text">'
                    'The model predicts classroom motion activity.'
                    '</p></div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="dashboard-card">'
                    '<div class="status-off">🔴 NO MOTION DETECTED</div>'
                    '<p class="small-text">'
                    'The model predicts no classroom motion activity.'
                    '</p></div>',
                    unsafe_allow_html=True
                )

        with result_col2:

            if motion_probability is not None:

                st.metric(
                    "Motion Probability",
                    f"{motion_probability * 100:.2f}%"
                )

                st.progress(
                    min(max(motion_probability, 0.0), 1.0)
                )

            else:

                st.metric(
                    "Predicted Class",
                    str(prediction)
                )

        # Save prediction history
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        history_record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Light Level": light_level,
            "Temperature": temperature,
            "Humidity": humidity,
            "CO2": co2,
            "Hour": hour,
            "Prediction": (
                "Motion Detected"
                if prediction == 1
                else "No Motion"
            ),
            "Motion Probability": (
                round(motion_probability * 100, 2)
                if motion_probability is not None
                else None
            )
        }

        st.session_state.prediction_history.append(
            history_record
        )

        st.success("Prediction completed successfully.")

        st.caption(
            "This prediction does not directly switch the classroom "
            "lights. Physical relay control and live sensor integration "
            "are separate system components."
        )

    except Exception as error:

        st.error("Prediction failed.")
        st.code(str(error))


st.divider()


# ============================================================
# 9. PREDICTION HISTORY
# ============================================================

st.subheader("📜 Prediction History")

if (
    "prediction_history" in st.session_state
    and st.session_state.prediction_history
):

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇️ Download Prediction History (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="smart_classroom_prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions yet. Enter sensor readings and run a prediction."
    )


# ============================================================
# 10. SYSTEM INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About This Smart Classroom System"):

    st.markdown(
        """
        **Project:** Smart Classroom Energy System

        **Machine Learning Model:** Logistic Regression
        (or the model name displayed in the sidebar).

        **Dataset:** KETI Smart Building System, Room 413.

        **Prediction Target:** PIR motion activity.

        **Input Features:**
        - Light level
        - Temperature
        - Humidity
        - CO₂
        - Hour of the day

        **Important limitations:**

        - The dashboard currently uses manual sensor inputs.
        - The model predicts motion activity, not actual light-switch status.
        - Live ESP8266 integration is not yet implemented here.
        - Relay-based physical light control is not implemented here.
        - Prediction history is stored in the current Streamlit session,
          not in a permanent database.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; padding:20px;">
        <p class="small-text">
        Smart Classroom Energy System |
        Machine Learning & IoT Project |
        2026
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
