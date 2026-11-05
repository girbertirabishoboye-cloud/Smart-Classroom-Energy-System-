
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# =========================================================
# SMART CLASSROOM ENERGY SYSTEM
# Machine Learning-Based Lighting Dashboard
# =========================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

# ---------------------------------------------------------
# DASHBOARD THEME
# ---------------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
[data-testid="stMetric"] {
    background-color: #1C2940;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #334155;
}
h1, h2, h3 {
    color: #F1F5F9;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

MODEL_PATH = "models/smart_classroom_decision_tree.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    loaded = joblib.load(MODEL_PATH)

    # Supports a saved dictionary containing model and features
    if isinstance(loaded, dict):
        model = loaded.get("model")
        features = loaded.get("features")
    else:
        model = loaded
        features = ["lux", "motion", "hour"]

    return model, features


loaded_model = load_model()

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("💡 Smart Classroom Energy System")
st.markdown("""
### Intelligent Lighting Prediction Dashboard
Monitor classroom conditions and predict whether lighting should be switched ON or OFF.
""")

st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("⚙️ Classroom Controls")

lux = st.sidebar.slider(
    "Ambient Light Level (Lux)",
    min_value=0,
    max_value=1000,
    value=150
)

motion = st.sidebar.selectbox(
    "Motion Detected",
    options=[1, 0],
    format_func=lambda x: "Motion Detected" if x == 1 else "No Motion"
)

hour = st.sidebar.slider(
    "Current Hour",
    min_value=0,
    max_value=23,
    value=datetime.now().hour
)

st.sidebar.caption(
    "Adjust the sensor inputs to see the predicted lighting action."
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

if loaded_model is not None:

    model, features = loaded_model

    input_data = pd.DataFrame(
        [[lux, motion, hour]],
        columns=["lux", "motion", "hour"]
    )

    # Reorder input columns to match model features
    input_data = input_data[features]

    prediction = model.predict(input_data)[0]

    # Convert output into readable ON/OFF
    if str(prediction).upper() in ["ON", "1", "LIGHT ON"]:
        light_status = "ON"
    else:
        light_status = "OFF"

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100

    # -----------------------------------------------------
    # STATUS CARDS
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Ambient Light",
            value=f"{lux} Lux"
        )

    with col2:
        st.metric(
            label="Motion",
            value="DETECTED" if motion == 1 else "NOT DETECTED"
        )

    with col3:
        st.metric(
            label="Current Hour",
            value=f"{hour}:00"
        )

    st.divider()

    # -----------------------------------------------------
    # MAIN LIGHT STATUS DISPLAY
    # -----------------------------------------------------

    if light_status == "ON":

        st.markdown("""
        <div style="
            background-color:#064E3B;
            padding:35px;
            border-radius:20px;
            text-align:center;
            border:3px solid #10B981;
        ">
            <h1 style="color:#6EE7B7;font-size:50px;">
                💡 LIGHT ON
            </h1>
            <h3 style="color:white;">
                The Decision Tree recommends switching the classroom light ON.
            </h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background-color:#7F1D1D;
            padding:35px;
            border-radius:20px;
            text-align:center;
            border:3px solid #EF4444;
        ">
            <h1 style="color:#FCA5A5;font-size:50px;">
                🌙 LIGHT OFF
            </h1>
            <h3 style="color:white;">
                The Decision Tree recommends switching the classroom light OFF.
            </h3>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------

    if confidence is not None:
        st.subheader("Prediction Confidence")
        st.progress(int(confidence))
        st.write(f"Model confidence: **{confidence:.2f}%**")

    # -----------------------------------------------------
    # PREDICTION HISTORY
    # -----------------------------------------------------

    if "history" not in st.session_state:
        st.session_state.history = []

    history_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Lux": lux,
        "Motion": "Detected" if motion == 1 else "Not Detected",
        "Hour": hour,
        "Predicted Lighting Status": f"LIGHT {light_status}"
    }

    st.session_state.history.append(history_record)

    # Avoid unlimited history growth
    st.session_state.history = st.session_state.history[-100:]

    st.divider()

    st.subheader("📊 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="smart_classroom_prediction_history.csv",
        mime="text/csv"
    )

else:

    st.error(
        "Decision Tree model not found. Please ensure the model file exists at:"
    )

    st.code("models/smart_classroom_decision_tree.joblib")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption("""
Smart Classroom Energy System | Machine Learning | Decision Tree Classification
""")
