
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# ==========================================================
# SMART CLASSROOM ENERGY SYSTEM
# Decision Tree Dashboard: LIGHT ON / LIGHT OFF
# ==========================================================

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

# ==========================================================
# DARK DASHBOARD DESIGN
# ==========================================================

st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: #F1F5F9;
}
[data-testid="stMetric"] {
    background-color: #1C2940;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #334155;
}
h1, h2, h3 {
    color: #F1F5F9;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD UPLOADED MODEL
# ==========================================================

MODEL_PATHS = [
    "models/smart_classroom_decision_tree_kaggle.joblib",
    "smart_classroom_decision_tree_kaggle(2).joblib",
    "smart_classroom_decision_tree_kaggle.joblib",
    "models/smart_classroom_decision_tree.joblib"
]

@st.cache_resource
def load_model():

    for path in MODEL_PATHS:

        if os.path.exists(path):

            saved_object = joblib.load(path)

            if isinstance(saved_object, dict):

                model = saved_object["model"]
                features = saved_object["features"]
                classes = saved_object.get(
                    "class_codes",
                    list(model.classes_)
                )

            else:
                model = saved_object
                features = [
                    "ambient_light_lux",
                    "motion_detected",
                    "temperature_celsius",
                    "occupancy_count",
                    "hour_of_day",
                    "day_of_week",
                    "time_of_day",
                    "weather_condition",
                    "special_event_flag",
                    "traffic_density",
                    "avg_pedestrian_speed"
                ]
                classes = list(model.classes_)

            return model, features, classes, path

    return None


loaded_model = load_model()

# ==========================================================
# HEADER
# ==========================================================

st.title("💡 Smart Classroom Energy System")

st.subheader("Intelligent Lighting Prediction Dashboard")

st.write(
    "Monitor classroom conditions and use the trained Decision Tree "
    "to predict the lighting action."
)

st.divider()

# ==========================================================
# SIDEBAR INPUTS
# ==========================================================

st.sidebar.title("⚙️ Classroom Sensor Inputs")

ambient_light_lux = st.sidebar.slider(
    "Ambient Light Level (Lux)",
    0, 1000, 150
)

motion_detected = st.sidebar.selectbox(
    "Motion Detected",
    [1, 0],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

temperature_celsius = st.sidebar.slider(
    "Temperature (°C)",
    -10.0, 60.0, 25.0
)

occupancy_count = st.sidebar.slider(
    "Occupancy Count",
    0, 100, 10
)

hour_of_day = st.sidebar.slider(
    "Hour of Day",
    0, 23, datetime.now().hour
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday",
     "Friday", "Saturday", "Sunday"]
)

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

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
    0.0, 100.0, 20.0
)

avg_pedestrian_speed = st.sidebar.slider(
    "Average Pedestrian Speed",
    0.0, 5.0, 1.2
)

# ==========================================================
# CLASS CODE MAPPING
# ==========================================================

st.sidebar.divider()
st.sidebar.subheader("💡 Lighting Class Mapping")

st.sidebar.warning(
    "Confirm the meanings of classes 0, 1, and 2 from your dataset "
    "documentation before using the dashboard for real control."
)

class_mapping = {}

if loaded_model is not None:

    model, features, classes, model_path = loaded_model

    for class_code in classes:

        class_mapping[class_code] = st.sidebar.selectbox(
            f"Class {class_code} means:",
            ["LIGHT OFF", "LIGHT ON"],
            key=f"class_mapping_{class_code}"
        )

# ==========================================================
# BUILD MODEL INPUT
# ==========================================================

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

# ==========================================================
# PREDICTION
# ==========================================================

if loaded_model is not None:

    model, features, classes, model_path = loaded_model

    input_df = pd.DataFrame([input_values])

    # Match the feature order expected by the trained model
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    predicted_status = class_mapping[prediction]

    # ======================================================
    # SENSOR METRICS
    # ======================================================

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

    # ======================================================
    # DISPLAY LIGHT ON / OFF
    # ======================================================

    if predicted_status == "LIGHT ON":

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
                The Decision Tree predicts that the classroom
                lighting should be switched ON.
            </h3>
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
                The Decision Tree predicts that the classroom
                lighting should be switched OFF.
            </h3>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Predicted original model class: {prediction}")

    # ======================================================
    # CONFIDENCE
    # ======================================================

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        confidence = max(probabilities) * 100

        st.subheader("📊 Prediction Confidence")

        st.progress(int(confidence))

        st.write(f"Model confidence: **{confidence:.2f}%**")

    # ======================================================
    # PREDICTION HISTORY
    # ======================================================

    if "history" not in st.session_state:

        st.session_state.history = []

    if st.sidebar.button("Save Current Prediction"):

        history_record = {
            "Timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Ambient Light (Lux)": ambient_light_lux,
            "Motion": motion_detected,
            "Temperature (°C)": temperature_celsius,
            "Occupancy": occupancy_count,
            "Hour": hour_of_day,
            "Original Model Class": prediction,
            "Lighting Status": predicted_status
        }

        st.session_state.history.append(history_record)

    st.divider()

    st.subheader("📋 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction History",
            data=csv_data,
            file_name="smart_classroom_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("Click 'Save Current Prediction' to record a prediction.")

    # ======================================================
    # MODEL DETAILS
    # ======================================================

    with st.expander("View Model Details"):

        st.write("Model file:", model_path)

        st.write("Algorithm: Decision Tree Classifier")

        st.write("Model features:")

        st.write(features)

        st.write("Original model classes:")

        st.write(classes)

        st.write("Configured class mapping:")

        st.write(class_mapping)

else:

    st.error(
        "Decision Tree model not found. Upload your .joblib model "
        "to the project folder or the models folder."
    )

    st.code(
        "models/smart_classroom_decision_tree_kaggle.joblib"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Smart Classroom Energy System | Decision Tree Classification "
    "| Intelligent Lighting Prediction"
)
