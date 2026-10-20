
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "smart_classroom_decision_tree.joblib"

st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="💡",
    layout="wide"
)

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]

st.title("💡 Smart Classroom Energy System")
st.caption("AI-assisted classroom lighting recommendations")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model, features = load_model()
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Classroom Sensor Inputs")

lux = st.sidebar.slider(
    "Light level (lux)", 0, 1000, 180, 10
)
motion = st.sidebar.selectbox(
    "Motion detected?", ["Yes", "No"]
)
hour = st.sidebar.slider(
    "Hour of day", 0, 23, 10
)
scheduled_day = st.sidebar.selectbox(
    "Scheduled school day?", ["Yes", "No"]
)

motion_value = 1 if motion == "Yes" else 0
schedule_value = 1 if scheduled_day == "Yes" else 0

values = {
    "lux": lux,
    "motion": motion_value,
    "hour": hour,
    "scheduled_day": schedule_value
}

try:
    input_data = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )
    prediction = str(model.predict(input_data)[0]).upper()
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

if prediction not in ["ON", "OFF"]:
    st.warning(f"The model returned: {prediction}")
    st.info("Check the label format used when training your model.")
    st.stop()

left, middle, right = st.columns(3)

left.metric("Light level", f"{lux} lux")
middle.metric("Motion", motion)
right.metric("School day", scheduled_day)

st.subheader("Recommended Lighting Action")

if prediction == "ON":
    st.success("💡 LIGHTS ON recommended")
else:
    st.info("🌙 LIGHTS OFF recommended")

st.caption(
    "This is a model recommendation, not a physical light controller. "
    "No real lights or sensors are connected."
)

if st.button("Save this prediction", type="primary"):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Light level (lux)": lux,
        "Motion": motion,
        "Hour": hour,
        "Scheduled school day": scheduled_day,
        "Recommendation": prediction
    }
    st.session_state.history.append(record)
    st.success("Prediction saved to this session.")

st.divider()
st.subheader("Prediction History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    counts = (
        history_df["Recommendation"]
        .value_counts()
        .rename_axis("Recommendation")
        .reset_index(name="Count")
    )
    fig = px.bar(
        counts,
        x="Recommendation",
        y="Count",
        title="Saved Lighting Recommendations"
    )
    st.plotly_chart(fig, use_container_width=True)

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv_data,
        file_name="classroom_prediction_history.csv",
        mime="text/csv"
    )
else:
    st.info("No predictions saved yet. Change the inputs and save a prediction.")

st.divider()
st.subheader("Estimated Energy Use")

st.write(
    "Enter your classroom's light power and daily operating hours "
    "to estimate energy consumption. These figures are estimates, "
    "not measured electricity use."
)

col1, col2, col3 = st.columns(3)

with col1:
    power_w = st.number_input(
        "Total lighting power (watts)",
        min_value=1.0,
        max_value=100000.0,
        value=300.0,
        step=10.0
    )

with col2:
    operating_hours = st.number_input(
        "Lighting operating hours per day",
        min_value=0.0,
        max_value=24.0,
        value=8.0,
        step=0.5
    )

with col3:
    days_per_month = st.number_input(
        "Operating days per month",
        min_value=1,
        max_value=31,
        value=22
    )

baseline_kwh = power_w * operating_hours * days_per_month / 1000

st.metric("Estimated baseline monthly use", f"{baseline_kwh:.2f} kWh")

if st.session_state.history:
    saved_df = pd.DataFrame(st.session_state.history)
    on_count = (saved_df["Recommendation"] == "ON").sum()
    total_count = len(saved_df)

    # Illustrative scenario only: assumes each saved prediction
    # represents one equal-duration observation.
    on_fraction = on_count / total_count
    scenario_kwh = baseline_kwh * on_fraction
    st.metric(
        "Illustrative use if ON observations represent operating time",
        f"{scenario_kwh:.2f} kWh"
    )
    st.caption(
        "This scenario is not a verified savings measurement. "
        "Reliable energy estimates require observation durations, "
        "actual lamp power, and a defined baseline."
    )

st.caption("Smart Classroom Energy System | Prototype dashboard")
