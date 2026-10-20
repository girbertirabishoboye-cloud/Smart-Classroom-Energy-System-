from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Classroom Energy System", page_icon="💡", layout="centered")

st.title("💡 Smart Classroom Energy System")
st.subheader("Decision Tree Classification")
st.write(
    "Enter classroom sensor readings to receive a recommended lighting action. "
    "This demonstration predicts ON/OFF; it does not directly control physical lights."
)

MODEL_PATH = Path(__file__).parent / "models" / "smart_classroom_decision_tree.joblib"

@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("Trained model file not found. Run the Jupyter training notebook first and save the model into the models folder.")
    st.stop()

artifact = load_artifact(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

st.info(
    "Demo dataset notice: the included model was trained on synthetic data and a predefined rule. "
    "Validate and retrain it with real classroom sensor readings before relying on it."
)

with st.form("classroom_inputs"):
    light_level = st.slider("Ambient light level (lux)", min_value=0, max_value=1000, value=180, step=10,
                            help="Use the measured ambient light level from a lux sensor.")
    motion = st.selectbox("Motion detected?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    hour = st.slider("Hour of day (24-hour clock)", min_value=0, max_value=23, value=10,
                     help="For example, 14 means 2:00 PM.")
    scheduled = st.selectbox("Is the classroom scheduled for use today?", options=[("Yes", 1), ("No", 0)],
                             format_func=lambda x: x[0])
    submitted = st.form_submit_button("Predict lighting action", type="primary")

if submitted:
    values = {
        "light_level_lux": light_level,
        "motion_detected": motion[1],
        "hour_of_day": hour,
        "scheduled_day": scheduled[1],
    }
    input_df = pd.DataFrame([{feature: values[feature] for feature in features}])
    prediction = model.predict(input_df)[0]

    if prediction == "ON":
        st.success("Recommended action: SWITCH LIGHTS ON")
        st.caption("The model predicts that the classroom lighting is needed under the learned policy.")
    else:
        st.warning("Recommended action: SWITCH LIGHTS OFF")
        st.caption("The model predicts that classroom lighting is not needed under the learned policy.")

    st.write("Input values used:")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        st.write("Model class probabilities (not a guarantee of real-world correctness):")
        st.dataframe(pd.DataFrame({"Action": model.classes_, "Probability": probabilities}),
                     use_container_width=True, hide_index=True)

st.divider()
st.markdown("**Model:** Decision Tree Classifier  \n**Inputs:** Light level, motion, time, scheduled day  \n**Output:** `ON` or `OFF`")
st.caption("Safety: this app only provides a recommendation. Physical lighting control requires a separate, properly isolated and rated hardware interface, manual override, and installation by a qualified person.")
