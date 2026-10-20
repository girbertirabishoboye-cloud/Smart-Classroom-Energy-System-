
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Classroom | Energy Management",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM PROFESSIONAL DESIGN
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    [data-testid="stSidebar"] {
        background-color: #101b2d;
    }

    [data-testid="stSidebar"] * {
        color: #f2f6fc;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #10243e, #174b70);
        padding: 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 5px 18px rgba(16, 36, 62, 0.12);
    }

    .hero h1 {
        color: white;
        font-size: 31px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #dce9f5;
        margin-bottom: 0;
        font-size: 15px;
    }

    .section-heading {
        color: #172b45;
        font-size: 22px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 14px;
    }

    .recommendation {
        padding: 23px;
        border-radius: 15px;
        margin: 8px 0 15px 0;
        border: 1px solid #dce4ed;
    }

    .recommendation-on {
        background: #e6f7ed;
        border-left: 6px solid #159957;
    }

    .recommendation-off {
        background: #edf2f8;
        border-left: 6px solid #56728f;
    }

    .recommendation h2 {
        margin: 0 0 8px 0;
        color: #172b45;
        font-size: 25px;
    }

    .recommendation p {
        margin: 0;
        color: #43566d;
        font-size: 14px;
    }

    .small-note {
        color: #68788b;
        font-size: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e7ef;
        padding: 17px;
        border-radius: 13px;
        box-shadow: 0 2px 7px rgba(20, 40, 70, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #53657a;
    }

    .footer {
        text-align: center;
        color: #78879a;
        font-size: 12px;
        padding-top: 18px;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. FIND AND LOAD THE MODEL
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_LOCATIONS = [
    PROJECT_DIR / "models" / "smart_classroom_decision_tree.joblib",
    PROJECT_DIR / "smart_classroom_decision_tree.joblib",
]

MODEL_PATH = next(
    (path for path in MODEL_LOCATIONS if path.is_file()),
    MODEL_LOCATIONS[0]
)


@st.cache_resource
def load_model(model_path, modified_time):
    saved = joblib.load(model_path)

    if isinstance(saved, dict):
        model = saved["model"]
        features = list(saved["features"])
    else:
        model = saved
        features = list(
            getattr(model, "feature_names_in_", [])
        )

        if not features:
            features = ["lux", "motion", "hour"]

    return model, features


# =========================================================
# 4. HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>💡 Smart Classroom Energy System</h1>
    <p>
        Intelligent lighting recommendations, classroom monitoring
        and energy-use analysis in one dashboard.
    </p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.is_file():
    st.error("The trained model file could not be found.")
    st.write("Expected locations:")
    for path in MODEL_LOCATIONS:
        st.code(str(path))
    st.info(
        "Ensure the model file is committed to your GitHub "
        "repository and deployed with this application."
    )
    st.stop()

try:
    model, features = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# =========================================================
# 5. SIDEBAR: CLASSROOM INPUTS
# =========================================================
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.caption("Configure the classroom conditions.")

st.sidebar.divider()

lux = st.sidebar.slider(
    "Ambient light level (lux)",
    min_value=0,
    max_value=1000,
    value=180,
    step=10,
    help="Lower values indicate a darker classroom."
)

motion = st.sidebar.selectbox(
    "Motion detected",
    ["Yes", "No"]
)

hour = st.sidebar.slider(
    "Hour of day",
    min_value=0,
    max_value=23,
    value=10,
    help="Use 24-hour time: 0 = midnight, 12 = noon."
)

scheduled_day = st.sidebar.selectbox(
    "Scheduled school day",
    ["Yes", "No"]
)

st.sidebar.divider()

st.sidebar.markdown("### Model status")
st.sidebar.success("Model loaded")

st.sidebar.caption(
    f"Model file: {MODEL_PATH.name}"
)

st.sidebar.caption(
    "Prototype mode · Manual sensor inputs"
)

# =========================================================
# 6. PREDICT LIGHTING ACTION
# =========================================================
motion_value = 1 if motion == "Yes" else 0
schedule_value = 1 if scheduled_day == "Yes" else 0

values = {
    "lux": lux,
    "motion": motion_value,
    "hour": hour,
    "scheduled_day": schedule_value,
}

try:
    missing_features = [
        feature for feature in features
        if feature not in values
    ]

    if missing_features:
        st.error(
            "The model requires unsupported inputs: "
            + ", ".join(missing_features)
        )
        st.stop()

    input_data = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features
    )

    prediction = str(model.predict(input_data)[0]).upper()

except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

if prediction not in ["ON", "OFF"]:
    st.error(
        f"Unexpected model output: {prediction}. "
        "Check the labels used during training."
    )
    st.stop()

# =========================================================
# 7. SESSION HISTORY
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# 8. OVERVIEW METRICS
# =========================================================
st.markdown(
    '<div class="section-heading">Classroom Overview</div>',
    unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric(
    "Ambient Light",
    f"{lux} lux",
    help="Current manually entered light-level reading."
)

kpi2.metric(
    "Motion Status",
    "Detected" if motion == "Yes" else "Not detected"
)

kpi3.metric(
    "School Schedule",
    "Active" if scheduled_day == "Yes" else "Inactive"
)

kpi4.metric(
    "Recommended Action",
    "LIGHTS ON" if prediction == "ON" else "LIGHTS OFF"
)

# =========================================================
# 9. LIGHTING RECOMMENDATION
# =========================================================
st.markdown(
    '<div class="section-heading">AI Lighting Recommendation</div>',
    unsafe_allow_html=True
)

if prediction == "ON":
    st.markdown("""
    <div class="recommendation recommendation-on">
        <h2>💡 Lights ON recommended</h2>
        <p>
            The trained Decision Tree model recommends turning
            the classroom lights on under the current inputs.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="recommendation recommendation-off">
        <h2>🌙 Lights OFF recommended</h2>
        <p>
            The trained Decision Tree model recommends keeping
            the classroom lights off under the current inputs.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.caption(
    "This recommendation is based on the trained model and "
    "provided inputs. It is not a command sent to a physical light."
)

# =========================================================
# 10. SAVE A PREDICTION
# =========================================================
st.markdown(
    '<div class="section-heading">Record This Observation</div>',
    unsafe_allow_html=True
)

save_col, info_col = st.columns([1, 2])

with save_col:
    if st.button(
        "＋ Save current prediction",
        type="primary",
        use_container_width=True
    ):
        record = {
            "Timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Light level (lux)": lux,
            "Motion": motion,
            "Hour": hour,
            "Scheduled school day": scheduled_day,
            "Recommendation": prediction,
        }

        st.session_state.history.append(record)
        st.success("Observation saved to this session.")

with info_col:
    st.info(
        "Saved observations remain in this Streamlit session. "
        "They are not permanently stored in a database."
    )

# =========================================================
# 11. ANALYTICS
# =========================================================
st.divider()

st.markdown(
    '<div class="section-heading">Monitoring & Analytics</div>',
    unsafe_allow_html=True
)

tab_history, tab_energy, tab_about = st.tabs([
    "📊 Prediction History",
    "⚡ Energy Analysis",
    "ℹ️ About the System"
])

with tab_history:
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        total = len(history_df)
        on_count = int(
            (history_df["Recommendation"] == "ON").sum()
        )
        off_count = total - on_count

        a, b, c = st.columns(3)
        a.metric("Total observations", total)
        b.metric("ON recommendations", on_count)
        c.metric("OFF recommendations", off_count)

        counts = pd.DataFrame({
            "Recommendation": ["ON", "OFF"],
            "Count": [on_count, off_count]
        })

        fig = px.bar(
            counts,
            x="Recommendation",
            y="Count",
            color="Recommendation",
            color_discrete_map={
                "ON": "#159957",
                "OFF": "#56728f"
            },
            text="Count",
            title="Distribution of Lighting Recommendations"
        )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_title="Recommended action",
            yaxis_title="Number of observations",
            margin=dict(l=15, r=15, t=60, b=15)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Observation Records")
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        download_col, clear_col = st.columns(2)

        with download_col:
            st.download_button(
                "⬇ Download observations (CSV)",
                data=csv_data,
                file_name="classroom_prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )

        with clear_col:
            if st.button(
                "Clear session history",
                use_container_width=True
            ):
                st.session_state.history = []
                st.rerun()

    else:
        st.info(
            "No observations have been saved yet. "
            "Adjust the classroom inputs and save a prediction "
            "to populate the analytics."
        )

with tab_energy:
    st.markdown("#### Monthly Lighting Consumption Estimate")

    st.write(
        "Enter the lighting specifications to estimate "
        "the energy used if the lights operate for the "
        "specified duration."
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        power_w = st.number_input(
            "Total lighting power (W)",
            min_value=1.0,
            max_value=100000.0,
            value=300.0,
            step=10.0
        )

    with e2:
        operating_hours = st.number_input(
            "Operating hours per day",
            min_value=0.0,
            max_value=24.0,
            value=8.0,
            step=0.5
        )

    with e3:
        days_per_month = st.number_input(
            "Operating days per month",
            min_value=1,
            max_value=31,
            value=22
        )

    baseline_kwh = (
        power_w * operating_hours * days_per_month / 1000
    )

    st.metric(
        "Estimated monthly energy",
        f"{baseline_kwh:.2f} kWh"
    )

    st.caption(
        "Formula: Energy (kWh) = Power (W) × Hours per day "
        "× Days per month ÷ 1,000."
    )

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        on_count = int(
            (history_df["Recommendation"] == "ON").sum()
        )
        total = len(history_df)

        st.markdown("#### Illustrative Operating Scenario")

        st.write(
            f"{on_count} of {total} saved observations "
            "recommended ON."
        )

        st.info(
            "A percentage of ON predictions is not a reliable "
            "measurement of energy savings unless the observations "
            "represent equal, known periods of time. Real savings "
            "require measured operating durations and a baseline."
        )

with tab_about:
    st.markdown("#### Project Information")

    st.write(
        "The Smart Classroom Energy System is a prototype "
        "that uses a trained Decision Tree model to recommend "
        "whether classroom lighting should be ON or OFF."
    )

    st.markdown("**Current capabilities**")
    st.markdown("""
    - Manual entry of classroom light level
    - Motion and school-schedule inputs
    - Model-based lighting recommendations
    - Session-based observation history
    - Interactive charts and CSV export
    - Estimated lighting energy consumption
    """)

    st.markdown("**Current limitations**")
    st.markdown("""
    - No physical sensors are connected.
    - No ESP32 or relay is controlling a light.
    - The training data may be simulated.
    - Session history is not saved permanently.
    """)

    st.warning(
        "If the model was trained on synthetic data, its "
        "predictions must be validated against real classroom "
        "measurements before practical deployment."
    )

# =========================================================
# 12. FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    Smart Classroom Energy System · AI-Assisted Energy Management
    <br>
    Prototype dashboard · Recommendations only · No physical
    control connected
</div>
""", unsafe_allow_html=True)
