
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Classroom Energy System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROFESSIONAL EMERALD, CHARCOAL & GOLD THEME
# =========================================================
st.markdown(
    """
    <style>
    :root {
        --background: #141A17;
        --surface: #1D2721;
        --surface-2: #25332A;
        --border: #3A5142;
        --green: #45A36B;
        --green-light: #78D99A;
        --gold: #E7C76A;
        --text: #F3F6F1;
        --muted: #B6C5B9;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        background: #1A241E;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .stApp h1, .stApp h2, .stApp h3,
    .stApp p, .stApp label {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(120deg, #1E4931 0%, #286B43 58%, #354F35 100%);
        border: 1px solid #548B62;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.2;
        margin: 8px 0 12px 0;
    }

    .hero p {
        color: #E0EEE2;
        font-size: 15px;
        line-height: 1.7;
        margin: 0;
    }

    .eyebrow {
        color: var(--gold);
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .section-heading {
        color: var(--gold);
        font-size: 21px;
        font-weight: 750;
        margin: 16px 0 12px 0;
    }

    .small-muted {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 120px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        color: var(--gold) !important;
    }

    .recommendation {
        border-radius: 16px;
        padding: 23px;
        margin: 8px 0 16px 0;
        background: var(--surface);
    }

    .recommendation-on {
        border: 1px solid #43865A;
        border-left: 6px solid var(--green-light);
        background: #20392A;
    }

    .recommendation-off {
        border: 1px solid #827548;
        border-left: 6px solid var(--gold);
        background: #302F25;
    }

    .recommendation h2 {
        color: #FFFFFF;
        font-size: 25px;
        margin: 0 0 8px 0;
    }

    .recommendation p {
        color: #D5E0D6;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }

    .status-pill {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.09);
        color: var(--gold);
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: #347C50;
        color: #FFFFFF;
        border: 1px solid #68A779;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: #438F60;
        color: #FFFFFF;
        border-color: var(--gold);
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"],
    .stTextInput div[data-baseweb="input"] {
        background: #25332A;
        border-color: #526B57;
    }

    .stSelectbox div[data-baseweb="select"] *,
    .stNumberInput input,
    .stTextInput input {
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        gap: 5px;
        padding: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #D4E0D5;
        border-radius: 8px;
    }

    .stTabs [aria-selected="true"] {
        background: #347C50 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        color: #AAB9AD;
        text-align: center;
        font-size: 12px;
        line-height: 1.8;
        padding: 25px 0 8px 0;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .hero h1 {
            font-size: 26px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD THE EXISTING TRAINED MACHINE-LEARNING MODEL
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_LOCATIONS = [
    PROJECT_DIR / "models" / "smart_classroom_decision_tree.joblib",
    PROJECT_DIR / "smart_classroom_decision_tree.joblib",
]

MODEL_PATH = next(
    (path for path in MODEL_LOCATIONS if path.is_file()),
    None,
)


@st.cache_resource
def load_model(model_path, modified_time):
    saved = joblib.load(model_path)

    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]
        features = list(saved.get("features", []))
    else:
        model = saved
        features = list(
            getattr(model, "feature_names_in_", [])
        )

    if not features:
        features = ["lux", "motion", "hour"]

    return model, features


if MODEL_PATH is None:
    st.error(
        "The trained model could not be found. Upload "
        "'smart_classroom_decision_tree.joblib' to the root "
        "of your GitHub repository, alongside app.py."
    )
    st.code(
        "Smart-Classroom-Energy-System-/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── smart_classroom_decision_tree.joblib"
    )
    st.stop()


try:
    model, model_features = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error(
        "The model file exists, but it could not be loaded. "
        "Check that it was saved correctly and that the "
        "scikit-learn version is compatible."
    )
    st.caption(f"Technical detail: {exc}")
    st.stop()


# =========================================================
# 4. HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART CAMPUS • INTELLIGENT ENERGY</div>
        <h1>Smart Classroom Energy System</h1>
        <p>
            An intelligent classroom lighting dashboard that uses a
            trained machine-learning model to recommend whether lights
            should be ON or OFF based on environmental and occupancy inputs.
            Monitor predictions, explore usage history, and review estimated
            energy costs from one dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<p class="small-muted">DECISION SUPPORT • MACHINE LEARNING • ENERGY AWARENESS</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        f'<p style="text-align:right;color:#B6C5B9;font-size:12px;">'
        f'Last page load: {datetime.now().strftime("%d %b %Y, %H:%M")}'
        f'</p>',
        unsafe_allow_html=True,
    )


# =========================================================
# 5. SIDEBAR INPUTS
# =========================================================
st.sidebar.markdown("## ⚙️ Control Centre")
st.sidebar.caption("Adjust the classroom conditions, then run a prediction.")
st.sidebar.divider()

with st.sidebar.form("classroom_input_form"):
    st.markdown("### Environmental inputs")

    lux = st.slider(
        "Light intensity (lux)",
        min_value=0,
        max_value=1000,
        value=180,
        step=10,
        help="Lower lux means the classroom is darker.",
    )

    motion_label = st.selectbox(
        "Motion detected?",
        options=["Yes", "No"],
        index=0,
    )
    motion = 1 if motion_label == "Yes" else 0

    hour = st.slider(
        "Hour of the day",
        min_value=0,
        max_value=23,
        value=10,
        help="24-hour clock: 10 means 10:00 AM.",
    )

    scheduled_label = st.selectbox(
        "Scheduled school day?",
        options=["Yes", "No"],
        index=0,
    )
    scheduled_day = 1 if scheduled_label == "Yes" else 0

    st.markdown("### Energy estimate settings")

    bulb_power = st.number_input(
        "Lighting power (watts)",
        min_value=1,
        max_value=5000,
        value=100,
        step=10,
        help="Estimated total power of the lighting being assessed.",
    )

    hours_per_day = st.slider(
        "Potential lighting hours per day",
        min_value=1,
        max_value=16,
        value=8,
    )

    days_per_month = st.slider(
        "Operating days per month",
        min_value=1,
        max_value=31,
        value=22,
    )

    electricity_rate = st.number_input(
        "Electricity rate (RWF per kWh)",
        min_value=0.0,
        max_value=10000.0,
        value=300.0,
        step=25.0,
        help="Replace the example with your applicable tariff.",
    )

    submitted = st.form_submit_button(
        "⚡ Run Prediction",
        use_container_width=True,
    )

st.sidebar.divider()
st.sidebar.markdown("### Model information")
st.sidebar.caption("Algorithm: saved machine-learning classifier")
st.sidebar.caption(f"Input features: {', '.join(model_features)}")
st.sidebar.caption(f"Model file: {MODEL_PATH.name}")


# =========================================================
# 6. PREPARE INPUTS AND RUN PREDICTION
# =========================================================
available_values = {
    "lux": lux,
    "motion": motion,
    "hour": hour,
    "scheduled_day": scheduled_day,
}

# Build the input in the exact feature order used during training.
try:
    input_data = pd.DataFrame(
        [{
            feature: available_values[feature]
            for feature in model_features
        }],
        columns=model_features,
    )

    raw_prediction = model.predict(input_data)[0]
    prediction = str(raw_prediction).strip().upper()

    if prediction not in ("ON", "OFF"):
        st.error(
            f"The model returned '{raw_prediction}' instead of ON or OFF. "
            "Check the labels used when training the model."
        )
        st.stop()

except Exception as exc:
    st.error(
        "Prediction failed. The feature names expected by the model "
        "may not match the dashboard inputs."
    )
    st.code(f"Model features: {model_features}")
    st.caption(f"Technical detail: {exc}")
    st.stop()


# Keep history in this browser session.
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = prediction

if submitted:
    st.session_state.last_prediction = prediction

    st.session_state.prediction_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Light intensity (lux)": lux,
        "Motion": motion_label,
        "Hour": hour,
        "Scheduled day": scheduled_label,
        "Prediction": prediction,
        "Lighting power (W)": bulb_power,
    })

display_prediction = st.session_state.last_prediction


# =========================================================
# 7. MAIN PREDICTION RESULT
# =========================================================
st.markdown(
    '<div class="section-heading">01 / Live classroom recommendation</div>',
    unsafe_allow_html=True,
)

if display_prediction == "ON":
    st.markdown(
        """
        <div class="recommendation recommendation-on">
            <div class="status-pill">LIGHTING RECOMMENDED</div>
            <h2>💡 Lights should be ON</h2>
            <p>
                According to the trained model, the current classroom
                conditions support switching the lights on. Review the
                sensor inputs and confirm the recommendation before acting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="recommendation recommendation-off">
            <div class="status-pill">ENERGY-SAVING OPPORTUNITY</div>
            <h2>🌙 Lights should be OFF</h2>
            <p>
                According to the trained model, the current conditions
                support keeping the lights off. Verify that natural light
                and classroom safety requirements are sufficient.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not submitted:
    st.info(
        "Showing the model's result for the default inputs. Adjust the "
        "controls in the sidebar and select **Run Prediction** to test "
        "another classroom scenario."
    )


# =========================================================
# 8. KPI CARDS
# =========================================================
monthly_energy_kwh = (
    bulb_power * hours_per_day * days_per_month / 1000
)
monthly_cost = monthly_energy_kwh * electricity_rate
daily_energy_kwh = bulb_power * hours_per_day / 1000

if display_prediction == "OFF":
    potential_energy_kwh = monthly_energy_kwh
    potential_cost = monthly_cost
else:
    potential_energy_kwh = 0
    potential_cost = 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "Model recommendation",
        display_prediction,
        help="Output from the saved machine-learning model.",
    )

with kpi2:
    st.metric(
        "Light intensity",
        f"{lux} lux",
        help="Current value entered in the dashboard.",
    )

with kpi3:
    st.metric(
        "Motion status",
        motion_label,
        help="Occupancy indicator entered by the user.",
    )

with kpi4:
    st.metric(
        "Lighting power",
        f"{bulb_power} W",
        help="Configured power used for the energy estimate.",
    )


# =========================================================
# 9. TABS: ANALYTICS, HISTORY, MODEL DETAILS
# =========================================================
tab_analytics, tab_history, tab_about = st.tabs([
    "📊 Energy Analytics",
    "🗂️ Prediction History",
    "🧠 Model & Project",
])


# =========================================================
# TAB 1: ENERGY ANALYTICS
# =========================================================
with tab_analytics:
    st.markdown(
        '<div class="section-heading">Energy use and cost estimates</div>',
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric(
            "Daily energy if ON",
            f"{daily_energy_kwh:.2f} kWh",
        )

    with a2:
        st.metric(
            "Monthly energy if ON",
            f"{monthly_energy_kwh:.2f} kWh",
        )

    with a3:
        st.metric(
            "Estimated monthly cost if ON",
            f"{monthly_cost:,.0f} RWF",
        )

    st.markdown(
        """
        <div class="panel">
            <strong>How to interpret these estimates</strong>
            <p class="small-muted">
                These calculations assume the configured lighting load runs
                for the selected hours and operating days. If the model
                recommends OFF, the displayed potential energy and cost
                represent a hypothetical upper estimate of energy avoided
                if that load would otherwise have run for the entire period.
                This is not a measured saving or a prediction of actual
                monthly performance.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    energy_data = pd.DataFrame({
        "Scenario": [
            "Lights ON for configured hours",
            "Hypothetical energy avoided if OFF",
        ],
        "Energy (kWh)": [
            monthly_energy_kwh,
            potential_energy_kwh,
        ],
    })

    fig_energy = px.bar(
        energy_data,
        x="Scenario",
        y="Energy (kWh)",
        color="Scenario",
        text_auto=".2f",
        title="Monthly energy scenario comparison",
        color_discrete_sequence=["#45A36B", "#E7C76A"],
    )

    fig_energy.update_layout(
        paper_bgcolor="#1D2721",
        plot_bgcolor="#1D2721",
        font_color="#F3F6F1",
        title_font_color="#E7C76A",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig_energy.update_xaxes(
        gridcolor="#35483A",
        tickfont_color="#D5E0D6",
    )
    fig_energy.update_yaxes(
        gridcolor="#35483A",
        title_font_color="#B6C5B9",
    )

    st.plotly_chart(fig_energy, use_container_width=True)

    st.caption(
        f"At the entered tariff, the configured lighting load would cost "
        f"approximately {monthly_cost:,.0f} RWF per month if it ran for "
        f"all configured hours and days."
    )


# =========================================================
# TAB 2: PREDICTION HISTORY
# =========================================================
with tab_history:
    st.markdown(
        '<div class="section-heading">Prediction records</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.prediction_history

    if history:
        history_df = pd.DataFrame(history)

        h1, h2 = st.columns(2)

        with h1:
            st.metric("Predictions recorded", len(history_df))

        with h2:
            off_count = int((history_df["Prediction"] == "OFF").sum())
            st.metric("OFF recommendations", off_count)

        st.dataframe(
            history_df.iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download history as CSV",
            data=csv_data,
            file_name="smart_classroom_prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        prediction_counts = (
            history_df["Prediction"]
            .value_counts()
            .rename_axis("Recommendation")
            .reset_index(name="Count")
        )

        fig_history = px.pie(
            prediction_counts,
            names="Recommendation",
            values="Count",
            title="Recorded ON and OFF recommendations",
            color="Recommendation",
            color_discrete_map={
                "ON": "#45A36B",
                "OFF": "#E7C76A",
            },
            hole=0.55,
        )

        fig_history.update_layout(
            paper_bgcolor="#1D2721",
            plot_bgcolor="#1D2721",
            font_color="#F3F6F1",
            title_font_color="#E7C76A",
        )

        st.plotly_chart(fig_history, use_container_width=True)

        if st.button("Clear session history", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

        st.caption(
            "History is stored in Streamlit's current session. It is not "
            "a permanent database and may be lost when the session resets."
        )
    else:
        st.info(
            "No prediction history yet. Use the sidebar and click "
            "'Run Prediction' to create your first record."
        )


# =========================================================
# TAB 3: MODEL AND PROJECT DETAILS
# =========================================================
with tab_about:
    st.markdown(
        '<div class="section-heading">About this system</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
            <h3>Project objective</h3>
            <p class="small-muted">
                Demonstrate how machine learning can support classroom
                lighting decisions using light intensity, motion and time
                of day. The dashboard presents a recommendation rather
                than directly controlling electrical equipment.
            </p>
        </div>

        <div class="panel">
            <h3>System workflow</h3>
            <p class="small-muted">
                1. Enter classroom conditions.<br>
                2. Arrange the input data using the model's saved feature names.<br>
                3. Run the trained classifier.<br>
                4. Display an ON/OFF recommendation.<br>
                5. Review energy estimates and export the recorded predictions.
            </p>
        </div>

        <div class="panel">
            <h3>Current limitations</h3>
            <p class="small-muted">
                This prototype uses manually entered inputs. No physical
                sensor, relay, microcontroller or lighting circuit is
                connected through this dashboard. Prediction history is
                session-based. Real-world performance must be validated
                using representative classroom data before operational use.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Loaded model details")

    detail1, detail2 = st.columns(2)

    with detail1:
        st.write("**Model file**")
        st.code(MODEL_PATH.name)

        st.write("**Input features**")
        st.code(", ".join(model_features))

    with detail2:
        st.write("**Classifier type**")
        st.code(type(model).__name__)

        st.write("**Current input row**")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

    st.warning(
        "If the model was trained on synthetic examples, its results "
        "demonstrate learning of those examples—not verified accuracy "
        "in a real classroom."
    )


# =========================================================
# 10. FOOTER
# =========================================================
st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>SMART CLASSROOM ENERGY SYSTEM</strong><br>
        Machine Learning • Classroom Energy Awareness • Decision Support<br>
        Academic prototype — recommendations require human verification.
    </div>
    """,
    unsafe_allow_html=True,
)
