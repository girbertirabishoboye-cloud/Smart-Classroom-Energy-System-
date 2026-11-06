
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
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
# 2. THEME
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

    .stApp {
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

    h1, h2, h3, p, label {
        color: var(--text);
    }

    .hero {
        background: linear-gradient(120deg, #1E4931, #286B43, #354F35);
        border: 1px solid #548B62;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        font-weight: 800;
    }

    .hero p {
        color: #E0EEE2;
        line-height: 1.7;
    }

    .section-heading {
        color: var(--gold);
        font-size: 21px;
        font-weight: 750;
        margin: 16px 0 12px 0;
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
        padding: 18px;
        min-height: 115px;
    }

    div[data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 800;
    }

    .recommendation {
        border-radius: 16px;
        padding: 23px;
        margin: 8px 0 16px 0;
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
        color: white;
        font-size: 25px;
    }

    .recommendation p {
        color: #D5E0D6;
        line-height: 1.6;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: #347C50;
        color: white;
        border: 1px solid #68A779;
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #438F60;
        border-color: var(--gold);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        gap: 5px;
        padding: 6px;
    }

    .stTabs [aria-selected="true"] {
        background: #347C50 !important;
        color: white !important;
    }

    .footer {
        color: #AAB9AD;
        text-align: center;
        font-size: 12px;
        line-height: 1.8;
        padding: 25px 0 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. LOAD MODEL
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parent

MODEL_LOCATIONS = [
    PROJECT_DIR / "models" / "smart_classroom_decision_tree.joblib",
    PROJECT_DIR / "models" / "smart_classroom_final_model.joblib",
    PROJECT_DIR / "smart_classroom_decision_tree.joblib",
    PROJECT_DIR / "smart_classroom_final_model.joblib",
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
        features = list(getattr(model, "feature_names_in_", []))

    if not features:
        features = ["lux", "motion", "hour"]

    return model, features


if MODEL_PATH is None:
    st.error("Trained model file not found.")

    st.code(
        "Smart-Classroom-Energy-System-/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── models/\n"
        "    └── smart_classroom_decision_tree.joblib"
    )

    st.stop()


try:
    model, model_features = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime,
    )
except Exception as exc:
    st.error("Unable to load the trained model.")
    st.caption(str(exc))
    st.stop()


# =========================================================
# 4. KAGGLE MODEL EVALUATION RESULTS
# =========================================================
# These results are entered from the user's Kaggle output.
# Confusion matrix order:
# [[True Negative, False Positive],
#  [False Negative, True Positive]]

EVALUATION = {
    "model_name": "Logistic Regression",
    "threshold": 0.724513,
    "accuracy": 0.8983,
    "precision_motion": 0.1555,
    "recall_motion": 0.6271,
    "f1_motion": 0.2492,
    "confusion_matrix": [
        [11592, 1206],
        [132, 222],
    ],
    "classification_report": {
        "No Motion": {
            "precision": 0.99,
            "recall": 0.91,
            "f1-score": 0.95,
            "support": 12798,
        },
        "Motion Detected": {
            "precision": 0.16,
            "recall": 0.63,
            "f1-score": 0.25,
            "support": 354,
        },
        "macro avg": {
            "precision": 0.57,
            "recall": 0.77,
            "f1-score": 0.60,
            "support": 13152,
        },
        "weighted avg": {
            "precision": 0.97,
            "recall": 0.90,
            "f1-score": 0.93,
            "support": 13152,
        },
    },
}


# =========================================================
# 5. HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div style="color:#E7C76A;font-weight:800;letter-spacing:2px;">
            SMART CAMPUS • INTELLIGENT ENERGY
        </div>
        <h1>Smart Classroom Energy System</h1>
        <p>
            An intelligent classroom lighting dashboard using machine
            learning to support lighting recommendations. Monitor
            predictions, explore usage history, estimate energy costs,
            and review model evaluation results.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

header_left, header_right = st.columns([3, 1])

with header_left:
    st.caption(
        "DECISION SUPPORT • MACHINE LEARNING • ENERGY AWARENESS"
    )

with header_right:
    st.caption(
        f"Last page load: {datetime.now().strftime('%d %b %Y, %H:%M')}"
    )


# =========================================================
# 6. SIDEBAR INPUTS
# =========================================================
st.sidebar.markdown("## ⚙️ Control Centre")
st.sidebar.caption("Adjust classroom conditions and run a prediction.")
st.sidebar.divider()

with st.sidebar.form("classroom_input_form"):

    st.markdown("### Environmental inputs")

    lux = st.slider(
        "Light intensity (lux)",
        min_value=0,
        max_value=1000,
        value=180,
        step=10,
    )

    motion_label = st.selectbox(
        "Motion detected?",
        ["Yes", "No"],
    )

    motion = 1 if motion_label == "Yes" else 0

    hour = st.slider(
        "Hour of the day",
        min_value=0,
        max_value=23,
        value=10,
    )

    scheduled_label = st.selectbox(
        "Scheduled school day?",
        ["Yes", "No"],
    )

    scheduled_day = 1 if scheduled_label == "Yes" else 0

    st.markdown("### Energy estimate settings")

    bulb_power = st.number_input(
        "Lighting power (watts)",
        min_value=1,
        max_value=5000,
        value=100,
        step=10,
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
    )

    submitted = st.form_submit_button(
        "⚡ Run Prediction",
        use_container_width=True,
    )

st.sidebar.divider()
st.sidebar.markdown("### Model information")
st.sidebar.caption(f"Classifier: {type(model).__name__}")
st.sidebar.caption(f"Features: {', '.join(model_features)}")
st.sidebar.caption(f"Model file: {MODEL_PATH.name}")


# =========================================================
# 7. PREDICTION
# =========================================================
available_values = {
    "lux": lux,
    "motion": motion,
    "hour": hour,
    "scheduled_day": scheduled_day,
}

try:
    input_data = pd.DataFrame(
        [
            {
                feature: available_values[feature]
                for feature in model_features
            }
        ],
        columns=model_features,
    )

    raw_prediction = model.predict(input_data)[0]
    prediction = str(raw_prediction).strip().upper()

    if prediction not in ("ON", "OFF"):
        st.error(
            f"Model returned '{raw_prediction}'. "
            "Expected ON or OFF. Check the model's target labels."
        )
        st.stop()

except Exception as exc:
    st.error("Prediction failed. Check the model's input features.")
    st.code(f"Model features: {model_features}")
    st.caption(str(exc))
    st.stop()


if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = prediction

if submitted:
    st.session_state.last_prediction = prediction

    st.session_state.prediction_history.append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Light intensity (lux)": lux,
            "Motion": motion_label,
            "Hour": hour,
            "Scheduled day": scheduled_label,
            "Prediction": prediction,
            "Lighting power (W)": bulb_power,
        }
    )

display_prediction = st.session_state.last_prediction


# =========================================================
# 8. MAIN PREDICTION RESULT
# =========================================================
st.markdown(
    '<div class="section-heading">01 / Live classroom recommendation</div>',
    unsafe_allow_html=True,
)

if display_prediction == "ON":
    st.markdown(
        """
        <div class="recommendation recommendation-on">
            <h2>💡 Lights should be ON</h2>
            <p>
                The trained model recommends switching the lights on
                under the current classroom conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="recommendation recommendation-off">
            <h2>🌙 Lights should be OFF</h2>
            <p>
                The trained model recommends keeping the lights off
                under the current classroom conditions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not submitted:
    st.info(
        "Showing a prediction for the default inputs. "
        "Change the sidebar values and click Run Prediction."
    )


# =========================================================
# 9. KPI CARDS
# =========================================================
monthly_energy_kwh = (
    bulb_power * hours_per_day * days_per_month / 1000
)

monthly_cost = monthly_energy_kwh * electricity_rate

daily_energy_kwh = bulb_power * hours_per_day / 1000

potential_energy_kwh = (
    monthly_energy_kwh if display_prediction == "OFF" else 0
)

potential_cost = (
    monthly_cost if display_prediction == "OFF" else 0
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Model recommendation", display_prediction)
kpi2.metric("Light intensity", f"{lux} lux")
kpi3.metric("Motion status", motion_label)
kpi4.metric("Lighting power", f"{bulb_power} W")


# =========================================================
# 10. DASHBOARD TABS
# =========================================================
tab_analytics, tab_history, tab_evaluation, tab_about = st.tabs(
    [
        "📊 Energy Analytics",
        "🗂️ Prediction History",
        "✅ Model Evaluation",
        "🧠 Model & Project",
    ]
)


# =========================================================
# TAB 1: ENERGY ANALYTICS
# =========================================================
with tab_analytics:

    st.markdown(
        '<div class="section-heading">Energy use and cost estimates</div>',
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)

    a1.metric("Daily energy if ON", f"{daily_energy_kwh:.2f} kWh")
    a2.metric("Monthly energy if ON", f"{monthly_energy_kwh:.2f} kWh")
    a3.metric("Estimated monthly cost if ON", f"{monthly_cost:,.0f} RWF")

    st.markdown(
        """
        <div class="panel">
            <strong>How to interpret these estimates</strong>
            <p>
                These calculations assume the lighting load operates
                for the selected hours and operating days. These are
                hypothetical estimates, not measured energy savings.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    energy_data = pd.DataFrame(
        {
            "Scenario": [
                "Lights ON for configured hours",
                "Hypothetical energy avoided if OFF",
            ],
            "Energy (kWh)": [
                monthly_energy_kwh,
                potential_energy_kwh,
            ],
        }
    )

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
    )

    st.plotly_chart(fig_energy, use_container_width=True)


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

        h1.metric("Predictions recorded", len(history_df))

        off_count = int((history_df["Prediction"] == "OFF").sum())
        h2.metric("OFF recommendations", off_count)

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

        if st.button("Clear session history"):
            st.session_state.prediction_history = []
            st.rerun()

    else:
        st.info(
            "No prediction history yet. Run a prediction using the sidebar."
        )


# =========================================================
# TAB 3: MODEL EVALUATION
# =========================================================
with tab_evaluation:

    st.markdown(
        '<div class="section-heading">Evaluation on the held-out test set</div>',
        unsafe_allow_html=True,
    )

    st.success(
        "Kaggle evaluation results loaded successfully. "
        "No separate metrics.json file is required."
    )

    st.markdown(
        f"**Selected model:** {EVALUATION['model_name']}  \n"
        f"**Classification threshold:** {EVALUATION['threshold']}"
    )

    # Main metrics
    e1, e2, e3, e4 = st.columns(4)

    e1.metric(
        "Accuracy",
        f"{EVALUATION['accuracy'] * 100:.2f}%",
    )

    e2.metric(
        "Motion Precision",
        f"{EVALUATION['precision_motion'] * 100:.2f}%",
    )

    e3.metric(
        "Motion Recall",
        f"{EVALUATION['recall_motion'] * 100:.2f}%",
    )

    e4.metric(
        "Motion F1-score",
        f"{EVALUATION['f1_motion'] * 100:.2f}%",
    )

    st.divider()

    # Confusion matrix
    st.markdown("### Confusion Matrix")

    cm = EVALUATION["confusion_matrix"]

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No Motion", "Actual Motion Detected"],
        columns=["Predicted No Motion", "Predicted Motion Detected"],
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale=[
            "#1D2721",
            "#347C50",
            "#78D99A",
        ],
        title="Confusion Matrix — Logistic Regression",
    )

    fig_cm.update_layout(
        paper_bgcolor="#1D2721",
        plot_bgcolor="#1D2721",
        font_color="#F3F6F1",
        title_font_color="#E7C76A",
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    st.caption(
        "Rows represent actual classes; columns represent predicted classes."
    )

    st.divider()

    # Classification report
    st.markdown("### Classification Report")

    report_df = pd.DataFrame(
        EVALUATION["classification_report"]
    ).T

    st.dataframe(
        report_df.style.format(
            {
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "f1-score": "{:.2f}",
                "support": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Performance interpretation")

    st.markdown(
        """
        - The model achieved **89.83% accuracy** on the held-out test set.
        - The No Motion class has stronger performance than the Motion Detected class.
        - The Motion Detected recall is **62.71%**, meaning the model identified
          approximately 63% of the actual motion cases.
        - Motion precision is **15.55%**, indicating that many predicted motion
          events were false positives.
        - The dataset contains substantially more No Motion samples than Motion
          Detected samples. This class imbalance affects the evaluation.
        """
    )

    st.warning(
        "These metrics describe performance on the Kaggle test dataset. "
        "They do not guarantee the same performance in a real classroom."
    )

    # Download evaluation report
    report_csv = report_df.to_csv().encode("utf-8")

    st.download_button(
        "⬇️ Download classification report CSV",
        data=report_csv,
        file_name="smart_classroom_classification_report.csv",
        mime="text/csv",
    )


# =========================================================
# TAB 4: MODEL AND PROJECT DETAILS
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
            <p>
                Demonstrate how machine learning can support classroom
                lighting decisions using classroom conditions and
                environmental information.
            </p>
        </div>

        <div class="panel">
            <h3>System workflow</h3>
            <p>
                1. Enter classroom conditions.<br>
                2. Prepare the input features.<br>
                3. Run the trained model.<br>
                4. Display the lighting recommendation.<br>
                5. Review energy estimates and prediction history.<br>
                6. Inspect the held-out test evaluation.
            </p>
        </div>

        <div class="panel">
            <h3>Current limitations</h3>
            <p>
                This prototype uses manually entered inputs.
                Physical sensor, relay, and lighting control integration
                must be verified separately. Prediction history is
                session-based. Real-world performance requires validation
                with representative classroom data.
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
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )

    st.warning(
        "The model's predictions should be validated against real "
        "classroom observations before operational deployment."
    )


# =========================================================
# 11. FOOTER
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
