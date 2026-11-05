from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smart Classroom Energy System", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background:#141A17;color:#F3F6F1}
[data-testid="stSidebar"] {background:#1A241E;border-right:1px solid #3A5142}
.block-container {max-width:1450px;padding-top:1.3rem}
.hero {background:linear-gradient(120deg,#1E4931,#286B43,#354F35);border:1px solid #548B62;border-radius:18px;padding:26px;margin-bottom:18px}
.hero h1 {color:white;font-size:32px;margin:5px 0}
.hero p {color:#E0EEE2}
div[data-testid="stMetric"] {background:#1D2721;border:1px solid #3A5142;border-radius:12px;padding:15px}
</style>
""", unsafe_allow_html=True)

ROOT=Path(__file__).resolve().parent
MODEL_CANDIDATES=[ROOT/"models"/"smart_classroom_decision_tree_kaggle.joblib",
                  ROOT/"smart_classroom_decision_tree_kaggle.joblib"]
MODEL_PATH=next((p for p in MODEL_CANDIDATES if p.exists()),None)
if MODEL_PATH is None:
    st.error("Model file not found. Put smart_classroom_decision_tree_kaggle.joblib in the models folder.")
    st.stop()

@st.cache_resource
def load_artifact(path,mtime):
    return joblib.load(path)
artifact=load_artifact(str(MODEL_PATH),MODEL_PATH.stat().st_mtime)
model=artifact["model"]
features=artifact["features"]

st.markdown("""
<div class="hero">
 <div style="letter-spacing:2px;color:#E7C76A;font-weight:800;font-size:12px">SMART CAMPUS • ENERGY AWARENESS</div>
 <h1>Smart Classroom Energy System</h1>
 <p>Interactive Decision Tree classification dashboard trained with the supplied secondary smart-lighting dataset.
 Predictions display the dataset's original class codes pending verification of their official label definitions.</p>
</div>
""",unsafe_allow_html=True)
st.caption(f"Model artifact: {MODEL_PATH.name} • Classifier: DecisionTreeClassifier pipeline • Dataset-driven class codes: 0, 1, 2")

with st.sidebar:
    st.header("Classroom input")
    with st.form("predict"):
        lux=st.slider("Ambient light (lux)",0.0,105000.0,500.0,step=50.0)
        motion=st.selectbox("Motion detected",["No","Yes"])
        temperature=st.slider("Temperature (°C)",-15.0,65.0,25.0,step=.5)
        occupancy=st.slider("Occupancy count",0,230,8)
        hour=st.slider("Hour of day",0,23,10)
        day=st.selectbox("Day of week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        tod=st.selectbox("Time of day",["Morning","Afternoon","Evening","Night"])
        weather=st.selectbox("Weather condition",["Clear","Cloudy","Foggy","Rainy"])
        special=st.selectbox("Special event",["No","Yes"])
        traffic=st.slider("Traffic density",0.0,70.0,22.0,step=.5)
        speed=st.slider("Average pedestrian speed",0.0,6.0,2.9,step=.1)
        submitted=st.form_submit_button("Run dataset-based prediction",use_container_width=True)

if "history" not in st.session_state: st.session_state.history=[]
if submitted:
    row={
      "ambient_light_lux":lux,"motion_detected":1 if motion=="Yes" else 0,
      "temperature_celsius":temperature,"occupancy_count":occupancy,"hour_of_day":hour,
      "day_of_week":day,"time_of_day":tod,"weather_condition":weather,
      "special_event_flag":1 if special=="Yes" else 0,
      "traffic_density":traffic,"avg_pedestrian_speed":speed
    }
    input_df=pd.DataFrame([row],columns=features)
    prediction=int(model.predict(input_df)[0])
    probabilities=model.predict_proba(input_df)[0]
    classes=list(model.classes_)
    conf=float(probabilities[classes.index(prediction)])
    st.session_state.latest={"input":row,"class":prediction,"confidence":conf,"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.session_state.history.append(st.session_state.latest)

latest=st.session_state.get("latest")
if latest:
    st.subheader("Prediction output")
    c1,c2,c3=st.columns(3)
    c1.metric("Predicted dataset class",f"Class {latest['class']}")
    c2.metric("Model confidence",f"{latest['confidence']*100:.1f}%")
    c3.metric("Prediction time",latest["timestamp"])
    st.info("Class 0/1/2 are displayed without ON/OFF naming because the official human-readable meanings of these codes must be verified from the Kaggle dataset documentation.")
    st.subheader("Submitted input")
    st.dataframe(pd.DataFrame([latest["input"]]),use_container_width=True,hide_index=True)
else:
    st.info("Choose the classroom conditions in the sidebar and click Run dataset-based prediction.")

st.divider()
st.subheader("Prediction history")
if st.session_state.history:
    history=pd.DataFrame([{"Timestamp":h["timestamp"],"Predicted class":h["class"],"Confidence":round(h["confidence"],4)} for h in st.session_state.history])
    st.dataframe(history.iloc[::-1],use_container_width=True,hide_index=True)
    st.download_button("Download history CSV",history.to_csv(index=False).encode("utf-8"),"smart_classroom_prediction_history.csv","text/csv")
    counts=history["Predicted class"].value_counts().sort_index().rename_axis("Class").reset_index(name="Count")
    fig=px.bar(counts,x="Class",y="Count",text="Count",title="Prediction counts by dataset class",color="Class")
    fig.update_layout(paper_bgcolor="#1D2721",plot_bgcolor="#1D2721",font_color="#F3F6F1")
    st.plotly_chart(fig,use_container_width=True)
else:
    st.caption("History appears after the first prediction; it is session-based.")

with st.expander("Methodology and limitations"):
    st.write("The classifier was trained using a stratified 80/20 split with random state 42. Inputs include ambient light, motion, temperature, occupancy, time-derived and contextual variables. The app is a demonstration interface. It does not operate a physical light relay.")
    st.write("Dataset citation and official class-code meanings should be confirmed against the original Kaggle dataset page before final academic submission.")
