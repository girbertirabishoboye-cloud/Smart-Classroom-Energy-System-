import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Smart Classroom Energy System", page_icon="💡", layout="wide")
MODEL_PATH=Path(__file__).parent/"models"/"smart_classroom_decision_tree_kaggle.joblib"

st.markdown("""
<style>
.stApp{background:#101827;color:#F1F5F9}
[data-testid="stMetric"]{background:#1C2940;border:1px solid #334155;border-radius:14px;padding:16px}
h1,h2,h3{color:#F1F5F9}
</style>
""",unsafe_allow_html=True)

@st.cache_resource
def load_artifact(path,mtime):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(f"Model file missing: {MODEL_PATH}")
    st.stop()
artifact=load_artifact(str(MODEL_PATH),MODEL_PATH.stat().st_mtime)
model=artifact['model']; features=artifact['features']; classes=artifact.get('class_codes',list(model.classes_))

st.title("💡 Smart Classroom Energy System")
st.subheader("Dataset-based Decision Tree prediction and automatic lighting recommendation")
st.write("The dashboard has no manual LIGHT ON/OFF switch. The status is computed automatically from the project conditions. Sidebar values are a simulation until live ESP8266 sensor readings are integrated.")
st.divider()

st.sidebar.header("Sensor/context inputs (simulation)")
lux=st.sidebar.slider("Ambient light (lux)",0,1000,150)
motion=st.sidebar.selectbox("Motion detected",[1,0],format_func=lambda v:"Yes" if v==1 else "No")
temp=st.sidebar.slider("Temperature (°C)",0.0,50.0,25.0)
occupancy=st.sidebar.slider("Occupancy count",0,100,10)
hour=st.sidebar.slider("Hour of day",0,23,datetime.now().hour)
day=st.sidebar.selectbox("Day of week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
if 5<=hour<12: period="Morning"
elif 12<=hour<17: period="Afternoon"
elif 17<=hour<21: period="Evening"
else: period="Night"
weather=st.sidebar.selectbox("Weather",["Clear","Cloudy","Rainy","Stormy","Other"])
event=st.sidebar.selectbox("Special event flag",[0,1],format_func=lambda v:"Yes" if v==1 else "No")
traffic=st.sidebar.slider("Traffic density",0.0,100.0,20.0)
speed=st.sidebar.slider("Average pedestrian speed (m/s)",0.0,5.0,1.2)

# Existing explicit project rule: ON iff movement, daytime window, and low lux
is_motion=motion==1; in_window=7<=hour<=18; dark=lux<300
lamp_status="ON" if (is_motion and in_window and dark) else "OFF"

m1,m2,m3,m4=st.columns(4)
m1.metric("Ambient light",f"{lux} lux")
m2.metric("Motion","DETECTED" if is_motion else "NOT DETECTED")
m3.metric("Temperature",f"{temp:.1f} °C")
m4.metric("Occupancy",occupancy)

if lamp_status=="ON":
 st.markdown("<div style='background:#064E3B;border:3px solid #10B981;border-radius:18px;padding:28px;text-align:center'><h1 style='color:#6EE7B7;font-size:48px'>💡 LIGHT ON</h1><h3 style='color:white'>Automatic condition rule recommends lighting ON</h3></div>",unsafe_allow_html=True)
else:
 st.markdown("<div style='background:#7F1D1D;border:3px solid #EF4444;border-radius:18px;padding:28px;text-align:center'><h1 style='color:#FCA5A5;font-size:48px'>🌙 LIGHT OFF</h1><h3 style='color:white'>Automatic condition rule recommends lighting OFF</h3></div>",unsafe_allow_html=True)

st.subheader("Automatic condition check")
cond=pd.DataFrame({'Condition':['Motion detected','Hour is within 07:00–18:00','Ambient light < 300 lux'],'Result':['PASS' if x else 'FAIL' for x in [is_motion,in_window,dark]]})
st.dataframe(cond,use_container_width=True,hide_index=True)

st.divider(); st.subheader("Decision Tree output from the supplied secondary dataset")
row={'ambient_light_lux':lux,'motion_detected':motion,'temperature_celsius':temp,'occupancy_count':occupancy,'hour_of_day':hour,'day_of_week':day,'time_of_day':period,'weather_condition':weather,'special_event_flag':event,'traffic_density':traffic,'avg_pedestrian_speed':speed}
X=pd.DataFrame([row])[features]
pred=model.predict(X)[0]
st.metric("Original target code prediction",str(pred))
st.caption("Target codes are displayed as stored (0/1/2). The provided file has no verified code-to-ON/OFF mapping, so this model class is not relabeled as lamp state. The LIGHT ON/OFF card above reflects the separately documented project rule.")
if hasattr(model,'predict_proba'):
 probs=model.predict_proba(X)[0]; st.write(f"Classifier confidence for predicted code: {max(probs)*100:.2f}%")

if 'history' not in st.session_state: st.session_state.history=[]
if st.button("Record current monitoring row"):
 st.session_state.history.append({'timestamp':datetime.now().isoformat(timespec='seconds'),'lux':lux,'motion':motion,'temperature_celsius':temp,'occupancy_count':occupancy,'hour_of_day':hour,'day_of_week':day,'time_of_day':period,'weather_condition':weather,'special_event_flag':event,'traffic_density':traffic,'avg_pedestrian_speed':speed,'automatic_lamp_recommendation':lamp_status,'model_class_code':pred})
st.subheader("Monitoring history")
if st.session_state.history:
 hist=pd.DataFrame(st.session_state.history); st.dataframe(hist,use_container_width=True,hide_index=True)
 st.download_button("Download CSV history",hist.to_csv(index=False).encode(),"smart_classroom_history.csv","text/csv")
else: st.info("Click Record current monitoring row to keep a session record.")
with st.expander("Model details"):
 st.write("Algorithm: Decision Tree classifier inside a preprocessing pipeline")
 st.write("Features used:",features)
 st.write("Target field:",artifact.get('target'))
 st.write("Original classes:",classes)
 st.write("Model file:",MODEL_PATH.name)
st.divider(); st.caption("Smart Classroom Energy System • Secondary dataset evaluation • Dashboard recommendation is not direct relay actuation")
