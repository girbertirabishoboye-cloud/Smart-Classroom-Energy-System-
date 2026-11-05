from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

BASE=Path(__file__).parent
DATA=BASE/'data'/'smart_lighting_dataset_2024.csv'
OUT=BASE/'models'; OUT.mkdir(exist_ok=True)
df=pd.read_csv(DATA)
df['hour_of_day']=pd.to_datetime(df['timestamp'],errors='coerce').dt.hour
features=['ambient_light_lux','motion_detected','temperature_celsius','occupancy_count','hour_of_day','day_of_week','time_of_day','weather_condition','special_event_flag','traffic_density','avg_pedestrian_speed']
target='lighting_action_class'
df=df.dropna(subset=features+[target]).copy()
X=df[features]; y=df[target].astype(int)
num=['ambient_light_lux','motion_detected','temperature_celsius','occupancy_count','hour_of_day','special_event_flag','traffic_density','avg_pedestrian_speed']
cat=['day_of_week','time_of_day','weather_condition']
pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat),('num','passthrough',num)])
model=Pipeline([('preprocessor',pre),('classifier',DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_leaf=10,class_weight='balanced',random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print('Accuracy:',accuracy_score(yte,pred)); print('Confusion matrix order [0,1,2]:\n',confusion_matrix(yte,pred,labels=[0,1,2])); print(classification_report(yte,pred,labels=[0,1,2],zero_division=0))
joblib.dump({'model':model,'features':features,'target':target,'class_codes':[0,1,2],'dataset_note':'Project-supplied dataset; codes preserved, semantic mapping not verified.','evaluation_random_state':42},OUT/'smart_classroom_decision_tree_kaggle.joblib')
