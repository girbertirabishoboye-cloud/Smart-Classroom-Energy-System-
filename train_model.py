from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "smart_classroom_energy_dataset.csv")
features = ["light_level_lux", "motion_detected", "hour_of_day", "scheduled_day"]
X, y = df[features], df["lights_action"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=5,
                               class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix (labels OFF, ON):")
print(confusion_matrix(y_test, pred, labels=["OFF", "ON"]))
out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": features, "classes": list(model.classes_)},
            out / "smart_classroom_decision_tree.joblib")
print("Model saved.")
