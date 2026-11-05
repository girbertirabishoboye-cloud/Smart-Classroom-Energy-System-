# Smart Classroom Energy System — Revision Package

## Included
- `Smart_Classroom_Kaggle_Training_Revised.ipynb`: retraining notebook built around the uploaded secondary CSV.
- `data/smart_lighting_dataset_2024.csv`: source CSV as supplied (with derived `hour_of_day` added).
- `models/smart_classroom_decision_tree_kaggle.joblib`: trained pipeline and feature metadata.
- `app.py`: revised dashboard using this model and its required inputs.
- `evaluation/`: test split report, confusion matrix CSV and image, and summary.
- `requirements.txt`: runtime libraries.

## Reproducible test metrics
Test split 20%, stratified, random state 42.
Test records: 2400
Accuracy: 0.8329 (83.29%)
Confusion matrix class order: [0, 1, 2]
[[615, 171, 12], [81, 1333, 5], [131, 1, 51]]

## Dataset and label documentation — important
CSV target is `lighting_action_class`, with numeric codes 0, 1, 2. The notebook uses the existing target without relabeling it. The original Kaggle page URL and human-readable class definitions were not available in the uploaded files; verify these on the source page and cite that exact page before submission. Do not claim the class codes mean ON/OFF/DIM unless supported by dataset documentation.

## Run dashboard locally
1. Install Python dependencies: `pip install -r requirements.txt`
2. Ensure model is in `models/`.
3. Run: `streamlit run app.py`

## Academic caveats
- This evaluation is from a held-out split of a secondary dataset, not measured physical classroom deployment.
- Dashboard predicts dataset class codes and does not command a relay.
- Replace this package's model with a hardware-connected model only after verifying the target mapping and safely testing the switching circuit.
