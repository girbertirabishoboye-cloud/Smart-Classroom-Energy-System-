# Smart Classroom Energy System — Improvement Package (PowerPoint excluded)

## Included
- `data/smart_lighting_dataset_2024.csv` — supplied secondary dataset (12,000 rows, 17 original columns).
- `Smart_Classroom_Kaggle_Training_Revised.ipynb` — reproducible notebook with dataset analysis, training, metrics, confusion matrix, tree and saved model.
- `train_model.py` — repeatable training script.
- `models/smart_classroom_decision_tree_kaggle.joblib` — retrained pipeline artifact.
- `app.py` — Streamlit dashboard; automatic rule card LIGHT ON/OFF plus original dataset class-code prediction.
- `evaluation/` — classification report, confusion matrix, feature-importance and tree visualizations.
- `documentation/` — evaluation narrative, data provenance/citation note, label creation note, member reflection drafts.

## Evaluation
Stratified 80/20 split, `random_state=42`; test accuracy 0.8329 (83.29%). Test records: 2400. Confusion matrix rows actual and columns predicted in class order [0,1,2]:

```
[[615, 171, 12], [81, 1333, 5], [131, 1, 51]]
```

Class-specific scores are in `evaluation/classification_report.csv`.

## Important target/data integrity notes
The source file's target is `lighting_action_class` with codes 0, 1, 2. The exact semantics of those codes are not verified by the CSV alone. We retain the original codes for supervised learning and display. Do not label a code as ON/OFF/DIM until the dataset's official source card verifies its meaning.

The dashboard separately displays a transparent rule-based LIGHT ON/OFF recommendation based on the project rule: ON iff motion=1, hour 07 through 18 inclusive, and ambient light <300 lux; otherwise OFF. This is an explicit derived-label rule, not a reinterpretation of target classes. The sidebar is simulation; physical switching needs live ESP8266 sensor data and relay firmware.

## Dataset citation
Stafford, G. A. (2020). *Environmental Sensor Telemetry Data*. Kaggle. https://www.kaggle.com/datasets/garystafford/environmental-sensor-data-132k (accessed 5 November 2026).

**Provenance caveat:** The supplied CSV (`smart_lighting_dataset_2024.csv`) has 12,000 rows and 17 columns, including engineered classroom context variables and a three-code target; it is not the raw 405,184-row Kaggle telemetry file schema. Treat it as a project-supplied/processed dataset unless its origin chain is confirmed. Cite the original Kaggle page as the intended secondary source only after confirming this file's lineage.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reflections
`documentation/Member_Reflections_Drafts.docx` contains draft reflections for group members. Replace member placeholders and personalize each reflection so it accurately describes each member's contribution.
