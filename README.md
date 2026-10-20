# Smart Classroom Energy System

A beginner-friendly machine-learning project that predicts whether classroom lights should be ON or OFF using a Decision Tree Classifier.

## Inputs and output

Inputs:
- `light_level_lux`: ambient light level, measured in lux
- `motion_detected`: `1` if motion is detected, otherwise `0`
- `hour_of_day`: integer from `0` to `23`
- `scheduled_day`: `1` if the classroom is scheduled for use, otherwise `0`

Output:
- `lights_action`: `ON` or `OFF`

The extra `scheduled_day` input is included because time alone does not tell the model whether the classroom is scheduled for use. If your lecturer strictly requires exactly three input features, remove this column consistently from the CSV, notebook, and Streamlit app.

## Important dataset note

The supplied CSV is **synthetic demonstration data**, not measurements collected from a real classroom. The labels follow this example policy:

Turn lights ON only when all conditions are true:
1. The classroom is scheduled for use (`scheduled_day = 1`).
2. Motion is detected (`motion_detected = 1`).
3. The time is from 07:00 through 17:59 (`7 <= hour_of_day < 18`).
4. Ambient light is below 400 lux (`light_level_lux < 400`).

Otherwise, the label is OFF. A model trained on these labels learns the demonstration policy; test accuracy on this dataset does not prove real-world accuracy or energy savings. Collect labelled real sensor data and validate the threshold/policy before operational use.

## Project structure

- `data/smart_classroom_energy_dataset.csv` — synthetic dataset
- `notebooks/Smart_Classroom_Energy_System_Training.ipynb` — training and evaluation notebook
- `models/` — location for saved trained model
- `app.py` — Streamlit interface
- `requirements.txt` — Python dependencies

## Run in Jupyter

1. Extract the ZIP file.
2. Open Anaconda Prompt or a terminal in the extracted `Smart_Classroom_Energy_System` folder.
3. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
4. Start Jupyter:
   ```bash
   jupyter notebook
   ```
5. Open `notebooks/Smart_Classroom_Energy_System_Training.ipynb`.
6. Run cells from top to bottom. The notebook saves the model to `models/smart_classroom_decision_tree.joblib`.

If the notebook cannot locate the dataset, run Jupyter from the project root and keep the provided folder structure.

## Run Streamlit locally

After the notebook has saved the model, run this command from the project root:
```bash
streamlit run app.py
```
The browser should open the app, usually at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `data/`, and the trained model file at `models/smart_classroom_decision_tree.joblib`.
2. Make sure the model file exists in the repository before deploying; the app will show an error if it is missing.
3. In Streamlit Community Cloud, create a new app and select the repository, branch, and `app.py`.
4. Deploy and test several input combinations.

Do not commit private credentials or API keys. This project does not require an API key.

## Physical hardware safety

The Streamlit app only recommends ON/OFF. It does not switch real lights. Physical control requires appropriate isolated drivers, correctly rated relays/contactors, protective enclosures, manual override, and qualified installation. Do not connect computer GPIO pins directly to mains lighting.
