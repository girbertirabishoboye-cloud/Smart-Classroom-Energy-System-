# Project Outline: Smart Classroom Energy System

## Problem statement
Classroom lights may remain ON when they are unnecessary, wasting electricity. A simple automated decision-support system can use ambient light, motion, and time context to recommend whether lights should be ON or OFF.

## Aim
To develop and evaluate a Decision Tree Classification model that recommends classroom lighting actions from sensor-related inputs.

## Objectives
1. Prepare a structured dataset containing light level, motion, and time information.
2. Train a Decision Tree Classifier to predict the lighting action.
3. Evaluate the classifier using a held-out test set, classification report, and confusion matrix.
4. Develop a Streamlit interface for entering readings and displaying the predicted action.
5. Identify the limitations of synthetic data and propose validation with real classroom readings.

## Method
The project uses supervised classification. The included synthetic labels are generated using a transparent demonstration policy: ON when a classroom is scheduled, motion is detected, the hour is between 07:00 and 17:59, and ambient light is below 400 lux; otherwise OFF. The dataset is split into training and test subsets. A bounded-depth Decision Tree Classifier is trained and evaluated. The trained artifact is loaded by a Streamlit app.

## Evaluation
Report test accuracy, precision, recall, F1-score, and the confusion matrix. Because the included data are synthetic and labels follow a programmed rule, metrics indicate how well the model reproduces that rule, not real-world energy savings or generalization.

## Limitations and future improvements
- Collect actual lux, motion, and timestamp readings over multiple classroom sessions.
- Have a responsible person label appropriate lighting actions and document classroom policies.
- Evaluate on data from different days or rooms to reduce leakage from near-identical scenarios.
- Compare Decision Tree with rule-based control and other models.
- Add manual override, sensor fault detection, minimum on/off duration, and safe failure behavior.
- Measure energy consumption before and after controlled deployment.

## Deliverables
Dataset CSV, Jupyter training notebook, saved Decision Tree model, Streamlit app, requirements file, and deployment instructions.
