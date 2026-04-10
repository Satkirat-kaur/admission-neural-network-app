# Admission Chance Neural Network App

## Project Overview

This project predicts whether a student has a **high admission chance** using a neural network classifier.

The original notebook-based solution was modularized into a clean Python project and deployed as an interactive Streamlit application. The app not only predicts the outcome, but also provides profile interpretation, improvement suggestions, what-if analysis, and a suggested university tier.

## Objectives

* Convert notebook-based neural network code into modular Python files
* Train and evaluate an admission prediction model
* Build an interactive Streamlit application
* Help students understand and improve their admission profile
* Publish the project to GitHub

## Dataset

The model uses the admission dataset with the following inputs:

* GRE_Score
* TOEFL_Score
* University_Rating
* SOP
* LOR
* CGPA
* Research

## Target Variable

The original admission chance was converted into a binary class:

* `1` = High Admission Chance
* `0` = Lower Admission Chance

This was done using the threshold:

* `Admit_Chance >= 0.8` → `1`
* otherwise → `0`

## Preprocessing

* Dropped `Serial_No`
* Converted `University_Rating` and `Research` to categorical
* Applied one-hot encoding
* Scaled features using **MinMaxScaler**
* Used train-test split with stratification

## Model Used

* **MLPClassifier**
* `hidden_layer_sizes=(3,)`
* `batch_size=50`
* `max_iter=500`
* `activation='tanh'`
* `random_state=123`

## Model Performance

* **Accuracy:** 90.00%
* **Confusion Matrix:** `[[63, 6], [4, 27]]`

This indicates strong performance in identifying both lower and high admission chance applicants.

## App Features

* Predicts whether a student has a high admission chance
* Shows the probability of a high admission chance
* Explains profile strengths and concerns
* Suggests ways to improve the profile
* Provides what-if scenario analysis
* Recommends a university tier based on probability

## Project Structure

```text 
admission-neural-network-app/
│
├── data/
│   └── Admission.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   ├── admission_mlp_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── logs/
│   └── app.log
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash 
python main.py
```

### 3. Run the Streamlit app

```bash 
streamlit run app.py
```

## Real-World Use Case

This app can be used as a student decision-support tool to:

* estimate admission competitiveness
* identify weak points in an application
* simulate how improvements affect admission likelihood
* guide students toward realistic university tiers

## Future Improvements

* compare with other neural network architectures
* use probability calibration
* add historical admit profile comparison
* support university-specific prediction models

## Author

**Satkirat Kaur**
Algonquin College

## GitHub Repository

[(repo link](https://github.com/Satkirat-kaur/admission-neural-network-app)
