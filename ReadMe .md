Diabetes Hospital Readmission: Machine Learning Project

## Project overview

This project explores whether information recorded for a diabetes-related hospital encounter can help distinguish encounters with and without a positive `readmission_flag`. It compares interpretable statistical models with more flexible machine-learning approaches and uses unsupervised learning to explore encounter profiles.

The work is exploratory. Predictive associations do not establish that a patient characteristic or hospital action causes readmission, and the models are not validated clinical decision tools.

## Project question

Can encounter-level patient, diagnosis, medication, admission, and hospital-utilization information help predict the dataset's readmission label, and how do the candidate methods compare in performance and interpretability?

## Methods

### Data review and preparation

The workflow reviews target balance, missing values, duplicates, variable distributions, and possible outliers. It prepares model features using the project's reusable helper functions and creates a reproducible stratified training/test split.

### Supervised models

- Logistic Regression baseline
- L1 (Lasso), L2 (Ridge), and Elastic Net Logistic Regression
- Linear and RBF Support Vector Machines
- Random Forest
- XGBoost
- Feedforward neural network

Candidate settings are compared with stratified cross-validation where implemented. Final supervised model metrics are calculated on a held-out test set. The reported measures include accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.

### Unsupervised and explanatory analyses

- K-Means explores groups of encounters using selected utilization and hospital-stay features. Cluster profiles and readmission rates are described after clustering.
- PCA summarizes encoded and scaled features for visualization and allows a cross-validation comparison of predictive performance with and without dimensionality reduction.
- Logistic Regression coefficients and permutation importance help describe model behavior. They are associations or model-reliance measures, not causal effects.

## Project files

Place these project files together in a working folder, or update the paths in the script to match your layout:

```text
project-folder/
├── Diabetes_Readmission_ML_Complete.py
├── diabetes_ml.py
├── requirements.txt
├── README.md
└── data/
    └── diabetes_clean.csv
```

`diabetes_ml.py` is a required local project module imported by the script. It is not a package installed by `requirements.txt`. The cleaned CSV is also a project input and is not included in this deliverable. Use the cleaned dataset from your project and keep it in a location you are authorized to access.

## Requirements

- Python with versions supported by the packages in `requirements.txt`
- The cleaned diabetes encounter CSV containing the expected target column, `readmission_flag`
- The project helper module, `diabetes_ml.py`
- The Python packages listed in `requirements.txt`

## Setup

Open PowerShell or a terminal in the project folder and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configure the data path

Before running the script, open `Diabetes_Readmission_ML_Complete.py` and update `DATA_PATH` so it points to your cleaned CSV. For example, if the file is in a `data` folder beside the script:

```python
DATA_PATH = Path(__file__).resolve().parent / "data" / "diabetes_clean.csv"
```

Make sure `diabetes_ml.py` is beside the script or otherwise available on Python's import path. The notebook version should be run from top to bottom in a Jupyter environment with the same dependencies, data path, and helper module available.

## Run the analysis

From the activated environment and project folder:

```powershell
python Diabetes_Readmission_ML_Complete.py
```

The script follows the converted notebook's cell order. It prints tables and metrics and displays plots. Some model-fitting steps, especially hyperparameter searches and neural-network training, may take time depending on the computer and installed libraries.

## Reading the results

- **ROC-AUC** summarizes ranking quality over classification thresholds. A score of 0.50 corresponds to chance-level ranking.
- **PR-AUC** focuses on positive-class retrieval; interpret it relative to the positive-label prevalence.
- **Recall** measures the share of positive encounters identified; **precision** measures the share of positive predictions that are correct.
- **F1-score** balances precision and recall. **Accuracy** is the overall share of correct predictions and should not be interpreted alone.
- Confusion matrices use `[[TN, FP], [FN, TP]]` in this project.
- K-Means clusters are exploratory groups, not confirmed clinical categories. PCA plots are projections of high-dimensional data and do not, by themselves, establish cluster separation.
- Permutation importance and model coefficients describe fitted model behavior; neither establishes causation.

## Limitations

The available cleaned data represents encounters and does not include a stable patient identifier. If individuals contribute multiple encounters, an encounter-level split may place related encounters in both training and test data. The readmission flag's exact time window and operational definition should be confirmed. Missing data, historical practices, documentation, population differences, and the selected classification threshold can affect results. Subgroup performance, calibration, and prospective performance are not established by this project.

Before considering real-world use, define the outcome and decision context, evaluate on patient-level and later-period data where possible, assess calibration and subgroup performance, and choose a threshold based on explicit consequences of missed cases and false alerts.
