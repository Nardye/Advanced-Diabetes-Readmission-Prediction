"""Reusable functions for the diabetes readmission machine-learning project.

The target is the dataset's readmission_flag. Rows represent encounters. This
module cannot make a patient-level split unless a stable patient identifier is
present in the data.
"""
from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    silhouette_score)
from sklearn.inspection import permutation_importance

TARGET = "readmission_flag"
LEAKAGE_OR_ID_COLUMNS = {
    TARGET, "readmitted", "rowID", "patient_nbr",
    "prior_utilization", "utilization_group", "stay_group",
    "age_avg_los", "los_difference", "los_vs_age_average"
}
CLUSTER_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the cleaned CSV and return it as a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return a copy plus basic missingness, duplicate, target, and outlier checks.

    IQR flags are descriptive only. This function does not delete flagged rows.
    """
    data = df.copy()
    report = {
        "shape": data.shape,
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_by_column": data.isna().sum().sort_values(ascending=False),
        "dtypes": data.dtypes.astype(str),
    }
    if TARGET in data.columns:
        report["target_counts"] = data[TARGET].value_counts(dropna=False)
        report["target_rate"] = data[TARGET].value_counts(normalize=True, dropna=False)
    outliers = {}
    for col in ["number_inpatient", "number_outpatient", "number_emergency"]:
        if col in data.columns:
            values = pd.to_numeric(data[col], errors="coerce").dropna()
            if len(values):
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers[col] = int(((values < q1 - 1.5 * iqr) |
                                     (values > q3 + 1.5 * iqr)).sum())
    report["iqr_outlier_flags"] = outliers
    return data, report


def prepare_data(df: pd.DataFrame, target: str = TARGET):
    """Create X, y, and a leakage-aware preprocessing ColumnTransformer.

    Numeric values are median-imputed and standardized. Categorical values are
    most-frequent-imputed and one-hot encoded. Text categories such as the
    literal 'None' stay categories; only actual missing values are imputed.
    """
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} is not in the dataset.")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    y = y.loc[valid].astype(int)
    # Always remove the target itself from predictors, along with known
    # duplicate-target, identifier, and leakage-risk columns.
    columns_to_exclude = set(LEAKAGE_OR_ID_COLUMNS) | {target}
    X = df.loc[valid].drop(
        columns=[c for c in columns_to_exclude if c in df.columns],
        errors="ignore"
    ).copy()
    if target in X.columns:
        raise RuntimeError(f"Target leakage: {target!r} remains in feature columns.")
    X = X.drop(columns=[c for c in X.columns if c.endswith("_desc")], errors="ignore")

    # Feature engineering: combine prior encounter counts into a total.
    prior_count_cols = [c for c in ["number_inpatient", "number_outpatient",
                                    "number_emergency"] if c in X.columns]
    if prior_count_cols:
        numeric_counts = X[prior_count_cols].apply(pd.to_numeric, errors="coerce")
        X["total_prior_visits_engineered"] = numeric_counts.sum(axis=1, min_count=1)

    # Avoid supplying a total and its component counts together.
    if "prior_utilization_count" in X.columns:
        X = X.drop(columns="prior_utilization_count")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    for col in categorical_cols:
        X[col] = X[col].astype("object")
        X[col] = X[col].where(X[col].notna(), np.nan)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])
    return X, y, preprocessor


def evaluate_model(model_name, fitted_pipeline, X_test, y_test) -> dict:
    """Calculate requested test metrics and confusion matrix."""
    predictions = fitted_pipeline.predict(X_test)
    if hasattr(fitted_pipeline, "predict_proba"):
        scores = fitted_pipeline.predict_proba(X_test)[:, 1]
    else:
        scores = fitted_pipeline.decision_function(X_test)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }


def _cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_regression_models(X_train, y_train, preprocessor):
    """Tune baseline, L2, L1 and Elastic Net Logistic Regression by CV ROC-AUC."""
    models = {}
    summaries = []
    specs = {
        "Logistic Regression baseline": (LogisticRegression(max_iter=3000, random_state=42), {}),
        "Logistic Regression L2": (LogisticRegression(solver="saga", max_iter=4000, random_state=42),
                                    {"model__C": [0.01, 0.1, 1, 10], "model__l1_ratio": [0]}),
        "Logistic Regression L1": (LogisticRegression(solver="saga", max_iter=4000, random_state=42),
                                    {"model__C": [0.01, 0.1, 1, 10], "model__l1_ratio": [1]}),
        "Logistic Regression Elastic Net": (LogisticRegression(solver="saga", max_iter=4000, random_state=42),
                                    {"model__C": [0.01, 0.1, 1, 10], "model__l1_ratio": [0.25, 0.5, 0.75]}),
    }
    for name, (classifier, grid) in specs.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", classifier)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=_cv(), n_jobs=-1,
                                  refit=True, return_train_score=True)
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            summaries.append({"model": name, "cv_roc_auc": search.best_score_,
                              "best_parameters": search.best_params_})
        else:
            # Baseline is also cross-validated for a fair CV comparison.
            search = GridSearchCV(pipe, {"model__C": [1.0]}, scoring="roc_auc",
                                  cv=_cv(), n_jobs=-1, refit=True, return_train_score=True)
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            summaries.append({"model": name, "cv_roc_auc": search.best_score_,
                              "best_parameters": {}})
        models[name] = fitted
    return models, pd.DataFrame(summaries)


def train_ml_models(X_train, y_train, preprocessor, include_xgboost=True):
    """Tune linear/RBF SVM and XGBoost with stratified CV ROC-AUC."""
    models, summaries = {}, []
    svm_specs = {
        "Linear SVM": {"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1, 10]},
        "RBF SVM": {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10],
                    "model__gamma": ["scale", 0.01, 0.1]},
    }
    for name, grid in svm_specs.items():
        pipe = Pipeline([("preprocessor", preprocessor),
                         ("model", SVC(probability=True, random_state=42))])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=_cv(), n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
        summaries.append({"model": name, "cv_roc_auc": search.best_score_,
                          "best_parameters": search.best_params_})

    if include_xgboost:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            warnings.warn("XGBoost is unavailable; install it in the notebook environment.")
        else:
            pipe = Pipeline([("preprocessor", preprocessor),
                ("model", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                         random_state=42, n_jobs=1))])
            grid = {
                "model__n_estimators": [150, 300],
                "model__max_depth": [3, 4, 6],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            }
            search = RandomizedSearchCV(pipe, grid, n_iter=10, scoring="roc_auc",
                cv=_cv(), n_jobs=-1, random_state=42, refit=True)
            search.fit(X_train, y_train)
            models["XGBoost"] = search.best_estimator_
            summaries.append({"model": "XGBoost", "cv_roc_auc": search.best_score_,
                              "best_parameters": search.best_params_})
    return models, pd.DataFrame(summaries)


def perform_clustering(df: pd.DataFrame, features=None, k_values=range(2, 7)):
    """Run K-Means on encounter-level utilization features; return labels/profiles."""
    features = features or [c for c in CLUSTER_FEATURES if c in df.columns]
    if not features:
        raise ValueError("None of the configured clustering features are present.")
    from sklearn.impute import SimpleImputer
    data = df[features].apply(pd.to_numeric, errors="coerce")
    matrix = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(data))
    scores, labels_by_k = {}, {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(matrix)
        scores[k] = silhouette_score(matrix, labels)
        labels_by_k[k] = labels
    best_k = max(scores, key=scores.get)
    result = df.copy()
    result["cluster"] = labels_by_k[best_k]
    aggregations = {f"{f}_median": (f, "median") for f in features}
    if TARGET in result.columns:
        aggregations["readmission_rate"] = (TARGET, "mean")
    profile = result.groupby("cluster").agg(encounters=("cluster", "size"), **aggregations)
    if "readmission_rate" in profile:
        profile["readmission_rate"] *= 100
    return {"scores": scores, "best_k": best_k, "data": result,
            "profile": profile, "features": features}


def perform_pca(X_train, X_test, preprocessor, n_components=2):
    """Fit preprocessing on training rows, then PCA and transform test rows."""
    train_encoded = preprocessor.fit_transform(X_train)
    test_encoded = preprocessor.transform(X_test)
    # PCA requires a dense matrix. For very wide data, check memory before conversion.
    if hasattr(train_encoded, "toarray"):
        train_encoded = train_encoded.toarray()
        test_encoded = test_encoded.toarray()
    pca = PCA(n_components=n_components, random_state=42)
    train_coordinates = pca.fit_transform(train_encoded)
    test_coordinates = pca.transform(test_encoded)
    return pca, train_coordinates, test_coordinates


def coefficient_table(fitted_pipeline, top_n=30):
    """Return largest absolute Logistic Regression coefficients and direction."""
    prep = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    names = prep.get_feature_names_out()
    coefs = model.coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefs})
    table["absolute_coefficient"] = table["coefficient"].abs()
    return table.sort_values("absolute_coefficient", ascending=False).head(top_n)


def permutation_importance_table(fitted_pipeline, X_test, y_test, top_n=20):
    """Estimate test-set permutation importance using ROC-AUC."""
    result = permutation_importance(fitted_pipeline, X_test, y_test,
        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    table = pd.DataFrame({"feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std})
    return table.sort_values("importance_mean", ascending=False).head(top_n)



def coefficient_stability_cv(X_train, y_train, preprocessor, C=0.1, l1_ratio=0.0):
    """Fit Logistic Regression on stratified folds and summarize coefficient stability.

    Encoded features absent from a fold are treated as coefficient zero.
    """
    from sklearn.base import clone
    feature_values = {}
    cv = _cv()
    for train_idx, _ in cv.split(X_train, y_train):
        pipe = Pipeline([("preprocessor", clone(preprocessor)),
            ("model", LogisticRegression(C=C, l1_ratio=l1_ratio, solver="saga",
                                          max_iter=4000, random_state=42))])
        pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        names = pipe.named_steps["preprocessor"].get_feature_names_out()
        values = pipe.named_steps["model"].coef_.ravel()
        for name, value in zip(names, values):
            feature_values.setdefault(name, []).append(float(value))
        # Record zero when a feature was absent from this fold's encoding.
        for name in set(feature_values) - set(names):
            feature_values[name].append(0.0)
    rows = []
    for name, values in feature_values.items():
        values = np.asarray(values, dtype=float)
        if len(values) < cv.n_splits:
            values = np.pad(values, (0, cv.n_splits - len(values)))
        rows.append({"feature": name, "mean_coefficient": values.mean(),
                     "coefficient_std": values.std(ddof=1),
                     "sign_consistency": max((values > 0).mean(), (values < 0).mean())})
    return pd.DataFrame(rows).sort_values("coefficient_std", ascending=False)


def compare_pca_cv(X_train, y_train, preprocessor, n_components=2):
    """Compare stratified CV ROC-AUC with and without PCA on training data.

    PCA is fit independently inside each fold after preprocessing, avoiding
    leakage from validation folds. Two components match the 2D visualization.
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import FunctionTransformer
    from scipy import sparse

    def dense_if_sparse(matrix):
        return matrix.toarray() if sparse.issparse(matrix) else matrix

    cv = _cv()
    baseline = Pipeline([("preprocessor", clone(preprocessor)),
                         ("model", LogisticRegression(max_iter=3000, random_state=42))])
    with_pca = Pipeline([("preprocessor", clone(preprocessor)),
        ("to_dense", FunctionTransformer(dense_if_sparse, accept_sparse=True)),
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("model", LogisticRegression(max_iter=3000, random_state=42))])
    baseline_scores = cross_val_score(baseline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=1)
    pca_scores = cross_val_score(with_pca, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=1)
    return pd.DataFrame({"approach": ["Logistic Regression, no PCA",
                                      f"Logistic Regression, PCA ({n_components} components)"],
                         "mean_cv_roc_auc": [baseline_scores.mean(), pca_scores.mean()],
                         "std_cv_roc_auc": [baseline_scores.std(ddof=1), pca_scores.std(ddof=1)]})

def train_neural_network(X_train, y_train, X_test, y_test, preprocessor,
                         epochs=100, batch_size=64):
    """Train a requested Keras feedforward binary classifier with early stopping."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Install TensorFlow in the notebook environment first.") from exc
    from sklearn.model_selection import train_test_split

    from sklearn.base import clone
    # Split raw training rows first. Fit imputers, encoders, and scaling only on
    # the inner training portion so validation information does not leak in.
    x_fit_raw, x_val_raw, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    fitted_preprocessor = clone(preprocessor)
    x_fit = fitted_preprocessor.fit_transform(x_fit_raw)
    x_val = fitted_preprocessor.transform(x_val_raw)
    test_encoded = fitted_preprocessor.transform(X_test)
    if hasattr(x_fit, "toarray"):
        x_fit = x_fit.toarray()
        x_val = x_val.toarray()
        test_encoded = test_encoded.toarray()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_fit.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                           tf.keras.metrics.AUC(name="roc_auc")])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(x_fit, y_fit, validation_data=(x_val, y_val),
        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)
    probabilities = model.predict(test_encoded, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model": "Neural Network",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }
    return model, history, metrics
