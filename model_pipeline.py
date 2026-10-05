"""
model_pipeline.py - Predictive engine of the Diabetes Digital Twin.

Task: from the data available at time t, predict the glucose 120 minutes later.
  * Classifier -> P(glucose at t+120 min > 180 mg/dL)   = "hyperglycemic spike" alert
  * Regressor  -> expected glucose value at t+120 min   (MAE / RMSE)

Feature fusion:
  STATIC EHR  (age, BMI, HbA1c, BP, comorbidities, medication, diet ...)
  + DYNAMIC   (current glucose, lags, trends, 1h/3h/24h aggregates, heart rate,
               steps, sleep quality, time since last meal, recent carbs, time of day)

All wearable features are CAUSAL (use only the past), so they can be computed for any
"current time" in the live dashboard without leakage.

Usage:
    python model_pipeline.py              # trains on data/ (generates it if missing)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from simulator import MED_INTENSITY

DATA_DIR = Path("data")
MODEL_PATH = Path("models/digital_twin_model.joblib")
METRICS_PATH = Path("models/metrics.json")

STEP_MIN = 15
HORIZON_STEPS = 8                  # 8 x 15 min = 120 minutes ahead
HORIZON_MIN = HORIZON_STEPS * STEP_MIN
SPIKE_THRESHOLD = 180.0            # mg/dL

EHR_NUMERIC = ["age", "gender_male", "bmi", "hba1c", "systolic_bp", "diastolic_bp",
               "family_history_diabetes", "diabetes_duration_yrs", "hypertension",
               "dyslipidemia", "smoker", "activity_code", "med_intensity"]
EHR_CATEGORICAL = ["medication", "diet_type"]


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def prepare_ehr(ehr: pd.DataFrame) -> pd.DataFrame:
    """Encode the static EHR table into model-ready columns (indexed by patient_id)."""
    e = ehr.copy()
    e["gender_male"] = (e["gender"] == "Male").astype(int)
    e["activity_code"] = e["activity_level"].map({"Sedentary": 0, "Light": 1, "Active": 2})
    e["med_intensity"] = e["medication"].map(MED_INTENSITY)
    return e.set_index("patient_id")[EHR_NUMERIC + EHR_CATEGORICAL]


def wearable_features(w: pd.DataFrame) -> pd.DataFrame:
    """Causal time-series features for ONE patient (rows sorted by time)."""
    w = w.sort_values("timestamp").reset_index(drop=True)
    g, hr, steps = w["glucose_mg_dl"], w["heart_rate_bpm"], w["steps"]
    carbs = w["meal_carbs_g"]
    n = len(w)
    f = pd.DataFrame(index=w.index)

    # current state + lags (15 / 30 / 60 / 120 min ago)
    f["glucose"] = g
    for k in (1, 2, 4, 8):
        f[f"glucose_lag_{k * STEP_MIN}m"] = g.shift(k)
    f["glucose_delta_30m"] = g - g.shift(2)
    f["glucose_delta_60m"] = g - g.shift(4)

    # short-term rolling stats
    f["glucose_mean_1h"] = g.rolling(4, min_periods=1).mean()
    f["glucose_std_1h"] = g.rolling(4, min_periods=2).std()
    f["glucose_max_3h"] = g.rolling(12, min_periods=1).max()
    f["glucose_mean_3h"] = g.rolling(12, min_periods=1).mean()

    # 24h aggregated wearable features (glycaemic control summary)
    roll24 = g.rolling(96, min_periods=8)
    f["glucose_mean_24h"] = roll24.mean()
    f["glucose_cv_24h"] = roll24.std() / roll24.mean()
    f["time_above_180_24h"] = (g > 180).astype(float).rolling(96, min_periods=8).mean()
    f["time_in_range_24h"] = ((g >= 70) & (g <= 180)).astype(float).rolling(96, min_periods=8).mean()

    # circadian
    hour = w["timestamp"].dt.hour + w["timestamp"].dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # heart rate / activity / sleep
    f["heart_rate"] = hr
    f["heart_rate_mean_1h"] = hr.rolling(4, min_periods=1).mean()
    f["steps_1h"] = steps.rolling(4, min_periods=1).sum()
    f["steps_3h"] = steps.rolling(12, min_periods=1).sum()
    f["steps_24h"] = steps.rolling(96, min_periods=8).sum()
    f["sleep_quality"] = w["sleep_quality_index"]

    # meal context (what was eaten, how long ago)
    is_meal = carbs > 0
    pos = pd.Series(np.where(is_meal, np.arange(n), np.nan)).ffill()
    f["mins_since_meal"] = ((np.arange(n) - pos) * STEP_MIN).fillna(600).clip(upper=600)
    f["last_meal_carbs"] = carbs.where(is_meal).ffill().fillna(0)
    f["carbs_last_3h"] = carbs.rolling(12, min_periods=1).sum()
    f["carbs_last_6h"] = carbs.rolling(24, min_periods=1).sum()
    return f


def patient_matrix(w: pd.DataFrame, ehr_row: pd.Series) -> pd.DataFrame:
    """Fuse wearable features with the patient's static EHR features (one row per timestamp)."""
    wf = wearable_features(w)
    for col, val in ehr_row.items():            # broadcast static EHR values
        wf[col] = val
    return wf


def build_dataset(ehr: pd.DataFrame, wearable: pd.DataFrame):
    """Fused training table for all patients -> X, y_glucose, y_spike, groups."""
    ehr_p = prepare_ehr(ehr)
    parts, ys, groups = [], [], []
    for pid, w in wearable.groupby("patient_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        X = patient_matrix(w, ehr_p.loc[pid])
        y = w["glucose_mg_dl"].shift(-HORIZON_STEPS)           # glucose 2h in the future
        keep = y.notna() & X["glucose_lag_120m"].notna()       # need a full look-back and a label
        parts.append(X[keep]); ys.append(y[keep])
        groups += [pid] * int(keep.sum())
    X = pd.concat(parts, ignore_index=True)
    y_reg = pd.concat(ys, ignore_index=True)
    return X, y_reg, (y_reg > SPIKE_THRESHOLD).astype(int), np.array(groups)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), EHR_CATEGORICAL)],
        remainder="passthrough", verbose_feature_names_out=False, sparse_threshold=0.0)


def _assemble(prep, model) -> Pipeline:
    """Wrap an already-fitted preprocessor + model as a single scikit-learn Pipeline."""
    return Pipeline([("prep", prep), ("model", model)])


def best_f1_threshold(y_true, proba) -> float:
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([f1_score(y_true, proba >= t) for t in ts]))])


def train(ehr: pd.DataFrame, wearable: pd.DataFrame, seed: int = 42) -> dict:
    X, y_reg, y_clf, groups = build_dataset(ehr, wearable)
    print(f"Fused dataset: {X.shape[0]:,} rows x {X.shape[1]} features | "
          f"spike prevalence {y_clf.mean():.1%}")

    # ---- split BY PATIENT (no patient appears in two sets -> honest generalisation) ----
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    tr_val, te = next(outer.split(X, y_clf, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_i, va_i = next(inner.split(X.iloc[tr_val], y_clf.iloc[tr_val], groups[tr_val]))
    tr, va = tr_val[tr_i], tr_val[va_i]
    print(f"Patients -> train {len(set(groups[tr]))} | val {len(set(groups[va]))} | test {len(set(groups[te]))}")

    prep = make_preprocessor().fit(X.iloc[tr])
    Xtr, Xva, Xte = (prep.transform(X.iloc[i]) for i in (tr, va, te))
    names = list(prep.get_feature_names_out())

    # ---- classifier: spike > 180 mg/dL in 120 min ----
    pos_weight = float((y_clf.iloc[tr] == 0).sum() / max((y_clf.iloc[tr] == 1).sum(), 1))
    clf = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=5, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0,
        scale_pos_weight=min(pos_weight, 4.0), eval_metric="aucpr",
        early_stopping_rounds=40, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, y_clf.iloc[tr], eval_set=[(Xva, y_clf.iloc[va])], verbose=False)

    # ---- regressor: glucose value at +120 min ----
    reg = xgb.XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=5, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0,
        eval_metric="rmse", early_stopping_rounds=40, random_state=seed, n_jobs=-1)
    reg.fit(Xtr, y_reg.iloc[tr], eval_set=[(Xva, y_reg.iloc[va])], verbose=False)

    # ---- decision threshold chosen on VALIDATION (not test) ----
    threshold = best_f1_threshold(y_clf.iloc[va], clf.predict_proba(Xva)[:, 1])

    # ---- evaluation on held-out PATIENTS ----
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= threshold).astype(int)
    yhat = reg.predict(Xte)
    y_true_reg, y_true_clf = y_reg.iloc[te], y_clf.iloc[te]
    persistence = X.iloc[te]["glucose"]                       # naive baseline: "glucose stays the same"
    tn, fp, fn, tp = confusion_matrix(y_true_clf, pred).ravel()
    metrics = {
        "horizon_minutes": HORIZON_MIN, "spike_threshold_mg_dl": SPIKE_THRESHOLD,
        "n_rows": int(len(X)), "n_patients": int(len(set(groups))), "test_rows": int(len(te)),
        "spike_prevalence": round(float(y_clf.mean()), 4),
        "decision_threshold": round(threshold, 2),
        "regression": {
            "MAE_mg_dl": round(float(mean_absolute_error(y_true_reg, yhat)), 2),
            "RMSE_mg_dl": round(float(np.sqrt(mean_squared_error(y_true_reg, yhat))), 2),
            "R2": round(float(r2_score(y_true_reg, yhat)), 3),
            "baseline_persistence_MAE": round(float(mean_absolute_error(y_true_reg, persistence)), 2),
            "baseline_persistence_RMSE": round(float(np.sqrt(mean_squared_error(y_true_reg, persistence))), 2),
        },
        "classification": {
            "accuracy": round(float(accuracy_score(y_true_clf, pred)), 3),
            "precision": round(float(precision_score(y_true_clf, pred)), 3),
            "recall_sensitivity": round(float(recall_score(y_true_clf, pred)), 3),
            "specificity": round(float(tn / (tn + fp)), 3),
            "f1": round(float(f1_score(y_true_clf, pred)), 3),
            "roc_auc": round(float(roc_auc_score(y_true_clf, proba)), 3),
            "pr_auc": round(float(average_precision_score(y_true_clf, proba)), 3),
            "baseline_persistence_f1": round(float(f1_score(y_true_clf, (persistence > SPIKE_THRESHOLD).astype(int))), 3),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "feature_importance_top15": dict(sorted(
            zip(names, map(lambda v: round(float(v), 4), clf.feature_importances_)),
            key=lambda kv: -kv[1])[:15]),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }

    bundle = {
        "classifier": _assemble(prep, clf),
        "regressor": _assemble(prep, reg),
        "threshold": threshold,
        "feature_columns": list(X.columns),
        "horizon_minutes": HORIZON_MIN,
        "spike_threshold": SPIKE_THRESHOLD,
        "metrics": metrics,
    }
    return bundle


def save_bundle(bundle: dict, path: Path = MODEL_PATH, metrics_path: Path = METRICS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    metrics_path.write_text(json.dumps(bundle["metrics"], indent=2))


def load_bundle(path: Path = MODEL_PATH) -> dict:
    return joblib.load(path)


# ----------------------------------------------------------------------------
# Inference helpers (used by the dashboard)
# ----------------------------------------------------------------------------
def predict_rows(bundle: dict, X: pd.DataFrame):
    """Return (spike_probability, predicted_glucose_in_120_min) arrays for rows of X."""
    X = X[bundle["feature_columns"]]
    return (bundle["classifier"].predict_proba(X)[:, 1], bundle["regressor"].predict(X))


def explain_row(bundle: dict, X_row: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Per-patient explanation: which features push the spike risk up/down?
    Uses XGBoost's native SHAP-style contributions (log-odds scale).
    """
    pipe = bundle["classifier"]
    Xt = pipe.named_steps["prep"].transform(X_row[bundle["feature_columns"]])
    names = pipe.named_steps["prep"].get_feature_names_out()
    contrib = pipe.named_steps["model"].get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[0][:-1]
    values = Xt[0]
    df = pd.DataFrame({"feature": names, "value": values, "contribution": contrib})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).head(top_k)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ehr_path, w_path = DATA_DIR / "ehr.csv", DATA_DIR / "wearable.csv"
    if not (ehr_path.exists() and w_path.exists()):
        from data_generation import generate_cohort
        print("Data not found -> generating synthetic cohort ...")
        DATA_DIR.mkdir(exist_ok=True)
        e, w = generate_cohort(100, 42)
        e.to_csv(ehr_path, index=False); w.to_csv(w_path, index=False)

    ehr = pd.read_csv(ehr_path)
    wearable = pd.read_csv(w_path, parse_dates=["timestamp"])
    bundle = train(ehr, wearable)
    save_bundle(bundle)
    print(json.dumps({k: v for k, v in bundle["metrics"].items() if k != "feature_importance_top15"}, indent=2))
    print(f"\nModel saved -> {MODEL_PATH}")
