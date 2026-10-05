"""
data_generation.py - Synthetic, fully anonymised data for a Type 2 Diabetes digital twin.

Two fused data streams (Synthea-inspired, custom time-series engine):

  1. STATIC / HISTORICAL EHR  -> data/ehr.csv
     Demographics (age, gender, Indian city/state), BMI, HbA1c, baseline BP,
     family history, comorbidities / past diagnoses, medication, diet, activity.

  2. DYNAMIC / WEARABLE       -> data/wearable.csv
     7 days x 96 readings/day (every 15 min) of:
       CGM glucose (mg/dL), heart rate (BPM), step count, sleep quality index (0-100),
       plus the logged meal carbohydrate (g) event.

No real patient data is used. IDs are random (DT-0001 ...), no names / contact details.

Usage:
    python data_generation.py --patients 100 --seed 42 --out data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from simulator import (MEDICATIONS, TwinParams, estimate_twin_params, simulate_glucose)

START = pd.Timestamp("2026-09-28 00:00")   # fixed anchor -> reproducible 7-day window
STEP_MIN = 15
STEPS_PER_DAY = 96

# (city, state, region, sampling weight) - T2D burden is high in South & Urban India
CITIES = [
    ("Vijayawada", "Andhra Pradesh", "South", 8), ("Visakhapatnam", "Andhra Pradesh", "South", 6),
    ("Hyderabad", "Telangana", "South", 9), ("Chennai", "Tamil Nadu", "South", 9),
    ("Bengaluru", "Karnataka", "South", 9), ("Kochi", "Kerala", "South", 6),
    ("Mumbai", "Maharashtra", "West", 8), ("Pune", "Maharashtra", "West", 5),
    ("Ahmedabad", "Gujarat", "West", 5), ("Delhi", "Delhi", "North", 9),
    ("Lucknow", "Uttar Pradesh", "North", 4), ("Chandigarh", "Chandigarh", "North", 4),
    ("Jaipur", "Rajasthan", "North", 4), ("Kolkata", "West Bengal", "East", 6),
    ("Bhubaneswar", "Odisha", "East", 3),
]

# Typical carbohydrate (g) per meal slot by diet pattern
DIETS = {
    "South Indian (rice-based)": {"B": 55, "L": 90, "S": 25, "D": 70, "gf": 1.08},
    "North Indian (roti-based)": {"B": 45, "L": 75, "S": 25, "D": 60, "gf": 1.00},
    "Mixed / lower-carb":        {"B": 35, "L": 55, "S": 15, "D": 45, "gf": 0.90},
}
# mean meal time (hour of day), sd (hours)
MEAL_TIMES = {"B": (8.0, 0.5), "L": (13.3, 0.5), "S": (17.0, 0.4), "D": (20.5, 0.6)}


# ----------------------------------------------------------------------------
# 1. STATIC EHR
# ----------------------------------------------------------------------------
def generate_ehr(n_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    weights = np.array([c[3] for c in CITIES], dtype=float)
    weights /= weights.sum()
    for i in range(n_patients):
        city, state, region, _ = CITIES[rng.choice(len(CITIES), p=weights)]
        gender = rng.choice(["Male", "Female"], p=[0.54, 0.46])
        age = int(np.clip(rng.normal(54, 11), 28, 82))
        bmi = float(np.clip(rng.normal(27.5, 4.2), 18.5, 42))          # Asian-Indian phenotype
        hba1c = float(np.clip(rng.normal(8.0, 1.3), 6.1, 12.0))
        duration = int(np.clip(rng.exponential(7), 0, max(1, age - 25)))
        family_hx = int(rng.random() < 0.62)                            # strong familial burden in India
        hypertension = int(rng.random() < (0.25 + 0.012 * (age - 40) + 0.01 * (bmi - 25)))
        dyslipidemia = int(rng.random() < (0.35 + 0.01 * (bmi - 25)))
        smoker = int(rng.random() < (0.22 if gender == "Male" else 0.03))
        sys_bp = float(np.clip(116 + 0.45 * (age - 40) + 0.8 * (bmi - 25) + 12 * hypertension
                               + rng.normal(0, 8), 95, 190))
        dia_bp = float(np.clip(0.55 * sys_bp + rng.normal(8, 5), 60, 115))

        # Regimen escalates with HbA1c and disease duration
        score = (hba1c - 6.5) + 0.12 * duration + rng.normal(0, 0.8)
        if score < 0.6:
            med = "None (lifestyle only)"
        elif score < 2.0:
            med = "Metformin"
        elif score < 3.2:
            med = rng.choice(["Metformin + Glimepiride", "Metformin + DPP-4 inhibitor"])
        else:
            med = "Metformin + Basal Insulin"

        diet = rng.choice(list(DIETS), p={"South": [.62, .12, .26], "West": [.15, .45, .40],
                                          "North": [.05, .75, .20], "East": [.30, .30, .40]}[region])
        activity = rng.choice(["Sedentary", "Light", "Active"], p=[0.40, 0.42, 0.18])

        dx = ["Type 2 Diabetes Mellitus"]
        if hypertension: dx.append("Essential Hypertension")
        if dyslipidemia: dx.append("Dyslipidemia")
        if bmi >= 30: dx.append("Obesity")
        if duration > 10 and rng.random() < 0.35: dx.append("Diabetic Peripheral Neuropathy")
        if duration > 8 and rng.random() < 0.25: dx.append("Diabetic Nephropathy (early)")

        rows.append({
            "patient_id": f"DT-{i + 1:04d}", "age": age, "gender": gender,
            "city": city, "state": state, "bmi": round(bmi, 1), "hba1c": round(hba1c, 1),
            "systolic_bp": round(sys_bp), "diastolic_bp": round(dia_bp),
            "family_history_diabetes": family_hx, "diabetes_duration_yrs": duration,
            "hypertension": hypertension, "dyslipidemia": dyslipidemia, "smoker": smoker,
            "medication": med, "diet_type": diet, "activity_level": activity,
            "past_diagnoses": "; ".join(dx),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2. DYNAMIC WEARABLE STREAM
# ----------------------------------------------------------------------------
def _ar1_noise(n: int, sd: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Autocorrelated noise (sensor drift + unmodelled physiology)."""
    out = np.zeros(n)
    eps = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def generate_wearable(ehr_row: pd.Series, rng: np.random.Generator, days: int = 7) -> pd.DataFrame:
    n = days * STEPS_PER_DAY
    index = pd.date_range(START, periods=n, freq=f"{STEP_MIN}min")
    hours = np.asarray(index.hour + index.minute / 60.0)
    diet = DIETS[ehr_row["diet_type"]]

    # ---- latent patient traits (hidden from the EHR -> keeps the ML task realistic) ----
    base = estimate_twin_params(ehr_row["hba1c"], ehr_row["bmi"], diet["gf"])
    true_params = TwinParams(
        fasting=base.fasting + rng.normal(0, 10),
        carb_gain=base.carb_gain * rng.uniform(0.75, 1.30),
        peak_min=float(rng.uniform(45, 70)),
    )
    # medication already lowers glucose -> shift to reflect current regimen
    fast_d, gain_m = MEDICATIONS[ehr_row["medication"]]
    true_params.fasting += 0.4 * fast_d
    true_params.carb_gain *= (0.5 + 0.5 * gain_m)
    indulgence = rng.uniform(0.85, 1.20)
    walk_prob = {"Sedentary": 0.15, "Light": 0.40, "Active": 0.70}[ehr_row["activity_level"]]

    # ---- sleep quality per day (index 0-100); poor sleep raises next-day fasting glucose ----
    sleep_base = rng.normal(70, 9)
    sleep_day = np.clip(rng.normal(sleep_base, 9, days), 30, 98)
    sleep_row = np.repeat(sleep_day, STEPS_PER_DAY)

    # ---- meals ----
    meal_rows, steps = [], np.zeros(n)
    for d in range(days):
        for slot in "BLSD":
            if slot == "S" and rng.random() > 0.7:
                continue
            mu, sd = MEAL_TIMES[slot]
            hr_ = float(np.clip(rng.normal(mu, sd), 6.0, 22.5))
            t = START + pd.Timedelta(days=d) + pd.Timedelta(minutes=round(hr_ * 60 / STEP_MIN) * STEP_MIN)
            carbs = max(8.0, diet[slot] * indulgence * rng.normal(1.0, 0.18))
            if rng.random() < 0.05:                     # festival / feast meal
                carbs *= 1.6
            walk = float(rng.integers(10, 26)) if rng.random() < walk_prob else 0.0
            meal_rows.append({"time": t, "carbs": round(carbs, 0), "walk_min": walk})
            if walk > 0:                                # walking bout ~ 95 steps/min starting 10 min post-meal
                i0 = int((t - START).total_seconds() // 60 // STEP_MIN) + 1
                for k in range(i0, min(n, i0 + 2)):
                    steps[k] += min(walk, 15) * 95 if k == i0 else max(walk - 15, 0) * 95
    meals = pd.DataFrame(meal_rows)

    # baseline daytime movement + optional morning walk
    awake = ((hours >= 6) & (hours < 22.5)).astype(float)
    activity_scale = {"Sedentary": 18, "Light": 32, "Active": 50}[ehr_row["activity_level"]]
    steps += rng.poisson(activity_scale * awake + 1)
    if ehr_row["activity_level"] == "Active":
        steps[(hours >= 6) & (hours < 6.75)] += rng.normal(1100, 150, ((hours >= 6) & (hours < 6.75)).sum())
    steps = np.round(np.clip(steps, 0, 2200)).astype(int)

    # ---- glucose = twin engine + AR(1) noise ----
    fasting_adj = (70 - sleep_row) * 0.30
    g = simulate_glucose(index, meals, true_params, fasting_adjust=fasting_adj)
    g = g + _ar1_noise(n, sd=5.0, phi=0.85, rng=rng) + rng.normal(0, 1.5, n)
    g = np.clip(g, 45, 400)

    # ---- heart rate ----
    rest = 60 + 0.12 * (ehr_row["age"] - 40) + 0.30 * (ehr_row["bmi"] - 25) + rng.normal(0, 3.5)
    asleep = ((hours >= 23) | (hours < 6)).astype(float)
    hr = (rest - 7 * asleep + np.minimum(0.022 * steps, 45) + 0.03 * (g - 120)
          + _ar1_noise(n, 2.0, 0.7, rng))
    hr = np.round(np.clip(hr, 42, 175)).astype(int)

    # ---- logged meal events on the 15-min grid ----
    meal_col = np.zeros(n)
    for _, m in meals.iterrows():
        meal_col[int((m["time"] - START).total_seconds() // 60 // STEP_MIN)] += m["carbs"]

    return pd.DataFrame({
        "patient_id": ehr_row["patient_id"], "timestamp": index,
        "glucose_mg_dl": np.round(g, 1), "heart_rate_bpm": hr, "steps": steps,
        "sleep_quality_index": np.round(sleep_row, 0).astype(int), "meal_carbs_g": meal_col,
    })


def generate_cohort(n_patients: int = 100, seed: int = 42, days: int = 7):
    rng = np.random.default_rng(seed)
    ehr = generate_ehr(n_patients, rng)
    wearable = pd.concat([generate_wearable(r, rng, days) for _, r in ehr.iterrows()],
                         ignore_index=True)
    return ehr, wearable


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic T2D EHR + wearable data")
    ap.add_argument("--patients", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ehr, wearable = generate_cohort(args.patients, args.seed)
    ehr.to_csv(out / "ehr.csv", index=False)
    wearable.to_csv(out / "wearable.csv", index=False)

    g = wearable["glucose_mg_dl"]
    print(f"EHR rows: {len(ehr)} | wearable rows: {len(wearable):,}")
    print(f"Glucose mean {g.mean():.0f} mg/dL | time >180: {(g > 180).mean():.1%} | "
          f"time <70: {(g < 70).mean():.1%}")
    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
