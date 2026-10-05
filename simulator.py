"""
simulator.py - Mechanistic "twin engine" for glucose dynamics.

This light-weight, physiology-inspired model is used in TWO places:
  1. data_generation.py  -> produces realistic synthetic CGM traces.
  2. app.py (What-If tab)-> projects the patient's glucose under different
                            diet / medication / activity scenarios.

Model (simplified, explainable to clinicians):

    G(t) = fasting + circadian(t) + sleep_adjust(t)
           + SUM_over_meals [ carb_gain * carbs * (1 - walk_effect) * gain_mult * K(t - t_meal) ]

    K(dt) = x^2 * exp(2 * (1 - x)),  x = dt / peak_min   (peak = 1 at x = 1, 0 for dt <= 0)

The ML model (model_pipeline.py) is the *data-driven* risk engine;
this file is the *mechanistic* what-if engine. Together = hybrid digital twin.

NOTE: Synthetic / educational only. Not a medical device.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Medication library: (change in fasting glucose mg/dL, multiplier on meal response)
# Values are illustrative, loosely inspired by typical drug-class effect sizes.
# --------------------------------------------------------------------------
MEDICATIONS = {
    "None (lifestyle only)": (0.0, 1.00),
    "Metformin": (-12.0, 0.88),
    "Metformin + Glimepiride": (-22.0, 0.80),
    "Metformin + DPP-4 inhibitor": (-20.0, 0.82),
    "Metformin + Basal Insulin": (-32.0, 0.74),
}
MED_INTENSITY = {name: i for i, name in enumerate(MEDICATIONS)}


@dataclass
class TwinParams:
    """Patient-specific parameters of the twin."""

    fasting: float          # baseline fasting glucose (mg/dL) under CURRENT regimen
    carb_gain: float        # mg/dL rise per gram of carbohydrate
    peak_min: float = 55.0  # minutes from meal to glucose peak
    dawn_amp: float = 10.0  # dawn-phenomenon amplitude (mg/dL)


def estimate_twin_params(hba1c: float, bmi: float, diet_glycemic_factor: float = 1.0) -> TwinParams:
    """Initial twin parameters derived ONLY from static EHR values."""
    eag = 28.7 * hba1c - 46.7                       # ADAG formula: HbA1c -> mean glucose
    fasting = float(np.clip(0.70 * eag + 12.0, 85, 230))
    gain = 0.80 + 0.02 * (bmi - 22.0) + 0.08 * (hba1c - 6.5)
    return TwinParams(fasting=fasting,
                      carb_gain=float(np.clip(gain, 0.5, 1.8)) * diet_glycemic_factor)


def meal_kernel(dt_min: np.ndarray, peak_min: float) -> np.ndarray:
    """Normalised post-meal glucose response shape (1.0 at peak)."""
    x = np.clip(dt_min, 0, None) / peak_min
    resp = (x ** 2) * np.exp(2.0 * (1.0 - x))
    return np.where(dt_min > 0, resp, 0.0)


def walk_effect(walk_min: float) -> float:
    """Fractional reduction of a meal's glucose rise from a post-meal walk (cap 40%)."""
    return float(min(0.40, 0.013 * max(walk_min, 0.0)))


def simulate_glucose(
    index: pd.DatetimeIndex,
    meals: pd.DataFrame,
    params: TwinParams,
    fasting_adjust: np.ndarray | None = None,
) -> np.ndarray:
    """
    Noise-free glucose trajectory (mg/dL) on `index`.

    meals: DataFrame with columns
        time (Timestamp), carbs (g), and optionally
        gain_mult (default 1.0), walk_min (default 0.0)
    Meals that happened BEFORE index[0] still contribute through their tail.
    """
    t0 = index[0]
    mins = ((index - t0).total_seconds() / 60.0).to_numpy()
    hours = np.asarray(index.hour + index.minute / 60.0)

    glucose = params.fasting + params.dawn_amp * np.exp(-0.5 * ((hours - 6.5) / 1.5) ** 2)
    if fasting_adjust is not None:
        glucose = glucose + fasting_adjust

    if meals is not None and len(meals):
        gain_mult = meals["gain_mult"].to_numpy() if "gain_mult" in meals else np.ones(len(meals))
        walks = meals["walk_min"].to_numpy() if "walk_min" in meals else np.zeros(len(meals))
        for t_meal, carbs, gm, wk in zip(meals["time"], meals["carbs"], gain_mult, walks):
            meal_min = (pd.Timestamp(t_meal) - t0).total_seconds() / 60.0
            amp = params.carb_gain * carbs * gm * (1.0 - walk_effect(wk))
            glucose = glucose + amp * meal_kernel(mins - meal_min, params.peak_min)
    return glucose


def calibrate_twin(
    index: pd.DatetimeIndex,
    meals: pd.DataFrame,
    observed: np.ndarray,
    init: TwinParams,
) -> tuple[TwinParams, float]:
    """
    Personalise the twin: grid-search a gain multiplier and fasting offset that
    best reproduce the patient's OBSERVED CGM trace (minimises RMSE).
    Returns (calibrated_params, rmse_mg_dl).
    """
    best, best_rmse = init, np.inf
    mask = ~np.isnan(observed)
    for m in np.linspace(0.5, 1.8, 27):
        for off in range(-40, 41, 8):
            cand = replace(init, carb_gain=init.carb_gain * m, fasting=init.fasting + off)
            sim = simulate_glucose(index, meals, cand)
            rmse = float(np.sqrt(np.mean((sim[mask] - observed[mask]) ** 2)))
            if rmse < best_rmse:
                best, best_rmse = cand, rmse
    return best, best_rmse
