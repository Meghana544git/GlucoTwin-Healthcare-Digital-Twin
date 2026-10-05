"""
app.py - Doctor-facing Digital Twin dashboard (Streamlit).

Run:  streamlit run app.py

Tabs
  1. Cohort Overview    - simulated patient list with live risk alerts
  2. Digital Twin       - fused EHR profile + wearable graphs + 2-hour spike prediction
  3. What-If Simulator  - diet / medication / activity scenarios on the personalised twin
  4. Model & Methods    - evaluation metrics, feature importance, safety notes

The sidebar "replay clock" simulates real-time streaming: move it and every alert,
graph and prediction is recomputed using ONLY data available up to that moment.

Synthetic data only. Educational proof-of-concept - not a medical device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import model_pipeline as mp
from data_generation import DIETS
from simulator import (MEDICATIONS, calibrate_twin, estimate_twin_params, simulate_glucose)

st.set_page_config(page_title="Diabetes Digital Twin", page_icon="🩺", layout="wide")

DATA_DIR = Path("data")
STEP = 15  # minutes between readings
FRIENDLY = {
    "glucose": "Current glucose", "glucose_delta_30m": "Glucose change (30 min)",
    "glucose_delta_60m": "Glucose change (60 min)", "glucose_lag_15m": "Glucose 15 min ago",
    "glucose_lag_30m": "Glucose 30 min ago", "glucose_lag_60m": "Glucose 1 h ago",
    "glucose_lag_120m": "Glucose 2 h ago", "glucose_mean_1h": "Mean glucose (1 h)",
    "glucose_mean_3h": "Mean glucose (3 h)", "glucose_max_3h": "Peak glucose (3 h)",
    "glucose_std_1h": "Glucose variability (1 h)", "glucose_mean_24h": "Mean glucose (24 h)",
    "glucose_cv_24h": "Glucose variability (24 h)", "time_above_180_24h": "Time >180 (24 h)",
    "time_in_range_24h": "Time in range (24 h)", "mins_since_meal": "Minutes since last meal",
    "last_meal_carbs": "Last meal carbs (g)", "carbs_last_3h": "Carbs in last 3 h (g)",
    "carbs_last_6h": "Carbs in last 6 h (g)", "hour_sin": "Time of day", "hour_cos": "Time of day",
    "heart_rate": "Heart rate", "heart_rate_mean_1h": "Heart rate (1 h mean)",
    "steps_1h": "Steps (last 1 h)", "steps_3h": "Steps (last 3 h)", "steps_24h": "Steps (24 h)",
    "sleep_quality": "Sleep quality index", "hba1c": "HbA1c", "bmi": "BMI", "age": "Age",
    "med_intensity": "Medication intensity", "diabetes_duration_yrs": "Diabetes duration",
}


# ----------------------------------------------------------------------------
# Data / model loading (cached; auto-generates + trains on the very first run)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing synthetic cohort and Digital Twin model (first run only) ...")
def load_everything():
    ehr_path, w_path = DATA_DIR / "ehr.csv", DATA_DIR / "wearable.csv"
    if not (ehr_path.exists() and w_path.exists()):
        from data_generation import generate_cohort
        DATA_DIR.mkdir(exist_ok=True)
        e, w = generate_cohort(100, 42)
        e.to_csv(ehr_path, index=False)
        w.to_csv(w_path, index=False)
    ehr = pd.read_csv(ehr_path)
    wearable = pd.read_csv(w_path, parse_dates=["timestamp"])

    if not mp.MODEL_PATH.exists():
        mp.save_bundle(mp.train(ehr, wearable))
    bundle = mp.load_bundle()

    ehr_p = mp.prepare_ehr(ehr)
    frames, matrices = {}, {}
    for pid, w in wearable.groupby("patient_id"):
        w = w.sort_values("timestamp").reset_index(drop=True)
        frames[pid] = w
        matrices[pid] = mp.patient_matrix(w, ehr_p.loc[pid])   # causal features for every timestamp
    return ehr.set_index("patient_id"), frames, matrices, bundle


EHR, FRAMES, MATRICES, BUNDLE = load_everything()
PIDS = list(EHR.index)
N_STEPS = len(FRAMES[PIDS[0]])
THR = BUNDLE["threshold"]


def risk_level(p: float) -> str:
    if p >= THR:
        return "High"
    return "Moderate" if p >= 0.5 * THR else "Low"


LEVEL_ICON = {"High": "🔴 High", "Moderate": "🟠 Moderate", "Low": "🟢 Low"}


@st.cache_data(show_spinner=False)
def cohort_snapshot(idx: int) -> pd.DataFrame:
    """Risk of every patient at replay time `idx` (uses only past data)."""
    rows = pd.concat([MATRICES[p].iloc[[idx]].assign(patient_id=p) for p in PIDS], ignore_index=True)
    proba, pred = mp.predict_rows(BUNDLE, rows)
    out = pd.DataFrame({
        "Patient": rows["patient_id"], "Age": rows["age"].astype(int),
        "Sex": np.where(rows["gender_male"] == 1, "M", "F"),
        "City": [EHR.loc[p, "city"] for p in rows["patient_id"]],
        "HbA1c": rows["hba1c"], "Medication": rows["medication"],
        "Glucose now": rows["glucose"].round(0), "30-min trend": rows["glucose_delta_30m"].round(0),
        "Predicted +2h": pred.round(0), "Spike risk": proba,
    })
    out["Alert"] = [LEVEL_ICON[risk_level(p)] for p in proba]
    return out.sort_values("Spike risk", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def calibrated_twin(pid: str, idx: int):
    """Personalise the mechanistic twin to the last <=3 days of the patient's CGM."""
    w = FRAMES[pid]
    start = max(0, idx - 288)
    index = pd.DatetimeIndex(w["timestamp"].iloc[start: idx + 1])
    logged = logged_meals(w.iloc[: idx + 1])
    e = EHR.loc[pid]
    init = estimate_twin_params(e["hba1c"], e["bmi"], DIETS[e["diet_type"]]["gf"])
    params, rmse = calibrate_twin(index, logged, w["glucose_mg_dl"].iloc[start: idx + 1].to_numpy(), init)
    return params, rmse


def logged_meals(w: pd.DataFrame) -> pd.DataFrame:
    m = w.loc[w["meal_carbs_g"] > 0, ["timestamp", "meal_carbs_g"]]
    return pd.DataFrame({"time": m["timestamp"].to_numpy(), "carbs": m["meal_carbs_g"].to_numpy()})


def meal_slot(hour: float) -> str:
    return "Breakfast" if hour < 11 else "Lunch" if hour < 15.5 else "Snack" if hour < 18.5 else "Dinner"


def typical_future_meals(w: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Habitual meal pattern (mean time & carbs per slot) projected over the next 24 h."""
    past = logged_meals(w[w["timestamp"] <= now])
    if past.empty:
        return pd.DataFrame(columns=["time", "carbs", "slot"])
    past["hour"] = pd.DatetimeIndex(past["time"]).hour + pd.DatetimeIndex(past["time"]).minute / 60
    past["slot"] = past["hour"].map(meal_slot)
    n_days = max(1, (now - w["timestamp"].iloc[0]).total_seconds() / 86400)
    rows = []
    for slot, g in past.groupby("slot"):
        if len(g) / n_days < 0.5:               # skip rarely-eaten slots (e.g. snack)
            continue
        for d in range(0, 3):
            t = now.normalize() + pd.Timedelta(days=d) + pd.Timedelta(hours=float(g["hour"].mean()))
            if now < t <= now + pd.Timedelta(hours=24):
                already = ((past["slot"] == slot) &
                           ((pd.DatetimeIndex(past["time"]) - t).total_seconds().to_series().abs().to_numpy() < 3 * 3600)).any()
                if not already:
                    rows.append({"time": t, "carbs": float(g["carbs"].mean()), "slot": slot})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True) if rows else \
        pd.DataFrame(columns=["time", "carbs", "slot"])


def glycemic_metrics(g: pd.Series) -> dict:
    mean = float(g.mean())
    return {"Mean glucose": f"{mean:.0f} mg/dL", "GMI (est. A1c)": f"{3.31 + 0.02392 * mean:.1f} %",
            "Time in range 70-180": f"{((g >= 70) & (g <= 180)).mean():.0%}",
            "Time above 180": f"{(g > 180).mean():.0%}", "Time below 70": f"{(g < 70).mean():.0%}",
            "Variability (CV)": f"{g.std() / mean:.0%}"}


# ----------------------------------------------------------------------------
# Sidebar: replay clock + patient picker
# ----------------------------------------------------------------------------
st.session_state.setdefault("t_idx", N_STEPS - 1)


def _shift(steps: int):
    st.session_state["t_idx"] = int(np.clip(st.session_state["t_idx"] + steps, 96, N_STEPS - 1))


def _jump_to_top_risk():
    st.session_state["pid_select"] = cohort_snapshot(st.session_state["t_idx"]).iloc[0]["Patient"]


st.sidebar.title("🩺 Diabetes Digital Twin")
st.sidebar.caption("Type 2 Diabetes · CGM & metabolic management · synthetic data")
st.sidebar.subheader("⏱ Live replay clock")
st.sidebar.slider("Simulated 'now' (reading #)", 96, N_STEPS - 1, key="t_idx",
                  help="Predictions use only data up to this moment.")
c1, c2, c3 = st.sidebar.columns(3)
c1.button("−15m", on_click=_shift, args=(-1,), width="stretch")
c2.button("+15m", on_click=_shift, args=(1,), width="stretch")
c3.button("+1h", on_click=_shift, args=(4,), width="stretch")
IDX = st.session_state["t_idx"]
NOW = FRAMES[PIDS[0]]["timestamp"].iloc[IDX]
st.sidebar.info(f"Now: **{NOW:%a %d %b, %H:%M}**")

st.sidebar.subheader("👤 Patient")
st.sidebar.button("🚨 Jump to highest-risk patient", on_click=_jump_to_top_risk, width="stretch")
PID = st.sidebar.selectbox(
    "Select patient", PIDS, key="pid_select",
    format_func=lambda p: f"{p} · {EHR.loc[p, 'age']}y {EHR.loc[p, 'gender'][0]} · {EHR.loc[p, 'city']}")
st.sidebar.warning("Synthetic data. Decision-support prototype - not for clinical use.")

# ----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🏥 Cohort Overview", "🧬 Digital Twin", "🔮 What-If Simulator", "ℹ️ Model & Methods"])

# ============================ TAB 1: COHORT ===================================
with tab1:
    snap = cohort_snapshot(IDX)
    st.subheader(f"Patient list · {NOW:%a %d %b %H:%M}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients monitored", len(snap))
    k2.metric("🔴 High spike risk (next 2 h)", int((snap["Alert"] == LEVEL_ICON["High"]).sum()))
    k3.metric("🟠 Moderate", int((snap["Alert"] == LEVEL_ICON["Moderate"]).sum()))
    k4.metric("Currently >180 mg/dL", int((snap["Glucose now"] > 180).sum()))

    f1, f2 = st.columns([2, 3])
    levels = f1.multiselect("Filter by alert level", list(LEVEL_ICON.values()), default=list(LEVEL_ICON.values()))
    query = f2.text_input("Search (patient ID, city, medication)")
    view = snap[snap["Alert"].isin(levels)]
    if query:
        mask = view[["Patient", "City", "Medication"]].astype(str).apply(
            lambda c: c.str.contains(query, case=False)).any(axis=1)
        view = view[mask]
    st.dataframe(
        view, hide_index=True, width="stretch", height=480,
        column_config={"Spike risk": st.column_config.ProgressColumn(
            "Spike risk (2 h)", min_value=0.0, max_value=1.0, format="percent")})
    st.caption("Pick a patient in the sidebar to open their Digital Twin. Sorted by predicted risk.")

# ============================ TAB 2: TWIN =====================================
with tab2:
    w, X, e = FRAMES[PID], MATRICES[PID], EHR.loc[PID]
    proba, pred = mp.predict_rows(BUNDLE, X.iloc[[IDX]])
    p, pred_g = float(proba[0]), float(pred[0])
    level, g_now = risk_level(p), float(w["glucose_mg_dl"].iloc[IDX])
    t_alert = NOW + pd.Timedelta(minutes=mp.HORIZON_MIN)

    st.subheader(f"Digital Twin · {PID}")

    # ---- real-time predictive alert ----
    if g_now < 70:
        st.error(f"🔵 Hypoglycaemia now: {g_now:.0f} mg/dL - treat immediately (15 g fast-acting carbohydrate).")
    if level == "High":
        st.error(f"🔴 **High Risk of Glucose Spike in {mp.HORIZON_MIN} mins** - predicted ≈ **{pred_g:.0f} mg/dL** "
                 f"around {t_alert:%H:%M} (risk {p:.0%}).")
    elif level == "Moderate":
        st.warning(f"🟠 **Moderate risk** of exceeding 180 mg/dL by {t_alert:%H:%M} "
                   f"(predicted ≈ {pred_g:.0f} mg/dL, risk {p:.0%}).")
    else:
        st.success(f"🟢 **Low risk** of a glucose spike in the next {mp.HORIZON_MIN} mins "
                   f"(predicted ≈ {pred_g:.0f} mg/dL, risk {p:.0%}).")

    # ---- EHR profile ----
    st.markdown("##### 📋 Static EHR profile")
    a, b, c, d, f_ = st.columns(5)
    a.metric("Age / Sex", f"{e['age']} / {e['gender']}")
    b.metric("Location", e["city"], e["state"], delta_color="off")
    c.metric("BMI", f"{e['bmi']}")
    d.metric("HbA1c (baseline)", f"{e['hba1c']} %")
    f_.metric("Baseline BP", f"{e['systolic_bp']}/{e['diastolic_bp']}")
    st.markdown(
        f"**Medication:** {e['medication']} &nbsp;|&nbsp; **Diet:** {e['diet_type']} &nbsp;|&nbsp; "
        f"**Activity:** {e['activity_level']} &nbsp;|&nbsp; **Duration:** {e['diabetes_duration_yrs']} yrs "
        f"&nbsp;|&nbsp; **Family history:** {'Yes' if e['family_history_diabetes'] else 'No'}  \n"
        f"**Past diagnoses:** {e['past_diagnoses']}")

    # ---- wearable graphs ----
    st.markdown("##### ⌚ Real-time wearable stream")
    win_label = st.radio("Window", ["Last 12 h", "Last 24 h", "Last 48 h", "All data"], index=1, horizontal=True)
    win = {"Last 12 h": 48, "Last 24 h": 96, "Last 48 h": 192, "All data": N_STEPS}[win_label]
    seg = w.iloc[max(0, IDX - win + 1): IDX + 1]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.2, 0.25],
                        subplot_titles=("CGM glucose (mg/dL)", "Heart rate (BPM)", "Steps per 15 min"))
    fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.07, line_width=0, row=1, col=1)
    fig.add_hline(y=180, line_dash="dash", line_color="crimson", row=1, col=1)
    fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["glucose_mg_dl"], mode="lines", name="Glucose",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    ml = seg[seg["meal_carbs_g"] > 0]
    fig.add_trace(go.Scatter(x=ml["timestamp"], y=ml["glucose_mg_dl"], mode="markers", name="Meal",
                             marker=dict(symbol="triangle-down", size=11, color="orange"),
                             text=[f"{c:.0f} g carbs" for c in ml["meal_carbs_g"]], hoverinfo="text+x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[NOW, t_alert], y=[g_now, pred_g], mode="lines+markers", name="AI forecast (+2 h)",
                             line=dict(color="crimson" if level == "High" else "gray", dash="dot"),
                             marker=dict(symbol=["circle", "star"], size=[7, 15])), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg["timestamp"], y=seg["heart_rate_bpm"], name="Heart rate",
                             line=dict(color="#d62728", width=1.5)), row=2, col=1)
    fig.add_trace(go.Bar(x=seg["timestamp"], y=seg["steps"], name="Steps", marker_color="#2ca02c"), row=3, col=1)
    fig.update_layout(height=640, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(range=[40, max(260, float(seg["glucose_mg_dl"].max()) + 20, pred_g + 20)], row=1, col=1)
    st.plotly_chart(fig, width="stretch")

    m1, m2 = st.columns([3, 2])
    with m1:
        st.markdown(f"**Glycaemic metrics ({win_label.lower()})**")
        gm = glycemic_metrics(seg["glucose_mg_dl"])
        cols = st.columns(len(gm))
        for col, (k, v) in zip(cols, gm.items()):
            col.metric(k, v)
        sleep_now = int(w["sleep_quality_index"].iloc[IDX])
        st.caption(f"Sleep quality index last night: **{sleep_now}/100** · steps in last 24 h: "
                   f"**{int(w['steps'].iloc[max(0, IDX - 95): IDX + 1].sum()):,}**")
    with m2:
        st.markdown("**Why this risk? (top drivers)**")
        ex = mp.explain_row(BUNDLE, X.iloc[[IDX]], top_k=6).iloc[::-1]
        ex["label"] = [f"{FRIENDLY.get(f, f)} = {v:.1f}" for f, v in zip(ex["feature"], ex["value"])]
        bar = go.Figure(go.Bar(x=ex["contribution"], y=ex["label"], orientation="h",
                               marker_color=["crimson" if c > 0 else "seagreen" for c in ex["contribution"]]))
        bar.update_layout(height=260, margin=dict(l=0, r=10, t=10, b=10),
                          xaxis_title="← lowers risk | raises risk →")
        st.plotly_chart(bar, width="stretch")

# ============================ TAB 3: WHAT-IF ==================================
with tab3:
    st.subheader(f"What-If Treatment Simulator · {PID}")
    st.caption("A mechanistic twin, personalised to this patient's recent CGM trace, projects the next 24 h "
               "under a modified plan. Compare it to the current plan.")
    w, e = FRAMES[PID], EHR.loc[PID]
    params, fit_rmse = calibrated_twin(PID, IDX)

    cL, cR = st.columns([1, 2])
    with cL:
        st.markdown("##### 🎛 Scenario controls")
        carb_scale = st.slider("Dietary carbohydrate intake", 50, 150, 100, 5, format="%d%%",
                               help="Scales every upcoming meal. 70% ≈ swapping part of the rice/roti for veg + protein.")
        meds = list(MEDICATIONS)
        med_new = st.selectbox("Medication regimen", meds, index=meds.index(e["medication"]))
        dose = st.slider("Dose / adherence intensity", 0.5, 1.5, 1.0, 0.1,
                         help="1.0 = current prescribed dose. 0.5 = missed/halved. 1.5 = up-titrated.")
        walk_min = st.slider("Post-meal walk (minutes)", 0, 30, 0, 5)
        st.caption(f"Twin personalisation error (RMSE vs last 3 days): **{fit_rmse:.1f} mg/dL** · "
                   f"carb sensitivity {params.carb_gain:.2f} mg/dL per g")

    # --- build baseline + scenario ---
    idx_future = pd.date_range(NOW - pd.Timedelta(hours=6), NOW + pd.Timedelta(hours=24), freq=f"{STEP}min")
    past = logged_meals(w[w["timestamp"] <= NOW])
    future = typical_future_meals(w, NOW)

    base_meals = pd.concat([past, future[["time", "carbs"]]], ignore_index=True)
    base_curve = simulate_glucose(idx_future, base_meals, params)

    cur_f, cur_g = MEDICATIONS[e["medication"]]
    new_f, new_g = MEDICATIONS[med_new]
    eff_f = new_f * dose
    eff_g = 1.0 - (1.0 - new_g) * dose
    scen_future = future[["time", "carbs"]].copy()
    scen_future["carbs"] *= carb_scale / 100
    scen_future["gain_mult"] = eff_g / cur_g
    scen_future["walk_min"] = float(walk_min)
    scen_meals = pd.concat([past.assign(gain_mult=1.0, walk_min=0.0), scen_future], ignore_index=True)
    ramp = np.clip((idx_future - NOW).total_seconds().to_numpy() / 7200, 0, 1)   # drug effect builds over 2 h
    scen_curve = simulate_glucose(idx_future, scen_meals, params, fasting_adjust=(eff_f - cur_f) * ramp)

    fut = idx_future >= NOW
    obs = w[(w["timestamp"] >= idx_future[0]) & (w["timestamp"] <= NOW)]

    with cR:
        fig = go.Figure()
        fig.add_hrect(y0=70, y1=180, fillcolor="green", opacity=0.07, line_width=0)
        fig.add_hline(y=180, line_dash="dash", line_color="crimson")
        fig.add_trace(go.Scatter(x=obs["timestamp"], y=obs["glucose_mg_dl"], name="Observed CGM",
                                 line=dict(color="black", width=2)))
        fig.add_trace(go.Scatter(x=idx_future[fut], y=base_curve[fut], name="Current plan (twin)",
                                 line=dict(color="#7f7f7f", width=2.5, dash="dash")))
        fig.add_trace(go.Scatter(x=idx_future[fut], y=scen_curve[fut], name="Scenario",
                                 line=dict(color="#2ca02c", width=3)))
        for _, m in future.iterrows():
            fig.add_annotation(x=m["time"], y=48, text=f"🍽 {m['slot']}", showarrow=False, font=dict(size=10))
        fig.add_vline(x=NOW, line_dash="dot", line_color="gray")
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Glucose (mg/dL)",
                          yaxis_range=[40, max(260, float(max(base_curve.max(), scen_curve.max())) + 20)],
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    def outcomes(curve):
        c = curve[fut]
        return {"peak": float(c.max()), "mean": float(c.mean()),
                "tir": float(((c >= 70) & (c <= 180)).mean() * 100), "tar_min": float((c > 180).sum() * STEP)}

    b, s = outcomes(base_curve), outcomes(scen_curve)
    st.markdown("##### 📊 Predicted impact over the next 24 h")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Peak glucose", f"{s['peak']:.0f} mg/dL", f"{s['peak'] - b['peak']:+.0f}", delta_color="inverse")
    o2.metric("Mean glucose", f"{s['mean']:.0f} mg/dL", f"{s['mean'] - b['mean']:+.0f}", delta_color="inverse")
    o3.metric("Time in range", f"{s['tir']:.0f}%", f"{s['tir'] - b['tir']:+.0f} pts")
    o4.metric("Time >180", f"{s['tar_min'] / 60:.1f} h", f"{(s['tar_min'] - b['tar_min']) / 60:+.1f} h", delta_color="inverse")

    t2h = np.searchsorted(idx_future, NOW + pd.Timedelta(minutes=mp.HORIZON_MIN))
    ml_pred = float(mp.predict_rows(BUNDLE, MATRICES[PID].iloc[[IDX]])[1][0])
    st.caption(f"Consistency check at +2 h → ML model: **{ml_pred:.0f}** mg/dL "
               f"· Twin (current plan): **{base_curve[t2h]:.0f}** mg/dL.")
    if s["tar_min"] < b["tar_min"] - 15:
        st.success(f"✅ This scenario cuts time above 180 mg/dL by {(b['tar_min'] - s['tar_min']) / 60:.1f} h "
                   f"and lowers the peak by {b['peak'] - s['peak']:.0f} mg/dL.")
    elif s["tar_min"] > b["tar_min"] + 15:
        st.warning("⚠️ This scenario increases hyperglycaemic exposure versus the current plan.")
    else:
        st.info("This scenario changes glucose exposure only marginally.")
    if scen_curve[fut].min() < 70:
        st.error("🔵 Scenario predicts glucose below 70 mg/dL - review hypoglycaemia risk before changing therapy.")
    st.caption("Medication effect sizes are illustrative. Final therapy decisions rest with the treating physician.")

# ============================ TAB 4: METHODS ==================================
with tab4:
    m = BUNDLE["metrics"]
    st.subheader("Model performance (held-out patients)")
    st.caption(f"Task: predict glucose and P(>{int(m['spike_threshold_mg_dl'])} mg/dL) {m['horizon_minutes']} min ahead · "
               f"{m['n_patients']} synthetic patients · {m['n_rows']:,} fused rows · patient-level train/val/test split.")
    r, c = m["regression"], m["classification"]
    x1, x2, x3, x4, x5, x6 = st.columns(6)
    x1.metric("MAE", f"{r['MAE_mg_dl']} mg/dL", f"{r['MAE_mg_dl'] - r['baseline_persistence_MAE']:+.1f} vs naive", delta_color="inverse")
    x2.metric("RMSE", f"{r['RMSE_mg_dl']} mg/dL")
    x3.metric("Accuracy", f"{c['accuracy']:.1%}")
    x4.metric("F1-score", f"{c['f1']:.3f}")
    x5.metric("Sensitivity", f"{c['recall_sensitivity']:.1%}")
    x6.metric("ROC-AUC", f"{c['roc_auc']:.3f}")
    left, right = st.columns(2)
    with left:
        imp = pd.Series(m["feature_importance_top15"]).iloc[::-1]
        fi = go.Figure(go.Bar(x=imp.values, y=[FRIENDLY.get(k, k) for k in imp.index], orientation="h"))
        fi.update_layout(height=420, title="Top feature importances (fused EHR + wearable)", margin=dict(l=0, r=10, t=40, b=10))
        st.plotly_chart(fi, width="stretch")
    with right:
        cm = c["confusion_matrix"]
        heat = go.Figure(go.Heatmap(z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], x=["No spike", "Spike"],
                                    y=["No spike", "Spike"], text=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                                    texttemplate="%{text}", colorscale="Blues", showscale=False))
        heat.update_layout(height=420, title="Confusion matrix (test set)", xaxis_title="Predicted",
                           yaxis_title="Actual", margin=dict(l=0, r=10, t=40, b=10))
        st.plotly_chart(heat, width="stretch")
    st.markdown(
        "**Hybrid Digital Twin** = data-driven risk engine (XGBoost on fused EHR + wearable features) "
        "+ mechanistic what-if engine (personalised meal-response model).  \n"
        "**Safety:** synthetic data only · decision *support*, not decision *making* · every alert shows its top drivers.")
