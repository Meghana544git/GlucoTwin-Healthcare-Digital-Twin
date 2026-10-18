# GlucoTwin - A Digital Twin for Type 2 Diabetes Management

> Happiest Health **Digital Twin Challenge 2026** (Unstop) - Proof-of-Concept submission

## 1. Team Details

| Field | Details |
|---|---|
| Team name | `CodeNCoffee` |
| College / Institution | `Prasad V. Potluri Siddhartha Institute of Technology` |
| Team leader | `Yara Meghana` |
| Member 2 | `Manogna Yara` |
| Demo video (unlisted YouTube) | `https://youtu.be/ltHecX_IwLU` |
| Live Link | `https://glucotwin-healthcare.streamlit.app/` |

---

## 2. Project Title & Problem Statement

**GlucoTwin: a hybrid (AI + physiology) Digital Twin that predicts glucose spikes two hours ahead and lets doctors test treatment changes virtually before prescribing them.**

India is home to over 100 million people living with diabetes, and most follow-ups are 10-minute visits every 1-3 months,
where decisions rely on a single HbA1c number. Between visits, glucose swings (post-meal spikes, dawn phenomenon, poor-sleep effects)
go unseen. Doctors have no way to ask *"what would happen to this patient if I reduce her rice portion, add a 15-minute walk, or up-titrate the drug?"*
without trying it on the real patient.

**Problem:** How can we fuse a patient's static medical history with continuous wearable data into a living virtual model that
(a) warns about adverse glucose events *before* they happen and (b) simulates "what-if" interventions safely?

---

## 3. Healthcare Use Case

| Item | Description |
|---|---|
| Condition | Type 2 Diabetes Mellitus (continuous glucose & metabolic management), prevalent across urban and semi-urban India |
| Users | Diabetologists / physicians (primary), care coaches (secondary) |
| Adverse event predicted | **Hyperglycaemic spike (> 180 mg/dL) 120 minutes ahead** |
| Why 2 hours? | Enough time for a meaningful action: a walk, a smaller next meal, a timely dose or a teleconsult nudge |
| Value to the doctor | Ranked patient list by live risk, explainable alerts, virtual testing of diet / drug / activity changes |
| Value to the patient | Fewer spikes, less long-term complication risk (neuropathy, nephropathy, retinopathy), personalised advice for Indian diets (rice-, roti- or mixed-based) |

**Indian context built into the data:** 15 cities across South/West/North/East India, regional diet patterns (South Indian rice-based, North Indian roti-based, mixed/lower-carb), early-onset phenotype with lower BMI thresholds, high family-history burden, and festival-meal carb surges.

---

## 4. Technical Architecture

### 4.1 System architecture (ASCII)

```
+------------------------------------------------------------------------------------+
|                               DATA LAYER (synthetic)                               |
|                                                                                    |
|  +---------------------------+              +-----------------------------------+  |
|  |  STATIC / HISTORICAL EHR  |              |   DYNAMIC WEARABLE STREAM (7 d)   |  |
|  |  age, sex, city/state     |              |  CGM glucose   (every 15 min)     |  |
|  |  BMI, HbA1c, BP           |              |  Heart rate    (BPM)              |  |
|  |  family hx, comorbidities |              |  Steps         (per 15 min)       |  |
|  |  medication, diet, activity|             |  Sleep quality index (0-100)      |  |
|  |        -> data/ehr.csv    |              |  Meal log (carbs g)               |  |
|  +-------------+-------------+              |        -> data/wearable.csv       |  |
|                |                            +----------------+------------------+  |
|                |  data_generation.py  (twin engine + AR(1) noise, Synthea-inspired) |
+----------------|-----------------------------------------|-------------------------+
                 |                                         |
                 v                                         v
+------------------------------------------------------------------------------------+
|                 FEATURE FUSION  (model_pipeline.py: patient_matrix)                |
|   EHR features  +  causal wearable features:                                       |
|   glucose lags (15/30/60/120m) - trends - 1h/3h/24h aggregates - TIR/TAR/CV        |
|   heart-rate & step windows - sleep - time since last meal - recent carbs - time   |
+----------------------------------------+-------------------------------------------+
                                         |
                  +----------------------+-----------------------+
                  v                                              v
+-----------------------------------------+   +------------------------------------+
|  DATA-DRIVEN RISK ENGINE (XGBoost)      |   |  MECHANISTIC TWIN ENGINE           |
|  - Classifier: P(glucose>180 in 120 min)|   |  (simulator.py)                    |
|  - Regressor : glucose value at +120 min|   |  G(t)=fasting+circadian+sleep      |
|  - Patient-level split, early stopping  |   |       + SUM carbs x gain x K(t)    |
|  - Explainability: SHAP-style drivers   |   |  Personalised per patient by       |
|  -> models/digital_twin_model.joblib    |   |  fitting the last 3 days of CGM    |
+--------------------+--------------------+   +----------------+-------------------+
                     |                                         |
                     +--------------------+--------------------+
                                          v
+------------------------------------------------------------------------------------+
|                    STREAMLIT DOCTOR DASHBOARD  (app.py)                            |
|  1. Cohort overview: ranked patients, live alerts     3. What-If simulator:        |
|  2. Digital Twin profile: EHR + wearable graphs,         diet / drug / dose /      |
|     "High risk of spike in 120 min" + top drivers        post-meal walk -> curves  |
|  Live replay clock simulates real-time streaming      4. Model metrics & methods   |
+------------------------------------------------------------------------------------+
```

### 4.2 Data flow

1. `data_generation.py` creates **100 anonymised patients**: static EHR + 7 days x 96 readings of CGM, HR, steps, sleep, meals.
2. `model_pipeline.py` fuses both streams into one table (**65,600 rows x 42 features**), splits **by patient** (train/val/test), trains the XGBoost classifier + regressor and saves one `joblib` bundle.
3. `app.py` loads the bundle; the **replay clock** exposes only data up to "now", features are recomputed, the model predicts the next 2 h, and alerts + drivers are displayed.
4. In the What-If tab, `simulator.py` is **calibrated to the patient's last 3 days of CGM**, then re-run with the doctor's changed diet / medication / activity to project 24 h glucose curves.

### 4.3 Why a *hybrid* twin?

| Engine | Strength | Used for |
|---|---|---|
| XGBoost (data-driven) | Accurate short-term forecasting, learns patterns across patients | Real-time 2-hour risk alerts |
| Mechanistic meal-response model | Responds sensibly to interventions never seen in the data | What-if treatment scenarios |

### 4.4 Results (reference run, seed 42, held-out patients)

| Metric | Value |
|---|---|
| MAE (glucose at +120 min) | **10.5 mg/dL** (naive "no change" baseline: 22.7) |
| RMSE | **15.8 mg/dL** (baseline: 32.2) |
| Accuracy | **93.9 %** |
| F1-score (spike) | **0.815** (baseline: 0.517) |
| Sensitivity / Specificity | 78.5 % / 97.1 % |
| ROC-AUC | 0.974 |

> Your numbers may differ slightly depending on library versions. Results are on **synthetic** data and show the pipeline works - not clinical validity.

---

## 5. Demo Video Script (Unlisted YouTube, target 4:30 - 5:00)

| Time | Screen | Narration |
|---|---|---|
| 0:00-0:25 | Title slide + team | "Hi, we are `[TEAM]` from `[COLLEGE]`. This is GlucoTwin, a Digital Twin for Type 2 Diabetes." |
| 0:25-0:55 | Problem slide | "India has over 100 million people with diabetes, but doctors see glucose in 3-month snapshots. Spikes between visits are invisible." |
| 0:55-1:30 | Architecture diagram | "We fuse static EHR with 7 days of wearable data, then run two engines: an XGBoost risk predictor and a mechanistic twin for what-if simulation." |
| 1:30-2:00 | Terminal: `python data_generation.py` then `python model_pipeline.py` | "Synthetic, anonymised data for 100 patients. The model predicts a >180 mg/dL spike 2 hours ahead: MAE 10.5 mg/dL, F1 0.82, tested on unseen patients." |
| 2:00-2:35 | Dashboard: **Cohort Overview** | "Doctors see all patients ranked by live risk. The sidebar replay clock simulates real-time streaming." Click **Jump to highest-risk patient**. |
| 2:35-3:25 | **Digital Twin** tab | "EHR profile, glucose, heart rate and steps. The red banner says *High risk of glucose spike in 120 mins*. The bar chart explains why: recent carbs, upward trend." Move the clock +1 h to show the alert updating. |
| 3:25-4:25 | **What-If** tab | "Reduce carbs to 70%, add a 20-minute post-meal walk, up-titrate medication. The green curve is the scenario; time above 180 drops by X hours. We also warn if hypoglycaemia becomes likely." |
| 4:25-4:50 | **Model & Methods** tab | "Patient-level validation, feature importance and a confusion matrix." |
| 4:50-5:00 | Closing slide | "Next steps: real CGM integration, FHIR/ABDM, clinical validation. Thank you." |

**Recording tips:** 1080p, hide notifications, zoom the browser to 110 %, upload as **Unlisted**, paste the link in the Team Details table above.

---

## 6. Setup & Execution

**Requirements:** Python 3.10 or newer.

```bash
# 1) (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) generate synthetic data  ->  data/ehr.csv, data/wearable.csv
python data_generation.py --patients 100 --seed 42 --out data

# 4) train and save the model  ->  models/digital_twin_model.joblib
python model_pipeline.py

# 5) launch the doctor dashboard
streamlit run app.py
```

Steps 3-4 are optional: `streamlit run app.py` generates data and trains the model automatically on first launch (about 1 minute).
Open the URL Streamlit prints (usually <http://localhost:8501>).

### Project structure

```
.
├── app.py                # Streamlit doctor dashboard
├── data_generation.py    # Synthetic EHR + wearable generator
├── model_pipeline.py     # Feature fusion, XGBoost training, evaluation, saving
├── simulator.py          # Mechanistic twin engine (used by generator and What-If tab)
├── requirements.txt
├── LICENSE               # MIT
├── README.md
└── SLIDES_OUTLINE.md     # Slide-by-slide outline for the PPT/PDF
```

---

## 7. Ethics, Privacy & Limitations

- **100 % synthetic data** - no real patient records; random IDs, no names or contact details.
- Decision *support* for physicians, not autonomous treatment.
- Medication effect sizes in the simulator are illustrative, not clinical guidance.
- Not clinically validated; synthetic data cannot capture the full variability of real patients.
- Roadmap: real CGM/wearable ingestion, FHIR / ABDM integration, calibration on real datasets, prospective clinical validation, fairness audits across regions.

---

## 8. License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 [Your Team Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
