# GlucoTwin - Presentation Outline (PPT / PDF)

Target: 12 slides, about 6-8 minutes. Style: one idea per slide, large visuals, 3-4 bullets maximum.
Use screenshots from the running dashboard wherever a slide says **[screenshot]**.

---

### Slide 1 - Title
- **GlucoTwin: A Digital Twin for Type 2 Diabetes**
- Happiest Health Digital Twin Challenge 2026
- Team name, college, member names, date

### Slide 2 - The Problem
- 100M+ Indians live with diabetes; follow-ups happen every 1-3 months
- Doctors see one HbA1c number, not the daily glucose swings
- Post-meal spikes cause long-term damage: neuropathy, nephropathy, retinopathy, heart disease
- No safe way to test "what if I change the diet or dose?"

### Slide 3 - Our Solution
- A living virtual model of each patient: **EHR history + wearable data = Digital Twin**
- Two jobs: **predict** a spike 2 hours ahead, **simulate** treatment changes
- Doctor-facing dashboard with explainable alerts

### Slide 4 - Healthcare Use Case
- Condition: Type 2 Diabetes, continuous glucose and metabolic management
- Adverse event: glucose above 180 mg/dL in 120 minutes
- Why 2 hours: time to act (walk, smaller meal, dose, teleconsult)
- Indian context: regional diets (rice vs roti), early-onset phenotype, high family history

### Slide 5 - Data: Two Fused Streams
- **Static EHR:** age, sex, city, BMI, HbA1c, BP, family history, diagnoses, medication
- **Dynamic wearable (7 days, every 15 min):** CGM glucose, heart rate, steps, sleep quality, meals
- 100 anonymised synthetic patients, 67,200 wearable readings
- Generated with a Synthea-inspired approach plus a custom time-series engine

### Slide 6 - System Architecture
- Insert the architecture diagram: Data -> Feature Fusion -> XGBoost + Mechanistic Twin -> Dashboard
- Callout: **hybrid twin** = data-driven accuracy + physiological what-if reasoning

### Slide 7 - ML Prediction Engine
- 42 fused features: glucose lags and trends, 1h/3h/24h aggregates, HR, steps, sleep, meal context, EHR
- XGBoost classifier (spike probability) and regressor (glucose at +120 min)
- Patient-level train/validation/test split, early stopping, decision threshold tuned on validation only
- Explainability: top risk drivers shown for every alert

### Slide 8 - Results
- MAE **10.5 mg/dL** vs 22.7 for the naive baseline
- RMSE **15.8 mg/dL**, accuracy **93.9%**, F1 **0.815**, ROC-AUC **0.974**
- **[screenshot]** Model & Methods tab: feature importance and confusion matrix
- Honest note: synthetic data, shows pipeline validity, not clinical validity

### Slide 9 - Dashboard Demo: Cohort and Twin Profile
- **[screenshot]** Cohort overview ranked by live risk
- **[screenshot]** Digital Twin tab: EHR profile, glucose/HR/steps graphs, banner "High Risk of Glucose Spike in 120 mins"
- Live replay clock simulates real-time streaming

### Slide 10 - What-If Treatment Simulator
- **[screenshot]** current plan vs scenario curves
- Controls: carbohydrate intake, medication regimen, dose, post-meal walk
- Outputs: peak glucose, time in range, hours above 180, hypoglycaemia safety warning
- Twin is calibrated to the patient's last 3 days of CGM

### Slide 11 - Impact, Ethics and Roadmap
- Impact: earlier intervention, fewer spikes, personalised advice, scalable follow-up
- Ethics: synthetic data only, decision support not decision making, explainable alerts
- Roadmap: real CGM/wearable integration, FHIR / ABDM, clinical validation, multilingual patient app, more conditions (hypertension, CKD)

### Slide 12 - Thank You
- Team members and contacts
- GitHub repository link and unlisted YouTube demo link
- Open source under the MIT License
