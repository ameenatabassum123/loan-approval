# Loan Approval (ML + Streamlit)

Predict whether a loan application should be **Approved** or **Rejected** using a machine learning pipeline and a clean, interactive **Streamlit** app.


# Highlights

* End‑to‑end ML workflow: data prep → model training → evaluation → serving
* One‑click **Streamlit** UI for quick predictions & charts
* Reproducible pipeline using `scikit-learn` + `joblib`
* Optional explainability with **LIME** (feature‑level reasons)
* GitHub‑ready single‑file setup for simplicity


## Requirements

* Python **3.10+**

Install dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt (example)**

```
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
lime
xgboost
```


##  Data

* File: `data.csv` (same as `EDA_Formatted_Data.csv`)
* Columns should match the features used during training.



##  Training + Inference in One File

`app.py` contains:

* **Data loading**
* **Preprocessing** (ColumnTransformer)
* **Model training** (Logistic Regression / XGBoost)
* **Prediction form** (Streamlit input fields)
* **Visualization** (bar/line charts after button click)
* **Optional LIME explanation**

---

##  Run the Streamlit App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal.

Features:

* Input form for applicant details
* Predict button → shows approval/rejection with probability
* Visuals update only after prediction
* Explainability (optional, with LIME)

---

## Evaluation

Metrics computed during training inside the file:

* Accuracy
* Precision / Recall / F1
* ROC‑AUC
* Confusion matrix (optional)

---

## ❗ Common Pitfalls

**Feature mismatch error** → always train + save a `pipeline.pkl`, then load in inference.

```python
# Save
joblib.dump(pipeline, "model.pkl")

# Load
pipeline = joblib.load("model.pkl")
```

**Charts not updating** → place plotting code inside button callback.

---

## Deploy

1. Push to GitHub
2. On Streamlit Cloud → New app → choose `app.py`
3. Ensure `requirements.txt` is present

---

##  Contact

Maintainer: **Ameena Tabassum**
