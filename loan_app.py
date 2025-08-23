import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular

# ---------------------------
# Load model
# ---------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Load training data
# ---------------------------
try:
    df_train = pd.read_csv("loan_approval_dataset.csv")
    df_train.columns = df_train.columns.str.strip()  # Clean column names
except FileNotFoundError:
    st.error("Dataset 'loan_approval_dataset.csv' not found. Place it in the same directory.")
    st.stop()

# ---------------------------
# Encode categorical features
# ---------------------------
le_education = LabelEncoder()
df_train['education'] = le_education.fit_transform(df_train['education'])

le_self = LabelEncoder()
df_train['self_employed'] = le_self.fit_transform(df_train['self_employed'])

# ---------------------------
# Prepare training features (no loan_id)
# ---------------------------
X_train = df_train.drop(columns=["loan_id", "loan_status"], errors="ignore")

# ---------------------------
# App title
# ---------------------------
st.title("💳 Loan Approval Prediction with LIME")

# ---------------------------
# User input
# ---------------------------
st.subheader("Enter Applicant Details")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=5, value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
education = ' ' + education   # keep leading space if dataset has it

self_employed = st.selectbox("Self Employed", ["No", "Yes"])
self_employed = ' ' + self_employed  # keep leading space if dataset has it

income_annum = st.number_input("Annual Income", min_value=200000, max_value=9900000, value=5000000)
loan_amount = st.number_input("Loan Amount", min_value=300000, max_value=39500000, value=10000000)
loan_term = st.number_input("Loan Term (Years)", min_value=2, max_value=20, value=10)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=600)
residential_assets_value = st.number_input("Residential Assets Value", min_value=-100000, max_value=29100000, value=5000000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, max_value=19400000, value=3000000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=300000, max_value=39200000, value=10000000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, max_value=14700000, value=3000000)

# ---------------------------
# Predict button
# ---------------------------
if st.button(" Predict Loan Approval"):
    # Create input dataframe
    data = pd.DataFrame(
        [[no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
          residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]],
        columns=[
            "no_of_dependents", "education", "self_employed", "income_annum", "loan_amount", "loan_term",
            "cibil_score", "residential_assets_value", "commercial_assets_value", 
            "luxury_assets_value", "bank_asset_value"
        ]
    )

    # Encode categorical inputs
    data['education'] = le_education.transform(data['education'])
    data['self_employed'] = le_self.transform(data['self_employed'])

    # Prediction
    try:
        prediction = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]  # Probability of approval
        if prediction == 1:
            st.success(f"✅ Loan Approved (Confidence: {proba:.2f})")
        else:
            st.error(f"❌ Loan Not Approved (Confidence: {proba:.2f})")
    except ValueError as e:
        st.error(f"Prediction failed: {str(e)}")


    data_for_lime = data.iloc[0].values
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=["Not Approved", "Approved"],
        mode="classification"
    )
    exp = explainer.explain_instance(data_for_lime, model.predict_proba, num_features=5)

    st.subheader("🔍 Model Explanation (LIME)")
    st.components.v1.html(exp.as_html(), height=800, scrolling=True)
