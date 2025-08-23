import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

# Encode categorical
le_education = LabelEncoder()
df['education'] = le_education.fit_transform(df['education'])

le_self = LabelEncoder()
df['self_employed'] = le_self.fit_transform(df['self_employed'])

# Drop loan_id
X = df.drop(columns=["loan_id", "loan_status"])
y = df["loan_status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New model.pkl saved without loan_id")
