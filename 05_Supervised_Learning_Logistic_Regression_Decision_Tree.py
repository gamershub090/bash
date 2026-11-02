"""
EXPERIMENT 5: Supervised Learning - Employee JobLevel Classification
Uses the IBM HR Analytics dataset from the practical.
Place WA_Fn-UseC_-HR-Employee-Attrition.csv in this folder before running.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CSV_FILE = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        "Download the IBM HR Analytics CSV from the practical's Kaggle dataset "
        "and place it beside this Python file: " + CSV_FILE
    )

df = pd.read_csv(CSV_FILE)
print("Dataset shape:", df.shape)
print(df.head())

# Remove identifier/constant columns if present
remove = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
df = df.drop(columns=[c for c in remove if c in df.columns])

# Separate target and input features
y = df["JobLevel"]
X = df.drop(columns=["JobLevel"])

numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])
preprocess = ColumnTransformer([
    ("num", num_pipe, numeric),
    ("cat", cat_pipe, categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

for name, classifier in models.items():
    model = Pipeline([("preprocess", preprocess), ("model", classifier)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
