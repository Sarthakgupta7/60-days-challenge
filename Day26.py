import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



data = {
    "Age": [25, 30, 35, 40, 28, np.nan, 50, 45, 38, 29],
    "Salary": [50000, 60000, 65000, 80000, 52000, 72000, 90000, np.nan, 67000, 58000],
    "City": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai",
             "Delhi", "Chennai", "Mumbai", "Delhi", "Chennai"],
    "Purchased": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("Purchased", axis=1)
y = df["Purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

numeric_features = ["Age", "Salary"]
categorical_features = ["City"]
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))