import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
data = {
    "review": [
        "I absolutely loved this product",
        "Worst experience ever",
        "Highly recommend this",
        "Not worth the money",
        "Very satisfied with the service",
        "Terrible customer support",
        "Amazing quality",
        "I am disappointed",
        "Excellent purchase",
        "Will not buy again"
    ],
    "sentiment": [
        "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative",
        "positive", "negative"
    ]
}

df = pd.DataFrame(data)

# Handle missing values
df.dropna(inplace=True)

# Encode labels
df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_review"] = df["review"].apply(clean_text)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["sentiment"],
    test_size=0.3,
    random_state=42
)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)

y_pred_default = pipeline.predict(X_test)
print("Default Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_default))
print("Precision:", precision_score(y_test, y_pred_default))
print("Recall:", recall_score(y_test, y_pred_default))
print("F1-score:", f1_score(y_test, y_pred_default))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))
cm = confusion_matrix(y_test, y_pred_default)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Default Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
param_grid = {
    "tfidf__ngram_range": [(1,1), (1,2)],
    "tfidf__max_df": [0.8, 1.0],
    "model__C": [0.1, 1, 10]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    verbose=1
)

grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Score:", grid.best_score_)
y_pred_tuned = grid.predict(X_test)


print("\nTuned Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Precision:", precision_score(y_test, y_pred_tuned))
print("Recall:", recall_score(y_test, y_pred_tuned))
print("F1-score:", f1_score(y_test, y_pred_tuned))