import pandas as pd
import re
import string
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# ======================================================
# 1️⃣ Sample Dataset (Spam vs Ham)
# ======================================================

data = {
    "text": [
        "Win a free iPhone now",
        "Limited time offer claim your prize",
        "Hello how are you doing today",
        "Let's meet tomorrow for lunch",
        "Congratulations you won lottery",
        "Project meeting at 10 AM",
        "Exclusive deal just for you",
        "Can you send me the report",
        "Click here to claim reward",
        "Are we still on for dinner",
        "Get rich quick scheme now",
        "Team meeting scheduled for Friday"
    ],
    "label": [
        "spam","spam","ham","ham","spam",
        "ham","spam","ham","spam","ham",
        "spam","ham"
    ]
}

df = pd.DataFrame(data)
df["label"] = df["label"].map({"ham": 0, "spam": 1})


# ======================================================
# 2️⃣ Train-Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42
)


# ======================================================
# 3️⃣ Create Pipeline (Vectorizer + Model)
# ======================================================

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])


# ======================================================
# 4️⃣ Define Hyperparameter Grid
# ======================================================

param_grid = {
    "tfidf__max_df": [0.8, 1.0],
    "tfidf__ngram_range": [(1,1), (1,2)],
    "clf__C": [0.1, 1, 10],
    "clf__solver": ["liblinear"]
}


# ======================================================
# 5️⃣ GridSearch with Cross-Validation
# ======================================================

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    verbose=1
)

grid.fit(X_train, y_train)


# ======================================================
# 6️⃣ Best Parameters
# ======================================================

print("\nBest Parameters Found:")
print(grid.best_params_)

print("\nBest Cross-Validation Score:")
print(grid.best_score_)

default_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)

tuned_pred = grid.predict(X_test)

print("\nDefault Model Accuracy:", accuracy_score(y_test, default_pred))
print("Tuned Model Accuracy:", accuracy_score(y_test, tuned_pred))

print("\nTuned Model Classification Report:")
print(classification_report(y_test, tuned_pred))
