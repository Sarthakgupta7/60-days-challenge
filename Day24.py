import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ======================================================
# 1️⃣ Create Slightly Imbalanced Dataset
# ======================================================

data = {
    "text": [
        "Win a free iPhone now",
        "Limited time offer claim your prize",
        "Congratulations you won lottery",
        "Exclusive deal just for you",
        "Click here to claim reward",
        "Get rich quick scheme now",
        "Hello how are you",
        "Let's meet tomorrow",
        "Project meeting at 10 AM",
        "Are we still on for dinner",
        "Please send the report",
        "Team meeting scheduled",
        "Lunch tomorrow?",
        "Can we reschedule meeting?"
    ],
    "label": [
        1,1,1,1,1,1,   # spam (6)
        0,0,0,0,0,0,0,0  # ham (8)
    ]
}

df = pd.DataFrame(data)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ======================================================
# 6️⃣ Evaluation Metrics
# ======================================================

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ======================================================
# 7️⃣ Confusion Matrix
# ======================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
