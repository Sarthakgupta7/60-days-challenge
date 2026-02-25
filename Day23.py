import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

df["clean_text"] = df["text"].apply(clean_text)


vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

print("TF-IDF Feature Matrix Shape:", X.shape)
print("Vocabulary Size:", len(vectorizer.get_feature_names_out()))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)



print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
