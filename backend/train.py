import os
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix

df = pd.read_csv("data/tamazight_qa.csv")
df = df.dropna(subset=["intent"])
df = df[df["intent"].map(df["intent"].value_counts()) >= 2]
X_raw = df["input"].fillna("").astype(str)
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)


def tokenize(text):
    return text.lower().split()

train_tokens = [tokenize(doc) for doc in X_train]

w2v = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)

def text_to_vector(text, model, vec_size=100):
    tokens = tokenize(text)
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if not vecs:
        return np.zeros(vec_size)
    return np.mean(vecs, axis=0)

X_train_vec = np.array([text_to_vector(t, w2v) for t in X_train])
X_test_vec  = np.array([text_to_vector(t, w2v) for t in X_test])

model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
with open("models/intent_classifier.pkl", "wb") as f:
    pickle.dump({
        "model":            model,
        "w2v":              w2v,
        "vector_size":      100,
    }, f)

classes = sorted(y.unique().tolist())
cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_title(f"Confusion matrix")
ax.set_ylabel("True")
ax.set_xlabel("Predicted")
plt.xticks(rotation=45, ha="right")
fig.tight_layout()
fig.savefig(f"models/confusion.png")
plt.close(fig)

