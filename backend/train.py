# train.py — Entraînement robuste du classificateur d'intention
# Modèle : FeatureUnion (Char-ngrams + Word-ngrams) + LogisticRegression
# Augmentation : Mixage linguistique (Mots questions x Entités x Particules)

import os, csv, pickle, logging, json, random, re
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.pkl")
CM_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("awalgpt.train")

# 1. Chargement des données sources
log.info("Chargement des sources...")
mots_cles = {}
with open(os.path.join(DATA_DIR, "mots_cles_intentions.json"), encoding="utf-8") as f:
    mots_cles = json.load(f)

# Extraction des noms de domaine (mots 3-20 chars du dataset)
noms_domaine = set()
X, y = [], []
with open(os.path.join(DATA_DIR, "dataset.csv"), encoding="utf-8", errors="replace") as f:
    for row in csv.DictReader(f):
        row = {k.strip().lstrip("\ufeff"): v for k, v in row.items() if k}
        text = next((row[c].strip() for c in ("normalized","input","tamazight") if row.get(c)), None)
        intent = next((row[c].strip().lower() for c in ("intent","category") if row.get(c)), None)
        if text and intent:
            X.append(text)
            y.append(intent)
            # Collecter des mots pour l'augmentation
            words = re.findall(r"\b\w{3,20}\b", text.lower())
            noms_domaine.update(words)

noms_domaine = list(noms_domaine)
log.info(f"Dataset : {len(X)} lignes. Entités trouvées : {len(noms_domaine)}")

# 2. Augmentation (Linguistic Mixing) - SUPPRIMÉ
log.info(f"Total pour l'entraînement : {len(X)} exemples")

from sklearn.feature_extraction.text import TfidfVectorizer

# 3. Pipeline avec FeatureUnion (TF-IDF Char 2-5 + Word 1-3)
pipe = Pipeline([
    ("union", FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", 
            ngram_range=(2, 5), 
            max_features=30000, 
            sublinear_tf=True, 
            min_df=2
        )),
        ("word", TfidfVectorizer(
            analyzer="word", 
            ngram_range=(1, 3), 
            max_features=15000, 
            sublinear_tf=True, 
            min_df=2
        ))
    ])),
    ("clf", LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

# 4. Entraînement et Evaluation
from collections import Counter
counts = Counter(y)
valides = {cls for cls, n in counts.items() if n >= 2}
X_f = [x for x, lbl in zip(X, y) if lbl in valides]
y_f = [lbl for lbl in y if lbl in valides]
X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.15, random_state=42, stratify=y_f)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
log.info(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
log.info(f"Classification Report :\n{classification_report(y_test, y_pred)}")

# 5. Sauvegarde
os.makedirs(MODEL_DIR, exist_ok=True)
metadata = {
    "model": pipe,
    "optimal_threshold": 0.55,
    "classes": list(pipe.classes_),
    "trained_on": len(X),
    "date": datetime.now().isoformat()
}
with open(MODEL_PATH, "wb") as f:
    pickle.dump(metadata, f)

# 6. Matrice de Confusion
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred, labels=pipe.classes_)
fig, ax = plt.subplots(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot(cmap="Blues", xticks_rotation=45, ax=ax, values_format='d')
plt.title("Confusion Matrix (Upgraded Intent Model)")
plt.tight_layout()
plt.savefig(CM_PATH)
log.info(f"Modèle et Matrice sauvegardés dans {MODEL_DIR}")