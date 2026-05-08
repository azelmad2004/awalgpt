import os
import re
import csv
import json
import pickle
import logging
import random
import time
import sys
import numpy as np
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
repertoire-base = os.path.dirname(os.path.abspath(__file__))
repertoire-modeles = os.path.join(repertoire-base, "models")
repertoire-donnees = os.path.join(repertoire-base, "data")
chemin-tfidf = os.path.join(repertoire-modeles, "tfidf_retriever.pkl")
chemin-index-faiss = os.path.join(repertoire-modeles, "faiss_index.bin")
chemin-docs-faiss = os.path.join(repertoire-modeles, "faiss_docs.pkl")
chemin-faiss-dico = os.path.join(repertoire-modeles, "faiss_dico.bin")
chemin-faiss-dico-docs = os.path.join(repertoire-modeles, "faiss_dico_docs.pkl")
chemin-classif = os.path.join(repertoire-modeles, "intent_classifier.pkl")
chemin-arabizi-map = os.path.join(repertoire-donnees, "arabizi_map.json")
seuil-pertinence-rag = 0.30
seuil-repli-rag = 0.10
boost-dictionnaire = 2.0
boost-semantique = 2.5
boost-dico = 3.0
MAX-MOTS-DICO = 4
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
for lib in ("sentence_transformers", "httpx", "faiss", "transformers", "urllib3"):
    logging.getLogger(lib).setLevel(logging.ERROR)
journal = logging.getLogger("awalgpt")
modele-embedding = None
index-faiss = None
documents-faiss = []
index-faiss-dico = None
documents-faiss-dico = []
moteur-tfidf = None
dicto-local = []
carte-arabizi = {}
modele-classif-intention = None
def charger-configuration-json() -> None:
    global carte-arabizi
    os.makedirs(repertoire-donnees, exist_ok=True)
    if not os.path.exists(chemin-arabizi-map):
        valeurs-defaut = {
            "normalisation": {
                "gh": "\u0263", "kh": "x", "sh": "c", "ch": "c",
                "7": "h", "9": "q", "3": "\u025b", "5": "x"
            },
            "output": {
                "\u0263": "4", "\u0194": "4",
                "\u025b": "3", "\u0190": "3",
                "\u1e25": "7", "\u1e24": "7",
                "x": "5", "X": "5",
                "q": "9", "Q": "9",
                "\u1e6d": "6", "\u1e6c": "6",
                "\u1e0d": "9", "\u1e0c": "9",
                "\u1e63": "9", "\u1e62": "9"
            },
            "outputMulti": {
                "gh": "4", "Gh": "4", "GH": "4",
                "kh": "5", "Kh": "5", "KH": "5"
            }
        }
        with open(chemin-arabizi-map, "w", encoding="utf-8") as f:
            json.dump(valeurs-defaut, f, ensure_ascii=False, indent=2)
    try:
        with open(chemin-arabizi-map, encoding="utf-8") as f:
            carte-arabizi = json.load(f)
    except Exception as e:
        journal.error(f"[CONFIG] arabizi_map.json : {e}")
def normaliser(texte: str) -> str:
    if not texte:
        return ""
    res = texte.strip()
    res = re.sub(r"<[^>]+>", " ", res)
    res = re.sub(
        r"[\U0001F000-\U0001FFFF\U00002600-\U000027BF\U0001F300-\U0001F9FF]",
        " ", res
    )
    res = re.sub(r"([a-z])([A-Z])", r"\1 \2", res)
    res = res.lower()
    mapping-norm = carte-arabizi.get("normalisation", {})
    for src, cible in sorted(mapping-norm.items(), key=lambda x: -len(x[0])):
        if len(src) > 1:
            res = res.replace(src, cible)
    for src, cible in mapping-norm.items():
        if len(src) == 1:
            res = res.replace(src, cible)
    res = re.sub(
        r"[^a-z0-9\u0263\u025b\u1e25\u1e6d\u1e95\u1e5b\u1e0d\u0600-\u06FF\s.,!?;:\-]",
        " ", res
    )
    res = re.sub(r"\s+", " ", res).strip()
    return res
def appliquerArabiziSortie(texte: str) -> str:
    if not texte:
        return texte
    res = texte
    for src, cible in carte-arabizi.get("outputMulti", {}).items():
        res = re.sub(re.escape(src), cible, res)
    for src, cible in carte-arabizi.get("output", {}).items():
        res = res.replace(src, cible)
    return res
motif-arabizi-chiffres = re.compile(r"[379]")
motif-fr-structure = re.compile(
    r"\b(le|la|les|de|du|des|un|une|au|aux|en|et|pour|dans|sur|avec|est|que|qui|"
    r"je|tu|il|elle|nous|vous|ils|elles|mon|ton|son|ma|ta|sa|mes|tes|ses|"
    r"ce|cet|cette|ces|dont|mais|ou|donc|car|ni|or|comment|pourquoi|quand|quel|quelle)\b",
    re.IGNORECASE
)
motif-fr-suffixes = re.compile(
    r"\b\w+(tion|ment|eur|euse|iere|ais|ait|ons|ez|ent|ique|isme|iste|age|ure)\b",
    re.IGNORECASE
)
def detecterLangueRequete(texte: str) -> str:
    if not texte or not texte.strip():
        return "tamazight"
    nb-arabe = sum(1 for c in texte if "\u0600" <= c <= "\u06FF")
    nb-latin = sum(1 for c in texte if c.isalpha() and c.isascii())
    nb-total = max(nb-arabe + nb-latin, 1)
    if nb-arabe / nb-total >= 0.5:
        return "darija" if motif-arabizi-chiffres.search(texte) else "arabe"
    if nb-latin / nb-total >= 0.5:
        hits-struct = len(motif-fr-structure.findall(texte.lower()))
        hits-suffixes = len(motif-fr-suffixes.findall(texte))
        if (hits-struct + hits-suffixes) >= 2:
            return "francais"
    return "tamazight"
def calculer-score-amazigh(mot: str) -> float:
    m = mot.lower()
    score = 0.5
    if re.search(r"gh|kh|[379]|[\u0263\u025b\u1e25\u1e6d\u1e95\u1e5b\u1e0d]", m):
        score += 0.2
    if re.match(r"^(ta|ti|as|am|ig|ul|ur|ad|agg|inn)", m):
        score += 0.15
    if re.search(r"(t|gh|en|in|an|wn)$", m):
        score += 0.1
    nb-voyelles = len(re.findall(r"[aeiou]", m))
    if nb-voyelles / max(len(m), 1) < 0.3:
        score += 0.1
    return min(score, 1.0)
def charger-pickle(chemin: str):
    if not os.path.exists(chemin):
        journal.warning(f"[MODELE] Fichier introuvable : {chemin}")
        return None
    try:
        with open(chemin, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        journal.error(f"[MODELE] Echec chargement {chemin} : {e}")
        return None
def chargerDictionnaire() -> None:
    global dicto-local
    chemin = os.path.join(repertoire-donnees, "dictionnaire.csv")
    if not os.path.exists(chemin):
        journal.warning(f"[DICO] Introuvable : {chemin}")
        return
    try:
        with open(chemin, "r", encoding="utf-8-sig") as f:
            dicto-local = list(csv.DictReader(f))
    except Exception as e:
        journal.error(f"[DICO] Erreur lecture : {e}")
def charger-moteur-semantique() -> None:
    global modele-embedding, index-faiss, documents-faiss
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import torch
        if modele-embedding is None:
            dispositif = "cuda" if torch.cuda.is_available() else "cpu"
            modele-embedding = SentenceTransformer("intfloat/multilingual-e5-small", device=dispositif)
        if os.path.exists(chemin-index-faiss):
            index-faiss = faiss.read_index(chemin-index-faiss)
            try:
                index-faiss.nprobe = 32
            except Exception:
                pass
            with open(chemin-docs-faiss, "rb") as f:
                documents-faiss = pickle.load(f)
    except ImportError:
        journal.warning("[FAISS] sentence_transformers ou faiss non installes")
    except Exception as e:
        journal.warning(f"[FAISS] Erreur chargement : {e}")
def chargerIndexFaissDico() -> None:
    global index-faiss-dico, documents-faiss-dico
    try:
        import faiss
        if os.path.exists(chemin-faiss-dico):
            index-faiss-dico = faiss.read_index(chemin-faiss-dico)
            try:
                index-faiss-dico.nprobe = 32
            except Exception:
                pass
            with open(chemin-faiss-dico-docs, "rb") as f:
                documents-faiss-dico = pickle.load(f)
    except ImportError:
        journal.warning("[DICO-FAISS] FAISS non installe")
    except Exception as e:
        journal.warning(f"[DICO-FAISS] Erreur chargement : {e}")
def charger-moteur-tfidf() -> None:
    global moteur-tfidf
    moteur-tfidf = charger-pickle(chemin-tfidf)
def chargerClassificateurIntention() -> None:
    global modele-classif-intention
    modele-classif-intention = charger-pickle(chemin-classif)
    if modele-classif-intention:
        classes = getattr(modele-classif-intention, "classes_", None)
        if classes is None:
            try:
                classes = modele-classif-intention.named_steps["clf"].classes_
            except Exception:
                classes = []
        journal.info(f"[CLASSIF] Classificateur LR charge — {len(classes)} classes : {list(classes)}")
def detecterIntention(texte: str) -> str:
    global modele-classif-intention
    if not texte or not texte.strip():
        return "general"
    if modele-classif-intention is None:
        chargerClassificateurIntention()
    if modele-classif-intention is None:
        return "general"
    try:
        texte-norm = normaliser(texte)
        if not texte-norm:
            return "general"
        prediction = modele-classif-intention.predict([texte-norm])[0]
        return str(prediction)
    except Exception as e:
        journal.warning(f"[CLASSIF] Erreur prediction : {e}")
        return "general"
def detecterIntentionAvecScore(texte: str) -> tuple:
    global modele-classif-intention
    if not texte or not texte.strip():
        return "general", 0.0
    if modele-classif-intention is None:
        chargerClassificateurIntention()
    if modele-classif-intention is None:
        return "general", 0.0
    try:
        texte-norm = normaliser(texte)
        if not texte-norm:
            return "general", 0.0
        prediction = modele-classif-intention.predict([texte-norm])[0]
        try:
            probas = modele-classif-intention.predict_proba([texte-norm])[0]
            confiance = float(probas.max())
        except Exception:
            confiance = 1.0
        return str(prediction), confiance
    except Exception as e:
        journal.warning(f"[CLASSIF] detecterIntentionAvecScore erreur : {e}")
        return "general", 0.0
def encoderRequete(texte: str):
    if modele-embedding is None:
        charger-moteur-semantique()
    return modele-embedding.encode(
        "query: " + texte,
        normalize_embeddings=True,
    ).astype("float32").reshape(1, -1)
def normaliserVecteurs(matrice) -> np.ndarray:
    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    normes = np.where(normes == 0, 1, normes)
    return matrice / normes
motif-particules = re.compile(
    r"^(gh|di|ar|ad|ur|gg|n|l|d|g|f|w|s|m|b)([a-z\u0600-\u06FF\u0263\u025b]{3,})$",
    re.IGNORECASE
)
seuil-protection-mot = 0.65
def separer-particules(texte: str) -> str:
    tokens-res = []
    for token in texte.split():
        if calculer-score-amazigh(token) >= seuil-protection-mot:
            tokens-res.append(token)
            continue
        match = motif-particules.match(token)
        if match:
            tokens-res.append(match.group(1) + " " + match.group(2))
        else:
            tokens-res.append(token)
    return " ".join(tokens-res)
def generer-formes-requete(requete: str) -> list:
    forme1 = requete.strip()
    forme2 = separer-particules(forme1)
    forme3 = normaliser(forme2)
    forme4 = normaliser(forme1)
    formes-finales = []
    vus = set()
    for forme in [forme1, forme2, forme3, forme4]:
        cle = forme.lower().strip()
        if cle and cle not in vus:
            formes-finales.append(forme)
            vus.add(cle)
    return formes-finales
def rechercherDictionnaire(requete: str, topK: int = 5) -> list:
    mots = set(requete.lower().split())
    resultats = []
    for entree in dicto-local:
        mot-amazigh = entree.get("amazigh", "").lower()
        if mot-amazigh in mots:
            traductions = {
                lang: entree.get(lang, "")
                for lang in ["fr", "ar", "en"]
                if entree.get(lang, "")
            }
            resultats.append({
                "tamazight": mot-amazigh,
                "score": boost-dictionnaire,
                "traductions": traductions,
            })
        if len(resultats) >= topK:
            break
    return resultats
def rechercherMotDictionnaire(mot: str) -> dict:
    mot-norm = mot.lower().strip()
    for entree in dicto-local:
        mot-amazigh = entree.get("amazigh", "").lower()
        if mot-amazigh == mot-norm:
            traductions = {
                lang: entree.get(lang, "")
                for lang in ["fr", "ar", "en"]
                if entree.get(lang, "")
            }
            if traductions:
                return {"tamazight": mot-amazigh, "traductions": traductions}
    return {}
def augmenter-requete-par-dictionnaire(requete: str) -> str:
    mots = set(requete.lower().split())
    termes-sup = []
    for entree in dicto-local:
        mot-amazigh = entree.get("amazigh", "").lower()
        match-pref = any(mot-amazigh.startswith(mot[:4]) for mot in mots if len(mot) >= 4)
        if mot-amazigh in mots or match-pref:
            for lang in ["fr", "ar", "en"]:
                val = entree.get(lang, "")
                if val and val.lower() not in mots:
                    termes-sup.append(val)
    if termes-sup:
        return requete + " " + " ".join(list(set(termes-sup))[:5])
    return requete
def rechercher-via-tfidf(requete: str, k: int, poids: float, fusion: dict) -> None:
    if not moteur-tfidf:
        return
    if "inverse" in moteur-tfidf:
        index-inverse = moteur-tfidf["inverse"]
        tokens-requete = set(normaliser(requete).split())
        for token in tokens-requete:
            if token in index-inverse:
                for idx-doc in index-inverse[token][:50]:
                    doc = moteur-tfidf["textes"][idx-doc]
                    fusion[doc] = fusion.get(doc, 0) + 0.6 * poids
    try:
        import scipy.sparse as sp
        vect-char = moteur-tfidf["char_vect"].transform([requete])
        vect-mot = moteur-tfidf["word_vect"].transform([requete])
        vect-final = sp.hstack([vect-char, vect-mot])
        vect-dense = moteur-tfidf["svd"].transform(vect-final)
        scores = np.dot(vect-dense, moteur-tfidf["matrice"].T).flatten()
        indices = np.argsort(-scores)[:k]
        for i in indices:
            if scores[i] > 0.05:
                doc = moteur-tfidf["textes"][i]
                fusion[doc] = fusion.get(doc, 0) + float(scores[i]) * poids
    except Exception as e:
        journal.warning(f"[TF-IDF] Canal-1 SVD erreur : {e}")
def rechercher-via-faiss(requete: str, k: int, poids: float, fusion: dict, seuil-score: float = 0.50) -> None:
    if index-faiss is None:
        return
    try:
        scores-sem, indices-sem = index-faiss.search(encoderRequete(requete), k)
        for score, idx in zip(scores-sem[0], indices-sem[0]):
            if idx < 0 or float(score) < seuil-score:
                continue
            texte = documents-faiss[idx]["tamazight"]
            fusion[texte] = fusion.get(texte, 0) + float(score) * boost-semantique * poids
    except Exception as e:
        journal.warning(f"[FAISS] Canal-2 erreur : {e}")
def rechercher-via-dico-faiss(requete: str, k: int, fusion: dict, meta-fusion: dict) -> None:
    if index-faiss-dico is None:
        return
    nb-mots-requete = len(requete.split())
    boost-exact = 2.0 if nb-mots-requete <= 3 else 1.0
    try:
        scores, indices = index-faiss-dico.search(encoderRequete(requete), k)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < 0.45:
                continue
            doc = documents-faiss-dico[idx]
            mot = doc.get("tamazight", "")
            if len(mot.split()) > MAX-MOTS-DICO:
                continue
            trad-fr = doc.get("fr", "").strip()
            trad-ar = doc.get("ar", "").strip()
            trad-en = doc.get("en", "").strip()
            parties = [mot]
            if trad-fr: parties.append(f"fr:{trad-fr}")
            if trad-ar: parties.append(f"ar:{trad-ar}")
            if trad-en: parties.append(f"en:{trad-en}")
            texte-enrichi = " | ".join(parties)
            mot-norm = normaliser(mot)
            requete-norm = normaliser(requete)
            multiplicateur = boost-dico * boost-exact
            if mot-norm == requete-norm:
                multiplicateur *= 2.0
            fusion[texte-enrichi] = fusion.get(texte-enrichi, 0) + float(score) * multiplicateur
            meta-fusion[texte-enrichi] = {"fr": trad-fr, "ar": trad-ar, "en": trad-en}
    except Exception as e:
        journal.warning(f"[DICO-FAISS] Canal-5 erreur : {e}")
def rechercher-par-mot(mot: str, fusion: dict, meta-fusion: dict) -> None:
    if len(mot) < 2:
        return
    if moteur-tfidf and "inverse" in moteur-tfidf:
        mot-norm = normaliser(mot)
        if mot-norm in moteur-tfidf["inverse"]:
            for idx-doc in moteur-tfidf["inverse"][mot-norm][:3]:
                doc = moteur-tfidf["textes"][idx-doc]
                fusion[doc] = fusion.get(doc, 0) + 0.4
    if index-faiss is not None:
        try:
            scores-s, indices-s = index-faiss.search(encoderRequete(mot), 3)
            for score, idx in zip(scores-s[0], indices-s[0]):
                if idx >= 0 and float(score) >= 0.50:
                    texte = documents-faiss[idx]["tamazight"]
                    fusion[texte] = fusion.get(texte, 0) + float(score) * 0.5
        except Exception:
            pass
    info-dico = rechercherMotDictionnaire(mot)
    if info-dico:
        cle = info-dico["tamazight"]
        trad = info-dico["traductions"]
        parties = [cle]
        for lang in ["fr", "ar", "en"]:
            v = trad.get(lang, "")
            if v:
                parties.append(f"{lang}:{v}")
        texte-enrichi = " | ".join(parties)
        fusion[texte-enrichi] = fusion.get(texte-enrichi, 0) + boost-dictionnaire
        meta-fusion[texte-enrichi] = trad
def rechercheHybride(requete: str, topK: int = 12, tokensAutorises: list = None) -> list:
    formes-requete = generer-formes-requete(requete)
    fusion: dict = {}
    meta-fusion: dict = {}
    journal.info(f"[HYBRIDE] requete={requete!r} formes={formes-requete} tokens-autorises={tokensAutorises}")
    for forme in formes-requete:
        forme-aug = augmenter-requete-par-dictionnaire(forme)
        rechercher-via-tfidf(forme-aug, topK, poids=1.2, fusion=fusion)
    rechercher-via-faiss(requete, topK, poids=1.0, fusion=fusion)
    for res in rechercherDictionnaire(requete, topK=5):
        mot = res["tamazight"]
        trad = res["traductions"]
        parties = [mot]
        for lang in ["fr", "ar", "en"]:
            v = trad.get(lang, "")
            if v:
                parties.append(f"{lang}:{v}")
        texte-enrichi = " | ".join(parties)
        fusion[texte-enrichi] = fusion.get(texte-enrichi, 0) + res["score"]
        meta-fusion[texte-enrichi] = trad
    if tokensAutorises is not None:
        mots-pour-canal4 = {mot for mot in tokensAutorises if len(mot) >= 2}
    else:
        mots-pour-canal4: set = set()
        for forme in formes-requete:
            for mot in forme.split():
                if len(mot) >= 3:
                    mots-pour-canal4.add(mot)
    for mot in sorted(mots-pour-canal4):
        rechercher-par-mot(mot, fusion, meta-fusion)
    rechercher-via-dico-faiss(requete, topK, fusion=fusion, meta-fusion=meta-fusion)
    if len(requete.split()) <= MAX-MOTS-DICO:
        for mot in requete.split():
            if len(mot) >= 2 and mot != requete:
                rechercher-via-dico-faiss(mot, 5, fusion=fusion, meta-fusion=meta-fusion)
    dedup: dict = {}
    for texte, score in fusion.items():
        cle = normaliser(texte.split("|")[0])
        if cle not in dedup or score > dedup[cle]["score"]:
            dedup[cle] = {"texte": texte, "score": score}
    trie = sorted(dedup.values(), key=lambda x: -x["score"])
    res-finaux = []
    for item in trie[:topK]:
        texte = item["texte"]
        trad = meta-fusion.get(texte, {})
        res-finaux.append({"tamazight": texte, "score": item["score"], "traductions": trad})
    return res-finaux
def est-rag-pertinent(correspondances: list, tokens-requete: set) -> bool:
    if not correspondances:
        return False
    score-max = max(r["score"] for r in correspondances)
    if score-max >= 0.80:
        return True
    return score-max >= seuil-pertinence-rag
def sonderRag(requete: str, intention: str = None, tokensAutorises: list = None) -> dict:
    debut = time.time()
    tokens-requete = set(requete.split())
    if intention is None:
        intention = detecterIntention(requete)
    correspondances = rechercheHybride(requete, topK=10, tokensAutorises=tokensAutorises)
    pertinent = est-rag-pertinent(correspondances, tokens-requete)
    if not pertinent:
        requete-aug = augmenter-requete-par-dictionnaire(requete)
        if requete-aug != requete:
            correspondances = rechercheHybride(requete-aug, topK=10, tokensAutorises=tokensAutorises)
            pertinent = est-rag-pertinent(correspondances, tokens-requete)
    score-max = max((r["score"] for r in correspondances), default=0.0)
    lignes-contexte = formater-contexte(correspondances)
    duree-ms = int((time.time() - debut) * 1000)
    journal.info(f"[RAG] sonderRag pertinent={pertinent} score-max={score-max:.3f} ({duree-ms}ms)")
    return {
        "correspondances": correspondances,
        "english_ctx": "\n".join(lignes-contexte),
        "pertinent": pertinent,
        "score_max": score-max,
        "intention": intention,
    }
def formater-contexte(correspondances: list) -> list:
    lignes = []
    for i, doc in enumerate(correspondances):
        texte = doc["tamazight"]
        trad = doc.get("traductions", {})
        extras = []
        for lang, label in [("fr", "FR"), ("ar", "AR"), ("en", "EN")]:
            val = trad.get(lang, "").strip()
            if val and f"{lang.lower()}:{val}" not in texte.lower():
                extras.append(f"{label}={val}")
        ligne = f"Source {i+1}: {texte}"
        if extras:
            ligne += f" [{', '.join(extras)}]"
        lignes.append(ligne)
    return lignes
def construireContexteAnglais(requete: str, intention: str = None, topK: int = 10, tokensAutorises: list = None) -> dict:
    res-rag = sonderRag(requete, intention=intention, tokensAutorises=tokensAutorises)
    correspondances = res-rag["correspondances"][:topK]
    lignes-contexte = formater-contexte(correspondances)
    return {
        "matches": correspondances,
        "english_ctx": "\n".join(lignes-contexte),
        "pertinent": res-rag["pertinent"],
        "score_max": res-rag["score_max"],
        "intention": res-rag["intention"],
    }
def obtenirTop2MotsSimilaires(texte: str, tokensAutorises: list = None) -> str:
    if index-faiss is None and index-faiss-dico is None:
        return ""
    if tokensAutorises is not None:
        mots = [m for m in tokensAutorises if len(m) > 2][:3]
    else:
        mots = [m for m in texte.split() if len(m) > 2][:3]
    if not mots:
        return ""
    res-mots = []
    for mot in mots:
        try:
            index-utilise = index-faiss-dico if (index-faiss-dico is not None and len(mot.split()) <= MAX-MOTS-DICO) else index-faiss
            docs-utilises = documents-faiss-dico if index-utilise is index-faiss-dico else documents-faiss
            if index-utilise is None:
                continue
            scores, indices = index-utilise.search(encoderRequete(mot), 2)
            correspondances = [
                (docs-utilises[idx].get("tamazight", "").split()[0], float(scores[0][j]))
                for j, idx in enumerate(indices[0])
                if idx >= 0
            ]
            if correspondances:
                textes-match = [c[0] for c in correspondances]
                res-mots.append(f"{mot} ({', '.join(textes-match)})")
        except Exception:
            pass
    return " | ".join(res-mots)
def obtenirProverbePourRequete(message: str) -> dict:
    proverbe-dispo = []
    chemin-csv = os.path.join(repertoire-donnees, "tamazight_latin.csv")
    if os.path.exists(chemin-csv):
        try:
            with open(chemin-csv, encoding="utf-8", errors="replace") as f:
                lecteur = csv.DictReader(f)
                for ligne in lecteur:
                    ligne-low = {k.lower(): v for k, v in ligne.items()}
                    categorie = (ligne-low.get("category", "") or "").lower()
                    tam = (ligne-low.get("tamazight", "") or ligne-low.get("text", "") or "").strip()
                    fr = (ligne-low.get("trans_fr", "") or ligne-low.get("fr", "") or "").strip()
                    if tam and fr and 4 <= len(tam.split()) <= 20:
                        if not categorie or "prov" in categorie:
                            proverbe-dispo.append({"tamazight": tam, "fr": fr})
        except Exception:
            pass
    if not proverbe-dispo:
        return {"tamazight": "", "fr": ""}
    if message:
        tokens-msg = set(normaliser(message).split())
        scores-prov = []
        for prov in proverbe-dispo:
            tokens-prov = (
                set(normaliser(prov["tamazight"]).split())
                | set(normaliser(prov["fr"]).split())
            )
            scores-prov.append((len(tokens-msg & tokens-prov), prov))
        scores-prov.sort(key=lambda x: -x[0])
        if scores-prov[0][0] > 0:
            return scores-prov[0][1]
    return random.choice(proverbe-dispo)
def construireRepliInformatif(requete: str) -> str:
    res-dico = rechercherDictionnaire(requete, topK=3)
    if not res-dico:
        proverbe = obtenirProverbePourRequete(requete)
        tam-fb = proverbe.get("tamazight", "Ur d-yettawi ubrid.")
        return f"[TAM] {tam-fb}\n[AR] لا توجد معلومات كافية حول هذا الموضوع."
    lignes-tam = []
    lignes-ar = []
    for res in res-dico:
        traductions = res.get("traductions", {})
        mot-princ = res["tamazight"].split()[0]
        if traductions.get("fr"):
            lignes-tam.append(f"* {mot-princ} -> {traductions['fr']}")
        if traductions.get("ar"):
            lignes-ar.append(f"* {mot-princ} -> {traductions['ar']}")
    repli-tam = "[TAM] " + " | ".join(lignes-tam) if lignes-tam else "[TAM] Ur d-yettawi ubrid."
    repli-ar = "[AR] " + " | ".join(lignes-ar) if lignes-ar else "[AR] لا توجد ترجمة."
    return f"{repli-tam}\n{repli-ar}"
charger-configuration-json()
chargerDictionnaire()
chargerClassificateurIntention()
chargerIndexFaissDico()