import os
import re
import pickle
import csv
import logging
import random
import sys
from collections import Counter
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from core import normaliser
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
rep_base = os.path.dirname(os.path.abspath(__file__))
rep_donnees = os.path.join(rep_base, "data")
chemin_modeles = os.path.join(rep_base, "models")
chemin_tfidf = os.path.join(chemin_modeles, "tfidf_retriever.pkl")
chemin_classif = os.path.join(chemin_modeles, "intent_classifier.pkl")
chemin_dico_csv = os.path.join(rep_donnees, "dictionnaire.csv")
chemin_faiss_corpus = os.path.join(chemin_modeles, "faiss_index.bin")
chemin_faiss_corpus_docs = os.path.join(chemin_modeles, "faiss_docs.pkl")
chemin_faiss_dico = os.path.join(chemin_modeles, "faiss_dico.bin")
chemin_faiss_dico_docs = os.path.join(chemin_modeles, "faiss_dico_docs.pkl")
RATIO_TEST = 0.3
GRAINE_ALEATOIRE = 42
TAUX_AUGMENTATION = 0.6
LIMITE_GENERAL_PAR_FICHIER = 5000
MAX_MOTS_DICO = 4
MIN_MOTS_CORPUS = 5
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("awalgpt.entrainement")

def est_corpus_valide(texte: str) -> bool:
    if not texte or not texte.strip():
        return False
    mots = texte.split()
    if len(mots) < 2:
        return False

    nb_alpha = sum(1 for c in texte if c.isalpha())
    if len(texte) > 0 and nb_alpha < len(texte) * 0.45:
        return False
    tokens_courts = sum(1 for m in mots if len(m) <= 2)
    if len(mots) > 3 and tokens_courts / len(mots) > 0.6:
        return False
    if len(mots) > 4:
        freq_max = Counter(mots).most_common(1)[0][1]
        if freq_max / len(mots) > 0.6:
            return False
    return True
def generer_exemples_synthetiques() -> tuple:
    exemples = []
    etiquettes = []
    fichiers_sources = [
        os.path.join(rep_donnees, "dataset.csv"),
        os.path.join(rep_donnees, "darija.csv"),
        os.path.join(rep_donnees, "tamazight_latin.csv"),
    ]
    compteur_par_intention: Counter = Counter()
    banque_exemples: dict = {}
    for chemin_fich in fichiers_sources:
        if not os.path.exists(chemin_fich):
            continue
        try:
            with open(chemin_fich, "r", encoding="utf_8", errors="replace") as f:
                lecteur = csv.DictReader(f)
                for ligne in lecteur:
                    texte = ""
                    for col in ["text", "input", "tamazight", "content", "message", "amazigh"]:
                        if col in ligne and ligne[col]:
                            texte = ligne[col].strip()
                            break
                    if not texte or len(texte.split()) > 6:
                        continue
                    if not est_corpus_valide(texte):
                        continue
                    intention = ""
                    for col in ["intent", "category", "type"]:
                        if col in ligne and ligne[col]:
                            intention = ligne[col].strip().lower()
                            break
                    if intention == "general":
                        continue
                    texte_norm = normaliser(texte)
                    if not texte_norm:
                        continue
                    if intention not in banque_exemples:
                        banque_exemples[intention] = []
                    if texte_norm not in banque_exemples[intention]:
                        banque_exemples[intention].append(texte_norm)
                    compteur_par_intention[intention] += 1
        except Exception as e:
            log.error(f"[SYNTH] Erreur {chemin_fich} : {e}")
    if not banque_exemples:
        return [], []
    moyenne_count = sum(compteur_par_intention.values()) / max(len(compteur_par_intention), 1)
    seuil_rare = moyenne_count * 0.4
    for intention, liste_ex in banque_exemples.items():
        est_rare = compteur_par_intention[intention] < seuil_rare
        for ex in liste_ex:
            exemples.append(ex)
            etiquettes.append(intention)
            if est_rare:
                ex_maj = ex.capitalize()
                if ex_maj != ex:
                    exemples.append(ex_maj)
                    etiquettes.append(intention)
                mots = ex.split()
                if 2 <= len(mots) <= 4:
                    ex_inv = " ".join(reversed(mots))
                    if ex_inv != ex:
                        exemples.append(ex_inv)
                        etiquettes.append(intention)
    return exemples, etiquettes
def charger_dico_augmentation() -> dict:
    dico = {}
    if not os.path.exists(chemin_dico_csv):
        return dico
    try:
        with open(chemin_dico_csv, "r", encoding="utf_8_sig") as f:
            lecteur = csv.DictReader(f)
            for ligne in lecteur:
                mot = ligne.get("amazigh", "").lower().strip()
                trad = [ligne.get(l, "").lower().strip()
                        for l in ["fr", "ar", "en"] if ligne.get(l, "").strip()]
                if mot and trad:
                    dico[mot] = trad
    except Exception as e:
        log.error(f"[DICO] Erreur : {e}")
    return dico
def augmenter_exemple(texte: str, dico: dict) -> list:
    variantes = []
    mots = texte.split()
    mots_s = mots.copy()
    nb_s = 0
    for i, mot in enumerate(mots_s):
        if mot in dico and nb_s < 2:
            trad = dico[mot]
            if trad:
                mots_s[i] = random.choice(trad)
                nb_s += 1
    v1 = " ".join(mots_s)
    if v1 != texte:
        variantes.append(v1)
    if 3 <= len(mots) <= 6:
        mots_p = mots.copy()
        i = random.randint(0, len(mots_p) - 2)
        mots_p[i], mots_p[i + 1] = mots_p[i + 1], mots_p[i]
        v2 = " ".join(mots_p)
        if v2 != texte and v2 not in variantes:
            variantes.append(v2)
    return variantes
def augmenter_donnees(textes: list, etiquettes: list, dico: dict) -> tuple:
    textes_aug = list(textes)
    etiquettes_aug = list(etiquettes)
    synt_textes, synt_etiq = generer_exemples_synthetiques()
    textes_aug.extend(synt_textes)
    etiquettes_aug.extend(synt_etiq)
    counts = Counter(etiquettes_aug)
    avg_count = sum(counts.values()) / max(len(counts), 1)
    nb_nouvelles = 0
    for idx in range(len(textes)):
        label = etiquettes[idx]
        if counts[label] < avg_count * 0.3:
            mult = 6
        elif counts[label] < avg_count * 0.6:
            mult = 3
        elif counts[label] < avg_count * 0.9:
            mult = 2
        else:
            mult = 1
        for _ in range(mult - 1):
            for v in augmenter_exemple(textes[idx], dico):
                textes_aug.append(v)
                etiquettes_aug.append(label)
                nb_nouvelles += 1
        if random.random() < TAUX_AUGMENTATION:
            for v in augmenter_exemple(textes[idx], dico):
                textes_aug.append(v)
                etiquettes_aug.append(label)
                nb_nouvelles += 1
    log.info(f"[AUGM] {nb_nouvelles} exemples ajoutes")
    return textes_aug, etiquettes_aug
def charger_corpus_equilibre() -> dict:
    textes = []
    etiquettes = []
    meta = []
    fichiers = [
        ("dataset.csv", LIMITE_GENERAL_PAR_FICHIER),
        ("darija.csv", LIMITE_GENERAL_PAR_FICHIER),
        ("tamazight_latin.csv", LIMITE_GENERAL_PAR_FICHIER),
    ]
    for nom_fich, lim_general in fichiers:
        chemin = os.path.join(rep_donnees, nom_fich)
        if not os.path.exists(chemin):
            continue
        cpt_general = 0
        try:
            with open(chemin, "r", encoding="utf_8", errors="replace") as f:
                lecteur = csv.DictReader(f)
                for ligne in lecteur:
                    ligne = {k.lstrip('\ufeff'): v for k, v in ligne.items() if k is not None}
                    texte = ""
                    for col in ["text", "input", "tamazight", "content", "message", "amazigh"]:
                        if col in ligne and ligne[col]:
                            texte = ligne[col].strip()
                            break
                    if not texte:
                        continue
                    texte_norm = normaliser(texte)
                    if len(texte_norm) < 3 or not est_corpus_valide(texte):
                        continue
                    intention = ""
                    for col in ["intent", "category", "type"]:
                        if col in ligne and ligne[col]:
                            intention = ligne[col].strip().lower()
                            break
                    if intention == "general":
                        if cpt_general >= lim_general:
                            continue
                        cpt_general += 1
                    textes.append(texte_norm)
                    etiquettes.append(intention)
                    if any("\u0600" <= c <= "\u06FF" for c in texte):
                        textes.append(f"__ar__ {texte_norm}")
                        etiquettes.append(intention)
                    meta.append({"tamazight": texte_norm, "intent": intention, "source": nom_fich})
        except Exception as e:
            log.error(f"[CORPUS] Erreur {nom_fich} : {e}")
    vu = {}
    textes_ded, etiq_ded = [], []
    for t, e in zip(textes, etiquettes):
        if t not in vu:
            textes_ded.append(t)
            etiq_ded.append(e)
            vu[t] = True
    return {"textes": textes_ded, "etiquettes": etiq_ded, "meta": meta}
def normaliser_vecteurs(matrice: np.ndarray) -> np.ndarray:
    normes = np.linalg.norm(matrice, axis=1, keepdims=True)
    normes = np.where(normes == 0, 1, normes)
    return matrice / normes
def entrainer_tfidf(textes: list) -> None:
    log.info("[TRAIN] TF_IDF hybride + SVD...")
    vect_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                max_features=35000, sublinear_tf=True,
                                min_df=3, strip_accents=None)
    vect_mot = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                               max_features=20000, sublinear_tf=True,
                               min_df=2, strip_accents=None)
    m_char = vect_char.fit_transform(textes)
    m_mot = vect_mot.fit_transform(textes)
    m_dense = sp.hstack([m_char, m_mot])
    n_comp = min(512, m_dense.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=GRAINE_ALEATOIRE)
    m_red = svd.fit_transform(m_dense).astype("float32")
    m_red = normaliser_vecteurs(m_red)
    log.info(f"[TRAIN] SVD {n_comp}D variance={sum(svd.explained_variance_ratio_):.2%}")
    index_inv = {}
    for idx, texte in enumerate(textes):
        for m in set(normaliser(texte).split()):
            if len(m) > 2:
                if m not in index_inv:
                    index_inv[m] = []
                if len(index_inv[m]) < 50:
                    index_inv[m].append(idx)
    donnees = {"char_vect": vect_char, "word_vect": vect_mot, "svd": svd,
               "matrice": m_red, "textes": textes, "inverse": index_inv}
    os.makedirs(chemin_modeles, exist_ok=True)
    with open(chemin_tfidf, "wb") as f:
        pickle.dump(donnees, f)
    log.info(f"[TRAIN] TF_IDF sauvegarde ({m_red.shape})")
def entrainer_classificateur(textes: list, etiquettes: list) -> None:
    log.info("[TRAIN] Classificateur LR d'intentions...")
    paires = [(t, e) for t, e in zip(textes, etiquettes) if e and e.strip()]
    if not paires:
        log.error("[TRAIN] Aucun exemple avec etiquette — annule")
        return
    textes_filt, etiq_filt = zip(*paires)
    textes_filt = list(textes_filt)
    etiq_filt = list(etiq_filt)
    counts = Counter(etiq_filt)
    classes_valides = {cls for cls, n in counts.items() if n >= 2}
    paires_f = [(t, e) for t, e in zip(textes_filt, etiq_filt) if e in classes_valides]
    if not paires_f:
        log.error("[TRAIN] Aucune classe avec >= 2 exemples — annule")
        return
    textes_filt, etiq_filt = zip(*paires_f)
    textes_filt = list(textes_filt)
    etiq_filt = list(etiq_filt)
    x_tr, x_te, y_tr, y_te = train_test_split(
        textes_filt, etiq_filt, test_size=RATIO_TEST,
        random_state=GRAINE_ALEATOIRE, stratify=etiq_filt
    )
    pipeline = Pipeline([
        ("union", FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                     max_features=35000, min_df=3,
                                     sublinear_tf=True, strip_accents=None)),
            ("mot", TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                     max_features=20000, min_df=2,
                                     sublinear_tf=True, strip_accents=None)),
        ])),
        ("clf", LogisticRegression(solver="saga", max_iter=800, C=3.0,
                                   class_weight="balanced",
                                   random_state=GRAINE_ALEATOIRE, tol=1e-3)),
    ])
    pipeline.fit(x_tr, y_tr)
    y_pred = pipeline.predict(x_te)
    acc = accuracy_score(y_te, y_pred)
    log.info(f"[EVAL] Precision : {acc:.2%}")
    log.info(f"[EVAL] Rapport :\n{classification_report(y_te, y_pred)}")
    noms_c = sorted(set(y_te))
    matrice = confusion_matrix(y_te, y_pred, labels=noms_c)
    print("\n── Matrice de confusion ────────────────────────────────")
    print("         " + "  ".join(f"{n[:6]:>6}" for n in noms_c))
    for i, ligne in enumerate(matrice):
        print(f"{noms_c[i][:8]:<8} " + "  ".join(f"{v:>6}" for v in ligne))
    print("────────────────────────────────────────────────────────\n")
    os.makedirs(chemin_modeles, exist_ok=True)
    with open(chemin_classif, "wb") as f:
        pickle.dump(pipeline, f)
    log.info(f"[TRAIN] Classificateur LR sauvegarde : {chemin_classif}")
def construire_et_sauvegarder_faiss(textes: list, chemin_index: str, chemin_docs: str, tag: str, meta_docs: list) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
    except ImportError as e:
        log.error(f"[FAISS-{tag}] Bibliotheques manquantes : {e}")
        return
    modele = SentenceTransformer("intfloat/multilingual-e5-small")
    log.info(f"[FAISS-{tag}] Encodage de {len(textes)} textes...")
    embeddings = modele.encode(
        ["passage: " + t for t in textes],
        normalize_embeddings=True, show_progress_bar=True, batch_size=128,
    )
    normes = np.linalg.norm(embeddings, axis=1)
    mask = normes > 1e-6
    emb = embeddings[mask].astype("float32")
    meta = [meta_docs[i] for i, m in enumerate(mask) if m]
    if len(emb) == 0:
        log.error(f"[FAISS-{tag}] Aucun vecteur valide — index non construit")
        return
    dim = emb.shape[1]
    n_list = min(max(int(4 * np.sqrt(len(emb))), 100), 1500)
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quant, dim, n_list, faiss.METRIC_INNER_PRODUCT)
    index.train(emb)
    index.add(emb)
    index.nprobe = 32
    os.makedirs(chemin_modeles, exist_ok=True)
    faiss.write_index(index, chemin_index)
    with open(chemin_docs, "wb") as f:
        pickle.dump(meta, f)
    log.info(f"[FAISS-{tag}] Sauvegarde : {chemin_index} ({len(meta)} docs)")
def construire_index_faiss(meta: list) -> None:
    docs_utiles = [
        m["tamazight"] for m in meta
        if len(m["tamazight"].split()) >= MIN_MOTS_CORPUS
        and est_corpus_valide(m["tamazight"])
    ]
    if not docs_utiles:
        log.error("[FAISS_CORPUS] Aucun document — index non construit")
        return
    construire_et_sauvegarder_faiss(
        docs_utiles, chemin_faiss_corpus, chemin_faiss_corpus_docs,
        tag="CORPUS", meta_docs=[{"tamazight": t} for t in docs_utiles]
    )
def construire_index_faiss_dico() -> None:
    log.info(f"[FAISS_DICO] Construction index dico mot_a_mot filtre <= {MAX_MOTS_DICO} mots")
    entrees = []
    if os.path.exists(chemin_dico_csv):
        try:
            with open(chemin_dico_csv, "r", encoding="utf_8_sig") as f:
                lecteur = csv.DictReader(f)
                for ligne in lecteur:
                    mot = (ligne.get("amazigh") or "").strip()
                    if not mot or len(mot.split()) > MAX_MOTS_DICO:
                        continue
                    trad_fr = (ligne.get("fr") or "").strip()
                    trad_ar = (ligne.get("ar") or "").strip()
                    trad_en = (ligne.get("en") or "").strip()
                    if not any([trad_fr, trad_ar, trad_en]):
                        continue
                    mot_norm = normaliser(mot)
                    parties = [mot_norm]
                    if trad_fr: parties.append(f"fr:{trad_fr}")
                    if trad_ar: parties.append(f"ar:{trad_ar}")
                    if trad_en: parties.append(f"en:{trad_en}")
                    texte_index = " | ".join(parties)
                    entrees.append({
                        "tamazight": mot,
                        "fr": trad_fr,
                        "ar": trad_ar,
                        "en": trad_en,
                        "texteIndex": texte_index,
                        "traductions": {"fr": trad_fr, "ar": trad_ar, "en": trad_en},
                    })
        except Exception as e:
            log.error(f"[FAISS_DICO] Erreur dictionnaire.csv : {e}")
    fichiers_sup = [
        os.path.join(rep_donnees, "darija.csv"),
        os.path.join(rep_donnees, "tamazight_latin.csv"),
    ]
    mots_deja = {e["tamazight"].lower() for e in entrees}
    for chemin_sup in fichiers_sup:
        if not os.path.exists(chemin_sup):
            continue
        try:
            with open(chemin_sup, "r", encoding="utf_8", errors="replace") as f:
                lecteur = csv.DictReader(f)
                for ligne in lecteur:
                    mot = ""
                    for col in ["amazigh", "tamazight", "text", "input", "content"]:
                        if col in ligne and ligne[col]:
                            mot = ligne[col].strip()
                            break
                    if not mot or len(mot.split()) > MAX_MOTS_DICO:
                        continue
                    if mot.lower() in mots_deja or not est_corpus_valide(mot):
                        continue
                    trad_fr = (ligne.get("fr") or ligne.get("trans_fr") or "").strip()
                    trad_ar = (ligne.get("ar") or ligne.get("trans_ar") or "").strip()
                    trad_en = (ligne.get("en") or ligne.get("trans_en") or "").strip()
                    if not any([trad_fr, trad_ar, trad_en]):
                        continue
                    mot_norm = normaliser(mot)
                    parties = [mot_norm]
                    if trad_fr: parties.append(f"fr:{trad_fr}")
                    if trad_ar: parties.append(f"ar:{trad_ar}")
                    if trad_en: parties.append(f"en:{trad_en}")
                    texte_index = " | ".join(parties)
                    entrees.append({
                        "tamazight": mot,
                        "fr": trad_fr,
                        "ar": trad_ar,
                        "en": trad_en,
                        "texteIndex": texte_index,
                        "traductions": {"fr": trad_fr, "ar": trad_ar, "en": trad_en},
                    })
                    mots_deja.add(mot.lower())
        except Exception as e:
            log.error(f"[FAISS_DICO] Erreur {chemin_sup} : {e}")
    if not entrees:
        log.error("[FAISS_DICO] Aucune entree — index non construit")
        return
    textes_index = [e["texteIndex"] for e in entrees]
    construire_et_sauvegarder_faiss(
        textes_index, chemin_faiss_dico, chemin_faiss_dico_docs,
        tag="DICO", meta_docs=entrees
    )
def verifier_classificateur() -> None:
    if not os.path.exists(chemin_classif):
        return
    try:
        with open(chemin_classif, "rb") as f:
            pipeline = pickle.load(f)
        classes = list(pipeline.named_steps["clf"].classes_)
        log.info(f"[VERIF] Classificateur OK — {len(classes)} classes : {classes}")
        test = normaliser("azul")
        pred = pipeline.predict([test])[0]
        proba = pipeline.predict_proba([test])[0].max()
        log.info(f"[VERIF] Test 'azul' -> {pred!r} (conf={proba:.2%})")
    except Exception as e:
        log.error(f"[VERIF] Erreur verificiation classificateur : {e}")
def verifier_index_dico() -> None:
    if not os.path.exists(chemin_faiss_dico) or not os.path.exists(chemin_faiss_dico_docs):
        return
    try:
        with open(chemin_faiss_dico_docs, "rb") as f:
            docs = pickle.load(f)
        nb_avec_trad = sum(1 for d in docs if d.get("traductions"))
        log.info(f"[VERIF] Index dico : {len(docs)} entrees, avec traductions={nb_avec_trad}")
    except Exception as e:
        log.error(f"[VERIF] Erreur verification index dico : {e}")
def main() -> None:
    os.makedirs(chemin_modeles, exist_ok=True)
    log.info("=" * 60)
    log.info("[MAIN] Demarrage pipeline entrainement AWAL GPT")
    log.info("=" * 60)
    donnees = charger_corpus_equilibre()
    if not donnees["textes"]:
        log.error("[MAIN] Aucun exemple — entrainement annule.")
        return
    dico = charger_dico_augmentation()
    textes_aug, etiq_aug = augmenter_donnees(donnees["textes"], donnees["etiquettes"], dico)
    entrainer_tfidf(textes_aug)
    entrainer_classificateur(textes_aug, etiq_aug)
    construire_index_faiss(donnees["meta"])
    construire_index_faiss_dico()
    verifier_classificateur()
    verifier_index_dico()
    log.info("=" * 60)
    log.info("[MAIN] Entrainement termine.")
    log.info(f"  TF_IDF         : {chemin_tfidf}")
    log.info(f"  Classificateur : {chemin_classif}")
    log.info(f"  Index corpus   : {chemin_faiss_corpus}")
    log.info(f"  Index dico     : {chemin_faiss_dico}")
    log.info("=" * 60)
if __name__ == "__main__":
    main()