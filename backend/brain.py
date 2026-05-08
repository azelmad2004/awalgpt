import os
import re
import time
import json
import pickle
import logging
import asyncio
import random
import sys
import uuid
from collections import Counter
import numpy as np
from groq import AsyncGroq
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import core
from core import (
    normaliser,
    appliquerArabiziSortie,
    construireContexteAnglais,
    obtenirProverbePourRequete,
    construireRepliInformatif,
    obtenirTop2MotsSimilaires,
    detecterLangueRequete,
    sonderRag,
    rechercherMotDictionnaire,
)
cleGroq = os.getenv("GROQ_API_KEY", "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")
repertoire-donnees = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
chemin-mots-sociaux = os.path.join(repertoire-donnees, "mots_sociaux.json")
chemin-mots-cles = os.path.join(repertoire-donnees, "mots_cles_intentions.json")
chemin-instructions = os.path.join(repertoire-donnees, "instructions_intentions.json")
chemin-reponses-sociales = os.path.join(repertoire-donnees, "reponses_sociales.json")
for lib in ("sentence_transformers", "httpx", "faiss", "transformers", "urllib3"):
    logging.getLogger(lib).setLevel(logging.ERROR)
journal = logging.getLogger("awalgpt")
client-groq = None
mots-sociaux-exacts = {}
mots-cles-par-intention = {}
instructions-par-intention = {}
reponses-sociales-rapides = {}
regles-compilees = []
def initialiser-client-groq() -> None:
    global client-groq
    if client-groq is None:
        client-groq = AsyncGroq(api_key=cleGroq)
def charger-configuration-intentions() -> None:
    global mots-sociaux-exacts, mots-cles-par-intention
    global instructions-par-intention, reponses-sociales-rapides, regles-compilees
    os.makedirs(repertoire-donnees, exist_ok=True)
    if not os.path.exists(chemin-mots-sociaux):
        journal.warning("[CONFIG] mots_sociaux.json absent")
        mots-sociaux-exacts = {}
    else:
        try:
            with open(chemin-mots-sociaux, encoding="utf-8") as f:
                mots-sociaux-exacts = json.load(f)
        except Exception as e:
            journal.error(f"[CONFIG] mots_sociaux.json : {e}")
            mots-sociaux-exacts = {}
    if not os.path.exists(chemin-mots-cles):
        journal.warning("[CONFIG] mots_cles_intentions.json absent")
        mots-cles-par-intention = {}
    else:
        try:
            with open(chemin-mots-cles, encoding="utf-8") as f:
                mots-cles-par-intention = json.load(f)
        except Exception as e:
            journal.error(f"[CONFIG] mots_cles_intentions.json : {e}")
            mots-cles-par-intention = {}
    if not os.path.exists(chemin-instructions):
        journal.warning("[CONFIG] instructions_intentions.json absent")
        instructions-par-intention = {}
    else:
        try:
            with open(chemin-instructions, encoding="utf-8") as f:
                instructions-par-intention = json.load(f)
        except Exception as e:
            journal.error(f"[CONFIG] instructions_intentions.json : {e}")
            instructions-par-intention = {}
    if not os.path.exists(chemin-reponses-sociales):
        journal.warning("[CONFIG] reponses_sociales.json absent")
        reponses-sociales-rapides = {}
    else:
        try:
            with open(chemin-reponses-sociales, encoding="utf-8") as f:
                donnees = json.load(f)
                reponses-sociales-rapides.clear()
                for intention, options in donnees.items():
                    reponses-sociales-rapides[intention] = [
                        (opt[0], opt[1]) for opt in options if len(opt) >= 2
                    ]
        except Exception as e:
            journal.error(f"[CONFIG] reponses_sociales.json : {e}")
            reponses-sociales-rapides = {}
    regles-compilees.clear()
    for intention, liste-mots in mots-cles-par-intention.items():
        if liste-mots:
            motif = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in liste-mots) + r")\b", re.I
            )
            regles-compilees.append((motif, intention))
def detecter-blocs-math(texte: str) -> list:
    motif-math = re.compile(
        r"\b(?:\d+[a-zA-Z]|[a-zA-Z]\d|[0-9+\-*/=()²³√]+[0-9a-zA-Z])\b"
        r"|[0-9]+\s*[+\-*/=]\s*[0-9]+"
    )
    return motif-math.findall(texte)
def nettoyer-pour-ml(texte: str) -> str:
    res = texte
    res = re.sub(
        r"[\U0001F000-\U0001FFFF\U00002600-\U000027BF\U0001F300-\U0001F9FF]",
        " ", res
    )
    res = re.sub(r"([!?.,]){2,}", r"\1", res)
    res = re.sub(r"^(Q:|q:|\d+\.|->|»|•|\*|>>)\s*", "", res.strip())
    res = re.sub(r"\s+", " ", res).strip().lower()
    return res
def detecter-acronymes(texte: str) -> list:
    motif-acronyme = re.compile(r"\b[A-Z]{2,5}\b")
    return motif-acronyme.findall(texte)
def detecter-run-on(texte: str) -> bool:
    mots = texte.strip().split()
    if len(mots) == 1 and len(texte.strip()) > 12:
        return bool(re.match(r"^[a-zA-Z\u0600-\u06FF]+$", texte.strip()))
    return False
def pretraiter-entree(message: str) -> dict:
    texte-original = message
    texte-pour-ml = nettoyer-pour-ml(message)
    blocs-math = detecter-blocs-math(message)
    acronymes = detecter-acronymes(message)
    est-run-on = detecter-run-on(message)
    nb-arabe = sum(1 for c in message if "\u0600" <= c <= "\u06FF")
    nb-latin = sum(1 for c in message if c.isalpha() and c.isascii())
    if nb-arabe > 0 and nb-latin > 0:
        type-script = "mixte-latin-arabe"
    elif nb-arabe > nb-latin:
        type-script = "arabe-pur"
    else:
        type-script = "latin-pur"
    message-nettoye = message.strip()
    est-ultra-court = len(message-nettoye) <= 2 and not any(c.isalnum() for c in message-nettoye)
    journal.info(
        f"[ETAPE-2] texte-original={texte-original!r} texte-pour-ml={texte-pour-ml!r} "
        f"blocs-math={blocs-math} acronymes={acronymes} est-run-on={est-run-on} "
        f"est-ultra-court={est-ultra-court} type-script={type-script}"
    )
    return {
        "texteOriginal": texte-original,
        "textePourMl": texte-pour-ml,
        "blocsMath": blocs-math,
        "acronymes": acronymes,
        "estUltraCourt": est-ultra-court,
        "estRunOn": est-run-on,
        "typeScript": type-script,
    }
def est-interrogatif(texte: str, intention-lr: str = "", confiance-lr: float = 0.0) -> bool:
    intentions-sociales = {"salutation", "remerciement", "au_revoir"}
    if intention-lr and confiance-lr >= 0.60:
        if intention-lr not in intentions-sociales:
            return True
    if "?" in texte:
        return True
    mots = texte.lower().split()
    mots-substantiels = [m for m in mots if len(m) > 4]
    if len(mots) > 4 and len(mots-substantiels) >= 2:
        return True
    motif-interrog = re.compile(
        r"\b(man[di]|me[lm]|ma[mt]|ri[gq]|bg?h|ac[hu]|is[q]|f[i]n[^\s]|an[dq]|ac[h]"
        r"|co[mq][b-z]|qu[aei]|wh[aeioy]|ho[wy]|whe|where|what|comb|qui|que|o[ùu])\w*\b",
        re.IGNORECASE
    )
    if motif-interrog.search(texte):
        return True
    return False
class DetecteurIntention:
    def __init__(self):
        self.modele = None
        self.est-charge = False
    def charger(self) -> bool:
        chemin = os.path.join(os.path.dirname(__file__), "models", "intent_classifier.pkl")
        if not os.path.exists(chemin):
            journal.warning(f"[BRAIN] Modele LR introuvable : {chemin}")
            return False
        try:
            with open(chemin, "rb") as f:
                self.modele = pickle.load(f)
            self.est-charge = True
            return True
        except Exception as e:
            journal.warning(f"[BRAIN] Modele LR non charge : {e}")
            return False
    def predire(self, texte: str, langue: str = "tamazight") -> dict:
        t = texte.lower().strip()
        mots = t.split()
        if t in mots-sociaux-exacts:
            return {"intent": mots-sociaux-exacts[t], "confidence": 0.99, "method": "exact-social"}
        intention-lr-prealable = ""
        confiance-lr-prealable = 0.0
        if self.est-charge and self.modele:
            try:
                probas = self.modele.predict_proba([t])[0]
                idx = int(np.argmax(probas))
                intention-lr-prealable = self.modele.classes_[idx]
                confiance-lr-prealable = float(probas[idx])
            except Exception:
                pass
        if len(mots) <= 2 and not est-interrogatif(t, intention-lr-prealable, confiance-lr-prealable):
            for mot in mots:
                if mot in mots-sociaux-exacts:
                    return {"intent": mots-sociaux-exacts[mot], "confidence": 0.97, "method": "word-social"}
        if self.est-charge and self.modele:
            try:
                probabilites = self.modele.predict_proba([texte])[0]
                indice = int(np.argmax(probabilites))
                confiance = float(probabilites[indice])
                seuil-ml = 0.75 if langue in {"arabe", "darija"} else 0.55
                if confiance >= seuil-ml:
                    return {
                        "intent": self.modele.classes_[indice],
                        "confidence": round(confiance, 3),
                        "method": "modele-lr",
                    }
            except Exception as e:
                journal.error(f"[ETAPE-4] Erreur LR : {e}")
        for intention, liste-mots in mots-cles-par-intention.items():
            if t in liste-mots:
                return {"intent": intention, "confidence": 0.90, "method": "keyword-exact"}
        for mot in mots:
            for intention, liste-mots in mots-cles-par-intention.items():
                if mot in liste-mots:
                    return {"intent": intention, "confidence": 0.80, "method": "keyword-word"}
        compteur-regles: dict = {}
        for motif, intention in regles-compilees:
            hits = len(motif.findall(t))
            if hits > 0:
                compteur-regles[intention] = compteur-regles.get(intention, 0) + hits
        if compteur-regles:
            meilleure = max(compteur-regles, key=lambda k: compteur-regles[k])
            nb-hits = compteur-regles[meilleure]
            confiance = min(0.75 + nb-hits * 0.05, 0.95)
            return {"intent": meilleure, "confidence": confiance, "method": "regex"}
        return {"intent": "general", "confidence": 0.50, "method": "fallback"}
detecteur = DetecteurIntention()
def obtenir-reponse-sociale-rapide(intention: str) -> str:
    options = reponses-sociales-rapides.get(intention)
    if not options:
        return "[TAM] Azul!\n[AR] مرحباً!"
    tam, ara = random.choice(options)
    tam-arabizi = appliquerArabiziSortie(tam)
    return f"[TAM] {tam-arabizi}\n[AR] {ara}"
def valider-reponse-llm(donnees: dict, identifiant: str = "init") -> bool:
    tam = donnees.get("tamazight", "").strip()
    ara = donnees.get("arabe", "").strip()
    if not tam:
        return False
    if len(tam.split()) < 3:
        return False
    if not ara or not re.search(r"[\u0600-\u06FF]", ara):
        return False
    if len(ara.split()) < 3:
        return False
    nb-non-latin = sum(
        1 for c in tam
        if "\u0600" <= c <= "\u06FF" or "\u2D30" <= c <= "\u2D7F"
    )
    ratio = nb-non-latin / max(len(tam), 1)
    return ratio < 0.35
def supprimer-echo-question(reponse: str, question: str) -> str:
    if re.search(r"\{.*\}", reponse) or re.search(r"\[TAM\]", reponse):
        return reponse.strip()
    tokens-question = set(normaliser(question).split())
    phrases-reponse = re.split(r"(?<=[.!?؟])\s+", reponse)
    phrases-finales = []
    for phrase in phrases-reponse:
        tokens-phrase = set(normaliser(phrase).split())
        ratio-commun = len(tokens-phrase & tokens-question) / max(len(tokens-phrase), 1)
        if ratio-commun > 0.85 and not phrases-finales:
            continue
        phrases-finales.append(phrase)
    return " ".join(phrases-finales).strip()
def detecter-remplissage(texte: str, seuil: float = 0.35) -> bool:
    mots = [m for m in texte.lower().split() if len(m) >= 3]
    if len(mots) < 10:
        return False
    _, nb-max = Counter(mots).most_common(1)[0]
    return (nb-max / len(mots)) > seuil
def nettoyer-cjk(texte: str) -> str:
    return re.sub(r"[\u4E00-\u9FFF\u3400-\u4DBF]", "", texte)
async def comprendre-requete(
    texte-brut: str,
    langue: str,
    identifiant: str = "init",
    acronymes: list = None,
) -> dict:
    initialiser-client-groq()
    langue-detectee = detecterLangueRequete(texte-brut)
    resultat = await extraire-hints-mots-etrangers(
        texte-brut, langue-detectee, identifiant, acronymes=acronymes or []
    )
    return resultat
async def extraire-hints-mots-etrangers(
    texte-brut: str,
    langue: str,
    identifiant: str = "init",
    acronymes: list = None,
) -> dict:
    acronymes = acronymes or []
    mots-sociaux-set = set(mots-sociaux-exacts.keys())
    if set(texte-brut.lower().split()) & mots-sociaux-set and len(texte-brut.split()) <= 3:
        return {
            "queryRag": normaliser(texte-brut),
            "tamTokensLow": [],
            "hintsLlm": "",
            "langueDetectee": langue,
        }
    tokens-classes: dict = {}
    contexte-acronymes = (
        f"\nACRONYMES DETECTES (ALL_CAPS) : {acronymes}"
        "\nCes tokens sont FORCEMENT NON_TAM HIGH — fournis l'expansion dans details."
        if acronymes else ""
    )
    prompt-systeme = (
        "You are a TOKEN CLASSIFIER for a Tamazight Latin NLP pipeline in Morocco.\n"
        "Classify EACH token from the message.\n\n"
        "OUTPUT FORMAT (strict JSON):\n"
        '{"tokens": {"word": {"part": "TAM_LATIN|NON_TAM", "confidence": "HIGH|LOW", "details": "..." or null}}}\n\n'
        "CLASSIFICATION RULES:\n"
        "  part='TAM_LATIN' -> Tamazight Latin script (including short particles n,d,i,g,s,w,a)\n"
        "  part='NON_TAM'   -> French, English, Arabic, city name, abbreviation, proper noun\n\n"
        "  confidence='HIGH' -> certain (>= 90%) -> provide details\n"
        "  confidence='LOW'  -> uncertain -> details = null -> needs RAG\n\n"
        "CRITICAL RULES:\n"
        "  1. SHORT PARTICLES (n, d, i, g, s, w, a) -> TAM_LATIN HIGH\n"
        "  2. ALL CAPS 2-5 chars (EST, BTS, DUT) -> NON_TAM HIGH, expand in details\n"
        "  3. CITY NAMES (khenifra, rabat) -> NON_TAM HIGH\n"
        "  4. FRENCH WORDS (les, filiers, des) -> NON_TAM HIGH\n"
        "  5. UNKNOWN TAM_LATIN words -> TAM_LATIN LOW, details=null\n\n"
        "EXAMPLES:\n"
        '  Input: "maydiyan les filiers n EST khenifra"\n'
        '  Output: {"tokens": {\n'
        '    "maydiyan": {"part": "TAM_LATIN", "confidence": "LOW", "details": null},\n'
        '    "les": {"part": "NON_TAM", "confidence": "HIGH", "details": "French article"},\n'
        '    "filiers": {"part": "NON_TAM", "confidence": "HIGH", "details": "French: filieres"},\n'
        '    "n": {"part": "TAM_LATIN", "confidence": "HIGH", "details": "Tamazight preposition"},\n'
        '    "EST": {"part": "NON_TAM", "confidence": "HIGH", "details": "Ecole Superieure de Technologie"},\n'
        '    "khenifra": {"part": "NON_TAM", "confidence": "HIGH", "details": "ville marocaine"}\n'
        "  }}\n"
        + contexte-acronymes
    )
    message-utilisateur = f"Classify each token in: {texte-brut}\nLanguage context: {langue}"
    journal.info(f"[{identifiant}] SOUS-ETAPE-3B envoi llama-instant texte={texte-brut!r}")
    try:
        debut-instant = time.time()
        reponse = await asyncio.wait_for(
            client-groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": prompt-systeme},
                    {"role": "user", "content": message-utilisateur},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=500,
            ),
            timeout=5.0,
        )
        duree-instant-ms = int((time.time() - debut-instant) * 1000)
        brut = reponse.choices[0].message.content or "{}"
        journal.info(f"[{identifiant}] SOUS-ETAPE-3C reponse brute ({duree-instant-ms}ms)\n{brut}")
        donnees = json.loads(brut)
        tokens-classes = donnees.get("tokens", {})
    except asyncio.TimeoutError:
        journal.warning(f"[{identifiant}] TIMEOUT 5s llama-instant")
        tokens-classes = {
            mot: {"part": "TAM_LATIN", "confidence": "LOW", "details": None}
            for mot in texte-brut.split() if len(mot) >= 2
        }
    except json.JSONDecodeError as e:
        journal.error(f"[{identifiant}] JSON invalide llama-instant : {e}")
        tokens-classes = {
            mot: {"part": "TAM_LATIN", "confidence": "LOW", "details": None}
            for mot in texte-brut.split() if len(mot) >= 2
        }
    except Exception as e:
        journal.warning(f"[{identifiant}] Erreur llama-instant : {e}")
        tokens-classes = {
            mot: {"part": "TAM_LATIN", "confidence": "LOW", "details": None}
            for mot in texte-brut.split() if len(mot) >= 2
        }
    tam-tokens-low: list = []
    lignes-entites: list = []
    tokens-ignores: list = []
    for mot, infos in tokens-classes.items():
        if not isinstance(infos, dict):
            tokens-ignores.append(mot)
            continue
        partie = str(infos.get("part", "")).upper()
        confiance = str(infos.get("confidence", "")).upper()
        details = infos.get("details")
        if partie == "TAM_LATIN" and confiance == "LOW":
            tam-tokens-low.append(mot)
        elif partie == "NON_TAM" and confiance == "HIGH" and details:
            lignes-entites.append(f"{mot} -> {details}")
        else:
            tokens-ignores.append(mot)
    journal.info(
        f"[{identifiant}] SOUS-ETAPE-3E tam-low={tam-tokens-low} entites={[l.split(' -> ')[0] for l in lignes-entites]} ignores={tokens-ignores}"
    )
    hints-llm = "\n".join(lignes-entites)
    if hints-llm:
        mots = hints-llm.split()
        if mots and mots.count(mots[0]) >= 3:
            hints-llm = ""
    nb-non-tam = sum(
        1 for v in tokens-classes.values()
        if isinstance(v, dict) and str(v.get("part", "")).upper() == "NON_TAM"
    )
    langue-detectee = (
        "mixte"
        if tokens-classes and nb-non-tam / max(len(tokens-classes), 1) > 0.5
        else langue
    )
    return {
        "queryRag": normaliser(texte-brut),
        "tamTokensLow": tam-tokens-low,
        "hintsLlm": hints-llm,
        "langueDetectee": langue-detectee,
    }
def rag-mot-individuel(mot: str, top-k: int = 2) -> list:
    if not mot or len(mot) < 2:
        return []
    fusion: dict = {}
    for res in sonderRag(mot, "general", tokensAutorises=[mot]).get("correspondances", [])[:top-k * 2]:
        fusion[res["tamazight"]] = max(fusion.get(res["tamazight"], 0), res["score"])
    mot-norm = normaliser(mot)
    if mot-norm != mot:
        for res in sonderRag(mot-norm, "general", tokensAutorises=[mot-norm]).get("correspondances", [])[:top-k * 2]:
            fusion[res["tamazight"]] = max(fusion.get(res["tamazight"], 0), res["score"])
    tries = sorted(fusion.items(), key=lambda x: -x[1])
    return [{"tamazight": t, "score": s} for t, s in tries[:top-k]]
def charger-exemples-few-shots() -> str:
    import csv as module-csv
    chemin = os.path.join(repertoire-donnees, "dataset.csv")
    entete = "EXEMPLES DE FORMAT ATTENDU :\n\n"
    if not os.path.exists(chemin):
        return entete
    intentions-cibles = set(mots-cles-par-intention.keys()) or {
        "salutation", "traduction", "question_info",
        "culture", "apprentissage", "au_revoir", "remerciement",
    }
    exemples-par-intention: dict = {}
    try:
        with open(chemin, encoding="utf-8", errors="replace") as f:
            lecteur = module-csv.DictReader(f)
            for ligne in lecteur:
                intention = (ligne.get("intent") or "").strip().lower()
                entree = (ligne.get("input") or ligne.get("text") or ligne.get("phrase") or "").strip()
                sortie = (ligne.get("output") or ligne.get("response") or "").strip()
                if intention not in intentions-cibles:
                    continue
                if intention in exemples-par-intention:
                    continue
                if not entree or not sortie or len(entree) > 80 or len(sortie) < 10:
                    continue
                if '"tamazight"' not in sortie and '"arabe"' not in sortie:
                    continue
                exemples-par-intention[intention] = (entree, sortie)
                if len(exemples-par-intention) >= len(intentions-cibles):
                    break
        if not exemples-par-intention:
            return entete
        lignes = [entete.rstrip()]
        for intention, (question, reponse) in exemples-par-intention.items():
            lignes.append(f"Q: {question}")
            lignes.append(f"R: {reponse}")
            lignes.append("")
        return "\n".join(lignes) + "\n"
    except Exception as e:
        journal.warning(f"[BRAIN] Erreur few-shots : {e}")
        return entete
def initialiser-prompt-systeme() -> str:
    few-shots = charger-exemples-few-shots()
    return (
        "Vous etes AWAL GPT, l'IA experte en langue et culture Tamazight.\n"
        "MISSION : Repondre aux questions avec des reponses COMPLETES au format JSON strict.\n\n"
        + few-shots
        + "REGLES ABSOLUES :\n"
        "1. Champ 'tamazight' : PHRASE COMPLETE en Tamazight Latin. Minimum 5 mots. INTERDIT : Tifinagh, Arabe, CJK.\n"
        "2. Champ 'arabe' : PHRASE COMPLETE en Arabe. Minimum 5 mots. INTERDIT : mots latins.\n"
        "3. Ne JAMAIS repeter la question dans la reponse.\n"
        "4. REPONDRE a la question posee — NE PAS la traduire mot a mot.\n"
        "5. Si RAG vide : utiliser [ENTITY_DETAILS] + connaissance generale.\n"
        "6. Blocs mathematiques [MATH] : resoudre d'abord, expliquer ensuite.\n"
        "7. Si [ENTITY_DETAILS] present : utiliser ces informations comme base.\n"
    )
prompt-systeme-base = ""
def construire-prompt(
    texte-brut: str,
    intention: str,
    langue: str,
    donnees-rag: dict,
    hints-llm: str,
    blocs-math: list,
    tam-tokens-low: list,
    identifiant: str,
) -> tuple:
    systeme = prompt-systeme-base
    systeme += f"\nCONTEXTE SPECIFIQUE : {instructions-par-intention.get(intention, 'Repondez de facon experte.')}"
    if langue == "français":
        systeme += "\nL'utilisateur ecrit en francais."
    elif langue in {"arabe", "darija"}:
        systeme += "\nL'utilisateur ecrit en arabe/darija. Champ 'tamazight' en Latin Tamazight."
    if not donnees-rag.get("pertinent"):
        systeme += (
            "\n\nIMPORTANT : Le corpus Tamazight local est vide pour cette requete. "
            "Utilisez [ENTITY_DETAILS] et votre connaissance generale. "
            "NE DITES PAS que vous ne savez pas."
        )
    prompt = f"Question: {texte-brut}\n\n"
    if hints-llm:
        prompt += (
            f"[ENTITY_DETAILS]\n{hints-llm}\n\n"
            "INSTRUCTION : Utilise les ENTITY_DETAILS pour construire une vraie reponse.\n\n"
        )
    prompt += f"Contexte RAG (Top 10):\n{donnees-rag['english_ctx']}\n\n"
    prompt += f"[INTENT] {intention}\n"
    indices-dico = []
    for mot in texte-brut.lower().split():
        if len(mot) > 2:
            indice = rechercherMotDictionnaire(mot)
            if indice:
                indices-dico.append(indice)
    if indices-dico:
        prompt += f"[DICTIONARY_HINTS] {', '.join(indices-dico)}\n"
    intentions-sociales = {"salutation", "remerciement", "au_revoir"}
    if len(texte-brut.split()) >= 3 and intention not in intentions-sociales:
        if tam-tokens-low:
            correspondances-faiss = obtenirTop2MotsSimilaires(" ".join(tam-tokens-low))
        else:
            correspondances-faiss = obtenirTop2MotsSimilaires(texte-brut)
        if correspondances-faiss:
            prompt += f"[WORD_MATCHES] {correspondances-faiss}\n"
    if intention == "culture":
        proverbe = obtenirProverbePourRequete(texte-brut)
        if proverbe and proverbe.get("tamazight"):
            prompt += f"[PROVERBE] {proverbe['tamazight']} ({proverbe.get('fr', '')})\n\n"
    if blocs-math:
        prompt += f"[MATH] Resous et explique : {', '.join(blocs-math)}\n"
    prompt += (
        "\nRepondez en JSON avec les cles 'tamazight' et 'arabe' UNIQUEMENT.\n"
        "OBLIGATOIRE : chaque champ doit etre une PHRASE COMPLETE (minimum 5 mots).\n"
    )
    return systeme, prompt
async def appeler-llm-avec-retry(
    systeme: str,
    prompt: str,
    historique: list,
    intention: str,
    identifiant: str,
) -> tuple:
    temperature-base = 0.40 if intention in {"traduction", "question_info"} else 0.70
    messages-base = [{"role": "system", "content": systeme}]
    if historique:
        messages-base.extend(historique[-10:])
    donnees-reponse = {}
    tentative = 0
    temps-llm-ms = 0
    while tentative < 2:
        messages-tour = messages-base.copy()
        messages-tour.append({"role": "user", "content": prompt})
        temp-courante = temperature-base + 0.1 * tentative
        debut-llm = time.time()
        reponse-groq = await asyncio.wait_for(
            client-groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages-tour,
                response_format={"type": "json_object"},
                temperature=temp-courante,
                max_tokens=800,
            ),
            timeout=20.0,
        )
        temps-llm-ms = int((time.time() - debut-llm) * 1000)
        contenu = reponse-groq.choices[0].message.content
        journal.info(f"[{identifiant}] ETAPE-7 reponse brute ({temps-llm-ms}ms)\n{contenu}")
        try:
            donnees-reponse = json.loads(contenu)
            if valider-reponse-llm(donnees-reponse, identifiant):
                break
            messages-base.append({"role": "assistant", "content": contenu})
            messages-base.append({
                "role": "user",
                "content": "REPONSE REFUSEE : phrase complete requise (minimum 5 mots). tamazight: Latin UNIQUEMENT. arabe: Arabe pur UNIQUEMENT.",
            })
        except json.JSONDecodeError as e:
            journal.warning(f"[{identifiant}] JSON invalide tentative {tentative + 1} : {e}")
        tentative += 1
    return donnees-reponse, tentative, temps-llm-ms
def valider-et-nettoyer(donnees: dict, texte-brut: str, tentative: int, identifiant: str) -> tuple:
    if tentative >= 2:
        journal.error(f"[{identifiant}] ECHEC apres 2 tentatives -> repli dictionnaire")
        texte-repli = construireRepliInformatif(texte-brut)
        tam-repli = texte-repli.split("[AR]")[0].replace("[TAM]", "").strip()
        ara-repli = texte-repli.split("[AR]")[1].strip() if "[AR]" in texte-repli else ""
        return tam-repli, ara-repli
    tam = supprimer-echo-question(donnees.get("tamazight", ""), texte-brut)
    tam = nettoyer-cjk(tam)
    ara = nettoyer-cjk(donnees.get("arabe", "").strip())
    tam = appliquerArabiziSortie(tam)
    if detecter-remplissage(tam):
        journal.warning(f"[{identifiant}] Remplissage detecte -> repli")
        texte-repli = construireRepliInformatif(texte-brut)
        tam-repli = texte-repli.split("[AR]")[0].replace("[TAM]", "").strip()
        tam = appliquerArabiziSortie(tam-repli)
        ara = texte-repli.split("[AR]")[1].strip() if "[AR]" in texte-repli else ""
    return tam, ara
async def generer-reponse(
    texte-brut: str,
    intention: str,
    query-rag: str,
    langue: str = "tamazight",
    historique: list = None,
    identifiant: str = "init",
    hints-llm: str = "",
    blocs-math: list = None,
    tam-tokens-low: list = None,
) -> dict:
    initialiser-client-groq()
    try:
        debut-rag = time.time()
        donnees-rag = construireContexteAnglais(
            query-rag, intention, topK=10,
            tokensAutorises=tam-tokens-low if tam-tokens-low else None
        )
        temps-rag-ms = int((time.time() - debut-rag) * 1000)
        journal.info(f"[{identifiant}] ETAPE-5 RAG pertinent={donnees-rag['pertinent']} score={donnees-rag['score_max']:.3f} ({temps-rag-ms}ms)")
        systeme, prompt = construire-prompt(
            texte-brut, intention, langue, donnees-rag,
            hints-llm, blocs-math or [], tam-tokens-low or [], identifiant
        )
        donnees-reponse, tentative, temps-llm-ms = await appeler-llm-avec-retry(
            systeme, prompt, historique or [], intention, identifiant
        )
        tam, ara = valider-et-nettoyer(donnees-reponse, texte-brut, tentative, identifiant)
        return {
            "reponse": f"[TAM] {tam}\n[AR] {ara}",
            "tamazight": tam,
            "arabe": ara,
            "tempsRagMs": temps-rag-ms,
            "tempsLlmMs": temps-llm-ms,
        }
    except Exception as e:
        journal.error(f"[{identifiant}] ERREUR GROQ : {type(e).__name__}: {e}")
        repli = construireRepliInformatif(texte-brut)
        return {"reponse": repli, "tamazight": repli, "arabe": "", "tempsRagMs": 0, "tempsLlmMs": 0}
async def traiter-message(
    message: str,
    domaine: str = None,
    historique: list = None,
) -> dict:
    identifiant = uuid.uuid4().hex[:8]
    message-nettoye = message.strip()
    journal.info(f"[{identifiant}] NOUVEAU MESSAGE : {message!r}")
    if len(message-nettoye) < 2:
        return {
            "reponse": "[TAM] Ur fhim ara. 3awed s wawal nni3en.\n[AR] لم أفهم. أعد السؤال بكلمات أخرى.",
            "intention": "incompris", "confiance": 1.0,
            "tempsRagMs": 0, "tempsLlmMs": 0, "langue": "tamazight",
        }
    pretraitement = pretraiter-entree(message)
    if pretraitement["estUltraCourt"] and message-nettoye not in mots-sociaux-exacts:
        return {
            "reponse": "[TAM] Ur fhim ara. 3awed s wawal nni3en.\n[AR] لم أفهم. أعد السؤال بكلمات أخرى.",
            "intention": "incompris", "confiance": 1.0,
            "tempsRagMs": 0, "tempsLlmMs": 0, "langue": "tamazight",
        }
    blocs-math = pretraitement["blocsMath"]
    acronymes = pretraitement["acronymes"]
    texte-pour-ml = pretraitement["textePourMl"]
    langue = detecterLangueRequete(message)
    norme = normaliser(message)
    comprehension = await comprendre-requete(message, langue, identifiant, acronymes=acronymes)
    query-rag = comprehension.get("queryRag", norme) or norme or message
    tam-tokens-low = comprehension.get("tamTokensLow", [])
    hints-llm = comprehension.get("hintsLlm", "")
    if not query-rag.strip():
        query-rag = "tamazight langue culture amazigh"
    resultat-intention = detecteur.predire(texte-pour-ml or norme or message, langue=langue)
    intention = resultat-intention["intent"]
    confiance = resultat-intention["confidence"]
    intentions-sociales = {"salutation", "remerciement", "au_revoir"}
    message-court = len(message.strip().split()) <= 4
    if (
        intention in intentions-sociales
        and confiance > 0.90
        and message-court
        and not est-interrogatif(message, intention, confiance)
    ):
        return {
            "reponse": obtenir-reponse-sociale-rapide(intention),
            "intention": intention, "confiance": confiance,
            "tempsRagMs": 0, "tempsLlmMs": 0, "langue": langue,
        }
    resultat-generation = await generer-reponse(
        texte-brut=message,
        intention=intention,
        query-rag=query-rag,
        langue=langue,
        historique=historique,
        identifiant=identifiant,
        hints-llm=hints-llm,
        blocs-math=blocs-math,
        tam-tokens-low=tam-tokens-low,
    )
    return {
        "reponse": resultat-generation["reponse"],
        "intention": intention,
        "confiance": confiance,
        "tempsRagMs": resultat-generation.get("tempsRagMs", 0),
        "tempsLlmMs": resultat-generation.get("tempsLlmMs", 0),
        "langue": langue,
    }
charger-configuration-intentions()
detecteur.charger()
prompt-systeme-base = initialiser-prompt-systeme()