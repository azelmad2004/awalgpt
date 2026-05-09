# core.py — Moteur de données minimaliste (RapidFuzz uniquement)
# Respect strict du pipeline : dict-local, cache LRU, fusion CSV, normalisation, arabizi.

import os, re, csv, json, time, logging
from collections import OrderedDict
from rapidfuzz import fuzz, process

# ═══════════════════════ Config inline ═══════════════════════
dict_local = {
    "config": {
        "score-cutoff-rapidfuzz": 70,
        "top-k-rag": 3,
        "cache-max-size": 500,
        "cache-ttl-sec": 300,
        "max-variantes-ortho": 5,
        "seuil-validation-vocab": 0.5,
        "seuil-protection-mot": 0.65,
    },
    "dict-mem": [],
    "fuzz-cache": OrderedDict(),
    "arabizi-map": {},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
log = logging.getLogger("awalgpt.core")

# ═══════════════════════ Chargement initial ═══════════════════════
def charger_donnees_fusionnees():
    # Priority: dicti.json (JSON) -> darija.csv
    fichiers = ["dicti.json", "darija.csv"]
    seen = {}
    for nom in fichiers:
        chemin = os.path.join(DATA_DIR, nom)
        if not os.path.exists(chemin):
            continue
            
        if nom.endswith(".json"):
            with open(chemin, encoding="utf-8") as f:
                data = json.load(f)
                for ligne in data:
                    mot = (ligne.get("amazigh") or "").strip()
                    if not mot: continue
                    mot_low = mot.lower()
                    if mot_low in seen: continue
                    rangee = {
                        "mot": mot,
                        "fr": (ligne.get("fr") or "").strip(),
                        "ar": (ligne.get("ar") or "").strip(),
                        "en": (ligne.get("en") or "").strip(),
                        "source": nom,
                    }
                    seen[mot_low] = len(dict_local["dict-mem"])
                    dict_local["dict-mem"].append(rangee)
        else:
            with open(chemin, encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                for ligne in reader:
                    mot = (ligne.get("amazigh") or ligne.get("tamazight") or ligne.get("mot") or ligne.get("text") or "").strip()
                    if not mot: continue
                    mot_low = mot.lower()
                    rangee = {
                        "mot": mot,
                        "fr": (ligne.get("fr") or ligne.get("trans_fr") or "").strip(),
                        "ar": (ligne.get("ar") or ligne.get("trans_ar") or "").strip(),
                        "en": (ligne.get("en") or ligne.get("trans_en") or "").strip(),
                        "source": nom,
                    }
                    if mot_low in seen:
                        idx = seen[mot_low]
                        for lang in ("fr", "ar", "en"):
                            if not dict_local["dict-mem"][idx][lang] and rangee[lang]:
                                dict_local["dict-mem"][idx][lang] = rangee[lang]
                        continue
                    seen[mot_low] = len(dict_local["dict-mem"])
                    dict_local["dict-mem"].append(rangee)
    log.info(f"[INIT] Fusion Data → {len(dict_local['dict-mem'])} entrées (Primary: dicti.json)")
    if not dict_local["dict-mem"]:
        log.error("[CRITICAL] RAG dict-mem is EMPTY. Check data files.")
        raise RuntimeError("RAG memory cannot be empty.")

def charger_arabizi_map():
    chemin = os.path.join(DATA_DIR, "normalisation.json")
    if os.path.exists(chemin):
        with open(chemin, encoding="utf-8") as f:
            dict_local["arabizi-map"] = json.load(f)

charger_arabizi_map()
charger_donnees_fusionnees()

# ═══════════════════════ Fonctions principales ═══════════════════════
def normaliser(texte: str) -> str:
    """Nettoyage phonétique ML."""
    if not texte:
        return ""
    res = texte.lower()
    # normalisation multi-char d'abord
    norm_map = dict_local["arabizi-map"].get("normalisation", {})
    for src, dst in sorted(norm_map.items(), key=lambda x: -len(x[0])):
        res = res.replace(src, dst)
    # Whitelist: Latin, Arabes, Chiffres, et caractères Tamazight spéciaux (ɣ, ɛ, ḥ, ṭ, etc.)
    res = re.sub(r"[^a-z0-9\u0600-\u06FF\u0263\u025b\u1e25\u1e6d\u1e95\u1e5b\u1e0d\u0194\u0190\s.,!?;:\-]", " ", res)
    final = re.sub(r"\s+", " ", res).strip()
    return final

def appliquer_arabizi_sortie(texte: str) -> str:
    """Conversion arabizi pour sortie (tamazight latin)."""
    if not texte:
        return texte
    res = texte
    for section in ("outputMulti", "output"):
        for src, dst in dict_local["arabizi-map"].get(section, {}).items():
            res = re.sub(re.escape(src), dst, res)
    if res != texte:
        log.debug(f"[CORE] Arabizi Output: '{texte[:40]}...' -> '{res[:40]}...'")
    return res

def rechercher_rapidfuzz(mot_brut: str) -> list:
    """RAG via RapidFuzz avec cache LRU — sans fallback loop pour performance."""
    top_k = dict_local["config"]["top-k-rag"]
    cutoff = dict_local["config"]["score-cutoff-rapidfuzz"]
    now = time.time()
    cache_key = f"{mot_brut}:{top_k}:{cutoff}"
    cache = dict_local["fuzz-cache"]
    if cache_key in cache:
        ts, val = cache[cache_key]
        if now - ts < dict_local["config"]["cache-ttl-sec"]:
            cache.move_to_end(cache_key)
            log.debug(f"[CORE] RAG Cache Hit: '{mot_brut}'")
            return val
        del cache[cache_key]
    variantes = {mot_brut.lower(), normaliser(mot_brut)}
    subs = {
        "7": ["h"], "h": ["7"],
        "3": ["ɛ"], "ɛ": ["3"],
        "9": ["q"], "q": ["9"],
        "4": ["ɣ"], "ɣ": ["4"],
        "5": ["x"], "x": ["5"],
    }
    for char, replacements in subs.items():
        if char in mot_brut.lower():
            for r in replacements:
                variantes.add(mot_brut.lower().replace(char, r))
    if len(mot_brut) > 3:
        variantes.add(mot_brut[1:])
        variantes.add(mot_brut[:-1])
    # Cutoff dynamique selon longueur du mot
    if len(mot_brut) < 4:
        d_cutoff = 45
    elif len(mot_brut) < 6:
        d_cutoff = 55
    else:
        d_cutoff = cutoff
    log.debug(f"[CORE] RAG Variants for '{mot_brut}': {variantes}")
    choix = [e["mot"].lower() for e in dict_local["dict-mem"]]
    resultats = []
    vus = set()
    for var in variantes:
        if not var:
            continue
        matches = process.extract(
            var, choix,
            scorer=fuzz.WRatio,
            limit=top_k,
            score_cutoff=d_cutoff,
        )
        for mot, score, _ in matches:
            # Protection : mot très court ne peut pas matcher mot long
            if len(mot) <= 2 and len(mot_brut) >= 4:
                continue
            if mot not in vus:
                vus.add(mot)
                entree = next(e for e in dict_local["dict-mem"] if e["mot"].lower() == mot)
                resultats.append({
                    "mot": entree["mot"],
                    "score": score,
                    "traductions": {
                        "fr": entree["fr"],
                        "ar": entree["ar"],
                        "en": entree["en"],
                    },
                })
    # ── Pas de fallback loop ── supprimé car cause 55s de latence
    resultats.sort(key=lambda x: (x["score"], -abs(len(x["mot"]) - len(mot_brut))), reverse=True)
    resultats = resultats[:top_k]
    if resultats:
        log.debug(f"[CORE] RAG Hits for '{mot_brut}': {[r['mot'] for r in resultats]}")
    if len(cache) >= dict_local["config"]["cache-max-size"]:
        cache.popitem(last=False)
    cache[cache_key] = (now, resultats)
    return resultats

def construire_repli_informatif(contexte: str) -> str:
    """Génère une réponse de secours basée sur les mots-clés trouvés dans le contexte/question."""
    tokens = re.findall(r"\b\w{3,}\b", contexte.lower())
    dict_mots = [e["mot"].lower() for e in dict_local["dict-mem"]]
    matches = []
    
    # On cherche les 3 meilleurs candidats RAG dans le texte
    for t in tokens:
        if len(matches) >= 3: break
        res = process.extractOne(t, dict_mots, score_cutoff=80)
        if res:
            entree = next(e for e in dict_local["dict-mem"] if e["mot"].lower() == res[0])
            matches.append(entree)
            
    if not matches:
        return "[TAM] Ur d-yettawi ubrid.\n[AR] لا توجد معلومات كافية حالياً."
        
    tam = "Awalen n ufares: " + ", ".join([m['mot'] for m in matches])
    ara = "الكلمات المفتاحية: " + ", ".join([m['ar'] for m in matches if m['ar']])
    return f"[TAM] {tam}\n[AR] {ara}"

def valider_vocabulaire(texte: str) -> bool:
    """Ratio de ≥50% mots (≥3 lettres) proches du dictionnaire (partial_ratio≥65)."""
    tokens = re.findall(r"\b\w{3,}\b", texte.lower())
    if not tokens:
        return False
    dict_mots = [e["mot"].lower() for e in dict_local["dict-mem"]]
    # Utilisation de extractOne pour la rapidité
    trouves = sum(1 for mot in tokens if process.extractOne(mot, dict_mots, scorer=fuzz.partial_ratio, score_cutoff=65))
    return (trouves / len(tokens)) >= dict_local["config"]["seuil-validation-vocab"]