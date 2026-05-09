# brain.py — Essentialist Orchestration
import os, re, json, time, logging, asyncio, pickle
from dotenv import load_dotenv
load_dotenv()
from groq import AsyncGroq
from rapidfuzz import process, fuzz
import core
import numpy as np

DATA_DIR = core.DATA_DIR
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "intent_classifier.pkl")
client_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

dict_local = {"mots-cles": {}, "instr": {}, "soc-rep": {}, "lr": None, "seuil": 0.55}

def charger_configs():
    for f, v in [("mots_cles_intentions.json", "mots-cles"),
                 ("instructions_intentions.json", "instr"), ("reponses_sociales.json", "soc-rep")]:
        p = os.path.join(DATA_DIR, f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as file: 
                data = json.load(file)
                if v == "mots-cles":
                    dict_local[v] = {k: [core.normaliser(m) for m in l] for k, l in data.items()}
                else:
                    dict_local[v] = data
                logging.info(f"[BRAIN] Load: {f} -> {len(dict_local[v])} items")
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            m = pickle.load(f)
            dict_local["lr"] = m["model"] if isinstance(m, dict) else m
            dict_local["seuil"] = m.get("optimal_threshold", 0.55) if isinstance(m, dict) else 0.55
    logging.info("[BRAIN] Configurations et modèle d'intention chargés")

async def pretraiter(m):
    t = m.strip().split()
    maths = re.findall(r"\d+[x-z]|[0-9+\-*/=()²³√]+", m)
    logging.debug(f"[BRAIN] Preproc: {len(t)} tokens, {len(maths)} math blocks | Tokens: {t}")
    return {"tokens": t, "maths": maths}

def detecter_intent(m, nb_tokens=1):
    n = core.normaliser(m)
    
    # 1. Keywords Match (including merged social ones)
    found_intent = None
    for intent, mots in dict_local["mots-cles"].items():
        if n in mots:
            found_intent = intent
            break
            
    # Social Shortcut : Uniquement si message très court et mot exact
    social_intents = ["salutation", "remerciement", "au_revoir"]
    if found_intent in social_intents and nb_tokens < 3:
        logging.debug(f"[BRAIN] Intent (Social Shortcut): {found_intent}")
        return {"intent": found_intent, "method": "social", "confiance": 1.0}

    # 2. LR Model (Priorisé pour messages longs ou complexes)
    if dict_local["lr"]:
        p = dict_local["lr"].predict_proba([n])[0]
        i = np.argmax(p)
        if p[i] >= dict_local["seuil"]: 
            intent = dict_local["lr"].classes_[i]
            logging.debug(f"[BRAIN] Intent (LR Model): {intent} with prob {p[i]:.4f}")
            return {"intent": intent, "method": "LR", "confiance": float(p[i])}

    # 3. Keyword Match Fallback (si LR est faible)
    if found_intent:
        logging.debug(f"[BRAIN] Intent (Keyword match): {found_intent}")
        return {"intent": found_intent, "method": "keywords", "confiance": 0.85}

    # 4. Fuzzy Keywords (Dernier recours)
    flat_kws = []
    for intent, mots in dict_local["mots-cles"].items():
        for kw in mots: flat_kws.append((kw, intent))
    
    if flat_kws:
        match = process.extractOne(n, [x[0] for x in flat_kws], scorer=fuzz.WRatio, score_cutoff=80)
        if match:
            kw_found = match[0]
            intent_found = next(x[1] for x in flat_kws if x[0] == kw_found)
            logging.debug(f"[BRAIN] Intent (Fuzzy): {intent_found} via '{kw_found}' (score={match[1]})")
            return {"intent": intent_found, "method": "fuzzy", "confiance": 0.75}

    logging.debug(f"[BRAIN] Intent (Fallback): general")
    return {"intent": "general", "method": "fallback", "confiance": 0.5}

async def traiter_message(message: str, historique: list = None):
    t0 = time.time()
    
    # 1. Préparation
    pre = await pretraiter(message)
    yield {"type": "step", "label": f"Analyse : {len(pre['tokens'])} tokens", "step": "preproc"}
    
    # 2. Intention
    intent = detecter_intent(message, nb_tokens=len(pre['tokens']))
    logging.info(f"[BRAIN] Message: '{message[:50]}...' | Intent: {intent['intent']} ({intent['confiance']:.2f})")
    yield {"type": "step", "label": f"Intention : {intent['intent']} ({int(intent['confiance']*100)}%)", "step": "intent"}

    if intent["method"] == "social":
        opts = dict_local["soc-rep"].get(intent["intent"], [["Azul", "أهلاً"]])
        # opts[0] est une paire [Tamazight, Arabe]
        pair = opts[0]
        tam_raw = pair[0] if isinstance(pair, list) and len(pair) > 0 else "Azul"
        ara_raw = pair[1] if isinstance(pair, list) and len(pair) > 1 else ""
        
        tam = core.appliquer_arabizi_sortie(tam_raw)
        yield {
            "type": "final", 
            "reponse": f"[TAM] {tam}\n\n[AR] {ara_raw}", 
            "intention": intent["intent"], 
            "confiance": intent["confiance"],
            "temps_total": int((time.time()-t0)*1000)
        }
        return

    # 3. RAG / Recherche
    t_rag_0 = time.time()
    rag_res = []
    for t in pre["tokens"]:
        if len(t) < 3: continue
        matches = core.rechercher_rapidfuzz(t)
        if matches: 
            rag_res.append(f"{t} -> {matches[0]['traductions']}")
            logging.debug(f"[BRAIN] RAG Hit: '{t}' matches '{matches[0]['mot']}' (score={matches[0]['score']})")
    
    t_rag_ms = int((time.time() - t_rag_0) * 1000)
    logging.info(f"[BRAIN] RAG: {len(rag_res)} matches found in {t_rag_ms}ms")
    yield {"type": "step", "label": f"RAG : {len(rag_res)} sources trouvées", "step": "rag"}
    
    # 4. Génération LLM
    yield {"type": "step", "label": "Synthèse Llama-3.3-70B...", "step": "llm"}
    t_llm_0 = time.time()
    
    instr = dict_local["instr"].get(intent["intent"], "Réponds de façon experte.")
    sys = (
        "Tu es AWAL GPT, expert reconnu en langue et culture Tamazight Amazighe.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 MISSION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Comprendre la question dans son ensemble et répondre avec précision.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ RÈGLE CRITIQUE — CONTEXTE RAG\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Le bloc [TRANSLATIONS RAG] contient des correspondances APPROXIMATIVES.\n"
        "Ces traductions peuvent être INCORRECTES ou HORS CONTEXTE.\n\n"
        "✅ UTILISE le RAG UNIQUEMENT si la traduction est cohérente avec la question\n"
        "❌ IGNORE le RAG si la traduction semble absurde ou hors sujet\n"
        "💡 EN CAS DE DOUTE → utilise ta propre connaissance du Tamazight/Darija\n\n"
        "Exemple de RAG INCORRECT à ignorer:\n"
        "  lmghrib → 'sunset prayer'  ← FAUX, lmghrib = le Maroc\n"
        "  righ → 'I Agree'           ← FAUX, righ = je veux / j'aime\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📝 RÈGLES DE LANGUE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. 📖 Champ 'tamazight' → Tamazight Latin UNIQUEMENT\n"
        "   • Autorisé : mélange avec termes français entre parenthèses\n"
        "   • Interdit : Tifinagh, arabe pur, CJK\n"
        "2. 🌍 Champ 'arabe' → Arabe فصحى structuré\n"
        "   • question_info : listes numérotées et sous-sections\n"
        "   • Ponctuation arabe : ، ؟ ؛\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 INTENTION DÉTECTÉE : {intent['intent']}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{instr}\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📦 FORMAT DE SORTIE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        'JSON strict UNIQUEMENT : {"tamazight": "...", "arabe": "..."}\n'
        "• Pas de markdown, pas d'explication hors JSON\n"
        "• Longueur adaptée à la complexité de la question\n"
    )
    if pre.get("maths"):
        sys += f"\n🔢 BLOCS MATHÉMATIQUES détectés: {pre['maths']} → Résoudre EN PREMIER, puis expliquer."
    usr = f"Message: {message}\nContexte RAG: {rag_res}"
    
    logging.debug(f"[BRAIN] LLM Prompt (System): {sys}")
    logging.debug(f"[BRAIN] LLM Prompt (User): {usr}")
    
    try:
        r = await client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[{"role":"system","content":sys}] + (historique or []) + [{"role":"user","content":usr}], 
            response_format={"type":"json_object"}
        )
        raw = json.loads(r.choices[0].message.content)
        logging.debug(f"[BRAIN] LLM Raw Response: {raw}")
    except Exception as e:
        logging.error(f"[BRAIN] LLM Error: {e}")
        raw = {"tamazight": "Ur d-yettawi ubrid.", "arabe": "خطأ في التوليد."}

    t_llm_ms = int((time.time() - t_llm_0) * 1000)
    logging.info(f"[BRAIN] LLM: Synthesis completed in {t_llm_ms}ms")
    tam = core.appliquer_arabizi_sortie(raw.get("tamazight", "Ur d-yettawi ubrid."))
    
    yield {
        "type": "final", 
        "reponse": f"[TAM] {tam}\n\n[AR] {raw.get('arabe','لا توجد معلومات.')}", 
        "intention": intent["intent"], 
        "confiance": intent["confiance"],
        "temps_rag_ms": t_rag_ms,
        "temps_llm_ms": t_llm_ms,
        "temps_total": int((time.time()-t0)*1000)
    }

charger_configs()