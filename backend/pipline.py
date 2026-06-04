import json
import os
import pickle
import re
import time
import numpy as np
import spacy
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
GGUF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "awal-chatbot-q4_k_m.gguf")
DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dictionaire.json")
KW_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mots_cles_intentions.json")
PKL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "intent_classifier.pkl")
NORM_MAP = {
    "gh": "ɣ", "kh": "x", "ch": "c", "sh": "š", "zh": "ž",
    "aa": "a", "ee": "e", "oo": "u",
    "é": "e", "è": "e", "ê": "e", "à": "a", "â": "a",
    "î": "i", "ù": "u", "ü": "u", "ç": "c",
}

# IPA/Tamazight → Arabizi (applied to LLM output so the latin line uses chat-style numbers)
ARABIZI_MAP = {
    "ɣ": "gh", "x": "kh",
    "ɛ": "3",  "ħ": "7",  "q": "9",  "ʔ": "2",
}

STOPWORDS = {
    "d", "n", "i", "u", "g", "k", "t", "m", "s", "l", "r", "a", "w", "f",
    "ay", "an", "in", "im", "is", "as", "ar", "ur", "wa", "wi", "di", "deg",
    "man", "may", "mayd", "ma", "ad", "gh", "iw", "aw", "righ", "nigh",
}

INSTRUCTION_MAP = {
    "traduction":    "Traduis précisément le mot demandé. Utilise le dictionnaire RAG.",
    "salutation":    "Réponds chaleureusement avec une salutation amazighe.",
    "question_info": "Réponds directement et précisément à la question.",
    "culture":       "Donne des faits culturels amazighs vérifiés.",
    "apprentissage": "Donne un exemple concret avec explication courte.",
    "remerciement":  "Réponds avec gratitude en Tamazight + Arabe.",
    "au_revoir":     "Réponds avec une formule de départ amazighe.",
    "general":       "Réponds naturellement et utilement.",
}

QUESTION_WORDS = ("maydyan", "mayd", "mamek", "manta", "acu", "mani", "amek")

TRANSLATION_SKIP = {"traduis", "translate", "tarjama", "s", "tmazight", "tamazight", "en", "in"}

STOP_TOKENS = ["<|im_end|>", "\n\nUser:", "User:"]

SYSTEM_PROMPT = """\
Tu es AWAL — assistant IA spécialisé en Tamazight Amazigh marocaine.
Tu réponds TOUJOURS en Tamazight Latin + Arabe.

━━━ CONTEXTE ━━━
INTENTION   : {intent}
INSTRUCTION : {instruction}
RAG         : {rag}
ENTITÉS     : {entities}
HISTORIQUE  : {history}

━━━ RÈGLES DE RÉPONSE ━━━
1. FORMAT STRICT — deux lignes uniquement :
   latin: [réponse en Tamazight Latin]
   ar: [réponse en Arabe]

2. LONGUEUR selon l'intention :
   - salutation / remerciement / au_revoir → 1 phrase chaleureuse suffit
   - traduction                            → mot(s) traduit(s) uniquement, pas de phrase
   - question_info / culture               → 2 à 4 phrases complètes, sois précis et informatif
   - apprentissage                         → explication + exemple concret, minimum 2 phrases
   - general                               → adapte la longueur à la complexité de la question

3. QUALITÉ :
   - Utilise le RAG en priorité. S'il contient la réponse, base-toi dessus.
   - Si le RAG est vide ou insuffisant et que tu ne sais pas → réponds :
       latin: Ur ssine3, ur ili umeslay-agi deg usebded inu.
       ar: لا أعلم، هذا الموضوع خارج نطاق معرفتي.
   - N'invente JAMAIS de faits, de noms, de lieux ou de traductions.
   - N'écris JAMAIS en anglais, même dans la ligne ar.
   - N'écris JAMAIS en Tifinagh (ⵣ ⵓ ⵎ...) — Latin uniquement pour la ligne latin.
   - La ligne latin et la ligne ar doivent dire exactement la même chose.

4. ARABIZI dans la ligne latin :
   - Utilise le style arabizi : écris 3 pour ɛ, 7 pour ħ, 9 pour q, 2 pour ʔ, gh pour ɣ.

━━━ EXEMPLES PAR INTENTION ━━━

# salutation
User: Azul fellak!
latin: Azul! Marhba bik, amek tellid?
ar: أهلاً وسهلاً! كيف حالك؟

# traduction
User: Traduis pain s tmazight
latin: aghrum
ar: الخبز

# question_info (avec RAG)
User: Manta tga Khenifra?
RAG: khenifra(100%)=FR:Khénifra (ville du Maroc)|AR:خنيفرة (مدينة مغربية)
latin: Khenifra d tadart tamqqrant deg waTTas n lAtlas Ufella, tzemmar d lqel3a n Zayan. Tella deg wammas n lMghrib.
ar: خنيفرة مدينة كبيرة في قلب الأطلس المتوسط، وهي عاصمة قبيلة زيان التاريخية، تقع في وسط المغرب.

# question_info (RAG vide, sujet inconnu)
User: Maydyan EST Khenifra?
RAG: Aucun
latin: Ur ssine3, ur ili umeslay-agi deg usebded inu.
ar: لا أعلم، هذا الموضوع خارج نطاق معرفتي.

# culture
User: Maydyan tga tafaska n Yennayer?
latin: Yennayer d aseggas amaynut amazigh, ittwaskar deg wass wis maraw d yan n Yennayer. D aselmed n tmazight d wudem n tsekla tamazight, ittwaskar s tmeghda d wuccen d ighsan.
ar: يناير هو رأس السنة الأمازيغية، يُحتفل به في الحادي عشر من يناير. يُعدّ مناسبة لإحياء الهوية الأمازيغية بالأكل التقليدي والاحتفالات الشعبية.

# apprentissage
User: Lemdeɣ awal amaynut: "amddakwel"
latin: Awal "amddakwel" yes3a amawal: "ami" ney "ameddukel". Ictawel: "Amddakwel inu d Yidir" — "Mon ami s'appelle Yidir".
ar: كلمة "أمضاكول" تعني "الصديق". مثال: "أمضاكولي هو يدير" — صديقي اسمه يدير.
"""
# ══════════════════════════════════════════════════════
# LOAD RESOURCES — called once at import
# ══════════════════════════════════════════════════════

def load_dictionary():
    raw = json.load(open(DICT_PATH, "r", encoding="utf-8"))
    dictionary = []
    word_index = {}
    seen = set()
    for entry in raw:
        word = (entry.get("amazigh") or entry.get("mot") or "").strip().lower()
        if not word or word in seen:
            continue
        seen.add(word)
        item = {
            "mot": word,
            "fr":  entry.get("fr", ""),
            "ar":  entry.get("ar", ""),
            "en":  entry.get("en", ""),
        }
        dictionary.append(item)
        word_index[word] = item
    print(f"[LOAD] Dictionary: {len(dictionary)} entries")
    return dictionary, word_index, list(word_index.keys())


def load_intent_keywords():
    raw = json.load(open(KW_PATH, "r", encoding="utf-8"))
    keywords = {
        intent: [kw.lower() for kw in kwlist if not kw.startswith("re:")]
        for intent, kwlist in raw.items()
    }
    print(f"[LOAD] Keywords: {list(keywords.keys())}")
    return keywords


def load_intent_classifier():
    data = pickle.load(open(PKL_PATH, "rb"))
    model = data.get("model") or data.get("lr") or data.get("clf")
    w2v   = data.get("w2v")
    size  = int(data.get("vector_size", getattr(w2v, "vector_size", 100)))
    tfidf = data.get("vectorizer") or data.get("tfidf")
    has_w2v   = w2v is not None
    has_tfidf = tfidf is not None
    print(f"[LOAD] Classifier: {type(model).__name__}  w2v={has_w2v}  tfidf={has_tfidf}")
    if not has_w2v:
        raise RuntimeError(
            "W2V model not found in PKL — pipeline requires word2vec embeddings. "
            "Retrain with w2v=True."
        )
    return model, w2v, size, tfidf  # tfidf kept for compat but unused


def load_llm():
    from llama_cpp import Llama
    if not os.path.exists(GGUF_PATH):
        raise FileNotFoundError(f"GGUF model not found: {GGUF_PATH}")
    llm = Llama(
        model_path   = GGUF_PATH,
        n_ctx        = 32768,
        n_gpu_layers = 0,
        n_threads    = max(1, (os.cpu_count() or 4) - 1),
        n_batch      = 1024,
        verbose      = False,
        use_mmap     = True,
        use_mlock    = False,
    )
    print(f"[LOAD] GGUF ready — model={GGUF_PATH}")
    return llm


print("[AWAL] Loading resources...")
DICTIONARY, WORD_INDEX, DICT_WORDS = load_dictionary()
INTENT_KEYWORDS                    = load_intent_keywords()
INTENT_MODEL, W2V_MODEL, W2V_SIZE, TFIDF_MODEL = load_intent_classifier()
NLP                                = spacy.load("xx_ent_wiki_sm")
LLM                                = load_llm()
print("[AWAL] Ready — backend=GGUF")


# ══════════════════════════════════════════════════════
# charger_configs — called by main.py lifespan
# FIX: was missing → AttributeError on startup
# ══════════════════════════════════════════════════════

def charger_configs():
    """
    Called by main.py lifespan at startup.
    All heavy resources are already loaded at module import time;
    this function exists so main.py can call pipline.charger_configs()
    without an AttributeError.
    It does a quick sanity check and returns a status dict.
    """
    status = {
        "dictionary_entries": len(DICTIONARY),
        "intent_keywords":    list(INTENT_KEYWORDS.keys()),
        "classifier":         type(INTENT_MODEL).__name__,
        "w2v":                W2V_MODEL is not None,
        "backend":            "GGUF",
        "spacy_model":        NLP.meta.get("name", "unknown"),
    }
    print(f"[CONFIG] charger_configs OK — {status}")
    return status


# ══════════════════════════════════════════════════════
# STEP 1 — NORMALISE
# Input:  raw string
# Output: normalised string (digraphs replaced)
# ══════════════════════════════════════════════════════

def normaliser(texte):
    result = texte.lower()
    for src, tgt in NORM_MAP.items():
        result = result.replace(src, tgt)
    return result

# ══════════════════════════════════════════════════════
# STEP 2 — TOKENISE
# Input:  string
# Output: list of lowercase tokens (no punct, no space)
# ══════════════════════════════════════════════════════

def tokeniser(texte):
    doc = NLP(texte)
    return [t.text.lower() for t in doc if not t.is_punct and not t.is_space]

# ══════════════════════════════════════════════════════
# STEP 3 — NER
# Input:  raw message string
# Output: comma-separated "word[LABEL]" string or "Aucune"
# ══════════════════════════════════════════════════════

def extractEntities(texte):
    doc = NLP(texte)
    results = []
    for ent in doc.ents:
        word    = ent.text.strip()
        word_lc = word.lower()

        if word_lc in WORD_INDEX:
            continue
        if word_lc in STOPWORDS or len(word) < 4:
            continue
        parts = word_lc.split()
        if all(p in WORD_INDEX or p in STOPWORDS or len(p) <= 2 for p in parts):
            continue
        if ent.label_ not in ("PERSON", "PER", "ORG", "GPE", "LOC"):
            continue

        label = "ORG" if ent.label_ == "ORG" else ("LOC" if ent.label_ in ("GPE", "LOC") else "PER")

        # enrich LOC with translation
        translation = ""
        if label == "LOC" and word_lc in WORD_INDEX:
            entry = WORD_INDEX[word_lc]
            translation = entry.get("fr", "") or entry.get("ar", "")

        tag = f"{word}[{label}]" + (f"={translation}" if translation else "")
        results.append(tag)

    return ", ".join(results) if results else "Aucune"

# ══════════════════════════════════════════════════════
# STEP 4 — INTENT DETECTION
# ══════════════════════════════════════════════════════

def _vectorize(text):
    """W2V-only vectorization (TF-IDF removed as requested)."""
    tokens = text.lower().split()
    vecs = [W2V_MODEL.wv[tok] for tok in tokens if tok in W2V_MODEL.wv]
    if not vecs:
        return np.zeros((1, W2V_SIZE))
    return np.mean(vecs, axis=0).reshape(1, -1)


def detecterIntent(message, tokens_norm):
    msg_lower = message.lower()

    # keyword scoring
    kw_scores = {
        intent: sum(1 for kw in kw_list if kw in msg_lower or kw in tokens_norm)
        for intent, kw_list in INTENT_KEYWORDS.items()
    }
    best_kw_intent = max(kw_scores, key=lambda k: kw_scores[k])
    best_kw_score  = kw_scores[best_kw_intent]

    # ML scoring (W2V only)
    ml_intent = "general"
    ml_conf   = 0.0
    if INTENT_MODEL is not None:
        X     = _vectorize(msg_lower)
        proba = INTENT_MODEL.predict_proba(X)[0]
        idx   = int(np.argmax(proba))
        ml_conf   = float(proba[idx])
        ml_intent = str(INTENT_MODEL.classes_[idx])

    # combined decision
    if best_kw_score >= 2:
        return best_kw_intent

    if best_kw_score == 1:
        if ml_intent == best_kw_intent:
            return best_kw_intent
        if ml_conf < 0.55:
            return best_kw_intent
        return ml_intent

    if ml_conf >= 0.45:
        return ml_intent

    if any(qw in msg_lower for qw in QUESTION_WORDS):
        return "question_info"

    return "general"

# ══════════════════════════════════════════════════════
# STEP 5 — RAG DICTIONARY LOOKUP
# ══════════════════════════════════════════════════════

def _format_entry(word, score):
    entry = WORD_INDEX[word]
    trans = []
    if entry["fr"]: trans.append(f"FR:{entry['fr']}")
    if entry["ar"]: trans.append(f"AR:{entry['ar']}")
    if entry["en"]: trans.append(f"EN:{entry['en']}")
    return f"{word}({score:.0f}%)={'|'.join(trans)}" if trans else None


def chercherRag(message, tokens_orig, intention, tokens_norm):
    if intention == "traduction":
        for tok in tokens_orig:
            if tok in TRANSLATION_SKIP or tok in STOPWORDS or len(tok) < 2:
                continue
            matches = fuzz_process.extract(tok, DICT_WORDS, scorer=fuzz.WRatio, limit=3, score_cutoff=82)
            for matched, score, _ in matches:
                result = _format_entry(matched, score)
                if result:
                    return result
        return "Aucun"

    # general lookup
    search_tokens = []
    seen_toks = set()
    for tok in tokens_norm + tokens_orig:
        if tok not in seen_toks and tok not in STOPWORDS and len(tok) >= 4:
            search_tokens.append(tok)
            seen_toks.add(tok)

    if not search_tokens:
        return "Aucun"

    results = []
    seen_matches = set()

    for token in search_tokens:
        threshold = 88 if len(token) <= 5 else 83
        matches = fuzz_process.extract(token, DICT_WORDS, scorer=fuzz.WRatio, limit=3, score_cutoff=threshold)
        for matched, score, _ in matches:
            if matched in STOPWORDS or matched in seen_matches:
                continue
            if len(matched) <= 2 and len(token) >= 4:
                continue
            seen_matches.add(matched)
            result = _format_entry(matched, score)
            if result:
                results.append(result)

    return "\n".join(results[:5]) if results else "Aucun"

# ══════════════════════════════════════════════════════
# STEP 6 — BUILD SYSTEM PROMPT
# ══════════════════════════════════════════════════════

def buildSystem(intention, rag, entities, history):
    return SYSTEM_PROMPT.format(
        intent      = intention,
        instruction = INSTRUCTION_MAP.get(intention, INSTRUCTION_MAP["general"]),
        rag         = rag,
        entities    = entities,
        history     = str(history[-4:]) if history else "[]",
    )

# ══════════════════════════════════════════════════════
# STEP 7 — GENERATE
# ══════════════════════════════════════════════════════

def generate_resp(system_prompt, user_message, history):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-4:]:
        # Support both {user/assistant} format (internal) and {role/content} format (from main.py)
        if turn.get("role") and turn.get("content"):
            messages.append({"role": turn["role"], "content": str(turn["content"])})
        else:
            if turn.get("user"):
                messages.append({"role": "user",      "content": str(turn["user"])})
            if turn.get("assistant"):
                messages.append({"role": "assistant", "content": str(turn["assistant"])})
    messages.append({"role": "user", "content": user_message})

    t0 = time.time()
    print("[GEN] GGUF prompt:")
    print("[GEN] SYSTEM PROMPT START")
    print(system_prompt)
    print("[GEN] SYSTEM PROMPT END")
    print("[GEN] USER MESSAGE START")
    print(user_message)
    print("[GEN] USER MESSAGE END")
    res = LLM.create_chat_completion(
        messages       = messages,  # type: ignore
        temperature    = 0.3,
        top_p          = 0.9,
        repeat_penalty = 1.15,
        max_tokens     = 512,
        stop           = STOP_TOKENS,
        stream         = False,
    )
    content = (res["choices"][0]["message"]["content"] or "").strip()  # type: ignore
    print(f"[GEN] GGUF {time.time()-t0:.1f}s")
    return content or "latin: Ur d-yelli usune.\nar: لا توجد إجابة."

# ══════════════════════════════════════════════════════
# STEP 8 — PARSE RESPONSE
# ══════════════════════════════════════════════════════

RE_LATIN = re.compile(r"latin\s*:\s*(.+?)(?=\nar\s*:|$)", re.DOTALL | re.IGNORECASE)
RE_AR    = re.compile(r"\bar\s*:\s*(.+?)$",               re.DOTALL | re.IGNORECASE)


def parse_response(raw):
    latin_match = RE_LATIN.search(raw)
    ar_match    = RE_AR.search(raw)

    latin = latin_match.group(1).strip() if latin_match else raw.strip()
    ar    = ar_match.group(1).strip()    if ar_match    else ""

    for src, tgt in ARABIZI_MAP.items():
        latin = latin.replace(src, tgt)

    return latin, ar

# ══════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════

def pipeline(message, history=None):
    hist = history or []

    normalised  = normaliser(message)
    tokens_orig = tokeniser(message)
    tokens_norm = tokeniser(normalised)
    print(f"[1] tokens: {tokens_orig}")

    entities = extractEntities(message)
    print(f"[2] entities: {entities}")

    intention = detecterIntent(message, tokens_norm)
    print(f"[3] intent: {intention}")

    rag = chercherRag(message, tokens_orig, intention, tokens_norm)
    print(f"[4] rag: {rag[:80]}")

    system_prompt = buildSystem(intention, rag, entities, hist)

    raw_answer = generate_resp(system_prompt, message, hist)

    latin_out, arabic_out = parse_response(raw_answer)
    print(f"[7] latin: {latin_out[:80]}")
    print(f"[7] ar:    {arabic_out[:60]}")

    return {
        "reponse":   f"{latin_out}\n{arabic_out}",
        "intention": intention,
        "entities":  entities,
        "rag_hits":  rag,
        "raw_llm":   raw_answer[:300],
    }


# ══════════════════════════════════════════════════════
# ASYNC GENERATOR — required by main.py
# main.py calls:  async for event in pipline.traiter_message(msg, historique=hist)
# Events:
#   {"type": "step",  "step": int,    "label": str}
#   {"type": "final", "reponse": str, "intention": str,
#    "confiance": float, "langue": str,
#    "temps_rag_ms": int, "temps_llm_ms": int}
# ══════════════════════════════════════════════════════

import asyncio

async def traiter_message(message: str, historique: list | None = None):
    """
    Async generator wrapping the sync pipeline() for use with FastAPI SSE.
    Runs the blocking pipeline in a thread executor so the event loop is never blocked.
    """
    hist = historique or []

    # Step 1 — normalisation + tokenisation
    yield {"type": "step", "step": 1, "label": "Normalisation & Tokenisation"}
    await asyncio.sleep(0)

    # Step 2 — NER
    yield {"type": "step", "step": 2, "label": "Détection d'entités (NER)"}
    await asyncio.sleep(0)

    # Step 3 — intent detection
    yield {"type": "step", "step": 3, "label": "Détection d'intention"}
    await asyncio.sleep(0)

    # Step 4 — RAG lookup
    yield {"type": "step", "step": 4, "label": "Recherche RAG dictionnaire"}
    await asyncio.sleep(0)

    # Step 5 — LLM generation (blocking — run in thread)
    yield {"type": "step", "step": 5, "label": "Génération LLM"}
    await asyncio.sleep(0)

    t0 = time.time()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline, message, hist)
    temps_llm_ms = int((time.time() - t0) * 1000)

    yield {
        "type":        "final",
        "reponse":     result["reponse"],
        "intention":   result["intention"],
        "confiance":   0.85,          # pipeline() does not expose proba — fixed value
        "langue":      "tamazight",
        "temps_rag_ms": 0,            # merged into pipeline(); not split
        "temps_llm_ms": temps_llm_ms,
    }


if __name__ == "__main__":
    for msg in ["Azul, amek tellid?", "righ adisingh maydyan l3asima n lmghrib", "Traduis pain s tmazight"]:
        result = pipeline(msg)
        print(result["reponse"])
        print()