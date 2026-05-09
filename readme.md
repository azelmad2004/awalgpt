┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🧭 PIPELINE AWAL GPT — ARCHITECTURE OPÉRATIONNELLE DÉTAILLÉE (4 FICHIERS)                             │
│  Contraintes immuables : dict-local (hyphens) • Zéro hardcode • RapidFuzz UNIQUEMENT • Zéro FAISS/TF-IDF│
│  Commentaires français obligatoires • Logging structuré • Config inline • LLM classifieur intégré       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  🔑 RÈGLES GLOBALES (ENFORCÉES À LA COMPILATION/EXÉCUTION)                                             │
│  • Nommage strict : dict-local["clef"] uniquement • Tirets autorisés • Zéro underscore dans les clés   │
│  • Données : Tout chargé depuis data/*.csv et data/*.json • Zéro liste/matrice codée en dur            │
│  • RAG : rapidfuzz.fuzz.WRatio + process.extract • Seuils dynamiques • Cache LRU avec TTL              │
│  • Interdits absolus : sentence_transformers, faiss, scipy, numpy (sauf LR), variables globales        │
│  • LLM : Groq AsyncGroq • llama-3.3-70b-versatile (génération) • llama-3.1-8b-instant (classifieur)   │
│  • Validation : Ratio ≥50% mots CSV • Retry temp=0.0 • Fallback dictionnaire si échec critique         │
│  • Arabizi : Mapping bidirectionnel depuis arabizi_map.json • Appliqué UNIQUEMENT sur champ "tamazight"│
│  • Logging : Préfixes obligatoires [INIT] [PREPROC] [INTENT] [LLM-CLASS] [RAG] [LLM] [VALIDATION]     │
│  [POST] [API] [TRAIN] • Niveaux INFO/DEBUG/ERROR • Métriques temps/mémoire/cache intégrées            │
─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  📁 FICHIER 1 : core.py — MOTEUR DE DONNÉES & RAPIDFUZZ (ALGORITHMIQUE)                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ INITIALISATION & FUSION CSV                                                                    │   │
│  │ • dict-local["config"] ← {max-tokens-llm:None, temp-min:0.0, temp-max:unlimited, timeout-groq:60,   │   │
│  │   score-cutoff-rapidfuzz:70, top-k-rag:3, max-variantes-ortho:3, seuil-protection-mot:0.65,   │   │
│  │   max-tentatives-llm:5, cache-max-size:500, cache-ttl-sec:300}                                │   │
│  │ • charger-donnees-fusionnees() → Lit dictionnaire.csv, darija.csv, tamazight-latin.csv        │   │
│  │   → Normalise colonnes en {mot, fr, ar, en, source} • Dé-duplique par mot.lower()           │   │
│  │   → Charge dans dict-local["dict-mem"] (liste de dict) • Indexe par mot pour accès O(1)     │   │
│  │ • dict-local["fuzz-cache"] → OrderedDict() • TTL 300s • Évince LRU si > cache-max-size      │   │
│  │ • Logging: [INIT] Fusion:3 CSV • Entrées:12k • Cache:LRU(500,TTL:300s) • Prêt              │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🔧 normaliser(texte: str) → str                                                                 │   │
│  │ • Docstring FR: """Applique le nettoyage phonétique et structurel pour le ML."""              │   │
│  │ • Étapes: 1. Suppression HTML/emojis/ponctuation excessive • 2. Section "normalisation" de    │   │
│  │   arabizi-map.json (multi-char → mono-char) • 3. Lowercase • 4. Regex [^a-z0-9\u0600-\u06FF]│   │
│  │   • 5. Réduction \s+ → " " • Retourne texte nettoyé                                           │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🔍 rechercher-rapidfuzz(mot: str, top-k: int = 3) → list                                      │   │
│  │ • Algorithme de variantes (max 5):                                                            │   │
│  │   1. mot-original (casing préservé) • 2. mot-normalisé (lower+arabizi)                       │   │
│  │   3. Suppression 1 char (indices 0, -1, milieu) • 4. Swap adjacent (i, i+1)                  │   │
│  │   5. Substitution phonétique (ɣ↔4, x↔5, ɛ↔3, ḥ↔7)                                          │   │
│  │ • Cache check: clé = f"{mot}:{hash(variantes)}" • Si hit → retour immédiat                  │   │
│  │ • RapidFuzz: process.extract(query=v, choices=[e["mot"] for e in dict-local["dict-mem"]],   │   │
│  │   scorer=fuzz.WRatio, limit=top-k, score_cutoff=dict-local["config"]["score-cutoff"])       │   │
│  │ • Post-traitement: Dé-duplique par mot normalisé • Trie décroissant • Extrait {mot, score,  │   │
│  │   traductions:{fr,ar,en}} • Stocke dans cache • Retourne TOP-K                              │   │
│  │ • Logging: [RAG] Mot:X • Variantes:Y • Cache:H/M • Matches:Z • MaxScore:S • Temps:T ms     │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ ✅ valider-vocabulaire(reponse-tam: str) → bool                                                 │   │
│  │ • Tokenisation: re.findall(r"\b\w{3,}\b", reponse-tam.lower()) • Filtre stopwords            │   │
│  │ • Matching: Pour chaque mot, fuzz.partial_ratio(mot, choix_dict_mem) ≥ 65                    │   │
│  │ • Calcul: ratio = mots_trouvés / mots_totaux • Seuil: ≥0.50 → True • Sinon → False (retry)  │   │
│  │ • Docstring FR: """Vérifie la cohérence lexicale de la réponse LLM contre le corpus source."""│   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  appliquer-arabizi-sortie(texte: str) → str                                                   │   │
│  │ • Ordre critique: 1. outputMulti ("gh"→"4") • 2. output ("ɣ"→"4") • 3. Reverse si nécessaire │   │
│  │ • Regex re.sub(re.escape(src), cible, texte) pour éviter partial match • Retourne texte      │   │
│  │   converti • Logging: [POST] Arabizi:N transfo • Champ:tamazight • OK                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  📁 FICHIER 2 : brain.py — ORCHESTRATION & LLM (CHAÎNE DE TRAITEMENT)                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────   │
│  │ ⚡ INITIALISATION                                                                                    │   │
│  │ • charger-configuration-intentions() → Charge 4 JSON • Compile regex dynamiques pour chaque       │   │
│  │   intention • Client Groq Async initialisé • LR model chargé via core • Logging: [INIT] Config    │   │
│  │   intentions:4 JSON • Regex:12 • LR:8 classes • Groq:Connecté • Ready                            │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🔤 pretraiter-entree(message: str) → dict                                                         │   │
│  │ • Conservation verbatim: dict-local["original-question"] • Détection run-on: len>12 ∧ ∧ espace   │   │
│  │   → Appel 8B-instant: prompt=["Split '{message}' for NLP. Return JSON array."] • Timeout 5s     │   │
│  │ • Extraction blocs: [MATH] \d+[x-z]|[0-9+\-*/=()²³√]+ • [URL] • [CODE] • Détection script:     │   │
│  │   comptage Unicode ranges → latin/arabe/mixte • Nettoyage ML: lowercase, emojis, ponctuation    │   │
│  │   excessive • Retour: {texte-original, texte-pour-ml, blocs-speciaux, tokens-analyse, type-script}│  │
│  │ • Logging: [PREPROC] Script:X • Run-on:Y • Tokens:Z • Blocs:{} • Original:Préservé           │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🎯 detecteur-intention(texte: str, langue: str) → dict                                          │   │
│  │ • Cascade 4 niveaux:                                                                             │   │
│  │   N0: Exact match dans mots-sociaux.json → {intent, conf:0.99, method:"exact"}                 │   │
│  │   N1: LR predict_proba → seuil adaptatif (arabe/darija:0.75, tamazight/fr:0.55, mixte:0.65)   │   │
│  │   N2: Exact match mots-cles-intentions.json → conf:0.90                                        │   │
│  │   N3: Regex pondérées → score=0.75+(hits×0.05) max 0.95 • N4: Fallback "general" conf:0.50    │   │
│  │   • Ultra-court (≤2 tokens ∧ pas social) → intent:"incompris" • conf:0.0 • Retour immédiat    │   │
│  │ • Logging: [INTENT] Intent:X • Conf:Y • Method:Z • Seuil:S • Validé:OK                       │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🤖 classifieur-tokens-llm(tokens: list, contexte: dict) → dict                                  │   │
│  │ • Modèle: llama-3.1-8b-instant • Timeout:5s • Temp:0.0 • Format:json_object                   │   │
│  │ • Prompt système:                                                                                │   │
│  │   "Classifie CHAQUE token: part(TAM_LATIN/NON_TAM) • confiance(HIGH≥90%/LOW) • HIGH→details   │   │
│  │   direct • LOW→details=null→éligible RAG • Règles: particules courtes→TAM HIGH • ALL CAPS 2-5 │   │
│  │   chars→NON_TAM HIGH • villes→NON_TAM HIGH • français/anglais→NON_TAM HIGH • inconnu tamazight│   │
│  │   →TAM LOW details=null"                                                                        │   │
│  │ • Parsing JSON: try/except JSONDecodeError/Timeout → Fallback: tout TAM_LATIN LOW            │   │
│  │ • Routage post-classif:                                                                          │   │
│  │   ────────────────────────────────────────────────────┐                                        │   │
│  │   │ HIGH+TAM_LATIN → utilisation directe (pas RAG)   │                                        │   │
│  │   │ HIGH+NON_TAM   → utilisation directe (pas RAG)   │                                        │   │
│  │   │ LOW+TAM_LATIN  → éligible RapidFuzz              │                                        │   │
│  │   │ LOW+NON_TAM    → ignoré                          │                                        │   │
│  │   │ entity_group   → entité unique dans prompt       │                                        │   │
│  │   └────────────────────────────────────────────────────┘                                        │   │
│  │ • Stocke dans dict-local["classification-tokens"] • Logging: [LLM-CLASS] Tokens:N • HIGH:X    │   │
│  │   • LOW:Y • Entities:Z • Temps:T ms • Modèle:8B • Fallback:0                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 📦 construire-prompt(contexte: dict, intention: str, rag: dict) → tuple                         │   │
│  │ • Assemblage blocs:                                                                             │   │
│  │   [SYSTEM] "Expert Tamazight. Réponds JSON {tamazight:latin, arabe:arabe} • ≥3 mots/champ •   │   │
│  │   NE PAS répéter question • Si RAG vide → connaissance générale"                              │   │
│  │   [ENTITY_DETAILS] Détails HIGH_CONF tokens (EST→École, khenifra→ville) • Format: "mot→détail"│   │
│  │   [TRANSLATIONS] Résultats RapidFuzz TOP-3 par mot LOW_CONF • Format: "mot→fr:X|ar:Y|en:Z(S)"│   │
│  │   [ORIGINAL_QUESTION] Question verbatim (casing préservé)                                     │   │
│  │   [INSTRUCTION] "Répondre directement. Phrases complètes. tamazight:latin only. arabe:arabic" │   │
│  │ • Retour: (systeme, prompt) • Logging: [PROMPT] Blocs:5 • Lignes:N • Entités:M • Tokens:K   │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────   │
│  │ 🤖 generer-avec-retry(prompt: str, systeme: str) → tuple                                        │   │
│  │ • Loop: tentative ∈ [1, dict-local["config"]["max-tentatives-llm"]]                           │   │
│  │   T1: 70B-versatile • temp=0.2-0.4 selon intention • max_tokens:illimité • timeout:120s      │   │
│  │   T2+: 8B-instant • temp=0.0 • prompt enrichi: "Erreur précédente: {erreur}. Corrige JSON." │   │
│  │ • Validation JSON immédiate: clés "tamazight","arabe" présentes • Type string • Non vide     │   │
│  │ • Retour: (donnees, tentative, temps-llm-ms) • Logging: [LLM] Modèle:X • Temp:Y • Tent:Z    │   │
│  │   • Temps:T ms • Status:OK/KO • Cache:K • Fallback:0                                        │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 🛡️ valider-et-nettoyer(donnees: dict, question: str, tentative: int) → tuple                  │   │
│  │ • Layer 1: Suppression écho → re.split phrases → ratio tokens commun >0.85 → skip première  │   │
│  │ • Layer 2: Validation vocabulaire CSV via rapidfuzz (≥50% mots trouvés) • Échec → retry     │   │
│  │ • Layer 3: Format JSON • tamazight:latin only (regex ^[a-z0-9345679\s.,!?;:-]+$) • arabe:   │   │
│  │   arabe only (regex [\u0600-\u06FF]) • ≥3 mots par champ                                    │   │
│  │ • Layer 4: Arabizi bidirectionnel sur champ tamazight • Nettoyage: \s+→" " • CJK supprimé   │   │
│  │ • Retour: (tam-propre, ar-propre) • Logging: [VALIDATION] Ratio:X% • Arabizi:Y • Format:OK │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  traiter-message(message: str, historique: list = None) → dict                                │   │
│  │ • Flux orchestré: Input → Preproc → Intent → LLM-Class → RAG → Prompt → LLM → Validation →  │   │
│  │   Arabizi → Output                                                                            │   │
│  │ • Circuit court social: si intent∈{salutation,remerciement,au_revoir} ∧ conf>0.90 ∧ msg court│   │
│  │   → réponse directe depuis reponses-sociales.json (pas LLM/RAG) • Retour immédiat           │   │
│  │ • Retour: {reponse, intention, confiance, metrics:{tokens, rag-matches, vocab-ratio,         │   │
│  │   llm-attempts, temps-total, cache-hits}} • Logging: [BRAIN] Intent:X • Temps:Y • OK       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  📁 FICHIER 3 : main.py — SERVEUR FASTAPI (7 ENDPOINTS + SSE + SQLITE)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ ⚡ INITIALISATION SERVEUR                                                                         │   │
│  │ • Config inline: secret-jwt, chemin-bd, domaines-valides • Middleware CORS • Lifespan: init   │   │
│  │   DB async • Charge brain.detecteur • Charge core.dict-local • Logging: [API] Serveur:OK    │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  AUTH (3 ENDPOINTS)                                                                               │   │
│  │ • POST /auth/register: Validation email/password/username • Hash SHA-256 email→uid • Hash     │   │
│  │   bcrypt password • Insert SQLite • Retour token JWT (30j) • Logging: [AUTH] Register:OK    │   │
│  │ • POST /auth/login: Recherche uid • Vérification bcrypt • Retour token JWT • Logging: [AUTH] │   │
│  │   Login:OK                                                                                      │   │
│  │ • GET /auth/me: Dépendance JWT • Retour profil utilisateur • Logging: [AUTH] Me:OK           │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 💬 POST /chat/stream (SSE)                                                                        │   │
│  │ • Auth JWT requise • Corps:{message, conversation-id, domaine} • Récupération historique(10)  │   │
│  │ • Appel brain.traiter-message • StreamingResponse générateur:                                 │   │
│  │   event:step → {"step":"intent","label":"Intention: X"} • sleep(0.2)                          │   │
│  │   event:step → {"step":"rag","label":"Recherche corpus..."} • sleep(0.2)                      │   │
│  │   event:step → {"step":"llm","label":"Génération Llama-3..."}                                 │   │
│  │   event:token → {"reponse":"[TAM]...\n[AR]..."}                                               │   │
│  │   event:done → {"intention":"X","confiance":Y,"temps-rag":Z,"temps-llm":W,"temps-total":V}    │   │
│  │ • Circuit court social: si intent social ∧ pas LLM → token direct sans étapes                │   │
│  │ • Sauvegarde échange SQLite (utilisateurs, messages, cache) • Logging: [API] Stream:OK      │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ 📂 CONVERSATIONS (3 ENDPOINTS)                                                                    │   │
│  │ • GET /conversations: Liste conversations utilisateur (GROUP BY + ORDER BY date DESC) • Retour│   │
│  │   {id, title, updated_at} • Logging: [API] ListConv:OK                                        │   │
│  │ • GET /conversations/{cid}/messages: Historique complet (ORDER BY id ASC) • Alternance user/ │   │
│  │   bot • Retour {sender, content, date} • Logging: [API] GetMsg:OK                            │   │
│  │ • DELETE /conversations/{cid}: Suppression messages conversation • Retour {status:ok} •      │   │
│  │   Logging: [API] DelConv:OK                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  📁 FICHIER 4 : train.py — ENTRAÎNEMENT LR (INTENT CLASSIFIER)                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────   │
│  │ ⚡ INITIALISATION & CHARGEMENT CORPUS                                                                 │   │
│  │ • Charger-corpus() → Lit dataset.csv,• Normalise colonnes •    │   │
│  │   Filtre: textes ≥2 mots • ≥45% alphabétiques • Pas spam/répétition • Équilibre classes       │   │
│  │   • Logging: [TRAIN-LOAD] Données:N • Classes:M • Prêt                                        │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  augmenter-donnees(textes, etiquettes, dico) → tuple                                              │   │
│  │ • Stratégies: 1. Remplacement synonymes via dictionnaire.csv • 2. Swap mots adjacents • 3.     │   │
│  │   Inversion ordre (phrases courtes) • 4. Capitalisation aléatoire • 5. Split stratifié (test=0.2)││
│  │ • Logging: [TRAIN-AUGM] Exemples:N • Augmentés:M • Split:Train:X/Test:Y                      │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │  entrainer-modele(train-x, train-y) → model                                                       │   │
│ • Modèle: LogisticRegression(solver="saga", max_iter=1000,              │   │
│  │   class_weight="balanced", C=3.0) • Évaluation: accuracy, f1-macro, matrice confusion        │   │
│  │   • Validation croisée: 5-fold • Sauvegarde: models/intent-classifier.pkl • Logging: [TRAIN- │   │
│  │   MODEL] Acc:X% • F1:Y • OK   
sauvgarde confusionmarice image avec matplotlib                                                                │   │
│  ├─────────────────────────────────────────────────────────────────────────────────────────────────┤   │
│  │ ✅ verifier-modele() → bool                                                                       │   │
│  │ • Chargement pickle •  │   │
│  │   • Calcul seuils adaptatifs par langue depuis distribution probabilités • Mise à jour       │   │
│  │   dict-local["config"]["seuils-langue"] • Logging: [TRAIN-VERIF] Stable • Seuils:OK • Prêt  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  🔄 FLUX DE DONNÉES GLOBAL (EXEMPLE CONCRET + GESTION D'ERREURS)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────   │
│  │ Question: "maydiyan les filiers n EST khenifra?"                                                │   │
│  │ [PREPROC] tokens:["maydiyan","les","filiers","n","est","khenifra"] • Script:mixte • Run-on:F  │   │
│  │ [INTENT] LR→"question_info" • Conf:0.82 • Seuil:0.55 → Validé                                │   │
│  │ [LLM-CLASS] 8B→maydiyan:TAM/LOW • les:NON/HIGH • EST:NON/HIGH • n:TAM/HIGH • khenifra:NON/HIGH│ │
│  │ [RAG] RapidFuzz→maydiyan→TOP-3:[maydiyan(95), maydiyen(88), maydyan(82)] • Boost:OK         │   │
│  │ [PROMPT] Assemblage→[SYSTEM]+[ENTITY]+[TRANS]+[ORIGINAL]+[INSTRUCTION]                       │   │
│  │ [LLM] 70B→temp:0.3 • T1:OK • JSON:{"tamazight":"...","arabe":"..."}                          │   │
│  │ [VALIDATION] Ratio:66%≥50% → OK • Écho:Supprimé • Arabizi:"taɣrmt"→"ta4rmt" • Format:OK     │   │
│  │ [SORTIE] [TAM] filiers n EST Khenifra: Informatique, 5alqat al-3lm, Gestion...              │   │
│  │         [AR] شعب EST خنيفرة: المعلوماتية، هندسة العلوم، التدبير...                         │   │
│  │ [METRICS] {tokens:6, intent:question_info, conf:0.82, rag:3, vocab:0.66, llm:1, total:912ms}│   │
│  │                                                                                                 │   │
│  │ 🛡️ GESTION D'ERREURS & FALLBACK:                                                                │   │
│  │ • Timeout Groq: Retry avec 8B-instant • temp=0.0 • Prompt enrichi erreur précédente          │   │
│  │ • JSON invalide: 2 tentatives max • Sinon → Fallback dictionnaire: "{mot} → {fr} | {ar}"    │   │
│  │ • RAG vide: Utilise ENTITY_DETAILS + connaissance générale • NE JAMAIS dire "Je ne sais pas" │   │
│  │ • Validation échouée: Retry temp=0.0 • Si échec critique → Réponse sécurisée: "Ur fhim ara" │   │
│  │ • Cache LRU: Évince entrées TTL>300s ou >500 éléments • Logging: [CACHE] Eviction:N • Hit:K │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                          │
│  📊 MATRICE LOGGING & VALIDATION (EXPANDUE)                                                              │
│  ┌──────────────┬──────────┬─────────────────────────────────────────────────────────────────────────┐ │
│  │ Préfixe      │ Niveau   │ Contenu obligatoire + Métriques                                         │ │
│  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────┤ │
│  │ [INIT]       │ INFO     │ Config:dict-local • CSV:3 fusionnés • Cache:LRU(500) • Modèles:Ready  │ │
│  │ [PREPROC]    │ INFO     │ Script:type • Run-on:bool • Tokens:list • Blocs:dict • Original:str   │ │
│  │ [INTENT]     │ INFO     │ Intent:str • Conf:float • Method:str • Seuil:float • Validé:bool      │ │
│  │ [LLM-CLASS]  │ INFO     │ Tokens:int • HIGH:int • LOW:int • Entities:list • Temps:ms • Fallback │ │
│  │ [RAG]        │ INFO     │ Mot:str • Variantes:list • Cache:hit/miss • Matches:list • MaxScore   │ │
│  │ [LLM]        │ INFO/WARN│ Modèle:str • Temp:float • Tentative:int • Temps:ms • Status:OK/KO     │ │
│  │ [VALIDATION] │ INFO/ERR │ Ratio:float • Echo:bool • Format:bool • Arabizi:int • Retry:bool      │ │
│  │ [POST]       │ INFO     │ Nettoyage:bool • Longueur:int • Prêt:bool • Format:str                │ │
│  │ [API]        │ INFO/ERR │ Endpoint:str • Auth:bool • SSE:events • BD:rows • Temps:ms • Erreur   │ │
│  │ [TRAIN-*]    │ INFO     │ Données:int • Augmentés:int • Split:train/test • Acc:float • F1:float │ │
│  └──────────────┴──────────┴─────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                          │
│  🔒 CONTRADICTIONS UPLOADÉES CORRIGÉES & ENFORCÉES                                                     │
│  • ❌ Supprimé: FAISS, TF-IDF, embeddings, sentence_transformers, scipy, variables globales modifiables│
│  • ✅ Imposé: RapidFuzz.WRatio UNIQUEMENT • 3 CSV fusionnés en mémoire • dict-local strict            │
│  • ✅ Nommage: dict-local["clef"] avec tirets • Zéro underscore • Commentaires français obligatoires  │
│  • ✅ LLM: Classifieur 8B-instant + Générateur 70B-versatile • Validation boucle CSV • Arabizi bidir  │
│  • ✅ Config inline • 7 endpoints FastAPI • SSE streaming • SQLite JWT • Fonctions minimisées         │
│  • ✅ Logging structuré • Métriques temps/mémoire/cache • Exemples réalistes • Gestion erreurs robuste │
│                                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘


 in train only dataset.csv remove the fuck tFIDF and vectorisatuon and enhnace les prompts more and more specifier more and more and use also my json files and apply the same pipline and remove fonctions and od logic are not sted in pipline yet like def obtenir_proverbe_pour_requete(message: str)   be smart and refractor with new logic new fonctions news things files just a help only 

try to be more optimized and minimalist and do not  use tr excpect andremove all otional things remove also the masseive verification are you do in every steep remove it also remove the massive logging  belive are imports are files are so good and just do the essential code 