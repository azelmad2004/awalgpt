# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# main.py — Serveur FastAPI AWAL GPT
# Rôle : Authentification JWT, streaming SSE, gestion conversations SQLite
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
import logging
# ── Logger Configuration (Must be before other imports) ──────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
for _lib in ("sentence_transformers", "httpx", "faiss", "transformers", "urllib3", "asyncio", "aiosqlite", "uvicorn", "rapidfuzz"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
log = logging.getLogger("awalgpt")

import os
from dotenv import load_dotenv
load_dotenv()
import time
import json
import hashlib
import asyncio
import sys
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import aiosqlite
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import brain

# ── Configuration ──────────────────────────────────────────────────────────
CLE_GROQ   = os.getenv("GROQ_API_KEY", "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")
SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_secret_key_2026_long_enough")
CHEMIN_BD  = os.getenv("DB_PATH", "awalgpt.db")

DOMAINES_VALIDES = {"health", "economy", "education", "culture", "tech", "daily"}


# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# BASE DE DONNÉES SQLite (aiosqlite)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

async def initialiser_base() -> None:
    """Crée toutes les tables SQLite au premier démarrage."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        # Optimisations SQLite
        await bd.execute("PRAGMA journal_mode=WAL")
        await bd.execute("PRAGMA synchronous=NORMAL")
        await bd.execute("PRAGMA cache_size=10000")
        
        await bd.executescript("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id TEXT PRIMARY KEY, nom TEXT, email TEXT UNIQUE,
                mot_de_passe TEXT, variete TEXT, cree_le TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utilisateur_id TEXT, conversation_id TEXT,
                message_utilisateur TEXT, message_bot TEXT,
                domaine TEXT, intention TEXT, langue TEXT,
                temps_ms INTEGER, horodatage TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(utilisateur_id);
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
            CREATE TABLE IF NOT EXISTS cache (
                cle TEXT PRIMARY KEY, reponse TEXT, horodatage TEXT
            );
        """)
        await bd.commit()
    log.info("[DB] Base initialisée")


async def bd_trouver_utilisateur(uid: str) -> dict:
    """Recherche un utilisateur par son identifiant SHA-256."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE id = ?", (uid,))
        ligne = await cur.fetchone()
        return dict(ligne) if ligne else None


async def bd_creer_utilisateur(
    uid: str, nom: str, email: str, hache: str, variete: str = None
) -> None:
    """Insère un nouvel utilisateur dans la base."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "INSERT INTO utilisateurs VALUES (?,?,?,?,?,?)",
            (uid, nom, email, hache, variete, datetime.now().isoformat())
        )
        await bd.commit()


async def bd_historique(uid: str, conv_id: str) -> list:
    """Récupère les 10 derniers échanges d'une conversation (format alternant user/assistant)."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT message_utilisateur, message_bot FROM messages "
            "WHERE utilisateur_id=? AND conversation_id=? "
            "ORDER BY id DESC LIMIT 10",
            (uid, conv_id)
        )
        lignes = await cur.fetchall()
    hist = []
    for ligne in reversed(lignes):
        if ligne[0]:
            hist.append({"role": "user",      "content": ligne[0]})
        if ligne[1]:
            hist.append({"role": "assistant", "content": ligne[1]})
    return hist


async def bd_inserer_message(
    uid: str, cid: str, msg_user: str, msg_bot: str,
    domaine: str, intention: str, langue: str, temps_ms: int
) -> None:
    """Sauvegarde un échange utilisateur/bot dans la base."""
    log.debug(f"[DB] Inserer message: CID={cid} | Intent={intention}")
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "INSERT INTO messages "
            "(utilisateur_id, conversation_id, message_utilisateur, message_bot, "
            "domaine, intention, langue, temps_ms, horodatage) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uid, cid, msg_user, msg_bot, domaine, intention, langue,
             temps_ms, datetime.now().isoformat())
        )
        await bd.commit()


async def bd_lister_conversations(uid: str) -> list:
    """Liste toutes les conversations d'un utilisateur (ordonnées par date desc)."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT conversation_id, MAX(horodatage) as d, message_utilisateur as t, COUNT(*) as nb "
            "FROM messages WHERE utilisateur_id=? "
            "GROUP BY conversation_id ORDER BY d DESC",
            (uid,)
        )
        return [
            {"id": l[0], "title": (l[2] or "Nouvelle discussion")[:30], "updated_at": l[1], "count": l[3]}
            for l in await cur.fetchall()
        ]


# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# AUTHENTIFICATION JWT
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

def creer_token(uid: str) -> str:
    """Génère un token JWT valable 30 jours."""
    return jwt.encode(
        {"sub": uid, "exp": datetime.now(timezone.utc) + timedelta(days=30)},
        SECRET_JWT,
        algorithm="HS256"
    )


async def obtenir_utilisateur(requete: Request) -> dict:
    """
    Dépendance FastAPI : valide le token JWT et retourne l'utilisateur.
    Lève HTTPException 401 si le token est absent, invalide ou expiré.
    """
    token = requete.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(401, "Token manquant")
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=["HS256"])
        utilisateur = await bd_trouver_utilisateur(payload["sub"])
        if utilisateur:
            return utilisateur
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expiré")
    except Exception:
        pass
    raise HTTPException(401, "Token invalide")


# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# APPLICATION FASTAPI
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage, nettoyage à l'arrêt."""
    log.info("[STARTUP] Initialisation Awal GPT...")
    await initialiser_base()
    
    # Initialisation des composants Brain/Core
    # Note: brain.core.charger_donnees_fusionnees() est déjà appelé lors de l'import
    brain.charger_configs()
    
    log.info("[STARTUP] Awal GPT prêt.")
    yield
    log.info("[SHUTDOWN] Awal GPT arrêté.")


app = FastAPI(title="Awal GPT — API Tamazight", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.get("/")
async def root():
    return {"name": "Awal GPT API", "status": "online", "version": "3.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# ── Endpoints d'authentification ──────────────────────────────────────────

@app.post("/auth/register")
async def inscrire(req: Request):
    """Crée un nouveau compte utilisateur. Valide email + password + username."""
    try:
        corps = await req.json()
    except Exception:
        raise HTTPException(400, "Corps JSON invalide")

    email = corps.get("email", "").lower().strip()
    nom   = corps.get("username", "").strip()
    mdp   = corps.get("password", "")

    if not email or not mdp or not nom:
        raise HTTPException(400, "email, password et username requis")
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "Format d'email invalide")
    if len(mdp) < 6:
        raise HTTPException(400, "Mot de passe trop court (minimum 6 caractères)")

    uid = hashlib.sha256(email.encode()).hexdigest()
    if await bd_trouver_utilisateur(uid):
        raise HTTPException(400, "Email déjà utilisé")

    hache = bcrypt.hashpw(mdp.encode(), bcrypt.gensalt()).decode()
    await bd_creer_utilisateur(uid, nom, email, hache, corps.get("preferred_variety"))
    log.info(f"[AUTH] Nouvel utilisateur : {nom}")
    return {"token": creer_token(uid), "user": {"id": uid, "username": nom}}


@app.post("/auth/login")
async def connexion(req: Request):
    """Authentifie un utilisateur et retourne un token JWT."""
    try:
        corps = await req.json()
    except Exception:
        raise HTTPException(400, "Corps JSON invalide")

    email = corps.get("email", "").lower().strip()
    mdp   = corps.get("password", "")
    uid   = hashlib.sha256(email.encode()).hexdigest()

    utilisateur = await bd_trouver_utilisateur(uid)
    if utilisateur and bcrypt.checkpw(mdp.encode(), utilisateur["mot_de_passe"].encode()):
        log.info(f"[AUTH] Connexion : {utilisateur['nom']}")
        return {
            "token": creer_token(utilisateur["id"]),
            "user":  {"id": utilisateur["id"], "username": utilisateur["nom"]},
        }
    raise HTTPException(401, "Email ou mot de passe incorrect")


@app.get("/auth/me")
async def profil_utilisateur(u=Depends(obtenir_utilisateur)):
    """Retourne les informations de l'utilisateur connecté."""
    return {"id": u["id"], "username": u["nom"], "email": u["email"]}


@app.get("/admin/users")
async def admin_lister_utilisateurs(key: str = None):
    """Route admin pour voir les inscrits (pour debug Railway)."""
    ADMIN_KEY = os.getenv("ADMIN_KEY", "awal_debug_2026")
    if key != ADMIN_KEY:
        raise HTTPException(403, "Accès refusé")
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT id, nom, email, cree_le FROM utilisateurs")
        utilisateurs = await cur.fetchall()
        return {"total": len(utilisateurs), "utilisateurs": [dict(u) for u in utilisateurs]}


# ── Endpoint de chat (Server-Sent Events) ─────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: Request, u=Depends(obtenir_utilisateur)):
    """
    Traite un message et retourne la réponse en Server-Sent Events (SSE).
    """
    try:
        corps = await req.json()
    except Exception:
        raise HTTPException(400, "Corps JSON invalide")

    message = corps.get("message", "").strip()
    if not message:
        raise HTTPException(400, "Message vide")

    cid = corps.get("conversation_id") or f"conv_{int(time.time())}"

    async def generer_stream():
        debut_total = time.time()
        done_sent = False
        final_meta = {}
        final_response = ""
        
        try:
            historique = await bd_historique(u["id"], cid)
            
            # brain.traiter_message est un async generator
            async for event in brain.traiter_message(message, historique=historique):
                if event["type"] == "step":
                    yield f"event: step\ndata: {json.dumps({'step': event['step'], 'label': event['label']})}\n\n"
                    # Petit délai pour la fluidité UI
                    await asyncio.sleep(0.05)
                
                elif event["type"] == "final":
                    final_response = event["reponse"]
                    yield f"event: token\ndata: {json.dumps(final_response, ensure_ascii=False)}\n\n"
                    
                    final_meta = {
                        "intention":    event.get("intention"),
                        "confiance":    event.get("confiance", 0),
                        "langue":       event.get("langue", "tamazight"),
                        "temps_rag_ms": event.get("temps_rag_ms", 0),
                        "temps_llm_ms": event.get("temps_llm_ms", 0),
                        "temps_total":  int((time.time() - debut_total) * 1000),
                    }

            if final_meta:
                log.info(f"[CHAT] Stream Done: {final_meta['intention']} | total: {final_meta['temps_total']}ms | user: {u['nom']}")
                yield f"event: done\ndata: {json.dumps(final_meta)}\n\n"
                done_sent = True

                # Sauvegarde en base
                await bd_inserer_message(
                    u["id"], cid, message, final_response,
                    corps.get("domain", "general"), 
                    final_meta["intention"],
                    final_meta["langue"], 
                    final_meta["temps_total"]
                )

        except Exception as erreur:
            log.error(f"[CHAT] Erreur stream : {erreur}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': 'Erreur serveur', 'detail': str(erreur)})}\n\n"
        finally:
            if not done_sent:
                yield f"event: done\ndata: {json.dumps({'error': True, 'temps_total': int((time.time()-debut_total)*1000)})}\n\n"

    return StreamingResponse(
        generer_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


# ── Endpoints de gestion des conversations ────────────────────────────────

@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    """Liste toutes les conversations de l'utilisateur connecté."""
    return {"conversations": await bd_lister_conversations(u["id"])}


@app.get("/conversations/{cid}/messages")
async def obtenir_messages(cid: str, u=Depends(obtenir_utilisateur)):
    """Retourne tous les messages d'une conversation ordonnés chronologiquement."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT message_utilisateur, message_bot, horodatage "
            "FROM messages WHERE utilisateur_id=? AND conversation_id=? ORDER BY id ASC",
            (u["id"], cid)
        )
        msgs = []
        for ligne in await cur.fetchall():
            msgs.append({"sender": "user", "content": ligne[0], "date": ligne[2]})
            msgs.append({"sender": "bot",  "content": ligne[1], "date": ligne[2]})
        return {"messages": msgs}


@app.delete("/conversations/{cid}")
async def supprimer_conversation(cid: str, u=Depends(obtenir_utilisateur)):
    """Supprime tous les messages d'une conversation."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "DELETE FROM messages WHERE utilisateur_id=? AND conversation_id=?",
            (u["id"], cid)
        )
        await bd.commit()
    return {"status": "ok"}