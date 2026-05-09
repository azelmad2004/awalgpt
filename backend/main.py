# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# main.py — Serveur FastAPI AWAL GPT (Version Optimisée)
# Rôle : Authentification JWT, Streaming SSE, Gestion Conversations SQLite & RAG
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

import logging

# ── Configuration du Logger ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
# On réduit le bruit des bibliothèques externes
for _lib in ("sentence_transformers", "httpx", "faiss", "uvicorn", "aiosqlite"):
    logging.getLogger(_lib).setLevel(logging.WARNING)

log = logging.getLogger("awalgpt")

import os
import time
import json
import hashlib
import asyncio
import uuid
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import aiosqlite
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Importation de la logique métier
import brain
import core

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# CONFIGURATION ET VARIABLES D'ENVIRONNEMENT
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_secret_key_2026_long_enough")
CHEMIN_BD  = os.getenv("DB_PATH", "awalgpt.db")
ALGORITHME = "HS256"

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# INITIALISATION DE LA BASE DE DONNÉES
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

async def initialiser_bd():
    """Crée les tables nécessaires si elles n'existent pas."""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        # Table Utilisateurs
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id TEXT PRIMARY KEY,
                nom TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                date_creation TEXT
            )
        """)
        # Table Messages (Historique)
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utilisateur_id TEXT,
                conversation_id TEXT,
                role TEXT,
                contenu TEXT,
                intention TEXT,
                langue TEXT,
                horodatage TEXT,
                FOREIGN KEY(utilisateur_id) REFERENCES utilisateurs(id)
            )
        """)
        await bd.commit()
    log.info(f"[DB] Base de données initialisée à : {CHEMIN_BD}")

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# SÉCURITÉ ET DÉPENDANCES
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

def generer_token(uid: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(days=7)
    payload = {"sub": uid, "exp": expiration}
    return jwt.encode(payload, SECRET_JWT, algorithm=ALGORITHME)

async def obtenir_utilisateur(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Non authentifié")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=[ALGORITHME])
        uid = payload.get("sub")
        
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            bd.row_factory = aiosqlite.Row
            cur = await bd.execute("SELECT * FROM utilisateurs WHERE id = ?", (uid,))
            user = await cur.fetchone()
            if not user:
                raise HTTPException(status_code=401, detail="Utilisateur introuvable")
            return dict(user)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expirée")
    except Exception:
        raise HTTPException(status_code=401, detail="Token invalide")

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# LIFESPAN (Gestion démarrage/arrêt)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Démarrage du serveur AWAL GPT...")
    await initialiser_bd()
    # On peut charger les moteurs ici si nécessaire
    yield
    log.info("🛑 Arrêt du serveur.")

app = FastAPI(title="AWAL GPT API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# ROUTES AUTHENTIFICATION
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@app.post("/auth/register")
async def register(req: Request):
    data = await req.json()
    email = data.get("email", "").lower().strip()
    nom = data.get("username", "").strip()
    pwd = data.get("password", "")

    if not email or not pwd or not nom:
        raise HTTPException(400, "Champs manquants")

    uid = hashlib.sha256(email.encode()).hexdigest()[:12]
    hashed_pwd = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()

    try:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            await bd.execute(
                "INSERT INTO utilisateurs (id, nom, email, password, date_creation) VALUES (?, ?, ?, ?, ?)",
                (uid, nom, email, hashed_pwd, datetime.now().isoformat())
            )
            await bd.commit()
    except aiosqlite.IntegrityError:
        raise HTTPException(400, "Cet email est déjà utilisé")

    return {"token": generer_token(uid), "user": {"id": uid, "nom": nom}}

@app.post("/auth/login")
async def login(req: Request):
    data = await req.json()
    email = data.get("email", "").lower().strip()
    pwd = data.get("password", "")

    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE email = ?", (email,))
        user = await cur.fetchone()

        if user and bcrypt.checkpw(pwd.encode(), user["password"].encode()):
            return {
                "token": generer_token(user["id"]),
                "user": {"id": user["id"], "nom": user["nom"]}
            }
    
    raise HTTPException(401, "Identifiants incorrects")

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# ROUTES CONVERSATIONS
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT conversation_id, MAX(horodatage) as last_msg, contenu "
            "FROM messages WHERE utilisateur_id = ? GROUP BY conversation_id ORDER BY last_msg DESC",
            (u["id"],)
        )
        rows = await cur.fetchall()
        return [{"id": r[0], "last_date": r[1], "preview": r[2][:50]} for r in rows]

@app.get("/conversations/{cid}")
async def detail_conversation(cid: str, u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT role, contenu, horodatage FROM messages WHERE conversation_id = ? AND utilisateur_id = ? ORDER BY id ASC",
            (cid, u["id"])
        )
        messages = await cur.fetchall()
        return [dict(m) for m in messages]

@app.delete("/conversations/{cid}")
async def supprimer_conversation(cid: str, u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("DELETE FROM messages WHERE conversation_id = ? AND utilisateur_id = ?", (cid, u["id"]))
        await bd.commit()
    return {"status": "success"}

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# CŒUR DU CHAT : STREAMING SSE
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@app.post("/chat/stream")
async def chat_stream(req: Request, u=Depends(obtenir_utilisateur)):
    body = await req.json()
    message_utilisateur = body.get("message", "").strip()
    cid = body.get("conversation_id") or f"conv_{uuid.uuid4().hex[:8]}"
    
    if not message_utilisateur:
        raise HTTPException(400, "Message vide")

    async def generateur_evenements():
        start_time = time.time()
        reponse_complete = ""
        meta_finale = {}

        # 1. Récupération de l'historique récent
        historique = []
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            bd.row_factory = aiosqlite.Row
            cur = await bd.execute(
                "SELECT role, contenu FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT 6",
                (cid,)
            )
            rows = await cur.fetchall()
            for r in reversed(rows):
                historique.append({"role": r["role"], "content": r["contenu"]})

        try:
            # 2. Appel au cerveau IA (brain.py doit retourner un async generator)
            async for event in brain.traiter_message_stream(message_utilisateur, historique=historique):
                
                if event["type"] == "step":
                    # Information sur l'étape en cours (RAG, Classification, etc.)
                    yield f"event: step\ndata: {json.dumps({'label': event['label']})}\n\n"
                
                elif event["type"] == "token":
                    # Morceau de texte généré
                    token = event.get("content", "")
                    reponse_complete += token
                    yield f"event: token\ndata: {json.dumps(token, ensure_ascii=False)}\n\n"
                
                elif event["type"] == "final":
                    # Métadonnées finales
                    meta_finale = {
                        "intention": event.get("intention"),
                        "langue": event.get("langue"),
                        "temps_ms": int((time.time() - start_time) * 1000)
                    }

            # 3. Envoi du signal de fin
            yield f"event: done\ndata: {json.dumps(meta_finale)}\n\n"

            # 4. Sauvegarde asynchrone en base de données
            async with aiosqlite.connect(CHEMIN_BD) as bd:
                ts = datetime.now().isoformat()
                await bd.executemany(
                    "INSERT INTO messages (utilisateur_id, conversation_id, role, contenu, intention, langue, horodatage) VALUES (?,?,?,?,?,?,?)",
                    [
                        (u["id"], cid, "user", message_utilisateur, None, None, ts),
                        (u["id"], cid, "assistant", reponse_complete, meta_finale.get("intention"), meta_finale.get("langue"), ts)
                    ]
                )
                await bd.commit()

        except Exception as e:
            log.error(f"[STREAM ERROR] {e}")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        generateur_evenements(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# LANCEMENT (COMPATIBLE RAILWAY)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

if __name__ == "__main__":
    import uvicorn
    # Railway définit la variable d'environnement PORT
    port = int(os.environ.get("PORT", 8000))
    
    log.info(f"🌍 AWAL GPT prêt sur le port {port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False, 
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
