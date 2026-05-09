# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# main.py — Serveur FastAPI AWAL GPT (Production Ready)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

import logging
import os
import time
import json
import hashlib
import asyncio
import jwt
import bcrypt
import aiosqlite
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

# --- Configuration du Cerveau AWAL (RAG/NLP) ---
import brain 

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("awalgpt-api")

app = FastAPI(title="AWAL GPT API")

# --- CONFIGURATION CRUCIALE : CORS ---
# Remplace ces URLs par tes URLs Railway réelles
origins = [
    "https://awal-gpt.up.railway.app", # Ton frontend
    "http://localhost:3000",           # Pour tes tests locaux
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration Base de Données & Sécurité ---
CHEMIN_BD = "awalgpt.db"
SECRET_KEY = os.getenv("JWT_SECRET", "ton_secret_tres_sur_123")
ALGORITHME = "HS256"

# --- Utilitaires de Sécurité ---
def generer_token(donnees: dict):
    expiration = datetime.now(timezone.utc) + timedelta(days=7)
    donnees.update({"exp": expiration})
    return jwt.encode(donnees, SECRET_KEY, algorithm=ALGORITHME)

async def obtenir_utilisateur(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Non autorisé")
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHME])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Session expirée")

# --- Initialisation de la Base de Données ---
@app.on_event("startup")
async def initialiser_bd():
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                utilisateur_id INTEGER,
                titre TEXT,
                date_creation DATETIME
            )
        """)
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                utilisateur_id INTEGER,
                message_utilisateur TEXT,
                message_bot TEXT,
                horodatage DATETIME
            )
        """)
        await bd.commit()

# --- ROUTES AUTHENTIFICATION ---

@app.post("/auth/register")
async def inscription(donnees: dict):
    username = donnees.get("username")
    email = donnees.get("email")
    password = donnees.get("password")
    
    if not username or not email or not password:
        raise HTTPException(status_code=400, detail="Champs manquants")
        
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    try:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            cur = await bd.execute(
                "INSERT INTO utilisateurs (username, email, password) VALUES (?, ?, ?)",
                (username, email, hashed)
            )
            uid = cur.lastrowid
            await bd.commit()
            
            user_data = {"id": uid, "username": username, "email": email}
            return {"token": generer_token(user_data), "user": user_data}
    except aiosqlite.IntegrityError:
        raise HTTPException(status_code=400, detail="Email ou pseudo déjà utilisé")

@app.post("/auth/login")
async def connexion(donnees: dict):
    email = donnees.get("email")
    password = donnees.get("password")
    
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE email = ?", (email,))
        user = await cur.fetchone()
        
        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
            user_data = {"id": user["id"], "username": user["username"], "email": user["email"]}
            return {"token": generer_token(user_data), "user": user_data}
            
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

# --- ROUTES CHAT & STREAMING ---

@app.post("/chat/stream")
async def chat_stream(request: Request, u=Depends(obtenir_utilisateur)):
    donnees = await request.json()
    message_u = donnees.get("message")
    conv_id = donnees.get("conversation_id")
    
    if not message_u:
        raise HTTPException(status_code=400, detail="Message vide")

    # Générer une réponse via le module brain.py (ton IA)
    async def generateur():
        reponse_complete = ""
        # On simule ou on utilise ton moteur de streaming AWAL
        try:
            async for chunk in brain.demander_stream(message_u):
                reponse_complete += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            # Sauvegarde en base après le stream
            async with aiosqlite.connect(CHEMIN_BD) as bd:
                await bd.execute(
                    "INSERT INTO messages (conversation_id, utilisateur_id, message_utilisateur, message_bot, horodatage) VALUES (?, ?, ?, ?, ?)",
                    (conv_id, u["id"], message_u, reponse_complete, datetime.now())
                )
                await bd.commit()
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generateur(), media_type="text/event-stream")

@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT * FROM conversations WHERE utilisateur_id = ? ORDER BY date_creation DESC",
            (u["id"],)
        )
        lignes = await cur.fetchall()
        return [dict(l) for l in lignes]

@app.post("/conversations")
async def creer_conversation(u=Depends(obtenir_utilisateur)):
    cid = hashlib.md5(f"{u['id']}{time.time()}".encode()).hexdigest()[:10]
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "INSERT INTO conversations (id, utilisateur_id, titre, date_creation) VALUES (?, ?, ?, ?)",
            (cid, u["id"], "Nouvelle discussion", datetime.now())
        )
        await bd.commit()
    return {"id": cid}

# --- POINT D'ENTRÉE POUR RAILWAY ---
if __name__ == "__main__":
    import uvicorn
    # Railway utilise la variable PORT, sinon 8080 par défaut
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
