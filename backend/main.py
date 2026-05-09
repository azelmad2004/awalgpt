import logging
import os
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional  # <--- هذا هو السطر الناقص الذي سبب المشكلة
from contextlib import asynccontextmanager

import aiosqlite
import bcrypt 
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

import brain

# ── Logger Configuration ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("awalgpt")

# ── Configuration ──────────────────────────────────────────────────────────
SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_secure_key_2026")
CHEMIN_BD  = os.getenv("DB_PATH", "awalgpt.db")

# ── Pydantic Models ────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    preferred_variety: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

# ── Base de Données ────────────────────────────────────────────────────────
async def initialiser_base():
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.executescript("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id TEXT PRIMARY KEY, nom TEXT, email TEXT UNIQUE,
                mot_de_passe TEXT, variete TEXT, cree_le TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utilisateur_id TEXT, conversation_id TEXT,
                message_utilisateur TEXT, message_bot TEXT,
                horodatage TEXT
            );
        """)
        await bd.commit()

async def bd_trouver_utilisateur(uid: str):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE id = ?", (uid,))
        ligne = await cur.fetchone()
        return dict(ligne) if ligne else None

# ── Authentification ──────────────────────────────────────────────────────
def creer_token(uid: str) -> str:
    return jwt.encode(
        {"sub": uid, "exp": datetime.now(timezone.utc) + timedelta(days=30)},
        SECRET_JWT, algorithm="HS256"
    )

async def obtenir_utilisateur(requete: Request):
    auth_header = requete.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Token manquant")
    token = auth_header.replace("Bearer ", "").strip()
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=["HS256"])
        u = await bd_trouver_utilisateur(payload["sub"])
        if u: return u
    except: pass
    raise HTTPException(401, "Token invalide")

# ── Lifespan & App ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialiser_base()
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/auth/register")
async def inscrire(data: UserRegister):
    uid = hashlib.sha256(data.email.lower().encode()).hexdigest()
    if await bd_trouver_utilisateur(uid):
        raise HTTPException(400, "Email déjà utilisé")
    hache = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("INSERT INTO utilisateurs VALUES (?,?,?,?,?,?)",
            (uid, data.username, data.email.lower(), hache, data.preferred_variety, datetime.now().isoformat()))
        await bd.commit()
    return {"token": creer_token(uid), "user": {"id": uid, "username": data.username}}

@app.post("/auth/login")
async def connexion(data: UserLogin):
    uid = hashlib.sha256(data.email.lower().encode()).hexdigest()
    u = await bd_trouver_utilisateur(uid)
    if u and bcrypt.checkpw(data.password.encode(), u["mot_de_passe"].encode()):
        return {"token": creer_token(uid), "user": {"id": uid, "username": u["nom"]}}
    raise HTTPException(401, "Identifiants incorrects")

@app.get("/auth/me")
async def profil(u=Depends(obtenir_utilisateur)):
    return {"id": u["id"], "username": u["nom"], "email": u["email"]}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, u=Depends(obtenir_utilisateur)):
    conv_id = request.conversation_id or str(uuid.uuid4())
    # هنا يتم استدعاء منطق AWAL GPT الخاص بك
    bot_res = f"Azul! Ceci est une réponse d'AWAL GPT pour: {request.message}"
    
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("""
            INSERT INTO messages (utilisateur_id, conversation_id, message_utilisateur, message_bot, horodatage)
            VALUES (?, ?, ?, ?, ?)
        """, (u["id"], conv_id, request.message, bot_res, datetime.now().isoformat()))
        await bd.commit()
    
    return {"response": bot_res, "conversation_id": conv_id}

@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        query = """
            SELECT conversation_id, MAX(horodatage) as last_update, message_utilisateur as title
            FROM messages WHERE utilisateur_id = ? 
            GROUP BY conversation_id ORDER BY last_update DESC
        """
        cur = await bd.execute(query, (u["id"],))
        rows = await cur.fetchall()
        return {"conversations": [{"id": r["conversation_id"], "title": r["title"][:40]} for r in rows]}

@app.get("/conversations/{cid}/messages")
async def get_messages(cid: str, u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM messages WHERE utilisateur_id=? AND conversation_id=? ORDER BY id ASC", (u["id"], cid))
        rows = await cur.fetchall()
        msgs = []
        for r in rows:
            msgs.append({"sender": "user", "text": r["message_utilisateur"]})
            msgs.append({"sender": "bot", "text": r["message_bot"]})
        return {"messages": msgs}

@app.delete("/conversations/{cid}")
async def delete_conv(cid: str, u=Depends(obtenir_utilisateur)):
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("DELETE FROM messages WHERE utilisateur_id=? AND conversation_id=?", (u["id"], cid))
        await bd.commit()
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
