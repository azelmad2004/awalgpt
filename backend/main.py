import os
import time
import json
import hashlib
import logging
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import aiosqlite
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import brain

# ── Configuration du Logger ────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("awalgpt")

# ── Variables d'Environnement ──────────────────────────────────────────────
SECRET_JWT = os.getenv("JWT_SECRET", "votre_cle_secrete_tres_longue")
CHEMIN_BD  = os.getenv("DB_PATH", "awalgpt.db")
ALGORITHME = "HS256"

# ── Initialisation de la Base de Données ───────────────────────────────────
async def initialiser_bd():
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS utilisateurs (
                id TEXT PRIMARY KEY,
                nom TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                date_creation TEXT
            )
        """)
        await bd.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                utilisateur_id TEXT,
                conversation_id TEXT,
                role TEXT,
                contenu TEXT,
                horodatage TEXT,
                FOREIGN KEY(utilisateur_id) REFERENCES utilisateurs(id)
            )
        """)
        await bd.commit()
    log.info(f"Database ready at {CHEMIN_BD}")

# ── Sécurité JWT ──────────────────────────────────────────────────────────
def generer_token(uid: str) -> str:
    payload = {"sub": uid, "exp": datetime.now(timezone.utc) + timedelta(days=7)}
    return jwt.encode(payload, SECRET_JWT, algorithm=ALGORITHME)

async def obtenir_utilisateur(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = auth.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=[ALGORITHME])
        uid = payload.get("sub")
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            bd.row_factory = aiosqlite.Row
            cur = await bd.execute("SELECT * FROM utilisateurs WHERE id = ?", (uid,))
            user = await cur.fetchone()
            if not user: raise HTTPException(401, "User not found")
            return dict(user)
    except Exception:
        raise HTTPException(401, "Invalid token")

# ── Lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialiser_bd()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes Auth ────────────────────────────────────────────────────────────
@app.post("/auth/register")
async def register(req: Request):
    data = await req.json()
    email, nom, pwd = data.get("email"), data.get("username"), data.get("password")
    uid = hashlib.sha256(email.encode()).hexdigest()[:10]
    hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
    try:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            await bd.execute("INSERT INTO utilisateurs VALUES (?,?,?,?,?)", 
                           (uid, nom, email, hashed, datetime.now().isoformat()))
            await bd.commit()
        return {"token": generer_token(uid), "user": {"id": uid, "nom": nom}}
    except: raise HTTPException(400, "Email already exists")

@app.post("/auth/login")
async def login(req: Request):
    data = await req.json()
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE email=?", (data.get("email"),))
        u = await cur.fetchone()
        if u and bcrypt.checkpw(data.get("password").encode(), u["password"].encode()):
            return {"token": generer_token(u["id"]), "user": {"id": u["id"], "nom": u["nom"]}}
    raise HTTPException(401, "Invalid credentials")

# ── Streaming Chat ─────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: Request, u=Depends(obtenir_utilisateur)):
    data = await req.json()
    msg = data.get("message", "")
    cid = data.get("conversation_id") or f"conv_{uuid.uuid4().hex[:6]}"

    async def event_generator():
        full_reply = ""
        try:
            # Appel à ta fonction dans brain.py
            async for chunk in brain.traiterMessage(msg, identifiant=u["id"]):
                if chunk:
                    full_reply += chunk
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            # Sauvegarde finale
            async with aiosqlite.connect(CHEMIN_BD) as bd:
                ts = datetime.now().isoformat()
                await bd.execute("INSERT INTO messages (utilisateur_id, conversation_id, role, contenu, horodatage) VALUES (?,?,?,?,?)",
                               (u["id"], cid, "user", msg, ts))
                await bd.execute("INSERT INTO messages (utilisateur_id, conversation_id, role, contenu, horodatage) VALUES (?,?,?,?,?)",
                               (u["id"], cid, "bot", full_reply, ts))
                await bd.commit()
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ── Lancement ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
