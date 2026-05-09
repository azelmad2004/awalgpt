# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# main.py — Serveur FastAPI AWAL GPT (Version Finale Corrigée)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
import logging
import os
import time
import json
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import aiosqlite
import bcrypt  # المصحح: استخدام bcrypt المباشر لتجنب مشاكل passlib
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr # المصحح: إضافة Pydantic لضمان عدم حدوث KeyError

import brain

# ── Logger Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
log = logging.getLogger("awalgpt")

# ── Configuration ──────────────────────────────────────────────────────────
SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_secret_key_2026_long_enough")
CHEMIN_BD  = os.getenv("DB_PATH", "awalgpt.db")

# ── Pydantic Models (الإصلاح الجوهري لمنع KeyError) ────────────────────────
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    preferred_variety: str = None

class UserLogin(BaseModel):
    email: str
    password: str

class ChatMessage(BaseModel):
    message: str
    conversation_id: str = None
    domain: str = "general"
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None  # تأكد أن هذا الحقل موجود لربط الرسائل بال
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# BASE DE DONNÉES SQLite (aiosqlite)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

async def initialiser_base() -> None:
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
                domaine TEXT, intention TEXT, langue TEXT,
                temps_ms INTEGER, horodatage TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(utilisateur_id);
            CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
        """)
        await bd.commit()
    log.info("[DB] Base initialisée")

async def bd_trouver_utilisateur(uid: str) -> dict:
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute("SELECT * FROM utilisateurs WHERE id = ?", (uid,))
        ligne = await cur.fetchone()
        return dict(ligne) if ligne else None

async def bd_creer_utilisateur(uid: str, nom: str, email: str, hache: str, variete: str = None) -> None:
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "INSERT INTO utilisateurs VALUES (?,?,?,?,?,?)",
            (uid, nom, email, hache, variete, datetime.now().isoformat())
        )
        await bd.commit()

async def bd_historique(uid: str, conv_id: str) -> list:
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
        if ligne['message_utilisateur']:
            hist.append({"role": "user", "content": ligne['message_utilisateur']})
        if ligne['message_bot']:
            hist.append({"role": "assistant", "content": ligne['message_bot']})
    return hist

async def bd_inserer_message(uid: str, cid: str, msg_user: str, msg_bot: str,
                            domaine: str, intention: str, langue: str, temps_ms: int) -> None:
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "INSERT INTO messages (utilisateur_id, conversation_id, message_utilisateur, "
            "message_bot, domaine, intention, langue, temps_ms, horodatage) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (uid, cid, msg_user, msg_bot, domaine, intention, langue, temps_ms, datetime.now().isoformat())
        )
        await bd.commit()

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# AUTHENTIFICATION & LIFESPAN
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

def creer_token(uid: str) -> str:
    return jwt.encode(
        {"sub": uid, "exp": datetime.now(timezone.utc) + timedelta(days=30)},
        SECRET_JWT, algorithm="HS256"
    )

async def obtenir_utilisateur(requete: Request) -> dict:
    auth_header = requete.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token manquant")
    token = auth_header.replace("Bearer ", "").strip()
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=["HS256"])
        utilisateur = await bd_trouver_utilisateur(payload["sub"])
        if utilisateur: return utilisateur
    except Exception: pass
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token invalide")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialiser_base()
    if hasattr(brain, 'charger_configs'): brain.charger_configs()
    yield

app = FastAPI(title="Awal GPT", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# ENDPOINTS
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@app.post("/auth/register")
async def inscrire(data: UserRegister): # تم الإصلاح: استخدام Pydantic
    uid = hashlib.sha256(data.email.lower().encode()).hexdigest()
    if await bd_trouver_utilisateur(uid):
        raise HTTPException(400, "Email déjà utilisé")
    
    # تشفير كلمة المرور بأسلوب متوافق مع Python 3.13
    hache = bcrypt.hashpw(data.password.encode(), bcrypt.gensalt()).decode()
    await bd_creer_utilisateur(uid, data.username, data.email.lower(), hache, data.preferred_variety)
    return {"token": creer_token(uid), "user": {"id": uid, "username": data.username}}

@app.post("/auth/login")
async def connexion(data: UserLogin): # تم الإصلاح: استخدام Pydantic لمنع KeyError
    uid = hashlib.sha256(data.email.lower().encode()).hexdigest()
    utilisateur = await bd_trouver_utilisateur(uid)
    
    if utilisateur and bcrypt.checkpw(data.password.encode(), utilisateur["mot_de_passe"].encode()):
        return {
            "token": creer_token(uid),
            "user": {"id": uid, "username": utilisateur["nom"]}
        }
    raise HTTPException(401, "Email ou mot de passe incorrect")

@app.get("/auth/me")
async def profil_utilisateur(u=Depends(obtenir_utilisateur)):
    return {"id": u["id"], "username": u["nom"], "email": u["email"]}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, current_user=Depends(get_current_user)):
    # 1. التحقق من البيانات المرسلة
    user_msg = request.message
    conv_id = request.conversation_id or str(uuid.uuid4()) # إنشاء معرف جديد إذا لم يوجد

    # 2. توليد رد الذكاء الاصطناعي (هنا تضع منطق AWAL GPT الخاص بك)
    bot_res = "هذا رد تجريبي من AWAL GPT" 

    # 3. حفظ في MySQL (Railway)
    db = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO messages (user_id, conversation_id, user_message, bot_response) 
            VALUES (:uid, :cid, :umsg, :bres)
        """), {
            "uid": current_user.id,
            "cid": conv_id,
            "umsg": user_msg,
            "bres": bot_res
        })
        db.commit()
    finally:
        db.close()

    return {"response": bot_res, "conversation_id": conv_id}
@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    """جلب قائمة المحادثات لليوزر الحالي ليظهر في القائمة الجانبية"""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        # سنجلب معرف المحادثة، آخر رسالة كعنوان، وتاريخ آخر تحديث
        query = """
            SELECT conversation_id, MAX(horodatage) as last_update, message_utilisateur as title
            FROM messages 
            WHERE utilisateur_id = ? 
            GROUP BY conversation_id 
            ORDER BY last_update DESC
        """
        cur = await bd.execute(query, (u["id"],))
        lignes = await cur.fetchall()
        
        return {
            "conversations": [
                {
                    "id": l["conversation_id"],
                    "title": (l["title"] or "Nouvelle discussion")[:40],
                    "updated_at": l["last_update"]
                } for l in lignes
            ]
        }

@app.get("/conversations/{cid}/messages")
async def obtenir_messages(cid: str, u=Depends(obtenir_utilisateur)):
    """جلب كل الرسائل داخل محادثة معينة عند الضغط عليها"""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        bd.row_factory = aiosqlite.Row
        cur = await bd.execute(
            "SELECT message_utilisateur, message_bot, horodatage FROM messages "
            "WHERE utilisateur_id=? AND conversation_id=? ORDER BY id ASC",
            (u["id"], cid)
        )
        lignes = await cur.fetchall()
        
        # تحويل البيانات لتناسب واجهة React (Sender: user/bot)
        msgs = []
        for l in lignes:
            msgs.append({"sender": "user", "text": l["message_utilisateur"], "date": l["horodatage"]})
            msgs.append({"sender": "bot", "text": l["message_bot"], "date": l["horodatage"]})
        
        return {"messages": msgs}

@app.delete("/conversations/{cid}")
async def supprimer_conversation(cid: str, u=Depends(obtenir_utilisateur)):
    """حذف محادثة بالكامل"""
    async with aiosqlite.connect(CHEMIN_BD) as bd:
        await bd.execute(
            "DELETE FROM messages WHERE utilisateur_id=? AND conversation_id=?",
            (u["id"], cid)
        )
        await bd.commit()
    return {"status": "success", "message": "Conversation supprimée"}
@app.get("/conversations")
def list_history(current_user=Depends(get_current_user)):
    db = SessionLocal()
    # جلب آخر رسالة من كل محادثة لتظهر كعنوان
    rows = db.execute(text("""
        SELECT conversation_id, user_message, created_at 
        FROM messages 
        WHERE user_id = :uid 
        AND id IN (SELECT MAX(id) FROM messages GROUP BY conversation_id)
        ORDER BY created_at DESC
    """), {"uid": current_user.id}).fetchall()
    db.close()

    return [
        {"id": r.conversation_id, "title": r.user_message[:30], "date": r.created_at} 
        for r in rows
    ]
# ── تحديث دالة chat_stream لحفظ التاريخ بشكل أفضل ──────────────────
# تأكد أن دالة chat_stream تقوم بحفظ الرسائل كما في الكود السابق الذي أعطيتك إياه
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
