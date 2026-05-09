# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# main.py — Serveur FastAPI AWAL GPT (Version Professionnelle MySQL)
# Rôle : Authentification JWT, Streaming SSE, RAG Tamazight & Persistance MySQL
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

import os
import time
import json
import hashlib
import logging
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# ── SQL Alchemy & MySQL ──────────────────────────────────────────────────
from sqlalchemy import (
    Column, String, Text, Integer, BigInteger, DateTime, 
    ForeignKey, select, delete, func, desc
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# ── FastAPI & Securité ─────────────────────────────────────────────────────
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Logique Métier (Brain) ────────────────────────────────────────────────
import brain
from core import chargerMoteurTfidf, chargerMoteurSemantique

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# CONFIGURATION ET LOGGER
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("awalgpt")

# Variables d'environnement
CLE_GROQ   = os.getenv("GROQ_API_KEY")
SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_secret_2026_ultra_long_key_for_security")
MYSQL_URL  = os.getenv("MYSQL_URL") # Format: mysql://user:pass@host:port/db

# Conversion de l'URL pour SQLAlchemy Async
if MYSQL_URL and MYSQL_URL.startswith("mysql://"):
    DATABASE_URL = MYSQL_URL.replace("mysql://", "mysql+aiomysql://")
else:
    # Fallback pour le développement local si MYSQL_URL n'est pas défini
    DATABASE_URL = "mysql+aiomysql://root:password@localhost:3306/awalgpt"

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# MODÈLES DE DONNÉES (SQLAlchemy)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

Base = declarative_base()

class User(Base):
    __tablename__ = "utilisateurs"
    id = Column(String(64), primary_key=True)
    nom = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, nullable=False, index=True)
    mot_de_passe = Column(String(255), nullable=False)
    variete_preferee = Column(String(50), default="standard")
    cree_le = Column(DateTime, default=datetime.utcnow)
    
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    utilisateur_id = Column(String(64), ForeignKey("utilisateurs.id"), index=True)
    conversation_id = Column(String(64), index=True)
    
    role = Column(String(20)) # 'user' ou 'assistant'
    contenu = Column(Text, nullable=False)
    
    # Métadonnées IA
    domaine = Column(String(50))
    intention = Column(String(50))
    langue = Column(String(20))
    temps_traitement_ms = Column(Integer, default=0)
    horodatage = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="messages")

# Configuration de l'Engine Async
engine = create_async_engine(
    DATABASE_URL, 
    pool_pre_ping=True, 
    pool_recycle=3600,
    echo=False
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# UTILITAIRES DE SÉCURITÉ ET JWT
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

def generer_token(uid: str) -> str:
    """Crée un token JWT valide 30 jours."""
    payload = {
        "sub": uid,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=30)
    }
    return jwt.encode(payload, SECRET_JWT, algorithm="HS256")

async def verifier_token(requete: Request) -> User:
    """Vérifie le token dans le header Authorization."""
    auth_header = requete.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token manquant ou format invalide")
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=["HS256"])
        uid = payload.get("sub")
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(User).where(User.id == uid))
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=401, detail="Utilisateur non trouvé")
            return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# GESTION DES CYCLES DE VIE (LIFESPAN)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage et nettoyage à l'arrêt."""
    log.info("[STARTUP] Connexion à MySQL et vérification des tables...")
    try:
        async with engine.begin() as conn:
            # Crée les tables si elles n'existent pas
            await conn.run_sync(Base.metadata.create_all)
        log.info("[STARTUP] Base de données MySQL prête.")
        
        # Initialisation des moteurs de recherche (RAG)
        log.info("[STARTUP] Chargement des moteurs IA...")
        brain.charger_configs()
        log.info("[STARTUP] Moteurs IA opérationnels.")
        
    except Exception as e:
        log.error(f"[STARTUP] Erreur critique : {e}")
        # On ne stoppe pas forcément, mais les requêtes échoueront
        
    yield
    
    log.info("[SHUTDOWN] Fermeture des connexions MySQL...")
    await engine.dispose()

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# ROUTES API
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

app = FastAPI(
    title="AWAL GPT - API Professionnelle",
    description="Backend IA pour la langue Tamazight avec RAG et MySQL",
    version="3.5.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return {
        "app": "AWAL GPT API",
        "status": "active",
        "db": "MySQL",
        "engine": "FastAPI + Groq"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}

# ── Authentification ───────────────────────────────────────────────────────

@app.post("/auth/register")
async def register(req: Request):
    data = await req.json()
    email = data.get("email", "").lower().strip()
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not email or not password or not username:
        raise HTTPException(400, "Tous les champs sont requis")

    uid = hashlib.sha256(email.encode()).hexdigest()
    hashed_pwd = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    async with AsyncSessionLocal() as session:
        # Vérifier si l'utilisateur existe déjà
        check = await session.execute(select(User).where(User.email == email))
        if check.scalar_one_or_none():
            raise HTTPException(400, "Cet email est déjà enregistré")

        new_user = User(
            id=uid,
            nom=username,
            email=email,
            mot_de_passe=hashed_pwd
        )
        session.add(new_user)
        await session.commit()
        
    log.info(f"[AUTH] Inscription : {username} ({email})")
    return {"token": generer_token(uid), "user": {"id": uid, "username": username}}

@app.post("/auth/login")
async def login(req: Request):
    data = await req.json()
    email = data.get("email", "").lower().strip()
    password = data.get("password", "")

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user and bcrypt.checkpw(password.encode(), user.mot_de_passe.encode()):
            log.info(f"[AUTH] Connexion : {user.nom}")
            return {
                "token": generer_token(user.id),
                "user": {"id": user.id, "username": user.nom}
            }
            
    raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

# ── Gestion des Conversations ──────────────────────────────────────────────

@app.get("/conversations")
async def get_conversations(current_user: User = Depends(verifier_token)):
    async with AsyncSessionLocal() as session:
        # On récupère l'ID de conversation unique et le dernier message pour chaque
        query = (
            select(
                Message.conversation_id, 
                func.max(Message.horodatage).label("last_date"),
                func.count(Message.id).label("count")
            )
            .where(Message.utilisateur_id == current_user.id)
            .group_by(Message.conversation_id)
            .order_by(desc("last_date"))
        )
        results = await session.execute(query)
        
        convs = []
        for row in results:
            convs.append({
                "id": row.conversation_id,
                "updated_at": row.last_date.isoformat(),
                "messages_count": row.count
            })
            
        return {"conversations": convs}

@app.get("/conversations/{cid}/messages")
async def get_messages(cid: str, current_user: User = Depends(verifier_token)):
    async with AsyncSessionLocal() as session:
        query = (
            select(Message)
            .where(Message.utilisateur_id == current_user.id, Message.conversation_id == cid)
            .order_by(Message.horodatage.asc())
        )
        result = await session.execute(query)
        messages = result.scalars().all()
        
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.contenu,
                    "timestamp": m.horodatage.isoformat(),
                    "meta": {"intent": m.intention, "lang": m.langue}
                } for m in messages
            ]
        }

@app.delete("/conversations/{cid}")
async def delete_conversation(cid: str, current_user: User = Depends(verifier_token)):
    async with AsyncSessionLocal() as session:
        q = delete(Message).where(Message.utilisateur_id == current_user.id, Message.conversation_id == cid)
        await session.execute(q)
        await session.commit()
    return {"status": "success", "message": "Conversation supprimée"}

# ── CŒUR IA : STREAMING SSE ────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: Request, current_user: User = Depends(verifier_token)):
    """Traitement par streaming avec sauvegarde asynchrone MySQL."""
    try:
        body = await req.json()
    except:
        raise HTTPException(400, "JSON invalide")

    user_message = body.get("message", "").strip()
    conversation_id = body.get("conversation_id") or f"conv_{uuid.uuid4().hex[:10]}"
    
    if not user_message:
        raise HTTPException(400, "Message vide")

    async def event_generator():
        start_time = time.time()
        full_response = ""
        metadata = {}
        
        # 1. Récupération de l'historique pour le contexte
        historique_contexte = []
        async with AsyncSessionLocal() as session:
            h_query = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(desc(Message.horodatage))
                .limit(8)
            )
            h_res = await session.execute(h_query)
            for m in reversed(h_res.scalars().all()):
                historique_contexte.append({"role": m.role, "content": m.contenu})

        try:
            # 2. Appel au moteur IA (Async Generator)
            async for event in brain.traiter_message(user_message, historique=historique_contexte):
                # Format SSE : 'event: [name]\ndata: [json]\n\n'
                
                if event["type"] == "step":
                    yield f"event: step\ndata: {json.dumps({'label': event['label']})}\n\n"
                    await asyncio.sleep(0.01)

                elif event["type"] == "token":
                    content = event.get("reponse", "")
                    full_response += content
                    yield f"event: token\ndata: {json.dumps(content, ensure_ascii=False)}\n\n"

                elif event["type"] == "final":
                    metadata = {
                        "intention": event.get("intention"),
                        "langue": event.get("langue", "tamazight"),
                        "domaine": event.get("domaine", "general"),
                        "confiance": event.get("confiance", 0)
                    }
            
            # 3. Calcul du temps et envoi du signal de fin
            total_duration = int((time.time() - start_time) * 1000)
            yield f"event: done\ndata: {json.dumps({'time_ms': total_duration, **metadata})}\n\n"

            # 4. Sauvegarde asynchrone dans MySQL
            async with AsyncSessionLocal() as session:
                # Sauvegarde message utilisateur
                m_user = Message(
                    utilisateur_id=current_user.id,
                    conversation_id=conversation_id,
                    role="user",
                    contenu=user_message,
                    horodatage=datetime.utcnow()
                )
                # Sauvegarde réponse assistant
                m_bot = Message(
                    utilisateur_id=current_user.id,
                    conversation_id=conversation_id,
                    role="assistant",
                    contenu=full_response,
                    intention=metadata.get("intention"),
                    langue=metadata.get("langue"),
                    domaine=metadata.get("domaine"),
                    temps_traitement_ms=total_duration,
                    horodatage=datetime.utcnow() + timedelta(seconds=1)
                )
                session.add_all([m_user, m_bot])
                await session.commit()
                log.info(f"[CHAT] Sauvegardé : {conversation_id} | User: {current_user.nom}")

        except Exception as e:
            log.error(f"[STREAM ERROR] {e}")
            yield f"event: error\ndata: {json.dumps({'message': 'Erreur interne de traitement'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════
# GESTION DU PORT ET LANCEMENT (RAILWAY FRIENDLY)
# #══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════#══════

if __name__ == "__main__":
    import uvicorn
    
    # Railway utilise la variable d'environnement PORT
    # Si elle n'existe pas, on utilise 8080 par défaut
    port_env = os.environ.get("PORT", "8080")
    try:
        target_port = int(port_env)
    except ValueError:
        target_port = 8080

    log.info(f"🚀 Lancement du serveur sur le port {target_port}...")
    
    # Configuration Uvicorn optimisée
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=target_port,
        reload=False,     # Désactiver reload en production
        workers=1,         # Railway Free/Hobby préfère 1 worker stable
        proxy_headers=True,
        forwarded_allow_ips="*"
    )
