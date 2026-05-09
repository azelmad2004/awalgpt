import os
import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, selectinload
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, select, delete

from pydantic import BaseModel, EmailStr
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

# --- استيراد محرك الذكاء الاصطناعي (AWAL GPT Brain) ---
import brain 

load_dotenv()

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("awalgpt-api")

app = FastAPI(
    title="AWAL GPT API",
    description="API pour le chatbot Amazigh AWAL GPT",
    version="2.0.0"
)

# --- Security & JWT Logic ---
SECRET_KEY = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_secret_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 Semaine

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- CORS CONFIGURATION (Updated for Railway) ---
origins = [
    "https://awal-gpt.up.railway.app",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE CONFIGURATION (SQLite to MySQL Transition) ---
# OLD SQLITE CODE (Commented as requested):
# DATABASE_URL = "sqlite+aiosqlite:///./awalgpt.db"
# engine = create_async_engine(DATABASE_URL)

# NEW MYSQL CODE FOR RAILWAY:
raw_url = os.getenv("MYSQL_URL")
if raw_url and raw_url.startswith("mysql://"):
    DATABASE_URL = raw_url.replace("mysql://", "mysql+aiomysql://")
else:
    DATABASE_URL = raw_url

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# --- DATABASE MODELS (Full Schema) ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String(100), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(255), default="Nouvelle conversation")
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), ForeignKey("conversations.id"))
    role = Column(String(20)) # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# --- UTILS & DEPENDENCIES ---

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_hashed_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    result = await db.execute(select(User).filter(User.username == username))
    user = result.scalars().first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# --- STARTUP: CREATE TABLES ---
@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("MySQL Database Initialized.")

# --- ENDPOINTS: AUTHENTICATION ---

@app.post("/auth/register")
async def register(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    # Check if user exists
    stmt = select(User).filter((User.email == data['email']) | (User.username == data['username']))
    existing_user = (await db.execute(stmt)).scalars().first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    new_user = User(
        username=data['username'],
        email=data['email'],
        hashed_password=get_hashed_password(data['password'])
    )
    db.add(new_user)
    await db.commit()
    return {"message": "Utilisateur créé avec succès"}

@app.post("/auth/login")
async def login(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    stmt = select(User).filter(User.username == data['username'])
    user = (await db.execute(stmt)).scalars().first()
    
    if not user or not verify_password(data['password'], user.hashed_password):
        raise HTTPException(status_code=401, detail="Identifiants incorrects")
    
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "username": user.username}

# --- ENDPOINTS: CONVERSATION HISTORY (HISTORIQUE) ---

@app.get("/conversations")
async def list_conversations(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    stmt = select(Conversation).filter(Conversation.user_id == current_user.id).order_by(Conversation.last_updated.desc())
    result = await db.execute(stmt)
    convs = result.scalars().all()
    return [{"id": c.id, "title": c.title, "last_updated": c.last_updated} for c in convs]

@app.post("/conversations")
async def create_conversation(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    new_conv = Conversation(user_id=current_user.id, title="Nouvelle discussion")
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return {"id": new_conv.id, "title": new_conv.title}

@app.get("/conversations/{conv_id}")
async def get_messages(conv_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    stmt = select(Message).filter(Message.conversation_id == conv_id).order_by(Message.timestamp.asc())
    result = await db.execute(stmt)
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in messages]

@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Conversation).filter(Conversation.id == conv_id, Conversation.user_id == current_user.id))
    await db.commit()
    return {"message": "Supprimé"}

# --- ENDPOINTS: CHAT ENGINE & STREAMING ---

@app.post("/chat/stream")
async def chat_stream(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    user_msg = data.get("message")
    conv_id = data.get("conversation_id")
    
    # Logic pour sauvegarder le message utilisateur (Historique)
    if conv_id:
        new_msg = Message(conversation_id=conv_id, role="user", content=user_msg)
        db.add(new_msg)
        await db.commit()

    async def event_generator():
        full_response = ""
        try:
            # Appel au cerveau de l'IA (AWAL)
            for chunk in brain.generate_response(user_msg):
                full_response += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            # Sauvegarder la réponse de l'IA dans l'historique à la fin
            if conv_id:
                async with AsyncSessionLocal() as background_db:
                    ai_msg = Message(conversation_id=conv_id, role="assistant", content=full_response)
                    background_db.add(ai_msg)
                    await background_db.commit()
                    
        except Exception as e:
            log.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': 'Erreur de génération'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- SERVER EXECUTION ---
if __name__ == "__main__":
    import uvicorn
    # Railway PORT logic
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
