import os
import logging
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import (
    Column, Integer, String, Text, ForeignKey, 
    DateTime, Boolean, select, delete, update, func
)

from pydantic import BaseModel, EmailStr, Field
from jose import JWTError, jwt
import bcrypt
from dotenv import load_dotenv

# --- Import AWAL GPT Brain Engine ---
try:
    import brain
except ImportError:
    # Fallback for development if brain.py is missing
    class MockBrain:
        def generate_response(self, text):
            yield "Azul! (Service brain non détecté - Mode test)"
    brain = MockBrain()

load_dotenv()

# ==========================================
# 1. CONFIGURATION & LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("AWAL-GPT-PRO")

app = FastAPI(
    title="AWAL GPT - API Professionnelle",
    description="Backend API pour le système Amazigh NLP - Projet PFE 2026",
    version="2.5.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# ==========================================
# 2. SECURITY & JWT CONFIG
# ==========================================
SECRET_KEY = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_2026_khénifra_est")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 Jours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# ==========================================
# 3. Pydantic SCHEMAS (Validation)
# ==========================================
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    password: Optional[str] = None

class UserOut(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None

class ConversationUpdate(BaseModel):
    title: str

# ==========================================
# 4. DATABASE MODELS (SQLAlchemy)
# ==========================================
Base = declarative_base()

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
    title = Column(String(255), default="Nouvelle discussion Amazigh")
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String(100), ForeignKey("conversations.id"))
    role = Column(String(20))  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")

# ==========================================
# 5. DB CONNECTION ENGINE
# ==========================================
raw_url = os.getenv("MYSQL_URL")
if raw_url and raw_url.startswith("mysql://"):
    DATABASE_URL = raw_url.replace("mysql://", "mysql+aiomysql://")
else:
    # Fallback to SQLite for local safety
    DATABASE_URL = "sqlite+aiosqlite:///./pfe_backup.db"

engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    pool_size=10, 
    max_overflow=20,
    pool_pre_ping=True
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ==========================================
# 6. MIDDLEWARES & CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 7. UTILITIES (Auth & Helpers)
# ==========================================
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

def get_hashed_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        log.error(f"Erreur de vérification password: {e}")
        return False

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    result = await db.execute(select(User).filter(User.username == username))
    user = result.scalars().first()
    if user is None:
        raise credentials_exception
    return user

# ==========================================
# 8. STARTUP & LIFECYCLE
# ==========================================
@app.on_event("startup")
async def on_startup():
    log.info("Démarrage de AWAL GPT API...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Base de données synchronisée sur Railway.")

# ==========================================
# 9. ENDPOINTS: AUTHENTICATION
# ==========================================

@app.post("/auth/register", status_code=201)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check for existing user
    stmt = select(User).filter((User.email == user_data.email) | (User.username == user_data.username))
    existing = await db.execute(stmt)
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Nom d'utilisateur ou email déjà utilisé.")
    
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=get_hashed_password(user_data.password)
    )
    db.add(new_user)
    try:
        await db.commit()
        await db.refresh(new_user)
        log.info(f"Nouvel utilisateur enregistré: {new_user.username}")
        return {"message": "Utilisateur créé avec succès", "id": new_user.id}
    except Exception as e:
        await db.rollback()
        log.error(f"Erreur lors de l'inscription: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.post("/auth/login")
async def login(login_data: UserLogin, db: AsyncSession = Depends(get_db)):
    stmt = select(User).filter(User.username == login_data.username)
    result = await db.execute(stmt)
    user = result.scalars().first()
    
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Identifiants invalides.")
    
    token = create_access_token(data={"sub": user.username})
    return {
        "access_token": token, 
        "token_type": "bearer", 
        "username": user.username,
        "email": user.email
    }

@app.get("/auth/me", response_model=UserOut)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.patch("/auth/update")
async def update_profile(
    update_data: UserUpdate, 
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    if update_data.email:
        current_user.email = update_data.email
    if update_data.password:
        current_user.hashed_password = get_hashed_password(update_data.password)
    
    await db.commit()
    return {"message": "Profil mis à jour"}

# ==========================================
# 10. ENDPOINTS: CONVERSATION MANAGEMENT
# ==========================================

@app.get("/conversations", response_model=List[Dict[str, Any]])
async def list_user_conversations(
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    stmt = select(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.last_updated.desc())
    
    result = await db.execute(stmt)
    convs = result.scalars().all()
    return [
        {
            "id": c.id, 
            "title": c.title, 
            "last_updated": c.last_updated.isoformat()
        } for c in convs
    ]

@app.post("/conversations")
async def start_new_conversation(
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    new_conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=current_user.id, 
        title="Discussion Amazigh - " + datetime.now().strftime("%H:%M")
    )
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    return {"id": new_conv.id, "title": new_conv.title}

@app.get("/conversations/{conv_id}/messages")
async def get_conversation_history(
    conv_id: str, 
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    # Security check: Does this conv belong to user?
    stmt_check = select(Conversation).filter(
        Conversation.id == conv_id, 
        Conversation.user_id == current_user.id
    )
    check = await db.execute(stmt_check)
    if not check.scalars().first():
        raise HTTPException(status_code=403, detail="Accès refusé.")

    stmt = select(Message).filter(
        Message.conversation_id == conv_id
    ).order_by(Message.timestamp.asc())
    
    result = await db.execute(stmt)
    messages = result.scalars().all()
    return [
        {
            "role": m.role, 
            "content": m.content, 
            "timestamp": m.timestamp.isoformat()
        } for m in messages
    ]

@app.put("/conversations/{conv_id}")
async def rename_conversation(
    conv_id: str, 
    update_data: ConversationUpdate,
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    stmt = update(Conversation).where(
        (Conversation.id == conv_id) & (Conversation.user_id == current_user.id)
    ).values(title=update_data.title)
    
    await db.execute(stmt)
    await db.commit()
    return {"status": "renamed"}

@app.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: str, 
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    stmt = delete(Conversation).where(
        (Conversation.id == conv_id) & (Conversation.user_id == current_user.id)
    )
    await db.execute(stmt)
    await db.commit()
    return {"message": "Conversation supprimée"}

# ==========================================
# 11. ENDPOINTS: CHAT ENGINE (STREAMING)
# ==========================================

@app.post("/chat/stream")
async def chat_with_awal_gpt(
    chat_data: ChatRequest, 
    current_user: User = Depends(get_current_user), 
    db: AsyncSession = Depends(get_db)
):
    user_msg = chat_data.message
    conv_id = chat_data.conversation_id
    
    # 1. Verification of conversation existence
    if conv_id:
        result = await db.execute(select(Conversation).filter(Conversation.id == conv_id))
        conv = result.scalars().first()
        if not conv:
             # Create one if ID doesn't exist
             conv = Conversation(id=conv_id, user_id=current_user.id)
             db.add(conv)
             await db.commit()
        
        # Save User Message
        db.add(Message(conversation_id=conv_id, role="user", content=user_msg))
        await db.commit()

    # 2. Generator for Event-Stream
    async def awal_response_generator():
        full_ai_response = ""
        try:
            # Connect to Brain Logic (brain.py)
            for chunk in brain.generate_response(user_msg):
                full_ai_response += chunk
                # SSE Format
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.01) # Small buffer for smoothness
            
            # Send Final Chunk
            yield f"data: {json.dumps({'done': True})}\n\n"

            # 3. Save AI response in background session
            if conv_id:
                async with AsyncSessionLocal() as bg_db:
                    ai_msg = Message(
                        conversation_id=conv_id, 
                        role="assistant", 
                        content=full_ai_response
                    )
                    bg_db.add(ai_msg)
                    # Update Title if it's the first message
                    count_stmt = select(func.count(Message.id)).where(Message.conversation_id == conv_id)
                    msg_count = (await bg_db.execute(count_stmt)).scalar()
                    if msg_count <= 2:
                        new_title = user_msg[:30] + "..."
                        await bg_db.execute(update(Conversation).where(Conversation.id == conv_id).values(title=new_title))
                    
                    await bg_db.commit()

        except Exception as e:
            log.error(f"Streaming Exception: {e}")
            yield f"data: {json.dumps({'error': 'Désolé, une erreur est survenue dans le moteur AWAL.'})}\n\n"

    return StreamingResponse(awal_response_generator(), media_type="text/event-stream")

# ==========================================
# 12. ENDPOINTS: SYSTEM & ANALYTICS
# ==========================================

@app.get("/system/stats")
async def get_app_stats(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # Stats for current user
    conv_count = await db.execute(select(func.count(Conversation.id)).where(Conversation.user_id == current_user.id))
    msg_count = await db.execute(select(func.count(Message.id)).join(Conversation).where(Conversation.user_id == current_user.id))
    
    return {
        "user": current_user.username,
        "total_conversations": conv_count.scalar(),
        "total_messages": msg_count.scalar(),
        "engine": "AWAL-NLP-v2",
        "status": "Online"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.5.0",
        "region": os.getenv("RAILWAY_ENVIRONMENT", "production")
    }

# ==========================================
# 13. ERROR HANDLERS
# ==========================================
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": True, "message": exc.detail},
    )

# ==========================================
# 14. SERVER EXECUTION
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # Logic for Dynamic Port on Railway
    server_port = int(os.getenv("PORT", 8080))
    log.info(f"Démarrage du serveur sur le port {server_port}")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=server_port, 
        reload=False, # Set False in production
        workers=4    # Multiprocessing for better performance
    )
