# main.py — AWAL GPT — FastAPI + MySQL (Railway)
# ════════════════════════════════════════════════════════════════════════════
# Fixes appliqués :
#   1. MySQL (aiomysql) via Railway env vars auto-détectés
#   2. Pool de connexions persistant (pas de reconnexion à chaque requête)
#   3. Fallback SQLite si MySQL indisponible (dev local)
#   4. Chat SSE fonctionnel branché sur brain.traiter_message
#   5. Sauvegarde correcte de TOUS les messages en base
# ════════════════════════════════════════════════════════════════════════════

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    force=True
)
for _lib in ("httpx", "urllib3", "asyncio", "aiosqlite", "uvicorn", "rapidfuzz", "groq", "httpcore", "aiomysql"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
log = logging.getLogger("awalgpt")

import os
from dotenv import load_dotenv
load_dotenv()

import time
import json
import hashlib
import asyncio
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import brain

# ── Configuration ──────────────────────────────────────────────────────────
SECRET_JWT = os.getenv("JWT_SECRET", "awal_gpt_ultra_secure_secret_key_2026_long_enough")

# ── MySQL / SQLite Detection ───────────────────────────────────────────────
# Railway injecte automatiquement MYSQL_URL ou DATABASE_URL ou les vars séparées
def _get_mysql_config():
    """Extrait la config MySQL depuis les variables Railway."""
    # Option 1 : URL complète (MYSQL_URL ou DATABASE_URL)
    for var in ("MYSQL_URL", "DATABASE_URL", "MYSQL_PUBLIC_URL", "MYSQL_PRIVATE_URL"):
        url = os.getenv(var, "")
        if url and ("mysql" in url.lower() or "mariadb" in url.lower()):
            parsed = urlparse(url)
            return {
                "host":   parsed.hostname,
                "port":   parsed.port or 3306,
                "user":   parsed.username,
                "password": parsed.password,
                "db":     parsed.path.lstrip("/"),
            }

    # Option 2 : Variables séparées Railway
    host = os.getenv("MYSQLHOST") or os.getenv("MYSQL_HOST")
    user = os.getenv("MYSQLUSER") or os.getenv("MYSQL_USER")
    pwd  = os.getenv("MYSQLPASSWORD") or os.getenv("MYSQL_PASSWORD")
    db   = os.getenv("MYSQLDATABASE") or os.getenv("MYSQL_DATABASE")
    port = int(os.getenv("MYSQLPORT") or os.getenv("MYSQL_PORT") or 3306)

    if host and user and db:
        return {"host": host, "port": port, "user": user, "password": pwd or "", "db": db}

    return None

MYSQL_CONFIG = _get_mysql_config()
USE_MYSQL = MYSQL_CONFIG is not None

if USE_MYSQL:
    log.info(f"[DB] MySQL détecté → {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['db']}")
else:
    log.warning("[DB] MySQL non configuré → Fallback SQLite (awalgpt.db)")
    import aiosqlite
    CHEMIN_BD = os.getenv("DB_PATH", "awalgpt.db")

# Pool global MySQL
_pool = None

# ── Init Base de Données ──────────────────────────────────────────────────

async def init_mysql_pool():
    global _pool
    import aiomysql
    _pool = await aiomysql.create_pool(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["user"],
        password=MYSQL_CONFIG["password"],
        db=MYSQL_CONFIG["db"],
        autocommit=True,
        minsize=2,
        maxsize=10,
        charset="utf8mb4",
        connect_timeout=10,
    )
    log.info("[DB] Pool MySQL créé")


async def initialiser_base():
    """Crée les tables si elles n'existent pas."""
    if USE_MYSQL:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS utilisateurs (
                        id VARCHAR(64) PRIMARY KEY,
                        nom VARCHAR(100),
                        email VARCHAR(200) UNIQUE,
                        mot_de_passe VARCHAR(200),
                        variete VARCHAR(50),
                        cree_le DATETIME DEFAULT CURRENT_TIMESTAMP
                    ) CHARACTER SET utf8mb4
                """)
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        utilisateur_id VARCHAR(64),
                        conversation_id VARCHAR(100),
                        message_utilisateur TEXT,
                        message_bot TEXT,
                        domaine VARCHAR(50),
                        intention VARCHAR(50),
                        langue VARCHAR(30),
                        temps_ms INT,
                        horodatage DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user (utilisateur_id),
                        INDEX idx_conv (conversation_id)
                    ) CHARACTER SET utf8mb4
                """)
        log.info("[DB] Tables MySQL vérifiées/créées")
    else:
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
        log.info("[DB] Tables SQLite vérifiées/créées")


# ── Helpers DB ────────────────────────────────────────────────────────────

async def db_fetch_one(query: str, params: tuple):
    """Exécute une requête SELECT et retourne une ligne (dict ou None)."""
    if USE_MYSQL:
        import aiomysql
        async with _pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params)
                return await cur.fetchone()
    else:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            bd.row_factory = aiosqlite.Row
            cur = await bd.execute(query, params)
            row = await cur.fetchone()
            return dict(row) if row else None


async def db_fetch_all(query: str, params: tuple):
    """Exécute une requête SELECT et retourne toutes les lignes."""
    if USE_MYSQL:
        import aiomysql
        async with _pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(query, params)
                return await cur.fetchall()
    else:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            bd.row_factory = aiosqlite.Row
            cur = await bd.execute(query, params)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def db_execute(query: str, params: tuple):
    """Exécute INSERT / UPDATE / DELETE."""
    if USE_MYSQL:
        async with _pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
    else:
        async with aiosqlite.connect(CHEMIN_BD) as bd:
            await bd.execute(query, params)
            await bd.commit()


# ── Requêtes métier ────────────────────────────────────────────────────────

async def bd_trouver_utilisateur(uid: str):
    return await db_fetch_one("SELECT * FROM utilisateurs WHERE id = %s" if USE_MYSQL else
                               "SELECT * FROM utilisateurs WHERE id = ?", (uid,))


async def bd_trouver_utilisateur_email(email: str):
    return await db_fetch_one(
        "SELECT * FROM utilisateurs WHERE email = %s" if USE_MYSQL else
        "SELECT * FROM utilisateurs WHERE email = ?", (email,))


async def bd_creer_utilisateur(uid, nom, email, hache, variete=None):
    ph = "%s" if USE_MYSQL else "?"
    await db_execute(
        f"INSERT INTO utilisateurs (id, nom, email, mot_de_passe, variete) VALUES ({ph},{ph},{ph},{ph},{ph})",
        (uid, nom, email, hache, variete)
    )


async def bd_historique(uid: str, conv_id: str) -> list:
    ph = "%s" if USE_MYSQL else "?"
    rows = await db_fetch_all(
        f"SELECT message_utilisateur, message_bot FROM messages "
        f"WHERE utilisateur_id={ph} AND conversation_id={ph} ORDER BY id DESC LIMIT 10",
        (uid, conv_id)
    )
    hist = []
    for r in reversed(rows):
        if r["message_utilisateur"]:
            hist.append({"role": "user",      "content": r["message_utilisateur"]})
        if r["message_bot"]:
            hist.append({"role": "assistant", "content": r["message_bot"]})
    return hist


async def bd_inserer_message(uid, cid, msg_user, msg_bot, domaine, intention, langue, temps_ms):
    ph = "%s" if USE_MYSQL else "?"
    now = datetime.now().isoformat()
    await db_execute(
        f"INSERT INTO messages (utilisateur_id, conversation_id, message_utilisateur, "
        f"message_bot, domaine, intention, langue, temps_ms, horodatage) "
        f"VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})",
        (uid, cid, msg_user, msg_bot, domaine, intention, langue, temps_ms, now)
    )


async def bd_lister_conversations(uid: str) -> list:
    ph = "%s" if USE_MYSQL else "?"
    rows = await db_fetch_all(
        f"SELECT conversation_id, MAX(horodatage) as d, "
        f"MIN(message_utilisateur) as t, COUNT(*) as nb "
        f"FROM messages WHERE utilisateur_id={ph} "
        f"GROUP BY conversation_id ORDER BY d DESC",
        (uid,)
    )
    return [
        {
            "id":         r["conversation_id"],
            "title":      (r["t"] or "Nouvelle discussion")[:30],
            "updated_at": str(r["d"]),
            "count":      r["nb"],
        }
        for r in rows
    ]


# ── JWT ──────────────────────────────────────────────────────────────────

def creer_token(uid: str) -> str:
    return jwt.encode(
        {"sub": uid, "exp": datetime.now(timezone.utc) + timedelta(days=30)},
        SECRET_JWT, algorithm="HS256"
    )


async def obtenir_utilisateur(requete: Request) -> dict:
    token = requete.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(401, "Token manquant")
    try:
        payload = jwt.decode(token, SECRET_JWT, algorithms=["HS256"])
        u = await bd_trouver_utilisateur(payload["sub"])
        if u:
            return u
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expiré")
    except Exception:
        pass
    raise HTTPException(401, "Token invalide")


# ── Lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[STARTUP] Initialisation Awal GPT...")
    if USE_MYSQL:
        await init_mysql_pool()
    await initialiser_base()
    brain.charger_configs()
    log.info("[STARTUP] Awal GPT prêt ✓")
    yield
    if USE_MYSQL and _pool:
        _pool.close()
        await _pool.wait_closed()
    log.info("[SHUTDOWN] Awal GPT arrêté.")


# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Awal GPT API", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ── Routes de base ────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"name": "Awal GPT API", "status": "online", "version": "4.0.0",
            "db": "mysql" if USE_MYSQL else "sqlite"}


@app.get("/health")
async def health():
    return {"status": "healthy", "db": "mysql" if USE_MYSQL else "sqlite",
            "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Auth ──────────────────────────────────────────────────────────────────

@app.post("/auth/register")
async def inscrire(req: Request):
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
    log.info(f"[AUTH] Inscription : {nom} ({email})")
    return {"token": creer_token(uid), "user": {"id": uid, "username": nom}}


@app.post("/auth/login")
async def connexion(req: Request):
    try:
        corps = await req.json()
    except Exception:
        raise HTTPException(400, "Corps JSON invalide")

    email = corps.get("email", "").lower().strip()
    mdp   = corps.get("password", "")
    uid   = hashlib.sha256(email.encode()).hexdigest()

    u = await bd_trouver_utilisateur(uid)
    if u and bcrypt.checkpw(mdp.encode(), u["mot_de_passe"].encode()):
        log.info(f"[AUTH] Connexion : {u['nom']}")
        return {"token": creer_token(u["id"]), "user": {"id": u["id"], "username": u["nom"]}}
    raise HTTPException(401, "Email ou mot de passe incorrect")


@app.get("/auth/me")
async def profil_utilisateur(u=Depends(obtenir_utilisateur)):
    return {"id": u["id"], "username": u["nom"], "email": u["email"]}


# ── Admin ─────────────────────────────────────────────────────────────────

@app.get("/admin/users")
async def admin_lister(key: str = None):
    if key != os.getenv("ADMIN_KEY", "awal_debug_2026"):
        raise HTTPException(403, "Accès refusé")
    ph = "%s" if USE_MYSQL else "?"
    rows = await db_fetch_all(
        f"SELECT id, nom, email, cree_le FROM utilisateurs", ()
    )
    return {"total": len(rows), "utilisateurs": rows}


# ── Chat SSE ──────────────────────────────────────────────────────────────

@app.post("/chat/stream")
async def chat_stream(req: Request, u=Depends(obtenir_utilisateur)):
    try:
        corps = await req.json()
    except Exception:
        raise HTTPException(400, "Corps JSON invalide")

    message = corps.get("message", "").strip()
    if not message:
        raise HTTPException(400, "Message vide")

    cid = corps.get("conversation_id") or f"conv_{int(time.time())}"
    log.info(f"[CHAT] {u['nom']} | CID={cid} | msg='{message[:50]}'")

    async def generer():
        debut = time.time()
        final_response = ""
        final_meta = {}
        done_sent = False

        try:
            historique = await bd_historique(u["id"], cid)

            async for event in brain.traiter_message(message, historique=historique):
                if event["type"] == "step":
                    yield f"event: step\ndata: {json.dumps({'step': event['step'], 'label': event['label']}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.03)

                elif event["type"] == "token":
                    yield f"event: token\ndata: {json.dumps(event.get('token', ''), ensure_ascii=False)}\n\n"

                elif event["type"] == "final":
                    final_response = event.get("reponse", "")
                    # Envoie la réponse complète comme token final
                    yield f"event: token\ndata: {json.dumps(final_response, ensure_ascii=False)}\n\n"

                    final_meta = {
                        "conversation_id": cid,
                        "intention":       event.get("intention", "general"),
                        "confiance":       event.get("confiance", 0),
                        "langue":          event.get("langue", "tamazight"),
                        "temps_rag_ms":    event.get("temps_rag_ms", 0),
                        "temps_llm_ms":    event.get("temps_llm_ms", 0),
                        "temps_total":     int((time.time() - debut) * 1000),
                    }

            if final_meta:
                yield f"event: done\ndata: {json.dumps(final_meta, ensure_ascii=False)}\n\n"
                done_sent = True

                # ✅ Sauvegarde en base (MySQL ou SQLite)
                await bd_inserer_message(
                    uid=u["id"],
                    cid=cid,
                    msg_user=message,
                    msg_bot=final_response,
                    domaine=corps.get("domain", "general"),
                    intention=final_meta["intention"],
                    langue=final_meta["langue"],
                    temps_ms=final_meta["temps_total"],
                )
                log.info(f"[CHAT] ✓ Sauvegardé | intention={final_meta['intention']} | {final_meta['temps_total']}ms")

        except Exception as err:
            log.error(f"[CHAT] Erreur stream: {err}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': str(err)}, ensure_ascii=False)}\n\n"
        finally:
            if not done_sent:
                yield f"event: done\ndata: {json.dumps({'error': True, 'temps_total': int((time.time()-debut)*1000)})}\n\n"

    return StreamingResponse(
        generer(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        }
    )


# ── Conversations ─────────────────────────────────────────────────────────

@app.get("/conversations")
async def lister_conversations(u=Depends(obtenir_utilisateur)):
    return {"conversations": await bd_lister_conversations(u["id"])}


@app.get("/conversations/{cid}/messages")
async def obtenir_messages(cid: str, u=Depends(obtenir_utilisateur)):
    ph = "%s" if USE_MYSQL else "?"
    rows = await db_fetch_all(
        f"SELECT message_utilisateur, message_bot, horodatage FROM messages "
        f"WHERE utilisateur_id={ph} AND conversation_id={ph} ORDER BY id ASC",
        (u["id"], cid)
    )
    msgs = []
    for r in rows:
        msgs.append({"sender": "user", "content": r["message_utilisateur"], "date": str(r["horodatage"])})
        msgs.append({"sender": "bot",  "content": r["message_bot"],          "date": str(r["horodatage"])})
    return {"messages": msgs}


@app.delete("/conversations/{cid}")
async def supprimer_conversation(cid: str, u=Depends(obtenir_utilisateur)):
    ph = "%s" if USE_MYSQL else "?"
    await db_execute(
        f"DELETE FROM messages WHERE utilisateur_id={ph} AND conversation_id={ph}",
        (u["id"], cid)
    )
    return {"status": "ok"}


# ── Entrée ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
