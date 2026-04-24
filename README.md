# ⴰⵡⴰⵍ — AWAL GPT v7.0 🤖

**AWAL GPT** is a high-performance, production-ready multilingual assistant specialized in the **Amazigh (Tamazight)** language and culture. It leverages cutting-edge LLMs and a custom RAG (Retrieval-Augmented Generation) pipeline to provide accurate, culturally-aware responses in Tamazight Latin script.

---

## 🚀 Key Features

*   **Multilingual Intelligence**: Native support for Tamazight (Latin & Tifinagh), French, Arabic (Darija/MSA), and English.
*   **Elite Tamazight Logic**: Advanced normalization, transliteration, and linguistic validation for high-quality Latin script output.
*   **Groq Acceleration**: Powered by **Groq's Llama 3.3 (70B)** for near-instantaneous response times (< 1s).
*   **Custom RAG Pipeline**: Local TF-IDF indexing for cultural facts, proverbs, and specialized domain vocabulary.
*   **Premium UI/UX**: Responsive React frontend with dark mode support, mobile-optimized sidebar overlay, and localized "Thinking" state animations.
*   **Secure API**: JWT-based authentication, rate-limiting, password hashing (Bcrypt), and MongoDB-backed conversation history.

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.13)
- **Database**: MongoDB (Motor async driver)
- **LLM**: Groq (Llama-3.3-70b-versatile)
- **ML Engine**: Scikit-Learn (Training) + NumPy (Optimized Runtime Cosine Similarity)
- **Security**: JWT, Bcrypt, Rate-Limiting Middleware

### Frontend
- **Framework**: React.js
- **Styling**: Vanilla CSS (Modern, Responsive Design)
- **Features**: "Thinking" state (Ar iswingim...), Sidebar Overlay for mobile, Multi-domain support.

---

## 🧠 System Architecture

The AWAL GPT pipeline follows a 5-stage transformation:

1.  **Normalization**: Cleans input text, handles numeric transliteration (e.g., `7` → `h`, `9` → `q`), and strips HTML/extra whitespace.
2.  **Intent Detection**: A dedicated ML model classifies the query (Salutation, Question, Translation, Culture, etc.).
3.  **Semantic RAG**:
    *   **TF-IDF Search**: Finds relevant cultural context or proverbs.
    *   **Multi-Lang Search**: Cross-references EN/FR/AR semantics to find verified Tamazight equivalents.
4.  **LLM Synthesis**: Generates a structured response via Groq with cross-lingual semantic anchors.
5.  **Post-Processing**: Final script validation and "salvage" logic to ensure the output is pure Tamazight Latin.

---

## ⚙️ Setup & Installation

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_key
MONGO_URI=mongodb://localhost:27017
JWT_SECRET=your_super_secret_key
ADMIN_TOKEN=your_admin_token
```

Initialize the indices and intent model:
```bash
python train.py
```

Run the server:
```bash
uvicorn main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm start
```

---

## 📡 API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/auth/register` | `POST` | Create a new user account |
| `/auth/login` | `POST` | Authenticate and receive JWT |
| `/chat` | `POST` | Send message and get AI response |
| `/conversations` | `GET` | List user's conversation history |
| `/health` | `GET` | System health and index status |

---

## 🌍 Language Support
- **Input**: Any length in Tamazight (Latin/Tifinagh), French, Arabic, English.
- **Output**:
    - `[TAM]` — Optimized Tamazight Latin.
    - `[FR]` — Corresponding French translation.
    - `[AR]` — Arabic equivalent.
    - `[EN]` — English context.

---

[TAM]
**AWAL GPT d asghiwel n tutlayt tamazight s tarrayt n AI tameqqrant.**
1. **Architecture**: N-isisfiw Stage-by-Stage pipeline i-xd-amn deg backend.
2. **Setup**: N-rni iwaliwen iwakken setup n project ad i-li d professional s 3 n steps.
3. **Optimized**: N-ssmras Groq d NumPy iwakken tazzla ad t-ili d tasfayt ikemlen.
