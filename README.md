# AI-Scholar: A RAG Framework for Research Paper Question Answering
 - Presentation/Demo:- https://drive.google.com/file/d/1SgHN49c0P5W3EmVsCPcTXfBjBo9Kb4b2/view

## Team Members:-
 - Adheil Gupta: 23BDS002
 - Arnav Gupta: 23BDS009
 - Atharva Agrawal: 23BDS010
 - Surya Narayana Rao: 23BDS025
 - Tejas Chalwadi: 23BDS063
  
## Overview

- React Native client → Express API (auth + history + SSE proxy) → FastAPI RAG microservice → ChromaDB + Gemini 2.0 Flash.
- Streaming over Server-Sent Events (SSE) with ordered chunks and a completion message.
- MongoDB stores users, chat titles, and message history.
- Supports both speech-to-text queries and text-to-speech playback.


## Contents

- **frontend/** – Expo React Native app (Android, iOS, Web)
- **backend/** – Express + TypeScript API (JWT auth, chat history, SSE proxy)
- **service/** – FastAPI RAG microservice (classification, retrieval, generation)
- **voice-module/** – Local React Native voice module consumed by the app


## Architecture

- **Authentication**: The app authenticates with Firebase (phone OTP or Google). The Express backend verifies the Firebase JWT and issues its own JWT for accessing `/chat/*` APIs.
- **Chat generation**: The mobile client opens an SSE connection to Express `/chat/generate`. Express proxies the request to FastAPI and relays streamed chunks back to the client, preserving order.
- **RAG**: FastAPI classifies the query, retrieves from ChromaDB, falls back to Semantic Scholar + in-memory FAISS hydration when needed, then streams grounded markdown from Gemini 2.0 Flash.
- **Persistence**: MongoDB Atlas stores users, chat titles, and per-turn message history.

Chain: React Native → Express (JWT + SSE) → FastAPI (RAG) → ChromaDB/Gemini → SSE back to client.


## End-to-End Interaction Flow

1. User logs in via Firebase (phone OTP or Google). Express issues a JWT stored in AsyncStorage.
2. Mobile client opens an SSE connection to `/chat/generate`, sending the prompt and JWT.
3. Express forwards the request to FastAPI and relays streamed chunks as they arrive.
4. FastAPI classifies, retrieves, and generates grounded markdown using Gemini.
5. Completion triggers chat persistence in MongoDB and refreshes history in the sidebar.
6. Users can copy responses, start/stop TTS playback, or revisit stored conversations.


## RAG Pipeline (FastAPI)

1. **Classify** the incoming question with `gemini-2.5-flash` into one of: specific-to-paper, generic-research, or non-research.
2. **Retrieve** context from the `research_papers` ChromaDB collection via LangChain using `gemini-embedding-001`.
3. **Fallback** when context is missing: query Semantic Scholar (search or paper autocomplete). If an open-access PDF is available, download the first ~10 pages, chunk with `RecursiveCharacterTextSplitter`, embed with LangChain FAISS, and run targeted similarity search.
4. **System prompt** strictly forbids fabrication and enforces the fallback: "Sorry but cannot answer your question at the moment" when evidence is insufficient.
5. **Stream generation** from `gemini-2.0-flash` back to Express, preserving chunk order and providing a final completion message.
6. **SSE payloads** to the mobile client use the shape: `{ chunk, finished, chatId?, error? }`.


## Ingestion & Knowledge Base

- **Persistent KB**: External ChromaDB HTTP server (`http://localhost:8000`) populated by the in-repo ingestion pipeline (`service/src/ingestion/`). Chunks are ~1k tokens with 200-token overlap; metadata includes title, authors, venue, year, categories, source, DOI/arxiv_id, and a `canonical_id` used for dedupe.
- **Embeddings**: GoogleGenerativeAiEmbeddingFunction with `gemini-embedding-001`.
- **Sources**: arXiv (Kaggle bootstrap + OAI-PMH delta), Semantic Scholar bulk search, and authenticated user PDF uploads from the app.
- **Workers**: RQ + Redis. A separate worker process consumes the `ingestion` queue and processes papers asynchronously; the SQLite ledger at `service/data/ingestion_ledger.sqlite` makes runs resumable.
- **Live Enrichment**: The specific-paper tool downloads up to 10 PDF pages via Semantic Scholar, chunks with `RecursiveCharacterTextSplitter`, embeds using LangChain FAISS, and answers directly from those slices.
- **MongoDB Atlas**: Stores users, chat titles, and message history (user/assistant turns with timestamps).

### Ingestion CLI

Run from `service/`:

```bash
# arXiv bootstrap from Kaggle JSONL dump
python -m src.ingestion.cli arxiv --bootstrap --kaggle path/to/arxiv-metadata.json.gz --limit 25000

# arXiv OAI-PMH delta sync
python -m src.ingestion.cli arxiv --oai --since 2025-01-01 --oai-set cs --limit 1000

# Semantic Scholar bulk search
python -m src.ingestion.cli s2 --query "transformer architecture" --limit 2000 --min-citations 5

# Single local PDF (smoke test, sync)
python -m src.ingestion.cli ingest-file ./paper.pdf

# RQ worker (in a separate process)
python -m src.ingestion.cli worker     # or: rq worker ingestion

# Ledger status
python -m src.ingestion.cli status
```

Add `--sync` to any source command to run inline (skip the queue) for smoke tests.


## Safety & Fidelity

- **Classification** short-circuits non-research prompts.
- **Generation prompt** mandates markdown structure, inline evidence, and a refusal fallback when retrieval fails.
- **Express middleware** verifies Firebase-issued JWTs before allowing chat access.
- **SSE layer** propagates structured errors that the app surfaces to users.
- **Secrets**: Gemini, Mongo, and JWT secrets live in environment variables; rotate any dev keys before production.


## Technologies by Component

- **Mobile app (`frontend/`)**: Expo React Native, Expo Router, Zustand, React Native Paper, `react-native-sse`, `@react-native-voice/voice` (speech input), `react-native-tts` (audio playback), Firebase Auth for OTP + Google sign-in, Axios for REST calls, AsyncStorage for JWT persistence.
- **Backend API (`backend/`)**: Express + TypeScript, Mongoose, JWT, CORS, dotenv. Endpoints: `/auth/signin`, `/chat/generate` (SSE proxy to FastAPI), `/chat/chats`, `/chat/:id`.
- **RAG microservice (`FastAPI/`)**: FastAPI, LangChain, Google genai SDK, Semantic Scholar REST, ChromaDB HTTP client, FAISS via LangChain, asyncio streaming.
- **Voice package (`voice/`)**: Custom React Native module (Android/iOS) for low-latency speech capture.
- **Data stores**: MongoDB Atlas, ChromaDB HTTP server, ephemeral FAISS vector stores.


## Environment Variables

### Backend (`backend/.env`)

- `MONGODB_CONNECTION_STRING` – MongoDB URI
- `JWT_SECRET` – Secret for signing server-issued JWTs
- `FASTAPI_BASE_URL` – e.g., `http://localhost:5432`

Example (`backend/.env`):

```
MONGODB_CONNECTION_STRING="mongodb+srv://user:pass@cluster/yourdb"
JWT_SECRET="change_me"
FASTAPI_BASE_URL="http://localhost:5432"
```

### Frontend (Expo)

- `EXPO_PUBLIC_BACKEND_URL` – e.g., `http://localhost:3000`
- `EXPO_PUBLIC_FIREBASE_APIKEY`
- `EXPO_PUBLIC_FIREBASE_AUTHDOMAIN`
- `EXPO_PUBLIC_FIREBASE_PROJECTID`
- `EXPO_PUBLIC_FIREBASE_STORAGEBUCKET`
- `EXPO_PUBLIC_FIREBASE_MESSAGINGSENDERID`
- `EXPO_PUBLIC_FIREBASE_APPID`
- `EXPO_PUBLIC_FIREBASE_MEASUREMENTID`

These map to `frontend/firebase-config.js` and API hooks (e.g., `${process.env.EXPO_PUBLIC_BACKEND_URL}/chat/chats`).

### FastAPI (`service/.env` — see `service/.env.example`)

- `GEMINI_KEY` – Gemini API key
- `GEMINI_MODEL` – classifier model (default `gemini-2.5-flash`)
- `GEMINI_GEN_MODEL` – generation model (default `gemini-2.0-flash`)
- `GEMINI_EMBED_MODEL` – embedding model (default `gemini-embedding-001`)
- `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION` – Chroma HTTP server
- `REDIS_URL`, `RQ_QUEUE` – ingestion queue
- `SEMANTIC_SCHOLAR_API_KEY` – optional (higher limits)
- `UPLOAD_DIR`, `DATA_DIR` – local storage paths (default under `service/data/`)
- `RERANKER_ENABLED` (default `true`), `RERANKER_MODEL` (default `BAAI/bge-reranker-base`)


## Setup & Run (Development)

1) Backend (Express)

```bash
cd backend
cp .env.example .env  # if present; otherwise create .env as above
npm install
npm run dev           # default port 3000
```

2) RAG Service (FastAPI)

```bash
cd service
python -m venv .venv && . .venv/bin/activate  # Unix/macOS: . .venv/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload --port 5432
```

3) ChromaDB (external server)

- Ensure a Chroma HTTP server runs at `http://localhost:8000` and the KB is seeded.

4) Frontend (Expo)

```bash
cd frontend
npm install
npx expo start
```

5) Voice module

- The app depends on the local tarball `../voice-module/react-native-voice-voice-3.2.4.tgz`. Ensure it exists and installs.


## API and Streaming

- **Express (`backend/`)**
  - `POST /api/v1/auth/login` → `{ token }`
  - `GET /api/v1/chat/chats` (Bearer token)
  - `POST /api/v1/chat/generate` (SSE proxy to FastAPI) → streams `{ chunk, finished }`
  - `POST /api/v1/chat/:cId` (Direct Gemini generation + Persistence) → streams `{ chunk, finished, chatId, title }`
  - `GET /api/v1/chat/:cId` → Fetch full chat history for a specific conversation

- **RAG Service (`service/`)**
  - `POST /fastapi/chat/generate` (internal) → streams model output consumed by Express
  - `POST /fastapi/ingest/upload` → accept a multipart PDF, enqueue ingestion, return `{ job_id, canonical_id }`
  - `GET  /fastapi/ingest/status/{job_id}` → RQ + ledger status for a queued ingestion
  - `GET  /fastapi/ingest/ledger` → recent ledger rows + counts
  - `GET  /fastapi/eval/run` → retrieval benchmark (hit@5, MRR@10, p50/p95 latency)

- **Backend additions for ingestion**
  - `POST /api/v1/papers/upload` (Bearer token, multipart PDF) → proxies to FastAPI ingest
  - `GET  /api/v1/papers/status/:jobId` → proxies to FastAPI ingest status

SSE payload envelope:

```json
{ "chunk": "text", "finished": false, "chatId": "optional", "error": null }
```


## Troubleshooting

- Android mic permissions: ensure they are declared and granted.
- React Native TTS: known issue/fix – https://github.com/ak1394/react-native-tts/pull/274
- Expo env: ensure `EXPO_PUBLIC_*` variables are present when running `expo start`.
- Default ports: Backend `3000`, FastAPI `5432`, Chroma `8000`.


## Current Gaps & Roadmap

- Hybrid retrieval (BM25 + vector) is intentionally deferred; reranker covers most of the gap until a corpus migration to a BM25-capable store.
- Full-text ingestion is capped at abstract + intro + first ~3 sections per paper to control embedding cost; expand once eval shows it's needed.
- Scanned PDFs are skipped (`status=skipped_ocr`); OCR is out of scope for v1.
- Multilingual retrieval relies on Gemini embeddings; UI translations not implemented.
- Model comparisons (multi-LLM) are out of scope for the current implementation.


## Outputs Surfaced to Users

- Markdown-formatted assistant response (<300 words) with inline citations or explicit fallback.
- Chat history (latest 10 chats + full conversation on selection).
- Optional audio playback per message, plus speech transcript capture on new prompts.
- Error toasts when FastAPI raises retrieval or generation failures.


## Repository Structure

```
AI-Scholar/
├─ backend/       # Express + TypeScript (Auth & Proxy)
├─ service/       # FastAPI + LangChain (RAG Pipeline)
├─ frontend/      # Expo React Native (Mobile App)
├─ voice-module/  # Native Voice Integration
└─ project.md     # Detailed Specs & Updates
```

## What's New in v2

Compared to v1, this release includes:
- **End-to-End RAG Service**: A new FastAPI pipeline with Gemini-powered query classification, ChromaDB retrieval, and Semantic Scholar fallbacks.
- **Real-time Chat Persistence**: Express now proxies SSE responses and saves each turn in MongoDB for history replay.
- **Voice-First Interaction**: Integrated native voice capture (`@react-native-voice/voice`) and TTS playback for hands-free use.
- **Secure Auth Flow**: Firebase phone OTP + Google sign-in wired into backend JWT issuance.
- **Mobile UX Overhaul**: Polished chat UI with markdown rendering, dark mode, and improved navigation (Drawer + Tabs).
