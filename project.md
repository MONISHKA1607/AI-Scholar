Problem definition:
Researchers across disciplines need quick, citation-backed answers to narrow questions without skimming entire papers. General-purpose chatbots hallucinate, mobile teams lack a shared workspace with grounded generation, and evidence trails rarely persist once the session ends.

Goal:
Ship a cross-platform (iOS/Android) research assistant that lets authenticated users chat, speak, and listen to grounded answers drawn from a curated scholarly knowledge base. Every answer must be sourced from retrieved context; otherwise the assistant must decline with a clear “cannot answer” fallback.

Solution snapshot:
React Native client → Express API (auth + history) → FastAPI RAG microservice → ChromaDB + Gemini 2.0 Flash. The chain streams responses over Server-Sent Events (SSE), stores conversations in MongoDB, and supports both speech-to-text queries and text-to-speech playback.

RAG pipeline (FastAPI):
1. Classify the incoming question with `gemini-2.5-flash` into **specific-to-paper**, **generic-research**, or **non-research**.  
2. Retrieve context from the `research_papers` ChromaDB collection via `kb_retrieval` (embeddings: `gemini-embedding-001`).  
3. If context is missing, fall back to Semantic Scholar (search or paper autocomplete) and, when an open-access PDF is available, hydrate the first ~10 pages into an in-memory FAISS index for targeted similarity search.  
4. Assemble a strict system prompt that forbids fabrication and enforces the “Sorry but cannot answer your question at the moment” response when evidence is insufficient.  
5. Stream the final answer from `gemini-2.0-flash` back to Express, preserving chunk order and providing a final completion message.  
6. Return structured SSE payloads `{chunk, finished, chatId?, error?}` to the mobile client.

Ingestion & knowledge base:
- Persistent KB: external ChromaDB HTTP server (`localhost:8000`) seeded offline with de-duplicated research paper chunks (~1k tokens, 200 token overlap) and metadata (title, authors, venue, URL/DOI).  
- Embeddings: `GoogleGenerativeAiEmbeddingFunction` with `gemini-embedding-001`.  
- Live enrichment: the specific-paper tool downloads up to ten PDF pages via Semantic Scholar, chunks with `RecursiveCharacterTextSplitter`, embeds using LangChain FAISS, and answers directly from those slices.  
- MongoDB Atlas stores users, chat titles, and message history (user/assistant turns with timestamps).

Safety & fidelity:
- Classification short-circuits non-research prompts.  
- Generation prompt mandates markdown structure, inline evidence, and a fallback refusal when retrieval fails.  
- Express middleware verifies Firebase-issued JWTs before allowing chat access.  
- SSE layer propagates structured errors that the app surfaces to users.  
- Sensitive API keys (Gemini, Mongo, JWT secret) live in environment variables; hard-coded keys in development code must be rotated before production.

Technologies by component:
- Mobile app (`frontend/`): Expo React Native, Expo Router, Zustand, React Native Paper, `react-native-sse`, `@react-native-voice/voice` (speech input), `react-native-tts` (audio playback), Firebase Auth for OTP + Google sign-in, Axios for REST calls, AsyncStorage for JWT persistence.  
- Backend API (`backend/`): Express + TypeScript, Mongoose, JWT, CORS, dotenv. Endpoints: `/auth/signin`, `/chat/generate` (SSE proxy to FastAPI), `/chat/chats`, `/chat/:id`.  
- RAG microservice (`FastAPI/`): FastAPI, LangChain, Google `genai` SDK, Semantic Scholar REST, ChromaDB HTTP client, FAISS via LangChain, asyncio streaming.  
- Voice package (`voice/`): custom React Native module (Android/iOS) for low-latency speech capture integrated into the app.  
- Data stores: MongoDB Atlas, ChromaDB HTTP server, ephemeral FAISS vector stores.

End-to-end interaction flow:
1. User logs in via Firebase (phone OTP or Google). Express issues a JWT stored in AsyncStorage.  
2. Mobile client opens an SSE connection to `/chat/generate`, sending the prompt and JWT.  
3. Express forwards the request to FastAPI and relays streamed chunks as they arrive.  
4. FastAPI classifies, retrieves, and generates grounded markdown using Gemini.  
5. SSE chunks update the UI in real time; completion triggers chat persistence in MongoDB and refreshes history in the sidebar.  
6. Users can copy responses, start/stop TTS playback, or revisit stored conversations.

Current gaps & roadmap notes:
- Chroma ingestion scripts live outside the repo; automation for PDF ingestion is planned.  
- Evaluation metrics (latency, retrieval hit rate) are not yet exposed, though SSE payloads carry space for diagnostics.  
- Multilingual retrieval relies on Gemini embeddings but UI translations are not implemented.  
- Model comparisons (multi-LLM) are out of scope for the current implementation.

Outputs surfaced to users:
- Markdown-formatted assistant response (<300 words) with inline citations or explicit fallback.  
- Chat history (latest 10 chats + full conversation on selection).  
- Optional audio playback per message, plus speech transcript capture on new prompts.  
- Error toasts when FastAPI raises retrieval or generation failures.

Supporting references & APIs:
- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP,” NeurIPS 2020.  
- Karpukhin et al., “Dense Passage Retrieval,” EMNLP 2020.  
- Semantic Scholar Graph API documentation (search + autocomplete).  
- Google Gemini 2.0 developer guide for streaming generation and embeddings.

Added features & improvements (v2 vs. v1):
- End-to-end RAG service: new FastAPI pipeline with Gemini-powered query classification, ChromaDB retrieval, Semantic Scholar fallbacks, and streaming generation that did not exist in v1.  
- Real-time chat streaming & persistence: Express now proxies SSE responses, saves each turn in MongoDB, and exposes `/chat/chats` + `/chat/:id` for history replay.  
- Voice-first interaction suite: integrated native voice capture (`@react-native-voice/voice`), custom `voice/` module, and `react-native-tts` playback controls for hands-free use.  
- Secure authentication flow: Firebase phone OTP + Google sign-in wired into backend JWT issuance, replacing the unauthenticated v1 chat experience.  
- Mobile UX overhaul: new drawer/tab navigation, polished chat window with markdown renderer, loaders, dark mode, scrolling fixes, and copy/playback controls.  
- Operational hardening: structured error handling over SSE, stricter prompt safety (“cannot answer” fallback), dotenv-based secrets, and modularized codebase (frontend/backend/FastAPI voice packages).

