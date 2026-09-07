import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
GEMINI_GEN_MODEL = os.getenv("GEMINI_GEN_MODEL") or "gemini-2.0-flash"
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL") or "gemini-embedding-001"

CHROMA_HOST = os.getenv("CHROMA_HOST") or "localhost"
CHROMA_PORT = int(os.getenv("CHROMA_PORT") or "8000")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION") or "research_papers"

REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
RQ_QUEUE = os.getenv("RQ_QUEUE") or "ingestion"

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB") or "ai_scholar"

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

UPLOAD_DIR = os.getenv("UPLOAD_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
)
DATA_DIR = os.getenv("DATA_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

RERANKER_ENABLED = (os.getenv("RERANKER_ENABLED") or "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL") or "BAAI/bge-reranker-base"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if GEMINI_KEY and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

ai = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
