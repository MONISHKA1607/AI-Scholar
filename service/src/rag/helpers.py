import io, requests, PyPDF2
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import faiss

from ..config import GEMINI_EMBED_MODEL  # noqa: F401  # ensures GOOGLE_API_KEY is set via config import

def extract_pdf(pdf_url: str, max_pages: int = 10) -> FAISS:
    try:    
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Create BytesIO object
        pdf_io = io.BytesIO(response.content)

        # 2. Extract per page with PyPDF2
        reader = PyPDF2.PdfReader(pdf_io)
        pages = reader.pages
        total_pages = len(pages)
        pages_to_read = min(total_pages, max_pages)

        page_docs: List[Document] = []
        for i in range(pages_to_read):
            text = pages[i].extract_text() or ""
            page_docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_url,
                    "page": i,  # 0-indexed
                    "total_pages": total_pages,
                }
            ))

        # 3. Chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,  # char offset in original page
        )
        chunks = splitter.split_documents(page_docs)
    
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_idx"] = idx
    
        embeddings = GoogleGenerativeAIEmbeddings(model=f"models/{GEMINI_EMBED_MODEL}")
        embedding_dim = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)
        
        vector_store = FAISS(
            embedding_function=embeddings,     # How to create vectors
            index=index,                      # Where to store vectors (needs to exist!)
            docstore=InMemoryDocstore(),      # Where to store documents
            index_to_docstore_id={},         # Mapping between them
        )

        vector_store.add_documents(chunks)

        return vector_store
    except Exception as e:
        return None