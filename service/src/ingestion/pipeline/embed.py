"""Embedding helpers. Chroma's GoogleGenerativeAiEmbeddingFunction handles
batching internally when documents are passed to collection.add. This module
exposes the shared embedding function and a direct embed helper for callers
that need vectors outside of Chroma (e.g., the reranker base layer).
"""
import chromadb.utils.embedding_functions as embedding_functions

from ...config import GEMINI_KEY, GEMINI_EMBED_MODEL


_embedding_fn = None


def get_embedding_function():
    """Returns a Chroma-compatible Google embedding function (singleton)."""
    global _embedding_fn
    if _embedding_fn is None:
        if not GEMINI_KEY:
            raise RuntimeError(
                "GEMINI_KEY is not set; cannot construct embedding function."
            )
        _embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GEMINI_KEY,
            model_name=GEMINI_EMBED_MODEL,
        )
    return _embedding_fn


def embed_texts(texts):
    """Embed a list of texts; returns a list of vectors."""
    fn = get_embedding_function()
    return fn(texts)
