"""Cross-encoder reranker. Lazy-loaded singleton so service startup
doesn't pay the model-load cost or fail if sentence-transformers isn't installed.
"""
from typing import List, Tuple

from ..config import RERANKER_ENABLED, RERANKER_MODEL


_model = None
_load_failed = False


def _get_model():
    """Returns a CrossEncoder or None if loading fails / disabled."""
    global _model, _load_failed
    if not RERANKER_ENABLED or _load_failed:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(RERANKER_MODEL)
        return _model
    except Exception as e:
        # Bag the error once; never retry inside hot retrieval path.
        print(f"[rerank] reranker disabled (load failed): {e}")
        _load_failed = True
        return None


def rerank(
    query: str,
    candidates: List[dict],
    *,
    top_k: int = 5,
    text_key: str = "text",
) -> List[dict]:
    """Rerank a list of candidate chunks against the query.

    Each candidate is a dict; rerank scores are stored under 'rerank_score'.
    If the reranker is unavailable, candidates are returned unmodified (truncated to top_k).
    """
    if not candidates:
        return []
    model = _get_model()
    if model is None:
        return candidates[:top_k]

    pairs: List[Tuple[str, str]] = [(query, c.get(text_key, "")) for c in candidates]
    try:
        scores = model.predict(pairs)
    except Exception as e:
        print(f"[rerank] predict failed, returning original order: {e}")
        return candidates[:top_k]

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: float(x[0]), reverse=True)
    out: List[dict] = []
    for score, cand in scored[:top_k]:
        c = dict(cand)
        c["rerank_score"] = float(score)
        out.append(c)
    return out
