from typing import Optional

from fastapi import APIRouter, HTTPException

from .runner import run_benchmark

router = APIRouter(prefix="/fastapi/eval", tags=["eval"])


@router.get("/run")
def run(benchmark_path: Optional[str] = None, top_k: int = 5):
    try:
        return run_benchmark(benchmark_path, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
