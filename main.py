import hashlib
import json
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis

from src.pipeline import LegalRAGPipeline

app = FastAPI(
    title="Legal AI RAG API",
    description="Backend API untuk Legal RAG System dengan Redis Caching",
    version="1.0.0",
)

# Inisialisasi Pipeline RAG
pipeline = LegalRAGPipeline()

# Inisialisasi Klien Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")  # Sesuaikan dengan nama service redis di docker
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
    )
    redis_client.ping()
    print("[*] Redis terhubung dengan sukses.")
except Exception as e:
    print(
        f"[!] Warning: Gagal terhubung ke Redis ({e}). API akan berjalan tanpa Caching."
    )
    redis_client = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    top_n: Optional[int] = 3
    use_cache: Optional[bool] = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    cached: bool


def _generate_cache_key(query: str, top_k: int, top_n: int) -> str:
    """Menghasilkan cache key unik berbasis MD5 (termasuk parameter top_k & top_n)."""
    clean_query = f"{query.strip().lower()}_k{top_k}_n{top_n}"
    query_hash = hashlib.md5(clean_query.encode("utf-8")).hexdigest()
    return f"legal_rag:cache:{query_hash}"


@app.get("/")
def root():
    return {"status": "ok", "message": "Legal AI RAG Service is running"}


@app.post("/api/v1/query", response_model=QueryResponse)
def handle_query(req: QueryRequest):
    cache_key = _generate_cache_key(req.query, req.top_k, req.top_n)

    # 1. Cek Cache di Redis (Cache HIT)
    if req.use_cache and redis_client:
        try:
            cached_data = redis_client.get(cache_key)
            if cached_data:
                print(f"[*] Cache HIT untuk kueri: '{req.query}'")
                parsed_res = json.loads(cached_data)
                parsed_res["cached"] = True
                return parsed_res
        except Exception as e:
            print(f"[!] Gagal membaca dari Redis: {e}")

    # 2. Cache MISS — Eksekusi RAG Pipeline
    print(
        f"[*] Cache MISS. Menjalankan RAG Pipeline untuk kueri: '{req.query}'"
    )
    try:
        result = pipeline.answer_query(
            query=req.query, top_k=req.top_k, top_n=req.top_n
        )
        result["cached"] = False

        # 3. Simpan Hasil ke Redis (TTL: 24 Jam)
        if redis_client:
            try:
                redis_client.setex(
                    cache_key,
                    time=86400,
                    value=json.dumps(result, ensure_ascii=False),
                )
            except Exception as e:
                print(f"[!] Gagal menulis ke Redis: {e}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))