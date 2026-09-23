import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder, SentenceTransformer

# Paksa Hugging Face transformers mengabaikan TensorFlow & menggunakan PyTorch
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"


def get_best_device() -> str:
    """Detect hardware acceleration: CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LegalRetriever:
    """Retriever for legal document using Qdrant DB and CrossEncoder Reranking."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: str = "indonesian_legal_docs",
        dense_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        # 1. Ambil host & port dari Environment Variable (fallback ke localhost jika tidak set)
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = collection_name
        self.device = device or get_best_device()

        print(
            f"[*] Connecting to Qdrant at {self.host}:{self.port} | Device: {self.device.upper()}"
        )

        # 2. Konek ke Qdrant Docker container
        self.client = QdrantClient(host=self.host, port=self.port)

        # 3. Load Embedding Model & Reranker
        print(f"[*] Loading Dense Model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(
            dense_model_name, device=self.device
        )

        print(f"[*] Loading Reranker Model: {reranker_model_name}...")
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        # 4. Prepare Database Collection
        self._setup_collection()

    def _setup_collection(self):
        """Creating collection in Qdrant or recreating it if vector dimension mismatches."""
        vector_size = self.dense_model.get_sentence_embedding_dimension()

        try:
            collections = [
                col.name for col in self.client.get_collections().collections
            ]
        except Exception as e:
            print(f"[!] Gagal terhubung ke Qdrant: {e}")
            raise e

        if self.collection_name in collections:
            try:
                collection_info = self.client.get_collection(
                    self.collection_name
                )
                existing_size = collection_info.config.params.vectors.size
                if existing_size != vector_size:
                    print(
                        f"[!] Dimensi vektor tidak cocok (DB: {existing_size}, Model: {vector_size})."
                    )
                    print(
                        f"[*] Menghapus dan membuat ulang koleksi '{self.collection_name}'..."
                    )
                    self.client.delete_collection(self.collection_name)
                    collections.remove(self.collection_name)
            except Exception:
                self.client.delete_collection(self.collection_name)
                collections.remove(self.collection_name)

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            print(
                f"[✓] Koleksi '{self.collection_name}' berhasil dibuat (Dimensi: {vector_size})."
            )

    def _generate_uuid(self, text_id: str) -> str:
        """Menghasilkan UUID v5 deterministic dari chunk_id agar tidak bentrok di Qdrant."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text_id))

    def index_processed_files(
        self,
        processed_dir: str = "data/processed",
        batch_size: int = 64,
    ):
        """Membaca berkas JSON hasil ekstraksi ingest.py dan mengindeksnya ke Qdrant secara bertahap."""
        processed_path = Path(processed_dir)
        files = list(processed_path.glob("*_processed.json"))

        if not files:
            print(
                f"[!] Tidak ada file terproses di {processed_path.resolve()}"
            )
            return

        all_chunks = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))

        print(f"[*] Mengindeks {len(all_chunks)} pasal ke Qdrant...")

        points = []
        for idx, chunk in enumerate(all_chunks):
            vector = self.dense_model.encode(chunk["content"]).tolist()

            # Buat UUID konsisten berdasarkan chunk_id (misal: "UU_11_2008_pasal_27_1")
            point_id = self._generate_uuid(
                chunk.get("chunk_id", f"chunk_{idx}")
            )

            payload = {
                "chunk_id": chunk.get("chunk_id", ""),
                "document_id": chunk.get("document_id", ""),
                "title": chunk.get("title", ""),
                "canonical_law_id": chunk.get("canonical_law_id", ""),
                "hierarchy_level": chunk.get("hierarchy_level", ""),
                "bab": chunk.get("bab", ""),
                "pasal": chunk.get("pasal", ""),
                "enactment_date": chunk.get("enactment_date", ""),
                "status": chunk.get("status", ""),
                "content": chunk.get("content", ""),
                "source_url": chunk.get("source_url", ""),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

            # Upsert per-batch agar efisien di memori
            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name, points=points
                )
                points = []

        # Upsert sisa poin
        if points:
            self.client.upsert(
                collection_name=self.collection_name, points=points
            )

        print(
            f"[✓] Berhasil menyimpan {len(all_chunks)} pasal ke database Qdrant."
        )

    def search_candidates(
        self,
        query: str,
        top_k: int = 10,
        filter_bab: Optional[str] = None,
    ) -> List[Dict]:
        """Tahap 1: Pencarian awal (candidate generation) menggunakan Cosine Similarity."""
        query_vector = self.dense_model.encode(query).tolist()

        query_filter = None
        if filter_bab:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="bab",
                        match=models.MatchValue(value=filter_bab),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )

        return [
            {"score": hit.score, "payload": hit.payload}
            for hit in response.points
        ]

    def rerank(
        self, query: str, candidates: List[Dict], top_n: int = 3
    ) -> List[Dict]:
        """Tahap 2: Menilai ulang kandidat menggunakan CrossEncoder agar lebih presisi."""
        if not candidates:
            return []

        pairs = [[query, item["payload"]["content"]] for item in candidates]
        scores = self.reranker.predict(pairs)

        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        return sorted(
            candidates, key=lambda x: x["rerank_score"], reverse=True
        )[:top_n]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 3,
        filter_bab: Optional[str] = None,
    ) -> List[Dict]:
        """Pipeline Utama: Pencarian Kandidat + Re-ranking."""
        candidates = self.search_candidates(
            query, top_k=top_k, filter_bab=filter_bab
        )
        return self.rerank(query, candidates, top_n=top_n)


if __name__ == "__main__":
    retriever = LegalRetriever()

    # Inisialisasi/Re-indeks
    retriever.index_processed_files()

    # Uji coba pencarian
    query_test = "Apa saja kewajiban Penyelenggara Sistem Elektronik?"
    results = retriever.retrieve(query=query_test, top_k=5, top_n=2)

    print(f"\n=== Hasil Pencarian: '{query_test}' ===")
    for res in results:
        p = res["payload"]
        print(
            f"\n[Skor Rerank: {res['rerank_score']:.4f}] {p['bab']} - {p['pasal']}"
        )
        print(f"Isi: {p['content']}...")