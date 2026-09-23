import os
from typing import Dict, List, Optional
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from src.retriever import LegalRetriever

load_dotenv(find_dotenv())


class LegalRAGPipeline:
    """Core RAG Pipeline that connects Qdrant Retriever with Groq Llama 3."""

    SYSTEM_PROMPT = """Kamu adalah Asisten Ahli Hukum Indonesia yang kritis, presisi, dan obyektif.
Tugas utama kamu adalah menjawab pertanyaan pengguna HANYA berdasarkan konteks dokumen hukum yang disediakan.

Aturan Ketat:
1. Jawab pertanyaan secara langsung, lugas, dan terstruktur.
2. Wajib mencantumkan rujukan hukum (misal: BAB X, Pasal Y) pada setiap klaim atau penjelasan yang kamu berikan.
3. JIKA informasi yang ditanyakan TIDAK ADA atau TIDAK DAPAT DISIMPULKAN dari konteks yang diberikan, kamu HARUS menyatakan: "Berdasarkan dokumen hukum yang tersedia, informasi mengenai hal tersebut tidak ditemukan."
4. Dilarang keras mengarang pasal, menafsirkan di luar konteks yang ada, atau menggunakan pengetahuan di luar teks dokumen.
"""

    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY cannot be found in environment variable or .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.retriever = LegalRetriever()

    def _expand_query_with_llm(self, original_query: str) -> str:
        """Menggunakan Groq Llama 3 untuk memperluas kueri awam menjadi istilah hukum formal."""
        system_prompt = (
            "Kamu adalah pakar hukum Indonesia. Tugasmu adalah mengubah/memperluas pertanyaan "
            "bahasa awam menjadi kata kunci dan istilah hukum formal yang biasa digunakan "
            "dalam Peraturan Perundang-undangan Indonesia (seperti UU ITE, PP 71/2019, Permenkominfo).\n"
            "Aturan:\n"
            "1. Berikan 3-5 istilah/frasa hukum formal atau sinonim teknis yang paling relevan.\n"
            "2. HANYA keluarkan frasa/kata kunci tambahan yang dipisahkan spasi, tanpa kalimat pembuka atau penutup."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Kueri awam: {original_query}"},
                ],
                temperature=0.2,
                max_tokens=60,
            )

            expanded_terms = response.choices[0].message.content.strip()
            expanded_query = f"{original_query} {expanded_terms}"
            print(f"[*] Expanded Query: '{expanded_query}'\n")
            return expanded_query
        except Exception as e:
            print(f"[!] Gagal ekspansi kueri: {e}. Menggunakan kueri asli.")
            return original_query

    def _build_context_string(self, retrieved_chunks: List[Dict]) -> str:
        """Format chunks from the retriever into a tidy context string for LLM."""
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, 1):
            payload = chunk["payload"]
            block = (
                f"[Dokumen {idx}]\n"
                f"Judul: {payload.get('title', 'Dokumen Hukum')}\n"
                f"Hirarki: {payload.get('bab', 'TANPA BAB')} | {payload.get('pasal', '')}\n"
                f"Isi Teks:\n{payload.get('content', '')}\n"
            )
            context_blocks.append(block)

        return "\n=" * 20 + "\n" + "\n".join(context_blocks)

    def answer_query(
        self,
        query: str,
        top_k: int = 10,
        top_n: int = 3,
        filter_bab: Optional[str] = None,
        use_query_expansion: bool = True,
    ) -> Dict:
        """Proses RAG lengkap: Query Expansion -> Retrieval -> Prompting -> Generasi Jawaban oleh LLM."""

        # 1. Perluas kueri pengguna jika fitur diaktifkan
        search_query = (
            self._expand_query_with_llm(query) if use_query_expansion else query
        )

        # 2. Ambil dokumen relevan dari Qdrant & CrossEncoder menggunakan kueri yang diekspansi
        retrieved_chunks = self.retriever.retrieve(
            query=search_query, top_k=top_k, top_n=top_n, filter_bab=filter_bab
        )

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "Tidak ditemukan dokumen hukum yang relevan di dalam database.",
                "sources": [],
            }

        # 3. Menyusun Konteks dan User Prompt (Gunakan kueri ASLI pengguna agar respons natural)
        context_str = self._build_context_string(retrieved_chunks)
        user_prompt = (
            f"KONTEKS DOKUMEN HUKUM:\n{context_str}\n\n"
            f"PERTANYAAN PENGGUNA:\n{query}\n\n"
            f"JAWABAN:"
        )

        # 4. Panggil Groq API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Konsisten dan meminimalkan halusinasi
            max_tokens=1024,
        )

        answer_text = response.choices[0].message.content

        # 5. Ambil metadata sumber untuk transparansi sitasi
        sources = [
            {
                "document_id": c["payload"].get("canonical_law_id") or c["payload"].get("document_id", ""),
                "bab": c["payload"].get("bab"),
                "pasal": c["payload"].get("pasal"),
                "score": c.get("rerank_score"),
                "content_preview": c["payload"].get("content", ""),
            }
            for c in retrieved_chunks
        ]

        return {"query": query, "answer": answer_text, "sources": sources}


if __name__ == "__main__":
    pipeline = LegalRAGPipeline()

    test_query = "Siapa yang wajib melakukan penapisan konten dalam Sistem Elektronik?"
    print(f"[*] Processing Question: '{test_query}'...\n")

    result = pipeline.answer_query(query=test_query)

    print("=== MODEL'S ANSWER ===")
    print(result["answer"])
    print("\n=== REFERENCE SOURCE ===")
    for src in result["sources"]:
        print(f"- {src['bab']} | {src['pasal']} (Score: {src['score']:.4f})")