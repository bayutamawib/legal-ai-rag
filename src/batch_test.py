import json
import time
from typing import List, Dict
from src.pipeline import LegalRAGPipeline


def run_batch_test(
    queries: List[str], output_file: str = "batch_test_results.json"
):
    """Menjalankan batch testing pada Legal RAG Pipeline dan menyimpan hasilnya ke JSON."""
    pipeline = LegalRAGPipeline()
    test_results: List[Dict] = []

    print(f"[*] Memulai Batch Testing untuk {len(queries)} pertanyaan...\n")
    start_total_time = time.time()

    for idx, query in enumerate(queries, 1):
        print(f"[{idx}/{len(queries)}] Processing Query: '{query}'")
        q_start = time.time()

        try:
            # Jalankan pipeline RAG
            result = pipeline.answer_query(query=query)

            q_elapsed = round(time.time() - q_start, 2)
            result["execution_time_seconds"] = q_elapsed
            test_results.append(result)

            print(
                f"    -> Selesai dalam {q_elapsed} detik | Rujukan ditemukan: {len(result['sources'])}"
            )
            print("=" * 60)

        except Exception as e:
            print(f"    [!] Error saat memproses kueri: {e}")
            test_results.append({"query": query, "error": str(e)})

    total_elapsed = round(time.time() - start_total_time, 2)
    print(f"\n[*] Batch Testing Selesai! Total waktu: {total_elapsed} detik.")

    # Simpan hasil pengujian ke file JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)

    print(f"[*] Hasil lengkap telah disimpan ke file: {output_file}\n")


if __name__ == "__main__":
    # Daftar pertanyaan uji coba (campuran bahasa awam & istilah teknis)
    test_questions = [
        "Siapa yang wajib melakukan penapisan konten dalam Sistem Elektronik?",
        "Apa sanksi bagi Penyelenggara Sistem Elektronik yang tidak melakukan pendaftaran?",
        "Bagaimana kewajiban PSE jika terjadi kegagalan perlindungan data pribadi?",
        "Apakah informasi elektronik dapat dijadikan alat bukti yang sah di pengadilan?",
        "Siapa yang berwenang melakukan pemblokiran atau pemutusan akses situs web dilarang?",
    ]

    run_batch_test(test_questions)