import json
import os
from typing import Dict, List, Optional
from datasets import Dataset
import pandas as pd
from src.evaluator import LegalRAGEvaluator


def load_batch_results(file_path: str = "batch_test_results.json") -> List[Dict]:
    """Membaca file JSON hasil batch testing."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File '{file_path}' tidak ditemukan! Jalankan 'python -m src.batch_test' terlebih dahulu."
        )

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_ragas_dataset(
    batch_data: List[Dict], ground_truths: Optional[Dict[str, str]] = None
) -> Dataset:
    """Mengubah format data hasil batch test menjadi Dataset Ragas.

    Args:
        batch_data: List dict dari batch_test_results.json
        ground_truths: Optional dict berisi pasangan {query: ground_truth_text}
    """
    questions = []
    contexts = []
    answers = []
    reference_answers = []

    ground_truths = ground_truths or {}

    for item in batch_data:
        # Abaikan item yang mengalami error
        if "error" in item or not item.get("answer"):
            continue

        query = item["query"]
        questions.append(query)
        answers.append(item["answer"])

        # Ekstrak semua preview/konten pasal dari sources sebagai contexts
        ctx_list = [
            f"{src.get('bab', '')} {src.get('pasal', '')}: {src.get('content_preview', '')}"
            for src in item.get("sources", [])
        ]
        # Jika tidak ada rujukan ditemukan, beri konteks kosong
        contexts.append(ctx_list if ctx_list else ["Kontek tidak ditemukan."])

        # Ambil ground truth jika disediakan, atau gunakan placeholder/jawaban asli
        gt = ground_truths.get(query, item["answer"])
        reference_answers.append(gt)

    dataset_dict = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
        "ground_truth": reference_answers,
    }

    return Dataset.from_dict(dataset_dict)


def evaluate_batch_file(
    input_json: str = "batch_test_results.json",
    output_json: str = "evaluation_report.json",
):
    """Membaca hasil batch test, menjalankan evaluasi Ragas, dan menyimpan laporan."""
    print(f"[*] Membaca data dari '{input_json}'...")
    raw_batch = load_batch_results(input_json)

    # Contoh ground truth opsional (bisa disesuaikan jika punya kunci jawaban hukum formal)
    custom_ground_truths = {
        "Siapa yang wajib melakukan penapisan konten dalam Sistem Elektronik?": (
            "Setiap Penyelenggara Sistem Elektronik (PSE) wajib melakukan pemutusan akses "
            "terhadap Informasi Elektronik yang memiliki muatan dilarang."
        )
    }

    print("[*] Mengonversi data ke format Ragas Dataset...")
    eval_dataset = prepare_ragas_dataset(
        raw_batch, ground_truths=custom_ground_truths
    )

    print(
        f"[*] Memulai evaluasi Ragas untuk {len(eval_dataset)} sampel kueri..."
    )
    evaluator = LegalRAGEvaluator()
    scores = evaluator.evaluate_pipeline(eval_dataset)

    print("\n=== RINGKASAN SKOR EVALUASI RAGAS ===")
    print(scores)

    # Konversi hasil evaluasi ke Pandas DataFrame untuk disimpan ke JSON/CSV
    df_results = scores.to_pandas()
    df_results.to_json(output_json, orient="records", indent=2)
    print(f"\n[*] Laporan rinci berhasil disimpan ke: '{output_json}'")


if __name__ == "__main__":
    evaluate_batch_file()