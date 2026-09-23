import os

# Mencegah bentrok dengan TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
# Import wrapper Ragas untuk LangChain
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import nama kelas metrik Ragas
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)


class LegalRAGEvaluator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY tidak ditemukan di file .env!")

        # 1. Inisialisasi model LangChain
        raw_llm = ChatGroq(
            groq_api_key=api_key, model_name="openai/gpt-oss-120b"
        )
        raw_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. Bungkus ke format yang dikenali oleh Ragas
        self.ragas_llm = LangchainLLMWrapper(raw_llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

        # 3. Oper ragas_llm dan ragas_embeddings secara eksplisit ke dalam metrik
        self.metrics = [
            Faithfulness(llm=self.ragas_llm),
            AnswerRelevancy(llm=self.ragas_llm, embeddings=self.ragas_embeddings),
            ContextPrecision(llm=self.ragas_llm),
            ContextRecall(llm=self.ragas_llm),
        ]

    def evaluate_pipeline(self, eval_dataset: Dataset):
        """Menjalankan evaluasi Ragas pada dataset uji."""
        print("[*] Running Ragas evaluation...")
        results = evaluate(
            dataset=eval_dataset,
            metrics=self.metrics,
        )
        return results


if __name__ == "__main__":
    # Sample Dataset untuk Uji Coba Evaluator
    sample_data = {
        "question": [
            "Siapa yang wajib melakukan penapisan konten dalam Sistem Elektronik?"
        ],
        "contexts": [
            [
                "Setiap Penyelenggara Sistem Elektronik wajib melakukan pemutusan akses terhadap Informasi Elektronik dan/atau Dokumen Elektronik yang memiliki muatan dilarang."
            ]
        ],
        "answer": [
            "Penyelenggara Sistem Elektronik (PSE) wajib melakukan pemutusan akses/penapisan terhadap konten yang memiliki muatan dilarang."
        ],
        "ground_truth": [
            "Penyelenggara Sistem Elektronik wajib melakukan pemutusan akses terhadap konten dilarang."
        ],
    }

    eval_dataset = Dataset.from_dict(sample_data)
    evaluator = LegalRAGEvaluator()
    scores = evaluator.evaluate_pipeline(eval_dataset)

    print("\n=== RAGAS EVALUATION SCORES ===")
    print(scores)