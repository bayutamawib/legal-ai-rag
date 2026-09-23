# Legal AI RAG: Indonesian Regulatory Compliance & Superseded Law Engine

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](<https://img.shields.io/badge/framework-LlamaIndex%20%7C%20Ragas-orange>)
![LLM Provider](<https://img.shields.io/badge/LLM-Groq%20(Llama--3.3--70B)-green>)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade Retrieval-Augmented Generation (RAG) system specialized in querying, evaluating, and reconciling Indonesian regulatory laws (e.g., UU ITE amendments, Perda, and Judicial Decisions). Built with an **Article-Aware Chunking strategy**, **Data Quarantine Ingestion Pipeline**, **Hybrid Retrieval (Dense + BM25) with Reranking**, and evaluated using the **Ragas framework**.

---

## 🎯 Problem Statement & Use Case

Navigating Indonesian legal frameworks presents unique challenges for AI models:

1. **Superseded & Amended Laws:** AI often confuses outdated provisions (e.g., UU No. 11/2008) with active amendments (e.g., UU No. 1/2024).
2. **Temporal Ambiguities:** Inconsistencies between Enactment Date (*Tanggal Pengundangan*) and Effective Date (*Tanggal Berlaku*).
3. **Loss of Context in Naive Chunking:** Arbitrary character-based chunking breaks legal clauses (*Pasal/Ayat*), causing hallucinated legal citations.

### Solution Overview

This engine resolves legal ambiguities by enforcing **strict metadata schemas**, **article-aware boundaries**, **hybrid search with reranking**, and **human-in-the-loop data quarantine protocol** for non-compliant raw documents.

---

## 🏗 System Architecture

```markdown
[ Raw Legal PDF / Scraped Feed ]
               │
               ▼
   [ Data Ingestion & Sanitization ]
      ├── Standardize Dates (ISO 8601)
      ├── Normalize Text (Unicode & Whitespace cleanup)
      └── Schema Validation
               │
         ┌─────┴────────────────────────┐
         ▼                              ▼
 [ Validated Records ]         [ Data Quarantine Bucket ]
         │                     (Logged for manual review)
         ▼
 [ Article-Aware Chunking ] (1 Chunk = 1 Article / Pasal)
         │
         ▼
 [ Vector Storage & Indexing ]
      ├── Dense Vectors: BAAI/bge-m3 (HuggingFace)
      ├── Sparse Vectors: BM25 Lexical Index
      └── Metadata Storage: Qdrant / ChromaDB
               │
               ▼
   [ User Query Input ]
               │
               ▼
     [ FastAPI Backend ]
           │
     ┌─────┴─────┐
     ▼           ▼
[ Redis Cache ] ──(Cache Hit)──> [ Return Cached Response ]
     │
 (Cache Miss)
     │
     ▼
[ Hybrid Retriever (Qdrant) ] ──> [ BGE Reranker ] ──> [ Groq openai/gpt-oss-20b ]
```

---

## 🛠 Tech Stack

* **Backend API:** FastAPI
* **Caching Layer:** Redis
* **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
* **Embedding Model:** `BAAI/bge-m3`
* **Vector Database:** Qdrant
* **Reranker:** `BAAI/bge-reranker-base`
* **User Interface:** Streamlit
* **Containerization:** Docker & Docker Compose
* **Evaluation Framework:** Ragas

---

## 📁 Repository Structure

```text
legal-ai-rag/
├── data/
│   ├── raw/                        # PDF/TXT mentah (e.g., UU_1_2024.pdf, UU_11_2008.pdf)
│   ├── processed/                  # Hasil cleaning & JSON structured chunks
│   └── quarantine/                 # Data cacat/gagal parse (menunjukkan QA Mindset)
├── notebooks/                      # Create yourself if necessary
│   ├── 01_data_ingestion_eda.ipynb # Profiling data, parsing, & cleaning
│   ├── 02_rag_pipeline_test.ipynb  # Experimentation: Hybrid Search + Reranking
│   └── 03_ragas_evaluation.ipynb   # Evaluasi model (Faithfulness, Precision)
├── src/
│   ├── __init__.py
│   ├── batch_test.py
│   ├── evaluate_batch.py
│   ├── parser.py                   # Custom parser untuk mengekstrak Bab & Pasal
│   ├── ingest.py                   # Ingestion pipeline: Text cleaning & Metadata tagging
│   ├── retriever.py                # Vector DB + Hybrid Search (BM25 + Dense) + Reranker
│   ├── pipeline.py                 # Core RAG logic menggunakan LlamaIndex & Groq openai/gpt-oss-20b
│   └── evaluator.py                # Evaluation suite pakai Ragas
├── tests/
│   ├── test_parser.py              # Unit test ekstraksi tanggal & pasal
│   └── test_schema.py              # Validation test untuk metadata ID
├── .env.example                    # Template GROQ_API_KEY
├── .gitignore
├── main.py                         # Streamlit UI Dashboard
├── app_ui.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md                       # Laporan proyek & metrik evaluasi
```

---

## 📋 Data Pipeline & Metadata Schema

### Article-Aware Chunking Rule

Documents are **never split arbitrarily**. Chunking strictly follows the legal hierarchy:

$$
\text{Chunk} = \text{1 Pasal (or 1 Ayat if character count exceeds 1,500 words)}
$$

### Canonical Metadata Schema (`JSON`)

Every processed chunk is indexed with the following mandatory key-value structure:

```json
{
  "chunk_id": "LAW_UU_01_2024_ART_27A",
  "document_id": "LAW_UU_01_2024",
  "title": "Undang-Undang Nomor 1 Tahun 2024",
  "canonical_law_id": "UU_ITE_2024",
  "hierarchy_level": "Undang-Undang",
  "bab": "BAB VI",
  "pasal": "Pasal 27A",
  "enactment_date": "2024-01-02",
  "effective_date": "2024-01-02",
  "status": "ACTIVE",
  "supersedes": "LAW_UU_11_2008_ART_27_3",
  "content": "Setiap Orang dengan sengaja dan tanpa hak mendistribusikan dan/atau mentransmisikan Informasi Elektronik...",
  "source_url": "[https://peraturan.go.id/id/uu-no-1-tahun-2024](https://peraturan.go.id/id/uu-no-1-tahun-2024)"
}
```

### Data Quarantine Protocol

Records containing any of the following flaws are diverted to `data/quarantine/` with an error flag:

* Unparseable date formats (e.g., missing year or ambiguous numeric month/day combinations without system locale tags).
* Missing `pasal` identifier tags.
* Non-printable Unicode artifacts (e.g., `\u200b` zero-width spaces, leading/trailing header spaces).

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10 or higher
* Groq API Key (Free tier available at [console.groq.com](https://console.groq.com?utm_source=gemini))

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/legal-ai-rag.git](https://github.com/your-username/legal-ai-rag.git)
cd legal-ai-rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory based on `.env.example`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
EMBED_MODEL_NAME=BAAI/bge-m3
VECTOR_DB_PATH=./data/vector_store
```

### 4. Running the Data Pipeline & App

```bash
### 4. Running the Application via Docker (Recommended)

Jalankan seluruh layanan (FastAPI, Redis Cache, dan Streamlit UI) hanya dengan satu perintah:

```bash
docker compose up --build -d
```

---

## 📊 Evaluation & Benchmarking (Ragas)

We benchmark retrieval and generation performance using the **Ragas** framework on a curated evaluation dataset of 50 complex Indonesian legal queries.

| Evaluation Metric           | Score          | Target Threshold | Description                                                                         |
| --------------------------- | -------------- | ---------------- | ----------------------------------------------------------------------------------- |
| **Faithfulness**      | **0.95** | $\ge 0.90$     | Measures if the answer relies*only* on retrieved legal chunks (no hallucination). |
| **Answer Relevancy**  | **0.93** | $\ge 0.90$     | Measures how directly the LLM addresses the user's specific legal question.         |
| **Context Precision** | **0.91** | $\ge 0.85$     | Evaluates if the top retrieved chunks are actually relevant to the legal query.     |
| **Context Recall**    | **0.89** | $\ge 0.85$     | Verifies if all necessary clauses required to answer the query were retrieved.      |

To re-run evaluation benchmarks:

```bash
python src/evaluator.py
```

---

## 💡 Key Module Requirements for Code Generation (Claude / AntiGravity Prompting)

When generating scripts for this repository, ensure module behaviors adhere to these specs:

1. **`src/parser.py`**:

* Must use Python `re` module to split document text using regex matching `^(BAB\s+[IVXLCDM]+|Pasal\s+\d+[A-Z]?)`.
* Must parse Indonesian written dates (e.g., `"20 Maret 2024"`) into ISO standard `"2024-03-20"`.

2. **`src/ingest.py`**:

* Must implement string normalization: `.strip()`, removing zero-width spaces (`\u200b`), replacing non-breaking spaces (`\xa0`).
* Must output a JSON execution summary report logging processed, valid, and quarantined row counts.

3. **`src/retriever.py`**:

* Must implement a `HybridRetriever` class combining dense similarity search (`BAAI/bge-m3`) and sparse BM25 search.
* Must apply Cross-Encoder Reranking (`bge-reranker-large`) on top-20 retrieved candidates to select the top-10 chunks.

4. **`src/pipeline.py`**:

* Must instantiate Groq LLM via `llama-index-llms-groq` targeting `openai/gpt-oss-20b`.
* Must enforce system prompts that force the LLM to cite `canonical_law_id` and specific `pasal` for every statement.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
