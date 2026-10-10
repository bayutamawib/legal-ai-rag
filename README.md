# Legal AI RAG: Indonesian Regulatory Compliance & Evaluation Pipeline

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Framework](<https://img.shields.io/badge/framework-LlamaIndex%20%7C%20Ragas-orange>)
![License](https://img.shields.io/badge/license-MIT-green)

A Retrieval-Augmented Generation (RAG) system for querying Indonesian regulatory documents, designed around **article-aware chunking**, **data-quality checks and quarantine**, **hybrid retrieval (dense + BM25)**, and **cross-encoder reranking**. The repository also contains an evaluation/QA layer that compares retrieved legal references against a ground-truth test set and reports retrieval, answer-quality, and latency metrics. Evaluation results can be analyzed in Google Sheets and visualized in Looker Studio.

> **Evaluation scope:** the current evaluation dataset is a small prototype set (currently 5 cases), not a statistically validated benchmark. Scores should be interpreted as diagnostic signals, not general performance guarantees. See [Evaluation &amp; Benchmarking](#-evaluation--benchmarking) for the metric definitions and limitations.

---

## 🎯 Problem Statement & Use Case

Indonesian legal documents present several challenges for retrieval and question answering:

1. **Amended and superseded laws:** similar provisions may appear in an original law and its amendments.
2. **Temporal ambiguity:** enactment dates and effective dates are not always interchangeable.
3. **Legal hierarchy and citations:** arbitrary text chunking can split a Pasal/Ayat, making relevant evidence harder to retrieve and legal references harder to validate.
4. **Document identity inconsistencies:** the same regulation can appear under different filename or metadata conventions, which can produce false retrieval misses if identifiers are compared literally.

### Solution overview

The system combines structured legal metadata, article-aware chunking, hybrid retrieval, reranking, and a separate evaluation layer. The evaluator canonicalizes known document-ID aliases before comparing expected and retrieved `(document_id, article)` pairs. It also records RAGAS metrics when they are successfully computed, so missing scores can be investigated instead of silently being treated as zero.

---

## 🏗 System Architecture

### RAG serving pipeline

```text
[ Raw Legal PDF / Scraped Feed ]
              │
              ▼
[ Data Ingestion & Sanitization ]
  ├── Date and text normalization
  ├── Metadata/schema validation
  └── Quarantine invalid records for review
              │
       ┌──────┴─────────┐
       ▼                ▼
[ Validated Records ]  [ Data Quarantine ]
       │
       ▼
[ Article-Aware Chunking ]
       │
       ▼
[ Vector Storage & Indexing ]
  ├── Dense embeddings: BAAI/bge-m3
  ├── Sparse lexical retrieval: BM25
  └── Metadata/vector storage: Qdrant
       │
       ▼
[ User Query ]
       │
       ▼
[ FastAPI / Streamlit ] ── [ Redis Cache ]
       │
       ▼
[ Hybrid Retrieval ] → [ Cross-Encoder Reranking ]
       │
       ▼
[ Groq LLM ] → [ Answer + Retrieved Sources ]
```

### Evaluation and analytics pipeline

```text
 data/evaluation/evaluation_cases.json
                 │
                 ▼
       src/evaluate_odin.py
                 │
                 ▼
       Existing Legal RAG pipeline
       (one execution per test case)
                 │
                 ▼
  ┌─────────────────────────────────────┐
  │ Expected/retrieved reference match  │
  │ Reference precision and recall      │
  │ RAGAS answer/context metrics        │
  │ Latency and status classification   │
  └─────────────────────────────────────┘
                 │
                 ▼
  evaluation/results/*.csv and *.json
                 │
                 ▼
         Google Sheets
                 │
                 ▼
          Looker Studio
```

The Google Sheets-to-Looker Studio workflow is the reporting layer. CSV import works without the optional Apps Script automation. If a Google Apps Script web endpoint is configured, it can be used to automate the synchronization of evaluation rows; do not commit its URL/token or local credentials.

---

## 🛠 Tech Stack

- **Backend API:** FastAPI
- **Caching layer:** Redis
- **LLM engine:** Groq API (configured in the RAG pipeline)
- **Embedding model:** `BAAI/bge-m3`
- **Vector database:** Qdrant
- **Reranker:** `BAAI/bge-reranker-base`
- **User interface:** Streamlit
- **Containerization:** Docker and Docker Compose
- **Evaluation framework:** RAGAS
- **Evaluation analysis:** Python, pandas, CSV/JSON artifacts
- **BI reporting:** Google Sheets and Looker Studio
- **Optional synchronization:** Google Apps Script web app

---

## 📁 Repository Structure

```text
legal-ai-rag/
├── app/                              # Application package
├── data/
│   ├── raw/                          # Source legal PDF/TXT files
│   ├── processed/                    # Cleaned/structured legal chunks
│   ├── quarantine/                   # Invalid records for review
│   └── evaluation/
│       └── evaluation_cases.json     # Ground-truth evaluation cases
├── evaluation/
│   └── results/
│       ├── evaluation_results.csv    # Per-case evaluation results
│       ├── failure_analysis.csv      # Failed cases for debugging
│       └── evaluation_summary.json   # Aggregate metrics/status counts
├── notebooks/                        # Analysis and experimentation
├── src/
│   ├── batch_test.py                 # Existing batch queries
│   ├── evaluate_batch.py             # Basic RAGAS evaluation
│   ├── evaluate_odin.py              # Extended RAG evaluation
│   ├── evaluator.py                  # RAGAS evaluator
│   ├── parser.py                     # Legal document parsing
│   ├── ingest.py                     # Ingestion and metadata tagging
│   ├── retriever.py                  # Hybrid search and reranking
│   └── pipeline.py                   # Core RAG orchestration
├── tests/                            # Automated tests
├── .gitignore
├── app_ui.py                         # Streamlit UI components
├── batch_test_results.json           # Existing batch-test output
├── docker-compose.yml
├── Dockerfile
├── evaluation_report.json            # Existing evaluation report
├── main.py                           # Application entry point
├── metadata_schema_example.json      # Example metadata schema
├── README.md
└── requirements.txt
```

Generated evaluation results may be recreated by running the evaluator. Include only artifacts that are safe to publish and useful for reproducing the analysis; never commit API keys, access tokens, private source documents, or `.sheets.env`.

---

## 📋 Data Pipeline & Metadata Schema

### Article-aware chunking

The chunking strategy follows legal structure rather than splitting text solely by arbitrary character count. A chunk generally represents one Pasal, with finer splitting where a long provision requires it and where the parser preserves the legal hierarchy.

### Canonical metadata schema

A processed chunk is expected to include structured metadata similar to:

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
  "content": "...",
  "source_url": "https://example.org/legal-source"
}
```

The values above illustrate the schema shape; they are not a substitute for verifying legal text and metadata against the authoritative document.

### Data quarantine protocol

Records should be quarantined for manual review when they contain problems such as:

- Unparseable or ambiguous dates.
- Missing legal article (`pasal`) identifiers where required.
- Text-normalization artifacts such as zero-width spaces or non-breaking spaces that affect parsing/search.
- Invalid or inconsistent metadata that cannot be safely canonicalized.

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher.
- A Groq API key for LLM-backed generation/evaluation.
- Access to the configured vector database and source corpus used by this repository.

### 2. Installation

```bash
git clone https://github.com/bayutamawib/legal-ai-rag.git
cd legal-ai-rag

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use a dedicated virtual environment rather than a heavily customized global/Conda `base` environment. LangChain integration packages must be compatible with one another; installing the newest `langchain-groq` into a LangChain 0.3 environment can cause import errors. If you see a `ModelProfile` import error, inspect versions with `python -m pip show langchain-groq langchain-core langchain ragas` and resolve the project dependency pins before rerunning.

### 3. Environment configuration

Create a `.env` file in the repository root based on `.env.example`. For example:

```env
GROQ_API_KEY=your_groq_api_key
EMBED_MODEL_NAME=BAAI/bge-m3
VECTOR_DB_PATH=./data/vector_store
```

Use the exact environment variables required by the current application configuration. Never commit `.env` or real API keys.

### 4. Run the application

To start the Docker Compose services:

```bash
docker compose up --build -d
```

Stop the services with:

```bash
docker compose down
```

---

## 📊 Evaluation & Benchmarking

The repository contains two evaluation paths:

- **Existing/basic evaluation:** `src/batch_test.py` and `src/evaluate_batch.py` produce the earlier batch/RAGAS outputs.
- **Extended evaluation:** `src/evaluate_odin.py` adds ground-truth reference matching, latency capture, retrieval-status classification, answer-quality status, and reporting artifacts. It reuses the existing RAG pipeline instead of rebuilding the RAG architecture.

### Evaluation case schema

Cases live in `data/evaluation/evaluation_cases.json`. Each case uses the following fields:

```json
{
  "case_id": "Q001",
  "query": "Example legal question",
  "expected_answer": "Verified reference answer",
  "expected_references": [
    {
      "document_id": "CANONICAL_DOCUMENT_ID",
      "articles": ["Pasal X", "Pasal Y"]
    }
  ],
  "query_type": "direct_lookup",
  "severity": "high"
}
```

`expected_references` explicitly maps each expected article to its document ID. The evaluator compares `(canonical_document_id, normalized_article)` pairs, avoiding false misses caused by common naming variations such as `PP_71_2019` versus `PP Nomor 71 Tahun 2019`. The expected references and answers must be verified against the source legal documents before a case is treated as ground truth.

### Run the extended evaluator

From the repository root:

```bash
python -m src.evaluate_odin
```

The script sets `USE_TF=0` before importing the RAG stack to avoid an unnecessary TensorFlow import in the PyTorch/sentence-transformers path. If your local version does not set this in the script, you can run:

```bash
USE_TF=0 python -m src.evaluate_odin
```

### Output artifacts

The evaluator writes:

| Artifact                                       | Purpose                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `evaluation/results/evaluation_results.csv`  | Per-case expected/retrieved references, scores, latency and statuses |
| `evaluation/results/failure_analysis.csv`    | Cases whose overall status is not`pass`                            |
| `evaluation/results/evaluation_summary.json` | Aggregate metrics and status counts                                  |

### Retrieval metrics

Expected and retrieved references are represented by unique `(document_id, article)` pairs.

- **Matched reference count:** number of expected pairs also present in retrieved sources.
- **Reference recall:** `matched_reference_count / expected_reference_count`. Measures how much of the annotated expected reference set was retrieved.
- **Reference precision:** `matched_reference_count / retrieved_reference_count`. Measures how many unique retrieved references match the expected set. With the current `top_n=3` and three unique returned references, this is equivalent to Precision@3; if fewer than three unique references are returned, the denominator is the actual retrieved-reference count.
- **Latency:** wall-clock duration of the RAG query call, recorded in seconds.

Document-ID canonicalization only normalizes known identifier aliases. It does not validate the legal correctness of an expected reference or replace the need for human verification.

### RAGAS metrics and thresholds

The extended evaluator uses the existing `LegalRAGEvaluator` to calculate these metrics when the required model calls and contexts are available:

| Metric            | Configured threshold | Meaning                                                                           |
| ----------------- | -------------------: | --------------------------------------------------------------------------------- |
| Faithfulness      |          `>= 0.90` | Whether the generated answer is supported by retrieved context                    |
| Answer relevancy  |          `>= 0.90` | Whether the answer addresses the query                                            |
| Context precision |          `>= 0.85` | Whether the retrieved context is relevant to the question/reference               |
| Context recall    |          `>= 0.85` | Whether the retrieved context covers information required by the reference answer |

Unavailable metric values are not equivalent to zero. RAGAS may return missing (`NaN`) results for individual metrics/cases; inspect those cases and the evaluator logs before interpreting averages. Aggregate averages only represent available values when some rows are missing.

### Status classification

The evaluator exposes separate fields to distinguish retrieval problems from answer-quality problems:

- `retrieval_empty`: no source returned.
- `retrieval_miss`: sources returned, but no expected reference pair matched.
- `partial_retrieval`: at least one reference matched, but reference recall is below `0.50`.
- `strong_retrieval`: reference recall is at least `0.50`.
- `answer_quality_status`: `pass`, `quality_failure`, or `evaluation_error`, based on the configured RAGAS thresholds and metric availability.
- `overall_status`: combined execution, retrieval, and answer-quality status for filtering and failure analysis.

The `0.50` recall cut-off is a project-defined triage policy for this prototype, not a universal retrieval-quality standard. Review `evaluation_results.csv` if `answer_quality_status` and `overall_status` appear inconsistent; an unavailable RAGAS score must not be interpreted as proof that the answer is poor.

### Current scope and limitations

- The current annotated set has **5 evaluation cases**. Expand and validate it before making broad claims about production performance.
- Some query cases expect multiple articles across several documents. Since the current evaluation configuration retrieves the top 3 sources, not every expected reference can necessarily appear in the returned set. Interpret recall alongside precision and retrieval depth.
- Legal references and expected answers require human verification against authoritative source documents.
- RAGAS metrics have shown missing values in some per-query outputs. Treat them as provisional until the cause of missing metrics is resolved and the contexts/ground truths are audited.
- Do not publish fixed benchmark scores unless the dataset, dependency versions, run configuration, and results are reproducible. Use `evaluation_summary.json` for the results of a specific run.

---

## 📈 Google Sheets & Looker Studio Dashboard

The evaluation artifacts can be imported into Google Sheets and used as a Looker Studio data source. This reporting layer supports query-level debugging as well as a high-level health check.

### Dashboard sections

1. **Executive summary — scorecards**

   - Total test cases (`Record Count`). If several runs are retained, filter by `run_id` to count a single run.
   - Average `latency_seconds`.
   - Average `faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` as four separate scorecards.
   - Missing RAGAS values should remain blank; averages may be based only on available metric values.
2. **Status and evaluation distribution — pie/donut charts**

   - Overall status: dimension `overall_status`, metric `Record Count`.
   - Retrieval status: dimension `retrieval_status`, metric `Record Count`.
   - These charts highlight retrieval misses/partial retrieval and answer-quality or evaluation failures.
3. **Performance trend across runs — time series/line chart**

   - Dimension: `updated_at`.
   - Metrics: average `context_precision` and `context_recall`.
   - Retain run identifiers/timestamps to compare runs. For manual CSV imports, add or populate `run_id` and `updated_at`; these fields are included by the optional Apps Script streaming workflow when configured.
4. **Query difficulty — bar chart**

   - Dimension: `query_type`.
   - Metric: average `answer_relevancy`.
   - Treat this as a preliminary view while per-query RAGAS values are incomplete.
5. **Failure analysis — interactive table**

   - Fields: `case_id`, `query`, `expected_references_flat`, `retrieved_references`, and `answer`.
   - Visual-level filter: `overall_status` is not `pass`.
   - Use this table to inspect expected versus retrieved legal references for each failed or partially evaluated case.

### Optional Google Sheets synchronization

A Google Apps Script web app can receive evaluation rows from Python and upsert them into an `evaluation_stream` worksheet using a run identifier and case ID. This automation is optional; CSV import remains a valid way to build/update the dashboard. If enabling synchronization:

- Store `SPREADSHEET_ID` and `API_TOKEN` in Apps Script **Script Properties**.
- Store the web app URL and the matching token locally in `.sheets.env`.
- Add `.sheets.env` to `.gitignore` and never commit tokens or the deployed endpoint configuration.
- Verify that an authenticated test row reaches the expected spreadsheet before running the full evaluation.

The Looker Studio connector may refresh according to its data-freshness/cache settings; data being written to Sheets does not guarantee that every chart refreshes immediately.

---

## 🔍 Engineering Notes

- Keep the core RAG pipeline separate from the evaluation layer. `src/evaluate_odin.py` should consume the existing pipeline outputs instead of re-implementing retrieval or generation.
- Run each evaluation query once and reuse that result for retrieval diagnostics and RAGAS inputs.
- Keep canonical document-ID aliases explicit and versioned; do not use article-only matching where the same Pasal number can occur in different laws.
- Do not fabricate expected answers or legal references to improve metrics.
- Avoid hard-coding one run's results in this README; commit evaluation artifacts only when they are safe to publish and reproducible.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
