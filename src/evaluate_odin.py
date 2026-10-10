from bokeh.layouts import row
from asyncio import selector_events
import os
os.environ["USE_TF"] = "0"

import math
import requests
import uuid
from dotenv import load_dotenv
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List
from ragas import evaluate

import pandas as pd
from datasets import Dataset

from src.pipeline import LegalRAGPipeline
from src.evaluator import LegalRAGEvaluator


# Run this script from the repository root:
#     python -m src.evaluate_odin
CASES_PATH = Path("data/evaluation/evaluation_cases.json")
OUTPUT_DIR = Path("evaluation/results")

RAGAS_THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.90,
    "context_precision": 0.85,
    "context_recall": 0.85,
}

# Load App Scripts .env to access API and Google Sheets ID
load_dotenv(".sheets.env")

APPS_SCRIPT_URL = os.getenv("APPS_SCRIPT_URL")
API_TOKEN = os.getenv("GOOGLE_SHEETS_API_TOKEN")
RUN_ID = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Streaming to Google Sheets API
# ---------------------------------------------------------------------------

def stream_to_google_sheet(row_data: dict, sync_status: str = "updated") -> None:
    """
    Sending 1 row of evaluation data into Google Sheets through Apps Script Web App.
    """
    payload = {
        "token": API_TOKEN,
        "run_id": RUN_ID,
        "sync_status": sync_status,
        "row": row_data
    }

    try:
        # Sending POST request with payload JSON
        response = requests.post(APPS_SCRIPT_URL, json=payload)

        # Google Apps Script Web App usually response with HTTP 200 or 302 (redirect)
        if response.status_code in (200,302):
            result = response.json()
            if not result.get("ok"):
                print(f"[!] Apps Script Error (case_id: {row_data.get('case_id')}): {result.get('error')}")
        else:
            print(f"[!] Failed to stream case_id: {row_data.get('case_id')} - HTTP Status: {response.status_code}")
    except Exception as e:
        print(f"[!] Request HTTP Error during streaming: {e}")

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def load_cases(path: Path = CASES_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def canonical_document_id(document_id: str) -> str:
    """
    Normalize common document ID naming variants so that
    equivalent legal documents are compared consistently.
    """

    normalized = normalize_text(document_id)

    aliases = {
        "pp_71_2019": "pp_71_2019",
        "pp nomor 71 tahun 2019": "pp_71_2019",

        "uu_11_2008": "uu_11_2008",
        "uu nomor 11 tahun 2008": "uu_11_2008",

        "uu nomor 1 tahun 2024": "uu_1_2024",

        "uu nomor 27 tahun 2022": "uu_27_2022",

        "uu nomor 20 tahun 2025": "uu_20_2025",

        "permenkominfo nomor 5 tahun 2020":
            "permenkominfo_5_2020",
    }

    normalized = re.sub(
        r"\.pdf$",
        "",
        normalized
    )

    normalized = normalized.strip()

    return aliases.get(
        normalized,
        normalized
    )


# ---------------------------------------------------------------------------
# Reference matching
# ---------------------------------------------------------------------------

def flatten_expected_references(
    expected_references: List[Dict[str, Any]],
) -> set[tuple[str, str]]:
    """
    Convert nested expected_references into a set of:
        (canonical_document_id, normalized_article)
    """
    references = set()

    for ref in expected_references or []:
        document_id = canonical_document_id(ref.get("document_id", ""))

        for article in ref.get("articles", []) or []:
            article_text = normalize_text(article)
            if document_id and article_text:
                references.add((document_id, article_text))

    return references


def get_retrieved_reference_pairs(
    sources: List[Dict[str, Any]],
) -> set[tuple[str, str]]:
    """
    Build (document_id, article) pairs from retrieved RAG sources.
    """
    pairs = set()

    for source in sources or []:
        document_id = canonical_document_id(source.get("document_id", ""))
        article = normalize_text(
            source.get("pasal")
            or source.get("article")
            or ""
        )

        if document_id and article:
            pairs.add((document_id, article))

    return pairs


def reference_match_stats(
    expected_references: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_pairs = flatten_expected_references(
        expected_references
    )
    retrieved_pairs = get_retrieved_reference_pairs(
        sources
    )

    matched_pairs = expected_pairs.intersection(retrieved_pairs)

    expected_count = len(expected_pairs)
    matched_count = len(matched_pairs)
    retrieved_count = len(retrieved_pairs)

    return {
        "expected_reference_count": expected_count,
        "matched_reference_count": matched_count,
        "retrieved_reference_count": retrieved_count,
        "reference_recall": (
            round(matched_count / expected_count, 4)
            if expected_count
            else float("nan")
        ),
        "reference_precision": (
            round(matched_count / retrieved_count, 4)
            if retrieved_count
            else float("nan")
        ),
        "expected_references": sorted(
            f"{doc} | {article}"
            for doc, article in expected_pairs
        ),
        "matched_references": sorted(
            f"{doc} | {article}"
            for doc, article in matched_pairs
        ),
        "retrieved_references": sorted(
            f"{doc} | {article}"
            for doc, article in retrieved_pairs
        ),
    }


# ---------------------------------------------------------------------------
# Abstention detection
# ---------------------------------------------------------------------------

def is_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)

    phrases = [
        "informasi mengenai hal tersebut tidak ditemukan",
        "informasi yang ditanyakan tidak ditemukan",
        "tidak ditemukan dokumen hukum yang relevan",
        "tidak ditemukan informasi yang relevan",
    ]

    return any(phrase in normalized for phrase in phrases)


# ---------------------------------------------------------------------------
# Context extraction for RAGAS
# ---------------------------------------------------------------------------

def extract_contexts(sources: List[Dict[str, Any]]) -> List[str]:
    """
    Extract textual contexts for RAGAS from the same source objects
    returned by the single RAG execution.

    Supports the current repo's content_preview plus common fallbacks.
    """
    contexts = []

    for source in sources or []:
        context = (
            source.get("content_preview")
            or source.get("content")
            or source.get("page_content")
            or source.get("text")
            or ""
        )

        if context:
            contexts.append(str(context))

    return contexts


# ---------------------------------------------------------------------------
# Status classification (replaces old classify_failure)
# ---------------------------------------------------------------------------

def classify_retrieval_status(row: Dict[str, Any]) -> str:
    """
    Independent retrieval quality status.

    Values:
        retrieval_empty   – no sources returned at all
        retrieval_miss    – sources returned but zero expected refs matched
        partial_retrieval – at least one match but recall < 0.50
        strong_retrieval  – recall >= 0.50
    """
    if row["source_count"] == 0:
        return "retrieval_empty"

    if row["matched_reference_count"] == 0:
        return "retrieval_miss"

    recall = row.get("reference_recall", 0.0)
    if pd.isna(recall):
        recall = 0.0

    if recall < 0.50:
        return "partial_retrieval"

    return "strong_retrieval"


def classify_answer_quality(row: Dict[str, Any]) -> str:
    """
    Independent answer quality status based on RAGAS thresholds.

    Values:
        evaluation_error  – RAGAS metrics are not available
        quality_failure   – at least one metric below threshold
        pass              – all four metrics meet threshold
    """
    metrics_available = 0
    metrics_passing = 0

    for metric, threshold in RAGAS_THRESHOLDS.items():
        value = row.get(metric)
        if value is not None and pd.notna(value):
            metrics_available += 1
            if value >= threshold:
                metrics_passing += 1

    if metrics_available == 0:
        return "evaluation_error"

    if metrics_available < len(RAGAS_THRESHOLDS):
        # Not all metrics computed — count what we have.
        # If any available metric is below threshold, fail.
        if metrics_passing < metrics_available:
            return "quality_failure"
        return "evaluation_error"

    # All four metrics available.
    if metrics_passing == len(RAGAS_THRESHOLDS):
        return "pass"

    return "quality_failure"


def classify_overall_status(row: Dict[str, Any]) -> str:
    """
    Combined status that brings retrieval and answer quality together.

    Values:
        execution_error        – RAG execution raised an exception
        retrieval_failure      – retrieval_empty or retrieval_miss
        retrieval_partial      – partial_retrieval
        answer_quality_failure – strong retrieval but answer quality not pass
        pass                   – strong retrieval + answer quality pass
    """
    if row.get("error"):
        return "execution_error"

    retrieval = row["retrieval_status"]

    if retrieval in ("retrieval_empty", "retrieval_miss"):
        return "retrieval_failure"

    if retrieval == "partial_retrieval":
        return "retrieval_partial"

    # retrieval == "strong_retrieval"
    if row["answer_quality_status"] != "pass":
        return "answer_quality_failure"
    
    #if row["answer_quality_status"] == "evaluation_error":
    #return "evaluation_error"

    #if row["answer_quality_status"] == "quality_failure":
    #return "answer_quality_failure"

    return "pass"


# ---------------------------------------------------------------------------
# Result row builders
# ---------------------------------------------------------------------------

def build_result_row(
    case: Dict[str, Any],
    result: Dict[str, Any],
    latency: float,
) -> Dict[str, Any]:
    sources = result.get("sources", [])
    expected_references = case.get("expected_references", [])

    stats = reference_match_stats(
        expected_references=expected_references,
        sources=sources,
    )

    answer = result.get("answer", "") or ""
    abstention = is_abstention(answer)

    contexts = extract_contexts(sources)

    row = {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_answer": case.get("expected_answer", ""),
        "expected_references": expected_references,
        "query_type": case.get("query_type", ""),
        "severity": case.get("severity", ""),
        "answer": answer,
        "source_count": len(sources),
        "latency_seconds": latency,
        "retrieved_articles": [
            source.get("pasal") or source.get("article") or ""
            for source in sources
        ],
        "retrieved_document_ids": [
            source.get("document_id", "")
            for source in sources
        ],
        "top_rerank_score": max(
            [
                source.get("score")
                for source in sources
                if source.get("score") is not None
            ],
            default=None,
        ),
        "expected_reference_count": stats[
            "expected_reference_count"
        ],
        "matched_reference_count": stats[
            "matched_reference_count"
        ],
        "retrieved_reference_count": stats[
            "retrieved_reference_count"
        ],
        "reference_recall": stats["reference_recall"],
        "reference_precision": stats["reference_precision"],
        "expected_references_flat": stats[
            "expected_references"
        ],
        "matched_references": stats[
            "matched_references"
        ],
        "retrieved_references": stats[
            "retrieved_references"
        ],
        "abstention": abstention,
        "unexpected_abstention": (
            abstention
            and stats["expected_reference_count"] > 0
        ),
        "error": "",
        "_ragas_contexts": contexts,
    }

    return row


def build_error_row(
    case: Dict[str, Any],
    latency: float,
    error_message: str,
) -> Dict[str, Any]:
    expected_references = case.get(
        "expected_references",
        [],
    )
    expected_count = len(
        flatten_expected_references(
            expected_references
        )
    )

    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "expected_answer": case.get("expected_answer", ""),
        "expected_references": expected_references,
        "query_type": case.get("query_type", ""),
        "severity": case.get("severity", ""),
        "answer": "",
        "source_count": 0,
        "latency_seconds": latency,
        "retrieved_articles": [],
        "retrieved_document_ids": [],
        "top_rerank_score": None,
        "expected_reference_count": expected_count,
        "matched_reference_count": 0,
        "retrieved_reference_count": 0,
        "reference_recall": 0.0,
        "reference_precision": float("nan"),
        "expected_references_flat": sorted(
            f"{doc} | {article}"
            for doc, article in flatten_expected_references(
                expected_references
            )
        ),
        "matched_references": [],
        "retrieved_references": [],
        "abstention": False,
        "unexpected_abstention": False,
        "error": error_message,
        "_ragas_contexts": [],
    }


# ---------------------------------------------------------------------------
# RAGAS runner
# ---------------------------------------------------------------------------

def run_ragas(
    raw_results: List[Dict[str, Any]],
) -> None:
    """
    Run RAGAS using contexts captured during the SAME RAG execution.
    This prevents a second RAG execution per query.
    """
    valid_rows = [
        row
        for row in raw_results
        if not row["error"]
        and row["answer"]
        and row["_ragas_contexts"]
    ]

    if not valid_rows:
        print("[!] No valid rows with answer + contexts for RAGAS.")
        return

    ragas_inputs = [
        {
            "question": row["query"],
            "contexts": row["_ragas_contexts"],
            "answer": row["answer"],
            "ground_truth": row["expected_answer"],
        }
        for row in valid_rows
    ]

    print(
        f"[*] Sending {len(ragas_inputs)} cases to RAGAS..."
    )

    dataset = Dataset.from_list(ragas_inputs)
    evaluator = LegalRAGEvaluator()
    scores = evaluator.evaluate_pipeline(dataset)
    #scores = evaluate(
    #    dataset=dataset,
    #    metrics=evaluator.metrics,
    #    raise_exceptions=True,
    #)
    ragas_df = scores.to_pandas()

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    for idx, row in enumerate(valid_rows):
        if idx >= len(ragas_df):
            continue

        ragas_row = ragas_df.iloc[idx]

        for metric in metric_names:
            value = ragas_row.get(metric)
            row[metric] = value


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading evaluation cases...")
    cases = load_cases()
    print(f"[*] Loaded {len(cases)} evaluation cases.")

    print("[2/5] Running Legal RAG pipeline...")
    pipeline = LegalRAGPipeline()

    raw_results: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] "
            f"{case['case_id']} - {case['query']}"
        )

        start = time.time()

        try:
            result = pipeline.answer_query(
                query=case["query"],
                top_k=10,
                top_n=3,
            )

            latency = round(
                time.time() - start,
                4,
            )

            row = build_result_row(
                case=case,
                result=result,
                latency=latency,
            )

        except Exception as exc:
            row = build_error_row(
                case=case,
                latency=round(
                    time.time() - start,
                    4,
                ),
                error_message=str(exc),
            )

        raw_results.append(row)

    print("[3/5] Running RAGAS evaluation...")
    run_ragas(raw_results)

    # ------------------------------------------------------------------
    # Classify each row with three independent status fields.
    # ------------------------------------------------------------------
    print("[4/5] Classifying statuses...")

    for row in raw_results:
        row["retrieval_status"] = classify_retrieval_status(row)
        row["answer_quality_status"] = classify_answer_quality(row)
        row["overall_status"] = classify_overall_status(row)

    # ------------------------------------------------------------------
    # Export results & Streaming into Google Sheets
    # ------------------------------------------------------------------
    print("[*] Preparing data and streaming into Google Sheets...")

    # Strip internal fields before writing CSV.
    export_rows = []
    for row in raw_results:
        clean_row = {
            key: value
            for key, value in row.items()
            if key != "_ragas_contexts"
        }

        for k, v in clean_row.items():
            if isinstance(v, float) and math.isnan(v):
                clean_row[k] = None

        # List conversion into string for easier insert into Google Sheet cell
        clean_row["retrieved_articles"] = ", ".join(clean_row["retrieved_articles"]) if isinstance(clean_row["retrieved_articles"], list) else clean_row["retrieved_articles"]
        clean_row["retrieved_document_ids"] = ", ".join(clean_row["retrieved_document_ids"]) if isinstance(clean_row["retrieved_document_ids"], list) else clean_row["retrieved_document_ids"]
        clean_row["expected_references_flat"] = ", ".join(clean_row["expected_references_flat"]) if isinstance(clean_row["expected_references_flat"], list) else clean_row["expected_references_flat"]
        clean_row["matched_references"] = ", ".join(clean_row["matched_references"]) if isinstance(clean_row["matched_references"], list) else clean_row["matched_references"]
        clean_row["retrieved_references"] = ", ".join(clean_row["retrieved_references"]) if isinstance(clean_row["retrieved_references"], list) else clean_row["retrieved_references"]
        
        export_rows.append(clean_row)

        # STREAMING: Send cleaned row into Google Sheets
        stream_to_google_sheet(clean_row)

    print("[*] Streaming completed.")

    df = pd.DataFrame(export_rows)

    results_path = OUTPUT_DIR / "evaluation_results.csv"
    df.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    failures_df = df[
        df["overall_status"] != "pass"
    ]

    failures_path = OUTPUT_DIR / "failure_analysis.csv"
    failures_df.to_csv(
        failures_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------

    summary = {
        "total_cases": int(len(df)),
        "retrieval_empty_cases": int(
            (df["retrieval_status"] == "retrieval_empty").sum()
        ),
        "retrieval_miss_cases": int(
            (df["retrieval_status"] == "retrieval_miss").sum()
        ),
        "partial_retrieval_cases": int(
            (df["retrieval_status"] == "partial_retrieval").sum()
        ),
        "strong_retrieval_cases": int(
            (df["retrieval_status"] == "strong_retrieval").sum()
        ),
        "answer_quality_failure_cases": int(
            (df["answer_quality_status"] == "quality_failure").sum()
        ),
        "passed_cases": int(
            (df["overall_status"] == "pass").sum()
        ),
        "average_reference_recall": round(
            df["reference_recall"].mean(),
            4,
        ),
        "average_reference_precision": round(
            df["reference_precision"].mean(),
            4,
        ),
        "average_latency_seconds": round(
            df["latency_seconds"].mean(),
            4,
        ),
    }

    # Add aggregate RAGAS metrics when available.
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        if metric in df.columns:
            summary[f"average_{metric}"] = (
                round(
                    df[metric].mean(),
                    4,
                )
            )

    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("[5/5] Evaluation completed.")
    print("\n=== SUMMARY ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResults: {results_path}")
    print(f"Failures: {failures_path}")


if __name__ == "__main__":
    main()