import pytest
from src.ingest import LegalIngestionPipeline

@pytest.fixture
def pipeline(tmp_path):
    # Use temporary directory so it doesn't contaminate pure folder
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    quarantine = tmp_path / "quarantine"
    return LegalIngestionPipeline(
        raw_dir=str(raw),
        processed_dir=str(processed),
        quarantine_dir=str(quarantine)
    )

def test_validate_and_enrich_chunk_success(pipeline):
    raw_chunk = {
        "bab": "BAB I",
        "pasal": "Pasal 1",
        "content": "Isi teks pasal yang valid"
    }
    doc_meta = {
        "document_id": "PP_71_2019",
        "title": "PP No 71 Tahun 2019",
        "canonical_law_id": "PP_71_2019",
        "enactment_date": "2019-10-10"
    }
    is_valid, enriched_chunk, error_reason = pipeline.validate_and_enrich_chunk(raw_chunk, doc_meta)

    assert is_valid is True
    assert error_reason == ""
    assert enriched_chunk["chunk_id"] == "PP_71_2019_PASAL_1"
    assert enriched_chunk["status"] == "ACTIVE"

def test_validate_and_enrich_chunk_missing_content(pipeline):
    raw_chunk = {
        "bab": "BAB I",
        "pasal": "Pasal 1",
        "content": ""
    }
    doc_meta = {
        "document_id": "PP_71_2019",
        "enactment_date": "2019-10-10"
    }

    is_valid, _, error_reason = pipeline.validate_and_enrich_chunk(raw_chunk, doc_meta)

    assert is_valid is False
    assert error_reason == "MISSING_PASAL_OR_CONTENT"

def test_validate_and_enrich_chunk_missing_date(pipeline):
    raw_chunk = {
        "bab": "BAB I",
        "pasal": "Pasal 1",
        "content": "Isi teks pasal."
    }
    doc_meta = {
        "bab": "BAB I",
        "pasal": "Pasal 1",
        "content": "Isi teks pasal."
    }

    is_valid, _, error_reason = pipeline.validate_and_enrich_chunk(raw_chunk, doc_meta)

    assert is_valid is False
    assert error_reason == "INVALID_OR_MISSING_ENACTMENT_DATE"