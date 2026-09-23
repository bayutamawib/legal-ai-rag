import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from src.parser import LegalDocumentParser
except ImportError:
    from parser import LegalDocumentParser

class LegalIngestionPipeline:
    """Pipeline pemrosesan data hukum, validasi skema, dan karantina data cacat."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        quarantine_dir: str = "data/quarantine",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.quarantine_dir = Path(quarantine_dir)

        # Buat direktori jika belum ada
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.parser = LegalDocumentParser()

    def validate_and_enrich_chunk(
        self, raw_chunk: Dict, doc_meta: Dict
    ) -> Tuple[bool, Dict, str]:
        """Validasi kesesuaian skema metadata dan identifikasi error."""
        # 1. Cek ketersediaan konten dan pasal
        if not raw_chunk.get("content") or not raw_chunk.get("pasal"):
            return False, raw_chunk, "MISSING_PASAL_OR_CONTENT"

        # 2. Cek validasi tanggal pengundangan
        enactment_date = doc_meta.get("enactment_date")
        if not enactment_date:
            return False, raw_chunk, "INVALID_OR_MISSING_ENACTMENT_DATE"

        # 3. Bentuk skema kanonikal jika lolos validasi
        pasal_clean = raw_chunk["pasal"].upper().replace(" ", "_")
        canonical_chunk = {
            "chunk_id": f"{doc_meta['document_id']}_{pasal_clean}",
            "document_id": doc_meta["document_id"],
            "title": doc_meta["title"],
            "canonical_law_id": doc_meta["canonical_law_id"],
            "hierarchy_level": doc_meta.get("hierarchy_level", "Peraturan Pemerintah"),
            "bab": raw_chunk.get("bab", "TANPA BAB"),
            "pasal": raw_chunk["pasal"],
            "enactment_date": enactment_date,
            "effective_date": doc_meta.get("effective_date", enactment_date),
            "status": doc_meta.get("status", "ACTIVE"),
            "supersedes": doc_meta.get("supersedes", None),
            "content": raw_chunk["content"],
            "source_url": doc_meta.get("source_url", "")
        }

        return True, canonical_chunk, ""

    def process_pdf_file(self, pdf_path: Path) -> Dict[str, int]:
        """Memproses satu berkas PDF, ekstraksi per pasal, dan pemisahan data bersih vs karantina."""
        doc_id = pdf_path.stem
        
        # 1. Ekstrak chunks dari PDF
        extracted_chunks = self.parser.parse_pp_pdf(str(pdf_path))
        if not extracted_chunks:
            return {"total_extracted": 0, "valid": 0, "quarantined": 0}

        # 2. Ekstrak tanggal pengundangan secara otomatis dari seluruh teks PDF
        full_text = " ".join([c["content"] for c in extracted_chunks])
        enactment_date = self.parser.parse_indonesian_date(full_text)

        # 3. Buat metadata dasar dokumen dari berkas PDF
        doc_meta = {
            "document_id": doc_id,
            "title": doc_id.replace("_", " "),
            "canonical_law_id": doc_id,
            "hierarchy_level": "Peraturan Pemerintah",
            "enactment_date": enactment_date,
            "status": "ACTIVE",
            "source_url": f"file://{pdf_path.resolve()}"
        }

        valid_chunks = []
        quarantined_chunks = []

        # 4. Validasi dan enriquecement per chunk
        for chunk in extracted_chunks:
            is_valid, processed_chunk, error_reason = self.validate_and_enrich_chunk(chunk, doc_meta)

            if is_valid:
                valid_chunks.append(processed_chunk)
            else:
                quarantined_chunks.append({
                    "raw_chunk": processed_chunk,
                    "document_id": doc_meta.get("document_id", "UNKNOWN"),
                    "error_reason": error_reason,
                })

        # 5. Simpan data valid
        if valid_chunks:
            out_path = self.processed_dir / f"{doc_id}_processed.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(valid_chunks, f, ensure_ascii=False, indent=2)

        # 6. Simpan data cacat ke karantina
        if quarantined_chunks:
            q_path = self.quarantine_dir / f"{doc_id}_quarantine.json"
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(quarantined_chunks, f, ensure_ascii=False, indent=2)

        return {
            "total_extracted": len(extracted_chunks),
            "valid": len(valid_chunks),
            "quarantined": len(quarantined_chunks),
        }

    def run_all(self):
        """Memproses seluruh PDF yang ada di folder raw."""
        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"[!] Tidak ada berkas PDF ditemukan di {self.raw_dir.resolve()}")
            return

        for pdf_path in pdf_files:
            print(f"[*] Memproses: {pdf_path.name}...")
            stats = self.process_pdf_file(pdf_path)
            print(f"    - Extracted : {stats['total_extracted']}")
            print(f"    - Valid     : {stats['valid']}")
            print(f"    - Quarantined: {stats['quarantined']}")

if __name__ == "__main__":
    pipeline = LegalIngestionPipeline()
    pipeline.run_all()