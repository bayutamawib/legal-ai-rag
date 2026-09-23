import re
import pdfplumber
from typing import Dict, List, Optional

class LegalDocumentParser:
    """Parser untuk mengekstrak dan memproses struktur dokumen hukum Indonesia."""

    INDONESIAN_MONTHS = {
        "januari": "01",
        "februari": "02",
        "maret": "03",
        "april": "04",
        "mei": "05",
        "juni": "06",
        "juli": "07",
        "agustus": "08",
        "september": "09",
        "oktober": "10",
        "november": "11",
        "desember": "12",
    }

    def __init__(self):
        # Regex untuk deteksi struktur BAB dan Pasal
        self.bab_pattern = re.compile(
            r"^\s*(BAB\s+[IVXLCDM]+)\b", re.IGNORECASE | re.MULTILINE
        )
        self.pasal_pattern = re.compile(
            r"^\s*(Pasal\s+\d+[A-Z]?)", re.IGNORECASE | re.MULTILINE
        )
        # Regex untuk ekstrak tanggal Indonesia (contoh: 20 Maret 2024)
        self.date_pattern = re.compile(
            r"(\d{1,2})\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+(\d{4})",
            re.IGNORECASE,
        )

    def sanitize_text(self, text: str) -> str:
        """Membersihkan karakter tak terlihat (unicode artifacts) dan spasi berlebih."""
        if not text:
            return ""

        # Hapus Zero-Width Space (\u200b) & Non-Breaking Space (\xa0)
        text = text.replace("\u200b", "").replace("\xa0", " ")
        # Normalisasi spasi dan baris baru
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def parse_indonesian_date(self, text: str) -> Optional[str]:
        """Mengubah teks tanggal Indonesia ke format ISO 8601 (YYYY-MM-DD)."""
        match = self.date_pattern.search(text)
        if not match:
            return None

        day, month_str, year = match.groups()
        day = day.zfill(2)
        month = self.INDONESIAN_MONTHS.get(month_str.lower())

        if not month:
            return None

        return f"{year}-{month}-{day}"

    def extract_articles(self, raw_text: str) -> List[Dict[str, str]]:
        """Memotong dokumen berdasarkan aturan per-Pasal dan mendeteksi BAB."""
        clean_text = self.sanitize_text(raw_text)
        matches = list(self.pasal_pattern.finditer(clean_text))
        chunks = []

        current_bab = "TANPA BAB"

        for i, match in enumerate(matches):
            pasal_title = match.group(1).strip()
            start_idx = match.start()

            # Tentukan batas akhir pasal
            end_idx = (
                matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
            )
            content = clean_text[start_idx:end_idx].strip()

            # Cari BAB terakhir sebelum posisi pasal saat ini
            text_before = clean_text[:start_idx]
            bab_matches = list(self.bab_pattern.finditer(text_before))
            if bab_matches:
                current_bab = bab_matches[-1].group(1).upper()

            chunks.append(
                {
                    "bab": current_bab,
                    "pasal": pasal_title.capitalize(),
                    "content": content,
                }
            )

        return chunks

    def parse_pp_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """Mengekstrak teks dari PDF Peraturan Pemerintah dan memotongnya per Pasal."""
        raw_text_lines = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue

                # Filter baris header/footer repetitif
                for line in text.split("\n"):
                    clean_line = line.strip()
                    if clean_line.startswith("www.peraturan.go.id") or clean_line.isdigit():
                        continue
                    raw_text_lines.append(clean_line)

        full_text = "\n".join(raw_text_lines)

        # Gunakan extract_articles agar logika pemotongan tetap konsisten
        return self.extract_articles(full_text)


if __name__ == "__main__":
    sample_text = """
    BAB I
    KETENTUAN UMUM
    
    Pasal 1
    Dalam Undang-Undang ini yang dimaksud dengan Informasi Elektronik adalah...
    
    BAB VI
    PERBUATAN YANG DILARANG
    
    Pasal 27A
    Setiap Orang dengan sengaja menyerang kehormatan...
    Diundangkan di Jakarta pada tanggal 2 Januari 2024.
    """

    parser = LegalDocumentParser()
    extracted_chunks = parser.extract_articles(sample_text)
    parsed_date = parser.parse_indonesian_date(sample_text)

    print(f"Parsed Date: {parsed_date}")
    print(f"Extracted Chunks: {len(extracted_chunks)}")
    for c in extracted_chunks:
        print(f"- {c['bab']} | {c['pasal']}: {c['content'][:50]}...")