import pytest
from src.parser import LegalDocumentParser

@pytest.fixture
def parser():
    return LegalDocumentParser()

def test_parse_indonesian_date_valid(parser):
    text = "Diundangkan di Jakarta pada tanggal 10 Oktober 2019"
    parsed_date = parser.parse_indonesian_date(text)
    assert parsed_date == "2019-10-10"

def test_parse_indonesian_date_invalid(parser):
    text = "Teks ini tidak mengandung informasi tanggal sama sekali."
    parsed_date = parser.parse_indonesian_date(text)
    assert parsed_date is None

def test_extract_articles_structure(parser):
    raw_text = """
    BAB I
    KETENTUAN UMUM

    Pasal 1
    Dalam Peraturan ini yang dimaksud dengan Sistem Elektronik...

    Pasal 2
    Penyelenggara Sistem Elektronik wajib menjaga kenadalan sistem.
    """
    chunks = parser.extract_articles(raw_text)

    assert len(chunks) == 2
    assert chunks[0]["pasal"] == "Pasal 1"
    assert chunks[0]["bab"] == "BAB I"
    assert "Sistem Elektronik" in chunks[0]["content"]
    assert chunks[1]["pasal"] == "Pasal 2"