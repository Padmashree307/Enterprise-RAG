import pytest
from pathlib import Path
from src.ingestion.text_parser import parse_text_file
from src.ingestion.chunker import chunk_documents

# Mock data
SAMPLE_TEXT_CONTENT = "Transaction Id: FIN-1001. Date: 2023-10-15. Department: Finance. Amount Eur: 1500.50. Description: Office supplies purchase."

def test_parse_text_file(tmp_path):
    # Create a temporary text file
    d = tmp_path / "financial"
    d.mkdir()
    p = d / "test_finance.txt"
    p.write_text(SAMPLE_TEXT_CONTENT, encoding="utf-8")
    
    docs = parse_text_file(p, "finance")
    
    assert len(docs) == 1
    doc = docs[0]
    assert doc["metadata"]["department"] == "finance"
    assert doc["metadata"]["source_type"] == "structured"
    assert doc["metadata"]["record_id"] == "FIN-1001"
    # Check content parsing
    assert "Amount Eur is 1500.50" in doc["text"]

def test_chunker_structured():
    # Structured docs should NOT be split
    docs = [{
        "text": "Some short text",
        "metadata": {"source_type": "structured", "department": "finance"}
    }]
    
    chunks = chunk_documents(docs)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["chunk_index"] == 0

def test_chunker_pdf():
    # PDF docs should be split if long
    long_text = "Word " * 1000 # ~5000 chars
    docs = [{
        "text": long_text,
        "metadata": {"source_type": "pdf", "department": "hr", "source_file": "test.pdf"}
    }]
    
    # Settings default chunk size is 500
    chunks = chunk_documents(docs)
    assert len(chunks) > 1
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["metadata"]["chunk_index"] == 1
