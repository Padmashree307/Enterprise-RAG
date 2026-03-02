import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.query_processor import query_processor
from src.retrieval.retriever import retriever

def test_detect_departments():
    # Direct keyword matches
    assert query_processor.detect_departments("budget report") == ["finance"]
    assert query_processor.detect_departments("hiring process") == ["hr"]
    assert query_processor.detect_departments("factory output") == ["manufacturing"]
    
    # Fallback
    fallback = query_processor.detect_departments("hello world")
    assert "finance" in fallback
    assert "hr" in fallback
    assert "manufacturing" in fallback

@patch("src.retrieval.retriever.chroma_manager")
@patch("src.retrieval.retriever.generate_embeddings")
def test_retrieve(mock_gen_embed, mock_chroma):
    # Mock embedding generation
    mock_gen_embed.return_value = [[0.1, 0.2, 0.3]]
    
    # Mock ChromaDB query results
    mock_chroma.query.return_value = {
        'ids': [['id1', 'id2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[{'dept': 'finance'}, {'dept': 'finance'}]],
        'documents': [['doc1 text', 'doc2 text']]
    }
    
    results = retriever.retrieve("test query", ["finance"], top_k=2)
    
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["score"] == 0.1
    assert results[0]["text"] == "doc1 text"
