import pytest
from unittest.mock import MagicMock, patch
from src.retrieval.query_processor import query_processor
from src.retrieval.retriever import retriever, reciprocal_rank_fusion

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

@patch("src.retrieval.retriever.bm25_indexer")
@patch("src.retrieval.retriever.chroma_manager")
@patch("src.retrieval.retriever.generate_embeddings")
def test_retrieve(mock_gen_embed, mock_chroma, mock_bm25):
    # Mock embedding generation
    mock_gen_embed.return_value = [[0.1, 0.2, 0.3]]
    
    # Mock ChromaDB query results
    mock_chroma.query.return_value = {
        'ids': [['id1', 'id2']],
        'distances': [[0.1, 0.2]],
        'metadatas': [[{'dept': 'finance'}, {'dept': 'finance'}]],
        'documents': [['doc1 text', 'doc2 text']]
    }
    
    # Mock BM25 returning empty (so vector-only fallback)
    mock_bm25.search.return_value = []
    
    results = retriever.retrieve("test query", ["finance"], top_k=2)
    
    assert len(results) == 2
    assert results[0]["id"] == "id1"
    assert results[0]["score"] == 0.1
    assert results[0]["text"] == "doc1 text"


def test_rrf_fusion():
    """Test that Reciprocal Rank Fusion correctly merges and re-ranks results."""
    # Vector results (sorted by distance ascending)
    vector_results = [
        {"id": "doc_A", "score": 0.1, "metadata": {}, "text": "Document A text", "department": "finance"},
        {"id": "doc_B", "score": 0.2, "metadata": {}, "text": "Document B text", "department": "finance"},
        {"id": "doc_C", "score": 0.5, "metadata": {}, "text": "Document C text", "department": "finance"},
    ]
    
    # BM25 results (sorted by bm25_score descending)
    bm25_results = [
        {"text": "Document B text", "metadata": {"record_id": "doc_B"}, "bm25_score": 5.0, "department": "finance"},
        {"text": "Document D text", "metadata": {"record_id": "doc_D"}, "bm25_score": 3.0, "department": "finance"},
        {"text": "Document A text", "metadata": {"record_id": "doc_A"}, "bm25_score": 1.0, "department": "finance"},
    ]
    
    fused = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
    
    # doc_A appears in both lists (rank 1 vector + rank 3 BM25) -> should have high combined score
    # doc_B appears in both lists (rank 2 vector + rank 1 BM25) -> should have highest combined score
    # doc_C appears only in vector (rank 3) -> lower score
    # doc_D appears only in BM25 (rank 2) -> lower score
    
    assert len(fused) == 4  # All unique docs
    
    # doc_B should be #1 (appears in both, high ranks in both)
    assert fused[0]["id"] == "doc_B"
    
    # doc_A should be #2 (appears in both, but lower BM25 rank)
    assert fused[1]["id"] == "doc_A"
    
    # All results should have rrf_score
    for result in fused:
        assert "rrf_score" in result
        assert result["rrf_score"] > 0


def test_rrf_fusion_empty_inputs():
    """Test RRF handles edge cases gracefully."""
    # Both empty
    result = reciprocal_rank_fusion([], [])
    assert result == []
    
    # Only vector results
    vector_only = [{"id": "v1", "score": 0.1, "metadata": {}, "text": "test", "department": "hr"}]
    result = reciprocal_rank_fusion(vector_only, [])
    assert len(result) == 1
    assert result[0]["id"] == "v1"
