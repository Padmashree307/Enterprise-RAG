import pytest
from unittest.mock import MagicMock, patch
from src.generation.llm_client import llm_client
from src.generation.prompt_builder import build_rag_prompt

def test_build_rag_prompt():
    docs = [
        {"text": "Context 1", "department": "finance", "score": 0.1},
        {"text": "Context 2", "department": "hr", "score": 0.2}
    ]
    prompt = build_rag_prompt("My Question", docs)
    
    assert "My Question" in prompt
    assert "Context 1" in prompt
    assert "Context 2" in prompt
    assert "Finance" in prompt
    assert "HR" in prompt

@patch("src.generation.llm_client.requests.post")
def test_llm_generate_success(mock_post):
    # Mock successful response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "response": "Generated Answer",
        "total_duration": 1000000000,
        "eval_count": 10
    }
    mock_post.return_value = mock_resp
    
    answer = llm_client.generate("prompt")
    assert answer == "Generated Answer"

@patch("src.generation.llm_client.requests.post")
def test_llm_generate_timeout(mock_post):
    import requests
    # Mock timeout
    mock_post.side_effect = requests.Timeout
    
    answer = llm_client.generate("prompt")
    assert "timed out" in answer
