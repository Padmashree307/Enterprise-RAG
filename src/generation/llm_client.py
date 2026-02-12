"""
Ollama LLM client for RAG generation.
Uses HTTP API to avoid heavy SDK dependencies.
"""
import json
import requests
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.helpers import timer, log_memory_usage

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL
        self.generate_url = f"{self.base_url}/api/generate"
        logger.info(f"OllamaClient initialized: model={self.model}, url={self.base_url}")

    def is_available(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            logger.error("Ollama server is not reachable.")
            return False

    @timer
    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 256) -> str:
        """
        Generate a response from the LLM.
        Uses streaming=False for simplicity on 8GB RAM.
        
        Args:
            prompt: The full prompt (system + context + question).
            temperature: Sampling temperature (low = more deterministic).
            max_tokens: Max tokens in the response.
            
        Returns:
            str: The generated response text.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,       # Context window (keep small for RAM)
                "num_thread": 4,       # CPU threads
            }
        }
        
        try:
            log_memory_usage("before_llm_generate")
            resp = requests.post(self.generate_url, json=payload, timeout=300)
            resp.raise_for_status()
            
            result = resp.json()
            response_text = result.get("response", "").strip()
            
            # Log stats
            total_duration = result.get("total_duration", 0) / 1e9  # ns to seconds
            eval_count = result.get("eval_count", 0)
            logger.info(f"LLM generation: {eval_count} tokens in {total_duration:.1f}s")
            log_memory_usage("after_llm_generate")
            
            return response_text
            
        except requests.Timeout:
            logger.error("LLM request timed out (300s limit)")
            return "[Error: LLM request timed out. Model may still be loading — try again.]"
        except requests.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            return "[Error: Cannot connect to Ollama. Run 'ollama serve' first.]"
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"[Error: {str(e)}]"

    def generate_stream(self, prompt: str, temperature: float = 0.1, max_tokens: int = 256):
        """
        Generate a streaming response from the LLM.
        Yields chunks of text as they are generated.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "num_thread": 4,
            }
        }
        
        try:
            with requests.post(self.generate_url, json=payload, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                
                for line in resp.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if not chunk.get("done"):
                                yield chunk.get("response", "")
                            else:
                                # Log stats from final chunk
                                total_duration = chunk.get("total_duration", 0) / 1e9
                                eval_count = chunk.get("eval_count", 0)
                                logger.info(f"LLM stream: {eval_count} tokens in {total_duration:.1f}s")
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"[Error: {str(e)}]"

# Singleton
llm_client = OllamaClient()
