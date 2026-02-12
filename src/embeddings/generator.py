from typing import List, Dict, Any
import gc
from sentence_transformers import SentenceTransformer
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.helpers import timer, log_memory_usage

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model

@timer
def generate_embeddings(documents: List[Dict[str, Any]], batch_size: int = 16) -> List[List[float]]:
    """
    Generates embeddings for a list of documents in batches.
    
    Args:
        documents (List[Dict]): List of chunked documents.
        batch_size (int): Batch size for encoding (default 16 for 8GB RAM safety).
        
    Returns:
        List[List[float]]: List of embedding vectors.
    """
    model = get_model()
    texts = [doc["text"] for doc in documents]
    embeddings = []
    
    total_docs = len(texts)
    logger.info(f"Generating embeddings for {total_docs} documents (batch_size={batch_size})...")
    
    for i in range(0, total_docs, batch_size):
        batch_texts = texts[i : i + batch_size]
        # Force numpy then tolist to get pure python floats
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, convert_to_tensor=False)
        embeddings.extend(batch_embeddings.tolist())
        
        if (i // batch_size) % 10 == 0:

            logger.info(f"Processed {i + len(batch_texts)}/{total_docs} documents")
            log_memory_usage("embedding_batch")
            
        # Explicit GC to be safe constraints
        gc.collect()
        
    logger.info(f"Generated {len(embeddings)} embeddings.")
    return embeddings
