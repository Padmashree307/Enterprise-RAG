from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.helpers import timer

@timer
def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Splits documents into smaller chunks.
    - PDF pages are split using RecursiveCharacterTextSplitter.
    - Structured records are KEPT AS IS (1 record = 1 chunk).
    
    Args:
        documents (List[Dict]): List of raw documents with 'text' and 'metadata'.
        
    Returns:
        List[Dict]: List of chunked documents.
    """
    chunked_docs = []
    
    # Initialize splitter for PDFs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len # Character count
    )
    
    for doc in documents:
        source_type = doc["metadata"].get("source_type", "unknown")
        
        if source_type == "structured":
            # Keep as is, just add chunk info
            new_doc = doc.copy()
            new_doc["metadata"] = doc["metadata"].copy()
            new_doc["metadata"]["chunk_index"] = 0
            chunked_docs.append(new_doc)
            
        elif source_type == "pdf":
            # Split text
            chunks = text_splitter.split_text(doc["text"])
            
            for i, chunk_text in enumerate(chunks):
                new_doc = {
                    "text": chunk_text,
                    "metadata": doc["metadata"].copy()
                }
                new_doc["metadata"]["chunk_index"] = i
                chunked_docs.append(new_doc)
                
        else:
            logger.warning(f"Unknown source type: {source_type}")
            
    logger.info(f"Chunked {len(documents)} docs into {len(chunked_docs)} chunks")
    return chunked_docs
