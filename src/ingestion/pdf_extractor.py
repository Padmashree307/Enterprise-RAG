import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
from src.utils.logger import logger
from src.utils.helpers import timer

@timer
def extract_text_from_pdf(file_path: Path, department: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF file page by page.
    
    Args:
        file_path (Path): Path to the PDF file.
        department (str): Department name for metadata.
        
    Returns:
        List[Dict]: List of document dictionaries with text and metadata.
    """
    documents = []
    
    if not file_path.exists():
        logger.error(f"PDF file not found: {file_path}")
        return []

    try:
        # Open PDF with PyMuPDF
        with fitz.open(file_path) as doc:
            logger.info(f"Extracting text from {file_path.name} ({len(doc)} pages)")
            
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                if not text.strip():
                    continue
                    
                # Create a document dict
                doc_record = {
                    "text": text,
                    "metadata": {
                        "department": department,
                        "source_type": "pdf",
                        "source_file": file_path.name,
                        "page_number": page_num + 1
                    }
                }
                documents.append(doc_record)
                
        logger.info(f"Extracted {len(documents)} pages from {file_path.name}")
        return documents

    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        return []
