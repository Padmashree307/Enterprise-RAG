import argparse
import json
from pathlib import Path
from typing import List, Dict

from src.config.settings import settings
from src.utils.logger import logger
from src.ingestion.pdf_extractor import extract_text_from_pdf
from src.ingestion.text_parser import parse_text_file
from src.ingestion.chunker import chunk_documents
from src.embeddings.generator import generate_embeddings
from src.vectorstore.chroma_manager import chroma_manager

def run_extraction() -> Dict[str, List[Dict]]:
    """
    Stage 1: Extracts raw text/records from all configured sources.
    Returns a dict of {department: [documents]}.
    Saves intermediate JSONs to data/processed/chunks/.
    """
    logger.info(">>> STARTING EXTRACTION PHASE <<<")
    results = {}
    
    for dept_key, config in settings.DEPARTMENTS.items():
        logger.info(f"Processing Department: {config.name}")
        dept_docs = []
        
        # 1. Extract PDF
        pdf_file = config.raw_path / config.file_types["pdf"]
        if pdf_file.exists():
            docs = extract_text_from_pdf(pdf_file, dept_key)
            dept_docs.extend(docs)
        else:
            logger.warning(f"File missing: {pdf_file}")
            
        # 2. Extract Text
        txt_file = config.raw_path / config.file_types["txt"]
        if txt_file.exists():
            docs = parse_text_file(txt_file, dept_key)
            dept_docs.extend(docs)
        else:
            logger.warning(f"File missing: {txt_file}")
            
        results[dept_key] = dept_docs
        
        # Save intermediate
        output_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_raw.json"
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dept_docs, f, indent=2, default=str)
            
        logger.info(f"Saved {len(dept_docs)} raw documents to {output_path.name}")
        
    return results

def run_chunking(extracted_data: Dict[str, List[Dict]] = None) -> Dict[str, List[Dict]]:
    """
    Stage 2: Chunks the extracted documents.
    If extracted_data is None, tries to load from intermediate JSONs.
    """
    logger.info(">>> STARTING CHUNKING PHASE <<<")
    results = {}
    
    # If no data provided, load from disk
    if not extracted_data:
        extracted_data = {}
        for dept_key in settings.DEPARTMENTS.keys():
            input_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_raw.json"
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as f:
                    extracted_data[dept_key] = json.load(f)
            else:
                logger.warning(f"No raw data found for {dept_key}, skipping chunking.")
    
    for dept_key, docs in extracted_data.items():
        if not docs:
            continue
            
        logger.info(f"Chunking {len(docs)} documents for {dept_key}...")
        chunked_docs = chunk_documents(docs)
        results[dept_key] = chunked_docs
        
        # Save intermediate
        output_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_chunked.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunked_docs, f, indent=2, default=str)
            
        logger.info(f"Saved {len(chunked_docs)} chunks to {output_path.name}")
        
    return results

def run_embedding(chunked_data: Dict[str, List[Dict]] = None) -> Dict[str, List[Dict]]:
    """
    Stage 3: Generates embeddings for chunks.
    Results are added to the document dicts under 'embedding'.
    """
    logger.info(">>> STARTING EMBEDDING PHASE <<<")
    results = {}
    
    # Load from disk if needed
    if not chunked_data:
        chunked_data = {}
        for dept_key in settings.DEPARTMENTS.keys():
            input_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_chunked.json"
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as f:
                    chunked_data[dept_key] = json.load(f)
            else:
                logger.warning(f"No chunked data found for {dept_key}, skipping embedding.")
    
    for dept_key, docs in chunked_data.items():
        if not docs:
            continue
            
        logger.info(f"Embedding {len(docs)} chunks for {dept_key}...")
        
        # Generate embeddings
        # generator.py now returns List[List[float]]
        embeddings = generate_embeddings(docs, batch_size=16)
        
        embedded_docs = []
        for doc, emb in zip(docs, embeddings):
            new_doc = doc.copy()
            new_doc["embedding"] = emb
            embedded_docs.append(new_doc)
            
        results[dept_key] = embedded_docs
        
        # Save intermediate
        output_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_embedded.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embedded_docs, f, indent=2, default=str)
        
        logger.info(f"Saved {len(embedded_docs)} embedded docs to {output_path.name}")
        
    return results

def run_indexing(embedded_data: Dict[str, List[Dict]] = None):
    """
    Stage 4: Indexes documents into ChromaDB.
    """
    logger.info(">>> STARTING INDEXING PHASE <<<")
    
    # Load from disk if needed
    if not embedded_data:
        embedded_data = {}
        for dept_key in settings.DEPARTMENTS.keys():
            input_path = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks" / f"{dept_key}_embedded.json"
            if input_path.exists():
                with open(input_path, "r", encoding="utf-8") as f:
                    embedded_data[dept_key] = json.load(f)
            else:
                logger.warning(f"No embedded data found for {dept_key}, skipping indexing.")
                
    for dept_key, docs in embedded_data.items():
        if not docs:
            continue
            
        collection_name = f"{dept_key}_kb"
        logger.info(f"Indexing {len(docs)} documents into '{collection_name}'...")
        
        chroma_manager.add_documents(collection_name, docs)
        
    logger.info("Indexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Pipeline")
    parser.add_argument("--stage", choices=["extract", "chunk", "embed", "index", "full"], default="full")
    args = parser.parse_args()
    
    # Initialize variables
    extracted_docs = None
    chunked_docs = None
    embedded_docs = None
    
    start_stage = args.stage
    
    # Pipeline Logic
    if start_stage == "full":
        extracted_docs = run_extraction()
        chunked_docs = run_chunking(extracted_docs)
        embedded_docs = run_embedding(chunked_docs)
        run_indexing(embedded_docs)
        
    elif start_stage == "extract":
        run_extraction()
        
    elif start_stage == "chunk":
        run_chunking(None)
        
    elif start_stage == "embed":
        run_embedding(None)
        
    elif start_stage == "index":
        run_indexing(None)
        
    logger.info(f"Pipeline finished for stage: {start_stage}")
