from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.helpers import timer

class ChromaManager:
    def __init__(self):
        self.db_path = str(settings.CHROMA_DB_PATH)
        logger.info(f"Initializing ChromaDB at {self.db_path}")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
    def get_collection(self, collection_name: str):
        """
        Get or create a collection. We use cosine distance by default.
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    @timer
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """
        Add documents to a collection.
        Expects documents to have 'text', 'embedding', and 'metadata'.
        """
        collection = self.get_collection(collection_name)
        
        ids = []
        embeddings = []
        metadatas = []
        docs = []
        
        for i, doc in enumerate(documents):
            # Generate a unique ID if not present
            # We use source_file + chunk_index or record_id
            dept = doc['metadata'].get('department', 'unknown')
            src = doc['metadata'].get('source_file', 'unknown')
            idx = doc['metadata'].get('chunk_index', i)
            rec_id = doc['metadata'].get('record_id', f"{src}_chk_{idx}")
            
            # Ensure unique ID format
            # If records already have record_id, use it. But PDF chunks need generated IDs.
            # Logic: if 'record_id' starts with 'FIN-', it's unique.
            # If it's a PDF chunk, it's 'UNIDO_Finance.pdf_chk_0'
            
            unique_id = str(rec_id)
            if 'record_id' not in doc['metadata']:
                # PDF chunk or other source without native record ID
                page = doc['metadata'].get('page_number')
                if page is not None:
                     unique_id = f"{src}_pg{page}_chk_{idx}"
                else:
                     unique_id = f"{src}_chk_{idx}"
            
            ids.append(unique_id)
            embeddings.append(doc['embedding'])
            
            # Clean metadata: Chroma only supports str, int, float, bool
            clean_meta = {}
            for k, v in doc['metadata'].items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    # Convert others to string (e.g. lists, dicts, None)
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)
            
            docs.append(doc['text'])
            
        # Batch upsert
        batch_size = 40  # Small batch for SQLite safety
        total = len(ids)
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            try:
                collection.upsert(
                    ids=ids[i:end],
                    embeddings=embeddings[i:end],
                    metadatas=metadatas[i:end],
                    documents=docs[i:end]
                )
            except Exception as e:
                logger.error(f"Error upserting batch {i} to {end}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error msg (truncated): {str(e)[:500]}...")
                # Log first item in batch to debug
                if i < total:
                    logger.error(f"Sample ID: {ids[i]}")

                    logger.error(f"Sample Metadata: {metadatas[i]}")
                raise e
            
        logger.info(f"Upserted {total} documents to collection '{collection_name}'")

    def query(self, collection_name: str, query_embedding: List[float], n_results: int = 5, where: Optional[Dict] = None):
        """
        Query the collection key. Supports optional metadata 'where' filters.
        """
        collection = self.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances'],
            where=where
        )
        return results

# Singleton
chroma_manager = ChromaManager()
