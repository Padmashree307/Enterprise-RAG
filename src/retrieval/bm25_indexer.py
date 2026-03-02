"""
BM25 Keyword Search Indexer for Hybrid Retrieval.
Builds per-department BM25 indices from existing chunked data.
Lazy-loads indices on first query to minimize startup cost.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.config.settings import settings
from src.utils.logger import logger
from src.utils.helpers import timer


class BM25Indexer:
    def __init__(self):
        self._indices: Dict[str, BM25Okapi] = {}
        self._corpora: Dict[str, List[Dict[str, Any]]] = {}
        self.chunks_dir = settings.CHROMA_DB_PATH.parent.parent / "processed" / "chunks"
        logger.info(f"BM25Indexer initialized. Chunks dir: {self.chunks_dir}")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer. Lowercases all tokens."""
        return re.findall(r'\w+', text.lower())

    def _load_department(self, department: str) -> bool:
        """
        Loads chunked documents for a department and builds the BM25 index.
        Returns True if successful, False otherwise.
        """
        if department in self._indices:
            return True  # Already loaded

        chunked_path = self.chunks_dir / f"{department}_chunked.json"
        if not chunked_path.exists():
            logger.warning(f"BM25: No chunked data found at {chunked_path}")
            return False

        try:
            with open(chunked_path, "r", encoding="utf-8") as f:
                docs = json.load(f)

            if not docs:
                logger.warning(f"BM25: Empty corpus for {department}")
                return False

            # Tokenize all documents
            tokenized_corpus = [self._tokenize(doc["text"]) for doc in docs]

            # Build BM25 index
            self._indices[department] = BM25Okapi(tokenized_corpus)
            self._corpora[department] = docs

            logger.info(f"BM25: Built index for '{department}' with {len(docs)} documents")
            return True

        except Exception as e:
            logger.error(f"BM25: Failed to build index for {department}: {e}")
            return False

    @timer
    def search(self, department: str, query: str, top_k: int = 10,
               record_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search a department's BM25 index.

        Args:
            department: Department key (e.g., 'finance').
            query: The user's query string.
            top_k: Number of top results to return.
            record_ids: Optional list of record IDs to filter results.

        Returns:
            List of dicts with 'text', 'metadata', 'bm25_score', 'department'.
        """
        if not self._load_department(department):
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        index = self._indices[department]
        corpus = self._corpora[department]

        # Get BM25 scores for all documents
        scores = index.get_scores(tokenized_query)

        # Pair scores with document indices and sort descending
        scored_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scored_docs:
            if score <= 0:
                continue  # Skip zero-relevance documents

            doc = corpus[doc_idx]

            # Apply record ID filter if specified
            if record_ids:
                doc_record_id = doc.get("metadata", {}).get("record_id", "")
                if isinstance(doc_record_id, str):
                    doc_record_id = doc_record_id.upper()
                if doc_record_id not in record_ids:
                    continue

            results.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "bm25_score": float(score),
                "department": department
            })

            if len(results) >= top_k:
                break

        return results


# Singleton
bm25_indexer = BM25Indexer()
