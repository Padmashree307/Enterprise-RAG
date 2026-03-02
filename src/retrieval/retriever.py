from typing import List, Dict, Any
from src.retrieval.query_processor import query_processor
from src.retrieval.bm25_indexer import bm25_indexer
from src.vectorstore.chroma_manager import chroma_manager
from src.embeddings.generator import generate_embeddings
from src.utils.logger import logger
from src.utils.helpers import timer


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion (RRF).
    RRF score = sum( 1 / (k + rank) ) for each list the document appears in.

    Args:
        vector_results: Results from ChromaDB vector search (sorted by distance ASC).
        bm25_results: Results from BM25 keyword search (sorted by score DESC).
        k: RRF constant (default 60, standard value from the original paper).

    Returns:
        List of merged results sorted by RRF score (descending).
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Score vector results (already sorted by distance, rank 1 = best)
    for rank, doc in enumerate(vector_results):
        doc_key = doc.get("id", doc["text"][:80])
        fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
        if doc_key not in doc_map:
            doc_map[doc_key] = doc

    # Score BM25 results (already sorted by bm25_score descending, rank 1 = best)
    for rank, doc in enumerate(bm25_results):
        doc_key = doc.get("metadata", {}).get("record_id", doc["text"][:80])
        fused_scores[doc_key] = fused_scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)
        if doc_key not in doc_map:
            # Convert BM25 result format to match vector result format
            doc_map[doc_key] = {
                "id": doc_key,
                "score": 0.0,  # No vector distance for BM25-only results
                "metadata": doc.get("metadata", {}),
                "text": doc["text"],
                "department": doc["department"]
            }

    # Sort by fused score descending (higher = more relevant)
    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    merged = []
    for doc_key in sorted_keys:
        result = doc_map[doc_key].copy()
        result["rrf_score"] = round(fused_scores[doc_key], 6)
        merged.append(result)

    return merged


class Retriever:
    @timer
    def retrieve(self, query: str, departments: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: combines ChromaDB vector search with BM25 keyword search
        using Reciprocal Rank Fusion (RRF).
        Supports exact ID filtering if an ID is detected in the query.
        """
        # 1. Generate query embedding
        query_doc = {"text": query}
        embeddings = generate_embeddings([query_doc], batch_size=1)
        query_embedding = embeddings[0]

        # 2. Check for specific Record IDs (hard filters)
        record_ids = query_processor.extract_record_ids(query)
        where_filter = None

        if record_ids:
            if len(record_ids) == 1:
                where_filter = {"record_id": record_ids[0]}
            else:
                where_filter = {"record_id": {"$in": record_ids}}

            logger.info(f"Applying hard ID filter for retrieval: {record_ids}")
            top_k = max(top_k, len(record_ids))

        # We fetch more candidates from each engine for better fusion
        fetch_k = top_k * 2

        all_vector_results = []
        all_bm25_results = []

        # 3. Query each department with BOTH engines
        for dept in departments:
            # --- Vector Search (ChromaDB) ---
            collection_name = f"{dept}_kb"
            try:
                results = chroma_manager.query(
                    collection_name,
                    query_embedding,
                    n_results=fetch_k,
                    where=where_filter
                )

                if results['ids'] and results['ids'][0]:
                    ids = results['ids'][0]
                    distances = results['distances'][0]
                    metadatas = results['metadatas'][0]
                    documents = results['documents'][0]

                    for i in range(len(ids)):
                        all_vector_results.append({
                            "id": ids[i],
                            "score": distances[i],
                            "metadata": metadatas[i],
                            "text": documents[i],
                            "department": dept
                        })

            except Exception as e:
                logger.error(f"Vector search error for {dept}: {e}")

            # --- BM25 Keyword Search ---
            try:
                bm25_results = bm25_indexer.search(
                    department=dept,
                    query=query,
                    top_k=fetch_k,
                    record_ids=record_ids if record_ids else None
                )
                all_bm25_results.extend(bm25_results)
            except Exception as e:
                logger.error(f"BM25 search error for {dept}: {e}")

        # 4. Sort vector results by distance (ascending = best first)
        all_vector_results.sort(key=lambda x: x["score"])

        logger.info(
            f"Hybrid search: {len(all_vector_results)} vector results, "
            f"{len(all_bm25_results)} BM25 results"
        )

        # 5. Fuse results using RRF
        if all_vector_results and all_bm25_results:
            fused = reciprocal_rank_fusion(all_vector_results, all_bm25_results)
            logger.info(f"RRF fusion produced {len(fused)} merged results")
            final_results = fused[:top_k * 2]
        elif all_vector_results:
            logger.info("No BM25 results, falling back to vector-only results")
            final_results = all_vector_results[:top_k * 2]
        elif all_bm25_results:
            logger.info("No vector results, falling back to BM25-only results")
            # Convert BM25 results to standard format
            converted = []
            for doc in all_bm25_results:
                converted.append({
                    "id": doc.get("metadata", {}).get("record_id", "unknown"),
                    "score": 0.0,
                    "metadata": doc.get("metadata", {}),
                    "text": doc["text"],
                    "department": doc["department"]
                })
            final_results = converted[:top_k * 2]
        else:
            return []

        # 6. Post-filter: when record IDs are in the query, ONLY keep
        #    chunks whose metadata record_id matches one of the requested IDs.
        #    This prevents unrelated PDF chunks from leaking into the response.
        if record_ids:
            id_set = set(rid.upper() for rid in record_ids)
            filtered = [
                doc for doc in final_results
                if doc.get("metadata", {}).get("record_id", "").upper() in id_set
            ]
            if filtered:
                logger.info(
                    f"Post-filter kept {len(filtered)}/{len(final_results)} "
                    f"results matching IDs {record_ids}"
                )
                return filtered
            else:
                # Fallback: if nothing matched metadata, return original
                # (safety net in case metadata field differs)
                logger.warning(
                    f"Post-filter found 0 results for IDs {record_ids}; "
                    f"returning unfiltered results"
                )

        return final_results


retriever = Retriever()
