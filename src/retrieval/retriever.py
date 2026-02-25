from typing import List, Dict, Any
from src.retrieval.query_processor import query_processor
from src.vectorstore.chroma_manager import chroma_manager
from src.embeddings.generator import generate_embeddings
from src.utils.logger import logger
from src.utils.helpers import timer

class Retriever:
    @timer
    def retrieve(self, query: str, departments: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents from specified departments.
        Supports exact ID filtering if an ID is detected in the query.
        """
        # 1. Generate query embedding
        query_doc = {"text": query}
        embeddings = generate_embeddings([query_doc], batch_size=1)
        query_embedding = embeddings[0]
        
        # 2. Check for specific Record IDs (Keyword-style hard filters)
        record_ids = query_processor.extract_record_ids(query)
        where_filter = None
        
        if record_ids:
            if len(record_ids) == 1:
                where_filter = {"record_id": record_ids[0]}
            else:
                # Use $in for multiple IDs
                where_filter = {"record_id": {"$in": record_ids}}
            
            logger.info(f"Applying hard ID filter for retrieval: {record_ids}")
            # Ensure we retrieve enough results to cover all requested IDs
            top_k = max(top_k, len(record_ids))
        
        all_results = []
        
        # 3. Query each department
        for dept in departments:
            collection_name = f"{dept}_kb"
            try:
                # Query with optional where filter
                results = chroma_manager.query(
                    collection_name, 
                    query_embedding, 
                    n_results=top_k,
                    where=where_filter
                )
                
                if not results['ids'] or not results['ids'][0]:
                    continue
                    
                ids = results['ids'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]
                
                for i in range(len(ids)):
                    all_results.append({
                        "id": ids[i],
                        "score": distances[i],
                        "metadata": metadatas[i],
                        "text": documents[i],
                        "department": dept
                    })
                    
            except Exception as e:
                logger.error(f"Error querying {dept}: {e}")
                
        # 4. Sort and return results
        all_results.sort(key=lambda x: x["score"])
        return all_results[:top_k * 2]

retriever = Retriever()
