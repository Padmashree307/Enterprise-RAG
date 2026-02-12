from typing import List, Dict, Any
from src.vectorstore.chroma_manager import chroma_manager
from src.embeddings.generator import generate_embeddings
from src.utils.logger import logger
from src.utils.helpers import timer

class Retriever:
    @timer
    def retrieve(self, query: str, departments: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant documents from specified departments.
        
        Args:
            query (str): The user query.
            departments (List[str]): List of departments to search.
            top_k (int): Number of results to return per department.
            
        Returns:
            List[Dict]: Combined list of results, sorted by distance (score).
        """
        # 1. Generate query embedding
        # Note: generate_embeddings expects a list of dicts with 'text'.
        # We wrap query in a dummy doc.
        
        # Optimization: We can just use model.encode directly if we exposed it, 
        # but generate_embeddings handles batching/logging. 
        # Let's make a helper or just use generate_embeddings.
        query_doc = {"text": query}
        embeddings = generate_embeddings([query_doc], batch_size=1)
        query_embedding = embeddings[0]
        
        all_results = []
        
        # 2. Query each department
        for dept in departments:
            collection_name = f"{dept}_kb"
            try:
                # Get more than needed to filter/rerank if we wanted
                results = chroma_manager.query(collection_name, query_embedding, n_results=top_k)
                
                # Unwrap output structure from Chroma
                # results is dict: {'ids': [[...]], 'distances': [[...]], 'metadatas': [[...]], 'documents': [[...]]}
                if not results['ids']:
                    continue
                    
                ids = results['ids'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0]
                documents = results['documents'][0]
                
                for i in range(len(ids)):
                    all_results.append({
                        "id": ids[i],
                        "score": distances[i], # Lower is better for cosine distance? 
                        # Chroma uses L2 by default? Or Cosine? 
                        # We set "hnsw:space": "cosine" in manager.
                        # For cosine distance, 0 is identical, 1 is orthogonal, 2 is opposite.
                        "metadata": metadatas[i],
                        "text": documents[i],
                        "department": dept
                    })
                    
            except Exception as e:
                logger.error(f"Error querying {dept}: {e}")
                
        # 3. Sort merged results by score (ascending for distance)
        all_results.sort(key=lambda x: x["score"])
        
        # Return top K overall? Or per dept?
        # Requirement says "Modular". Usually we show top results overall.
        # But if we search Multiple depts, we might return top_k * num_depts or top_k total.
        # Let's return top_k total for now to be concise, or allow caller to decide.
        # We'll return full list mostly, maybe slice top 2*k.
        
        return all_results[:top_k * 2] # Return substantial context

retriever = Retriever()
