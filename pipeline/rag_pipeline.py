"""
Main RAG Pipeline: Ties together retrieval and generation.
"""
import argparse
from src.retrieval.query_processor import query_processor
from src.retrieval.retriever import retriever
from src.generation.llm_client import llm_client
from src.generation.prompt_builder import build_rag_prompt
from src.utils.logger import logger
from src.utils.helpers import timer, log_memory_usage


@timer
def query_rag(question: str, top_k: int = 5, departments: list = None) -> dict:
    """
    Full RAG pipeline: detect departments -> retrieve -> generate.
    
    Args:
        question: The user's question.
        top_k: Number of documents to retrieve per department.
        departments: Optional list of departments to search. Auto-detected if None.
        
    Returns:
        dict with 'answer', 'sources', 'departments'.
    """
    # 1. Preprocess query
    clean_query = query_processor.preprocess_query(question)
    
    # 2. Detect departments (or use provided)
    if not departments:
        departments = query_processor.detect_departments(clean_query)
        logger.info(f"Auto-detected departments: {departments}")
    else:
        logger.info(f"Hard-filtering for department: {departments}")
    
    # 3. Retrieve relevant documents
    retrieved_docs = retriever.retrieve(clean_query, departments, top_k=top_k)
    logger.info(f"Retrieved {len(retrieved_docs)} documents")
    
    if not retrieved_docs:
        return {
            "answer": "No relevant documents found for your query.",
            "sources": [],
            "departments": departments
        }
    
    # 4. Build prompt (use ID formatting only if record IDs were in the query)
    record_ids = query_processor.extract_record_ids(clean_query)
    prompt = build_rag_prompt(clean_query, retrieved_docs, has_record_ids=bool(record_ids))
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    # 5. Generate answer
    answer = llm_client.generate(prompt, temperature=0.1, max_tokens=512)
    
    # 6. Compile sources
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "department": doc["department"],
            "source_file": doc["metadata"].get("source_file", "unknown"),
            "score": round(doc["score"], 4)
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "departments": departments
    }


def query_rag_stream(question: str, top_k: int = 5, departments: list = None):
    """
    Streaming RAG pipeline.
    Returns: (generator, sources, departments)
    """
    # 1. Preprocess
    clean_query = query_processor.preprocess_query(question)
    
    # 2. Detect departments
    if not departments:
        departments = query_processor.detect_departments(clean_query)
        logger.info(f"Auto-detected departments: {departments}")
    else:
        logger.info(f"Hard-filtering for department: {departments}")
        
    # 3. Retrieve
    retrieved_docs = retriever.retrieve(clean_query, departments, top_k=top_k)
    
    # Compile sources
    sources = []
    for doc in retrieved_docs:
        sources.append({
            "department": doc["department"],
            "source_file": doc["metadata"].get("source_file", "unknown"),
            "score": round(doc["score"], 4)
        })

    if not retrieved_docs:
        def empty_gen(): yield "No relevant documents found."
        return empty_gen(), sources, departments
    
    # 4. Build prompt (use ID formatting only if record IDs were in the query)
    record_ids = query_processor.extract_record_ids(clean_query)
    prompt = build_rag_prompt(clean_query, retrieved_docs, has_record_ids=bool(record_ids))
    
    # 5. Generate stream
    stream = llm_client.generate_stream(prompt, temperature=0.1, max_tokens=512)
    
    return stream, sources, departments


def interactive_mode():
    """Run the RAG system in interactive CLI mode."""
    print("=" * 60)
    print("  UNIDO Enterprise RAG System")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    # Check Ollama availability
    if not llm_client.is_available():
        print("\n[WARNING] Ollama is not running. Start it with 'ollama serve'.")
        print("Retrieval will still work, but answers won't be generated.\n")
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
                
            print("\nSearching...")
            log_memory_usage("query_start")
            
            result = query_rag(question)
            
            print("\n" + "-" * 50)
            print("ANSWER:")
            print("-" * 50)
            print(result["answer"])
            print("\n" + "-" * 50)
            print("SOURCES:")
            for s in result["sources"][:5]:
                print(f"  - [{s['department']}] {s['source_file']} (relevance: {1-s['score']:.2f})")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n[Error] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UNIDO RAG Query System")
    parser.add_argument("--query", "-q", type=str, help="Single query mode")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive CLI mode")
    args = parser.parse_args()
    
    if args.query:
        result = query_rag(args.query, top_k=args.top_k)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for s in result["sources"][:5]:
            print(f"  - [{s['department']}] {s['source_file']}")
    else:
        interactive_mode()
