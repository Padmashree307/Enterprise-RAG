import json
import argparse
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.rag_pipeline import query_rag
from src.utils.logger import logger

def evaluate(dataset_path: str = "evaluation/eval_dataset.json"):
    logger.info(f"Starting evaluation with dataset: {dataset_path}")
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    results = []
    total_dept_hits = 0
    total_keyword_hits = 0
    total_queries = len(dataset)
    total_time = 0
    
    for i, item in enumerate(dataset):
        query = item["question"]
        truth_dept = item["ground_truth_dept"]
        keywords = item["expected_keywords"]
        
        logger.info(f"evaluating query {i+1}/{total_queries}: {query}")
        
        # Run RAG
        start_time = time.time()
        # We assume Ollama might be slow, so we just start.
        # But for evaluation, we might want to skip generation if valid retrieval is the goal?
        # The prompt says "Run evaluation". Better to run full pipeline.
        
        # Note: running full generation for 10 queries might take 10 * 30s = 5 mins.
        # Let's do it.
        
        try:
            rag_result = query_rag(query, top_k=3)
            duration = time.time() - start_time
            total_time += duration
            
            # 1. Department Accuracy
            # Check if retrieved docs contain the ground truth department
            retrieved_depts = {source['department'] for source in rag_result['sources']}
            dept_hit = truth_dept in retrieved_depts
            if dept_hit:
                total_dept_hits += 1
                
            # 2. Answer Relevance (Keyword Match)
            answer = rag_result["answer"].lower()
            keyword_hit = any(k.lower() in answer for k in keywords)
            if keyword_hit:
                total_keyword_hits += 1
                
            results.append({
                "query": query,
                "truth_dept": truth_dept,
                "retrieved_depts": list(retrieved_depts),
                "dept_match": dept_hit,
                "keyword_match": keyword_hit,
                "duration": duration
            })
            
        except Exception as e:
            logger.error(f"Error evaluating query '{query}': {e}")
            results.append({
                "query": query,
                "error": str(e)
            })
            
    # Report
    dept_accuracy = total_dept_hits / total_queries * 100
    keyword_accuracy = total_keyword_hits / total_queries * 100
    avg_latency = total_time / total_queries
    
    report = {
        "metrics": {
            "department_retrieval_accuracy": f"{dept_accuracy:.1f}%",
            "answer_keyword_relevance": f"{keyword_accuracy:.1f}%",
            "average_latency_seconds": f"{avg_latency:.1f}"
        },
        "details": results
    }
    
    # Save report
    output_path = Path("evaluation/report.json")
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\n=== Evaluation Report ===")
    print(json.dumps(report["metrics"], indent=2))
    logger.info(f"Evaluation complete. Report saved to {output_path}")

if __name__ == "__main__":
    evaluate()
