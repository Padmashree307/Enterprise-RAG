"""Phase 6 Verification: Test retrieval pipeline."""
from src.retrieval.query_processor import query_processor
from src.retrieval.retriever import retriever

def run_test(label, query, top_k=3):
    print(f"\n=== {label} ===")
    print(f"Query: {query}")
    depts = query_processor.detect_departments(query)
    print(f"Detected departments: {depts}")
    results = retriever.retrieve(query, depts, top_k=top_k)
    for i, r in enumerate(results[:top_k]):
        dept = r["department"]
        score = r["score"]
        text_preview = r["text"][:120].replace("\n", " ")
        print(f"  {i+1}. [{dept}] score={score:.4f} | {text_preview}...")
    if not results:
        print("  (No results found)")
    return results

if __name__ == "__main__":
    run_test("TEST 1: Finance Query",
             "What is the total budget expenditure?")

    run_test("TEST 2: HR Query",
             "How many employees were recruited last year?")

    run_test("TEST 3: Manufacturing Query",
             "What is the production output of the factory?")

    run_test("TEST 4: General Query (all depts)",
             "What are the key challenges?")

    print("\n=== ALL TESTS COMPLETE ===")
