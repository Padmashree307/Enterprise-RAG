"""
Prompt builder for RAG pipeline.
Constructs prompts with retrieved context for the LLM.
"""

SYSTEM_PROMPT = """You are a knowledgeable assistant for UNIDO (United Nations Industrial Development Organization).
You answer questions based ONLY on the provided context documents.
If the context does not contain enough information to answer the question, say so explicitly.
Always cite which department (Finance, HR, or Manufacturing) your answer is based on.
Reproduce all numerical values, amounts, IDs, dates, and codes EXACTLY as they appear in the context. Do NOT reformat, round, or add/remove any digits from numbers.
Be concise and factual."""

RAG_PROMPT_TEMPLATE = """<s>[INST] {system}

### Context Documents:
{context}

### Question:
{question}

### Instructions:
- Answer based ONLY on the context above.
- If the answer is not in the context, say "I don't have enough information to answer this."
- Cite the department source when possible.
- Reproduce ALL numbers, amounts, IDs, and dates EXACTLY as they appear in the context. Do NOT reformat or alter any numerical values.
- Be concise. [/INST]"""


def build_rag_prompt(question: str, retrieved_docs: list) -> str:
    """
    Build a complete RAG prompt with retrieved context.
    
    Args:
        question: The user's question.
        retrieved_docs: List of retrieved document dicts with 'text', 'department', 'score'.
        
    Returns:
        str: The formatted prompt ready for the LLM.
    """
    # Format context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        dept = doc.get("department", "Unknown")
        text = doc.get("text", "")
        score = doc.get("score", 0.0)
        context_parts.append(f"[Document {i+1} | Department: {dept} | Relevance: {1-score:.2f}]\n{text}")
    
    context_str = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
    
    prompt = RAG_PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context_str,
        question=question
    )
    
    return prompt
