# Enterprise RAG System: Technical Documentation

**Version:** 1.0.0  
**Author:** Senior Solutions Architect  
**Project:** UNIDO Enterprise Knowledge Management (EKM)

---

## 1. System Architecture
The Enterprise RAG System is a modular, privacy-centric application designed to provide context-aware responses from internal department documents (Finance, HR, Manufacturing). The architecture is specifically optimized for local execution on commodity hardware (8GB RAM).

### High-Level Data Flow
1.  **Query Processing**: Incoming user queries are analyzed for department-specific intent.
2.  **Hybrid Retrieval**: A combination of semantic search (vector-based) and structured record filtering is used to fetch the most relevant context.
3.  **Prompt Augmentation**: The retrieved data is injected into a specialized instruction template.
4.  **Local Generation**: A quantized Large Language Model (LLM) generates a precise answer based **only** on the provided context.

### Optimization Strategy
-   **Quantization**: Uses 4-bit quantized Mistral-7B (~4.1GB VRAM/RAM) to maintain low memory overhead.
-   **Dimensionality Control**: Employs `all-MiniLM-L6-v2` embeddings (384-dimensional) for rapid inference and small indexing footprint.
-   **Batch Handling**: Embedding generation is capped at a batch size of 16 to prevent Out-of-Memory (OOM) errors.

---

## 2. Ingestion Layer
The ingestion layer transforms raw unstructured and semi-structured data into searchable vector representations.

### Document Parsing
-   **Unstructured (PDF)**: Utilizes `PyMuPDF` for high-fidelity text extraction from departmental documents.
-   **Semi-Structured (TXT/CSV)**: A custom `TextParser` identifies `key: value` pairs, preserving the integrity of individual records.

### Chunking & Serialization
-   **Strategy**: `RecursiveCharacterTextSplitter`.
-   **Parameters**:
    -   Chunk Size: 500 characters.
    -   Chunk Overlap: 100 characters.
-   **Logic**: PDF pages are split into overlapping segments. In contrast, structured records are treated as atomic units; one record equals one chunk to prevent loss of relational information.
-   **Normalization**: The `Serializer` converts structured dictionaries into natural language strings to align with the embedding model's training distribution.

---

## 3. Retrieval Strategy
Retrieval is governed by a multi-collection architecture using **ChromaDB** as the persistent vector store.

### Database Setup
-   **Isolation**: Separate collections (`finance_kb`, `hr_kb`, `manufacturing_kb`) ensure department-level data privacy and reduce search noise.
-   **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384d).

### Search Mechanics
-   **Semantic Search**: Cosine similarity is the primary distance metric.
-   **Intent Routing**: The `QueryProcessor` uses keyword-based heuristics to identify relevant departments, restricting search to specific collections.
-   **Hard Identification Filtering**: If a query contains specific Record IDs, the system applies a metadata filter (`where` clause in ChromaDB) to bypass semantic uncertainty.
-   **Parameters**: `top_k=5` (adjustable based on context window limits).

---

## 4. Augmentation & Generation
The generation layer bridges the gap between retrieved context and user-facing answers.

### LLM Integration
-   **Model**: Mistral-7B-Instruct-v0.1 (via Ollama).
-   **Execution**: Local inference through a dedicated HTTP client to minimize SDK overhead.
-   **Parameters**:
    -   `temperature`: 0.1 (deterministic output).
    -   `max_tokens`: 256.
    -   `num_ctx`: 2048.

### Prompt Engineering
The system uses a strictly controlled prompt template:
```text
<s>[INST] {system_prompt}
### Context Documents:
{context}
### Question:
{question}
[/INST]
```
**Constraints**: The system prompt enforces a "No-Info-Disclosure" policy (refusing to answer if context is missing) and mandates departmental citations.

---

## 5. Evaluation Metrics
Performance is measured via an internal benchmarking suite (`evaluation/evaluate.py`).

| Metric | Description | Target |
| :--- | :--- | :--- |
| **Dept. Retrieval Accuracy** | Success rate of the Query Processor in selecting the correct knowledge base. | > 95% |
| **Keyword Relevance** | Verification of ground-truth entities within the LLM's final response. | > 85% |
| **Response Latency** | End-to-end time from query submission to answer display. | < 15s (CPU) |

---

## 6. Setup & Deployment
The system is built for portability and reproducibility.

### Environment Configuration
Key variables in `.env`:
- `OLLAMA_BASE_URL`: URL of the local model server (default: `http://localhost:11434`).
- `CHUNK_SIZE`: 500.
- `CHUNK_OVERLAP`: 100.

### Containerization
The provided `Dockerfile` encapsulates the environment:
1.  **Base**: `python:3.11-slim`.
2.  **Dependencies**: Lightweight build with `pip install --no-cache-dir`.
3.  **Networking**: Port `8501` exposed for the Streamlit UI.
4.  **Host Communication**: Uses `host.docker.internal` to bridge to the Ollama service running on the host OS.

---
*Documentation generated by Enterprise RAG Solutions Architecture Unit.*
