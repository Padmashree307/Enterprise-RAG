# System Architecture

## Overview
The Enterprise RAG System is a modular application designed to ingest department-specific documents (Finance, HR, Manufacturing) and provide accurate, context-aware answers using a local LLM.

## Components

### 1. Data Ingestion Layer (`src/ingestion`)
- **PDF Extractor**: Users `PyMuPDF` to extract text from PDFs.
- **Text Parser**: Custom parser for structured text records (`key: value`).
- **Chunker**: `RecursiveCharacterTextSplitter` (500 chars, 100 overlap). Preserves structured records as single chunks.
- **Serializer**: Converts structured dicts to natural language strings for embedding.

### 2. Vector Database (`src/vectorstore`)
- **ChromaDB**: Local persistent vector store.
- **Collections**: Separate collections for `finance_kb`, `hr_kb`, `manufacturing_kb`.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384d). Batched generation (size 16) for RAM efficiency.

### 3. Retrieval Layer (`src/retrieval`)
- **Query Processor**: Detects intent/department using keyword matching.
- **Retriever**: Queries the relevant ChromaDB collection(s). Returns top-k results sorted by cosine similarity.

### 4. Generation Layer (`src/generation`)
- **LLM Client**: Connects to local Ollama instance (`mistral:7b-instruct-q4_0`).
- **Prompt Builder**: Constructs Mistral-compatible prompts with retrieved context.
- **Pipeline**: Coordinates the retrieve-then-generate flow.

## Data Flow
1. **User Query** -> **Query Processor** (Detects Dept)
2. **Retriever** -> **ChromaDB** (Sematic Search)
3. **Relevant Chunks** -> **Prompt Builder** (Context Window)
4. **Prompt** -> **Ollama LLM** (Generation)
5. **Answer** -> **UI/CLI**

## Design Decisions for 8GB RAM
- **Quantized LLM**: Uses 4-bit quantized Mistral (~4.1GB) to fit in memory.
- **Small Embeddings**: `all-MiniLM-L6-v2` is very lightweight (~90MB).
- **Batching**: Embedding generation uses small batch size (16) to avoid OOM.
- **Lazy Loading**: LLM is loaded only when needed (though Ollama keeps it in memory).
