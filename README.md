# Enterprise RAG System for UNIDO

A modular Retrieval-Augmented Generation (RAG) system designed for Finance, HR, and Manufacturing departments. Optimized for local deployment with 8GB RAM constraints.

## Features
- **Multi-Department Retrieval**: Automatically routes queries to Finance, HR, or Manufacturing knowledge bases.
- **Local Privacy**: Uses local LLM (Ollama Mistral-7B) and Vector DB (ChromaDB) — no data leaves the infrastructure.
- **Resource Optimized**: Batch processing, efficient embeddings (`all-MiniLM-L6-v2`), and memory-aware pipeline.
- **Interactive UI**: Streamlit web interface for easy interaction.

## Prerequisites
- **Python 3.10+**
- **Ollama**: Installed and running (`mistral:7b-instruct-q4_0` model pulled).
- **RAM**: Minimum 8GB (16GB recommended for faster generation).

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd enterprise-rag-ekm
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama**:
   ```bash
   ollama serve
   ollama pull mistral:7b-instruct-q4_0
   ```

4. **Ingest Data**:
   Process PDFs/text files and populate the vector database.
   ```bash
   python -m pipeline.ingest_pipeline --stage full
   ```

## Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### CLI Mode
```bash
# Interactive Chat
python -m pipeline.rag_pipeline --interactive

# Single detailed query
python -m pipeline.rag_pipeline --query "What is the budget for 2024?"
```

## Project Structure
- `src/`: Core logic (ingestion, embeddings, vectorstore, retrieval, generation).
- `pipeline/`: Orchestration scripts (`ingest_pipeline.py`, `rag_pipeline.py`).
- `data/`: Raw documents (PDFs/TXTs) and processed chunks.
- `tests/`: Unit tests.
- `evaluation/`: Evaluation metrics and reports.

## Architecture
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design decisions.
