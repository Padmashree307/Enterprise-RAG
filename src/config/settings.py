import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed. Using existing environment variables.")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

@dataclass
class DepartmentConfig:
    name: str
    raw_path: Path
    file_types: Dict[str, str]  # e.g. {"pdf": "filename.pdf", "txt": "filename.txt"}

@dataclass
class Settings:
    # LLM Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mistral:7b-instruct-q4_0")
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Database Settings
    CHROMA_DB_PATH: Path = Path(os.getenv("CHROMA_DB_PATH", str(VECTOR_DB_DIR / "chroma_db")))
    
    # Chunking Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))
    
    # Departments
    DEPARTMENTS: Dict[str, DepartmentConfig] = field(default_factory=lambda: {
        "finance": DepartmentConfig(
            name="Finance",
            raw_path=RAW_DIR / "financial",
            file_types={
                "pdf": "UNIDO Finance.pdf",
                "txt": "Finance_and_Accounting.txt"
            }
        ),
        "hr": DepartmentConfig(
            name="HR",
            raw_path=RAW_DIR / "hrm",
            file_types={
                "pdf": "UNIDO HR.pdf",
                "txt": "Human_Resource_Management.txt"
            }
        ),
        "manufacturing": DepartmentConfig(
            name="Manufacturing",
            raw_path=RAW_DIR / "manufacture",
            file_types={
                "pdf": "UNIDO Manufacturing.pdf",
                "txt": "Manufacturing_and_Production.txt"
            }
        )
    })

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

# Singleton instance
settings = Settings()

# Ensure directories exist
for path in [PROCESSED_DIR, VECTOR_DB_DIR, PROCESSED_DIR / "chunks"]:
    path.mkdir(parents=True, exist_ok=True)
