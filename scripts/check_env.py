import sys
import shutil
import importlib
import requests
from pathlib import Path

# ANSI colors for Windows terminal (modern Windows supports this)
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def check(name, status, details=""):
    symbol = f"{GREEN}✓{RESET}" if status else f"{RED}✗{RESET}"
    print(f"{symbol} {name:<20} {details}")

def verify_python():
    v = sys.version_info
    valid = v.major == 3 and v.minor >= 9
    check("Python Version", valid, f"{v.major}.{v.minor}.{v.micro}")

def verify_imports():
    packages = [
        "langchain_community",
        "langchain_core",
        "chromadb",
        "sentence_transformers",
        "fitz",  # PyMuPDF
        "dotenv",
        "psutil"
    ]
    print("\nChecking Libraries:")
    for pkg in packages:
        try:
            importlib.import_module(pkg)
            check(pkg, True, "Installed")
        except ImportError:
            check(pkg, False, "Missing - run 'pip install -r requirements.txt'")

def verify_ollama():
    print("\nChecking Ollama:")
    try:
        # Check if running
        res = requests.get("http://localhost:11434", timeout=2)
        if res.status_code == 200:
            check("Ollama Service", True, "Running at localhost:11434")
        else:
            check("Ollama Service", False, f"Responded with {res.status_code}")
            return
            
        # Check for model using 'ps' or 'list' (API varies, using simple generate check is robust)
        # Actually, let's just list tags
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        if res.status_code == 200:
            models = [m['name'] for m in res.json().get('models', [])]
            target = "mistral:7b-instruct-q4_0"
            # Normalize model names (remove :latest if present implied)
            target_found = any(target in m for m in models)
            if target_found:
                check("Mistral Model", True, "Found mistral:7b-instruct-q4_0")
            else:
                check("Mistral Model", False, f"Missing. Found: {', '.join(models[:3])}...")
                print(f"  {RED}Run: ollama pull mistral:7b-instruct-q4_0{RESET}")
    except Exception as e:
        check("Ollama Service", False, "Not reachable (is it running in WSL?)")
        print(f"  Error: {e}")

def verify_chroma_persistence():
    print("\nChecking Storage:")
    db_path = Path("./data/vector_db/chroma_db")
    if db_path.exists():
        check("ChromaDB Path", True, str(db_path))
    else:
        check("ChromaDB Path", False, "Not created yet (normal for fresh install)")

if __name__ == "__main__":
    print(f"Environment Verification Script\n{'='*30}")
    verify_python()
    verify_imports()
    verify_ollama()
    verify_chroma_persistence()
    print(f"\n{'='*30}\nDone.")
