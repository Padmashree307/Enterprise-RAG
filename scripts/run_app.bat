@echo off
echo Starting Enterprise RAG System...

:: Check if Ollama is running
curl -s http://localhost:11434/api/tags > NUL
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not accessible!
    echo Please run 'ollama serve' in another terminal.
    pause
)

:: Run Streamlit App
echo Launching Web UI...
streamlit run app.py
pause
