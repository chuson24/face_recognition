@echo off
cd /d "%~dp0"

REM Chay bang Python 3.11 global (co onnxruntime-gpu + CUDA) thay vi venv (CPU).
echo Dang khoi dong (GPU)... Mo trinh duyet vao http://localhost:8000
start http://localhost:8000
py -3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
