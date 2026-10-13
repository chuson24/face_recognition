@echo off
cd /d "%~dp0"

if not exist venv (
    echo Chua cai dat. Vui long chay setup.bat truoc.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Dang khoi dong... Mo trinh duyet vao http://localhost:8000
start http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
