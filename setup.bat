@echo off
cd /d "%~dp0"

if not exist venv (
    echo Dang tao moi truong ao Python...
    py -3.12 -m venv venv
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================
echo  Cai dat xong. Chay run.bat de khoi dong chuong trinh.
echo ================================================
pause
