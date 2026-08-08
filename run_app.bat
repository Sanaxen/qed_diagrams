@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
.venv\Scripts\python.exe -c "import streamlit, matplotlib, networkx, numpy, feynman, pyfeyn2, PIL" >nul 2>&1
if errorlevel 1 (
  echo Installing missing Python libraries...
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installation failed. Check the error above.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
if errorlevel 1 (
  echo.
  echo The app stopped because of an error. Check the message above.
  pause
)
