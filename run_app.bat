@echo off
cd /d "%~dp0"

rem Graphviz is optional, but required when the Graphviz layout checkbox is used.
where neato >nul 2>&1
if errorlevel 1 (
  echo Graphviz was not found. It is required for the optional Graphviz layout.
  where winget >nul 2>&1
  if errorlevel 1 (
    echo winget is not available, so Graphviz cannot be installed automatically.
    echo Install it manually from https://graphviz.org/download/ and restart this script.
  ) else (
    choice /C YN /N /M "Install Graphviz now with winget? [Y/N] "
    if errorlevel 2 goto graphviz_done
    winget install --id Graphviz.Graphviz -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
      echo Graphviz installation failed. The app will use the QED layout instead.
    ) else (
      rem Make a new installation visible without requiring this script to restart.
      if exist "%ProgramFiles%\Graphviz\bin\neato.exe" set "PATH=%ProgramFiles%\Graphviz\bin;%PATH%"
      if exist "%LOCALAPPDATA%\Programs\Graphviz\bin\neato.exe" set "PATH=%LOCALAPPDATA%\Programs\Graphviz\bin;%PATH%"
      where neato >nul 2>&1
      if errorlevel 1 echo Graphviz was installed, but a Windows sign-out or terminal restart may be required.
    )
  )
)
:graphviz_done

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)
.venv\Scripts\python.exe -c "import streamlit, matplotlib, networkx, numpy, feynman, pyfeyn2, PIL, pydot" >nul 2>&1
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
