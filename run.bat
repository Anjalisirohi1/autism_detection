@echo off
REM Double-click this file to start. It sets everything up by itself.
setlocal
cd /d "%~dp0"
title Analysis run
color 0F

echo.
echo ===================================================================
echo   Starting up. The first time, this takes a few minutes while it
echo   downloads what it needs. After that it is quick.
echo ===================================================================
echo.

set "PY="
for %%P in (py python python3) do (
  if not defined PY (
    %%P -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY=%%P"
  )
)

if not defined PY (
  echo.
  echo   PYTHON IS NOT INSTALLED ^(or is too old^).
  echo.
  echo   1. Go to   https://www.python.org/downloads/
  echo   2. Click the big yellow "Download Python" button.
  echo   3. Run the installer. IMPORTANT: tick the box that says
  echo      "Add python.exe to PATH" on the first screen.
  echo   4. When it finishes, double-click this file again.
  echo.
  goto :stayopen
)

if not exist ".venv" (
  echo Setting up ^(one time only^)...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo.
    echo   Could not create the working folder. Try moving this whole
    echo   folder to your Desktop and running it again.
    echo.
    goto :stayopen
  )
)

echo Installing what it needs ^(this is the slow part, one time only^)...
call ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
call ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.lock.txt
if errorlevel 1 (
  echo.
  echo   The install did not finish. This is almost always the internet
  echo   connection. Check you are online and double-click this again.
  echo.
  goto :stayopen
)

echo.
call ".venv\Scripts\python.exe" run_all.py %*

:stayopen
echo.
echo ===================================================================
echo   Finished. Read the message above.
echo   This window will stay open so you can read it.
echo ===================================================================
echo.
pause
endlocal
