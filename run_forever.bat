@echo off
:: ================================================================
::  run_forever.bat
::  Auto-restart wrapper for Trading Firm OS (Windows).
::
::  Usage:
::    run_forever.bat
::    run_forever.bat --mode paper --capital 10000
::
::  Behaviour:
::    - Activates venv if present
::    - Restarts on crash (exit code != 0)
::    - Exits cleanly if main.py returns 0
::    - Stops after MAX_RESTARTS consecutive crashes
:: ================================================================

setlocal enabledelayedexpansion

set MAX_RESTARTS=10
set RESTART_DELAY=30
set RESTART_COUNT=0
set LOG_FILE=logs\restart.log
set ARGS=%*
if "%ARGS%"=="" set ARGS=--mode paper

if not exist logs mkdir logs

:: Activate virtual environment if available
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo [run_forever] venv activated
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    echo [run_forever] .venv activated
)

echo ======================================== >> %LOG_FILE%
echo run_forever.bat started: %DATE% %TIME%  >> %LOG_FILE%
echo Args: %ARGS%                            >> %LOG_FILE%
echo ======================================== >> %LOG_FILE%

:loop
echo [run_forever] Starting main.py (restart #!RESTART_COUNT!)  %DATE% %TIME%
echo %DATE% %TIME% -- START (restart #!RESTART_COUNT!)          >> %LOG_FILE%

python main.py %ARGS%
set EXIT_CODE=%ERRORLEVEL%

echo %DATE% %TIME% -- EXIT code=%EXIT_CODE%                     >> %LOG_FILE%

:: Clean exit
if %EXIT_CODE%==0 (
    echo [run_forever] Clean exit (code 0). Goodbye.
    echo %DATE% %TIME% -- Clean exit.                           >> %LOG_FILE%
    exit /b 0
)

set /a RESTART_COUNT+=1

if !RESTART_COUNT! geq %MAX_RESTARTS% (
    echo [run_forever] Max restarts (%MAX_RESTARTS%) reached. Giving up.
    echo %DATE% %TIME% -- ABORT after %MAX_RESTARTS% restarts.  >> %LOG_FILE%
    exit /b 1
)

echo [run_forever] Crash (code %EXIT_CODE%). Restarting in %RESTART_DELAY%s... ^
 (!RESTART_COUNT!/%MAX_RESTARTS%)
echo %DATE% %TIME% -- Restart in %RESTART_DELAY%s              >> %LOG_FILE%

timeout /t %RESTART_DELAY% /nobreak >nul
goto loop
