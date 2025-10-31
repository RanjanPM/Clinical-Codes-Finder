@echo off
REM Demo wrapper for Windows Command Prompt
REM Allows: lookup diabetes
REM Instead of: python main.py "diabetes"

if "%*"=="" (
    echo Usage: lookup ^<clinical term^>
    echo.
    echo Examples:
    echo   lookup diabetes
    echo   lookup blood sugar test
    echo   lookup metformin 500 mg
    exit /b 1
)

REM Run main.py with all arguments
python main.py %*
