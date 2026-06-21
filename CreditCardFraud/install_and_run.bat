@echo off
cd /d "%~dp0"
echo ================================================
echo   Credit Card Fraud Detection Dashboard
echo   Setup: Installing required libraries
echo ================================================
echo.

:: Install all required libraries from requirements.txt
echo Installing Python libraries (this may take a few minutes)...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Installation failed. Make sure Python and pip are installed.
    echo Download Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo ================================================
echo   All libraries installed successfully!
echo   Launching dashboard...
echo ================================================
echo.

call "%~dp0run_dashboard.bat"
