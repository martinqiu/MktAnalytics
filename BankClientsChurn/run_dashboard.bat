@echo off
cd /d "%~dp0"
echo ================================================
echo   Bank Client Churn Dashboard
echo ================================================
echo.
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
streamlit run bank_dashboard.py
pause
