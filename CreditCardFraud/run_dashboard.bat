@echo off
cd /d "%~dp0"
echo ================================================
echo   Credit Card Fraud Detection Dashboard
echo ================================================
echo.
echo Open your browser at: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.
streamlit run fraud_dashboard.py
pause
